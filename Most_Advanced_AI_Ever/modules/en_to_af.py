


"""
en_to_af.py - patched

English ↔ Afrikaans translator helper (auto-safe max_length).
- Adds a small backward-compatibility shim for huggingface_hub.cached_download -> hf_hub_download.
- Fixes device selection bug.
- Enables tokenizer-aware safe max_length scaling (uses existing _safe_max_for_chunk).
- Lazy-loads Hugging Face pipeline (Helsinki-NLP/opus-mt-en-af and opus-mt-af-en).
- Preserves newline runs and spacing.

Drop this module into your project and import translate / translate_batch as before.
"""
from typing import List, Optional, Dict, Any
import logging
import re
import math

_LOG = logging.getLogger(__name__)

# === Backward compatibility shim for huggingface_hub.cached_download ===
# Place this at top so any downstream import that expects cached_download works.
try:
    # If cached_download exists nothing to do
    from huggingface_hub import cached_download  # type: ignore
except Exception:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        hf_hub_download = None

    def cached_download(*args, **kwargs):
        """
        Compatibility shim that delegates to hf_hub_download.
        Attempts to support common call patterns of the older cached_download.
        """
        if hf_hub_download is None:
            raise ImportError("hf_hub_download not available to emulate cached_download")
        # common older call pattern: cached_download(repo_id, filename, ...)
        if len(args) >= 2 and 'repo_id' not in kwargs and 'filename' not in kwargs:
            repo_id, filename, *rest = args
            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        return hf_hub_download(*args, **kwargs)

    # Inject into huggingface_hub namespace so imports succeed
    try:
        import importlib
        _hf_mod = importlib.import_module("huggingface_hub")
        setattr(_hf_mod, "cached_download", cached_download)
    except Exception:
        # Best-effort only — if module import fails there's nothing more we can do here.
        pass


# Default models
_MODEL_EN_AF = "Helsinki-NLP/opus-mt-en-af"
_MODEL_AF_EN = "Helsinki-NLP/opus-mt-af-en"

# Cached translators per model name
_TRANSLATORS: Dict[str, object] = {}

# Default chunk size in characters
_DEFAULT_CHUNK_CHARS = 2000


# --- Pre-convert common English dates to Afrikaans to avoid MT mangling ---
def _preconvert_english_dates(text: str) -> str:
    if not text:
        return text

    months_en_to_af = {
        "january": "Januarie", "february": "Februarie", "march": "Maart",
        "april": "April", "may": "Mei", "june": "Junie", "july": "Julie",
        "august": "Augustus", "september": "September", "october": "Oktober",
        "november": "November", "december": "Desember"
    }

    def af_day_ordinal(day_int: int) -> str:
        # Afrikaans ordinal: default "de", use "ste" for 1,21,31 except 11
        if day_int % 10 == 1 and day_int % 100 != 11:
            suf = "ste"
        else:
            suf = "de"
        return f"{day_int}{suf}"

    # permissive patterns (case-insensitive) covering the forms you showed
    patterns = [
        # "the 21st of October 2025" or "21st of October 2025"
        re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE),
        # "October 21, 2025" or "Oct 21, 2025"
        re.compile(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b", re.IGNORECASE),
        # "21 October 2025" or "21 Oct 2025"
        re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE),
    ]

    def repl_pat1(m):
        day = int(m.group(1)); month_en = m.group(2); year = m.group(3)
        af_month = months_en_to_af.get(month_en.lower()); 
        if not af_month: return m.group(0)
        return f"{af_day_ordinal(day)} van {af_month} {year}"

    def repl_pat2(m):
        month_en = m.group(1); day = int(m.group(2)); year = m.group(3)
        af_month = months_en_to_af.get(month_en.lower()); 
        if not af_month: return m.group(0)
        return f"{af_day_ordinal(day)} van {af_month} {year}"

    def repl_pat3(m):
        day = int(m.group(1)); month_en = m.group(2); year = m.group(3)
        af_month = months_en_to_af.get(month_en.lower()); 
        if not af_month: return m.group(0)
        return f"{af_day_ordinal(day)} van {af_month} {year}"

    # apply sequentially
    text = patterns[0].sub(repl_pat1, text)
    text = patterns[1].sub(repl_pat2, text)
    text = patterns[2].sub(repl_pat3, text)

    return text

# --- Short-circuit renderer for full English date sentences ---
def _render_direct_date_translation(text: str) -> Optional[str]:
    """
    If `text` matches a common English full-date sentence (weekday optional),
    return a fully-formed Afrikaans sentence, otherwise return None.
    Examples handled:
      - "The current date is Tuesday the 21st of October 2025"
      - "I said the current date is Tuesday the 21st of October 2025"
      - "Current date: Tuesday 21 October 2025"
    """
    if not text:
        return None

    # maps
    months = {
        "january":"Januarie","february":"Februarie","march":"Maart","april":"April","may":"Mei",
        "june":"Junie","july":"Julie","august":"Augustus","september":"September","october":"Oktober",
        "november":"November","december":"Desember"
    }
    weekdays = {
        "monday":"Maandag","tuesday":"Dinsdag","wednesday":"Woensdag","thursday":"Donderdag",
        "friday":"Vrydag","saturday":"Saterdag","sunday":"Sondag"
    }

    def af_ordinal(day: int) -> str:
        # Afrikaans ordinal heuristic used earlier (1,21,31 -> ste except 11)
        if day % 10 == 1 and day % 100 != 11:
            suf = "ste"
        else:
            suf = "de"
        return f"{day}{suf}"

    # permissive regex capturing optional leading text, optional weekday, day (with optional ordinal),
    # optional 'the' and 'of', month (word) and 4-digit year.
    pat = re.compile(
        r"""(?ix)                # case-insensitive, verbose
        ^.*?                     # allow any leading text (we'll keep whole sentence)
        (?P<prefix>.*?)?         # capture prefix to optionally preserve leading text
        (?P<weekday>\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)?  # optional weekday
        [,\s:]*                  # separators
        (?:the\s+)?              # optional 'the'
        (?P<day>\d{1,2})(?:st|nd|rd|th)?   # day with optional ordinal
        (?:\s+of\s+|\s+)         # ' of ' or space
        (?P<month>[A-Za-z]{3,})  # month name (at least 3 letters)
        [\s,]+
        (?P<year>\d{4})\b        # 4-digit year
        .*$
        """, re.UNICODE
    )

    m = pat.match(text.strip())
    if not m:
        return None

    # extract
    weekday_en = (m.group("weekday") or "").strip().lower()
    day = int(m.group("day"))
    month_en = m.group("month").strip().lower()
    year = m.group("year").strip()

    af_month = months.get(month_en)
    if not af_month:
        # standardize common short names like "oct" -> "october"
        month_long = {
            "jan":"january","feb":"february","mar":"march","apr":"april","jun":"june",
            "jul":"july","aug":"august","sep":"september","oct":"october","nov":"november","dec":"december"
        }.get(month_en)
        af_month = months.get(month_long) if month_long else None
    if not af_month:
        return None  # unknown month -> don't short-circuit

    af_week = weekdays.get(weekday_en) if weekday_en else None
    af_day = af_ordinal(day)

    # Build Afrikaans phrase. We preserve a leading phrase if there was text like "I said" or "Response:"
    prefix = (m.group("prefix") or "").strip()
    if prefix:
        # Trim trailing separators like ":" or "-" from prefix
        prefix = re.sub(r"[:\-\s]+$", "", prefix).strip()

    # Core sentence: "Die huidige datum is Dinsdag die 21ste van Oktober 2025"
    # But if prefix contains "The current date is" variants, replace with canonical 'Die huidige datum is'
    # Recognize some common English prefix forms and map them to canonical Afrikaans.
    prefix_lower = prefix.lower()
    english_prefixes = [
        "the current date is", "i said the current date is",
        "current date", "response:", "i said", "reply:"
    ]
    if any(prefix_lower.endswith(ep) for ep in english_prefixes):
        af_prefix = "Die huidige datum is"
    elif prefix:
        # translate a simple prefix conservatively by returning it untranslated then adding the date
        af_prefix = prefix  # leave original (less ideal) — user can fine-tune if needed
    else:
        af_prefix = "Die huidige datum is"

    # compose
    if af_week:
        result = f"{af_prefix} {af_week} die {af_day} van {af_month} {year}"
    else:
        result = f"{af_prefix} die {af_day} van {af_month} {year}"

    # ensure spacing looks sane
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


# ------------------------------- Translator loader -------------------------------

def _get_translator(model_name: str, device: Optional[int] = None):
    """Lazy-load and cache Hugging Face translation pipeline.

    device semantics:
      - device=None: use GPU 0 if available else CPU (-1)
      - device=int: pass that device index to pipeline (use -1 for CPU)
    """
    global _TRANSLATORS
    if model_name in _TRANSLATORS:
        return _TRANSLATORS[model_name]

    try:
        import torch
        from transformers import pipeline
    except Exception as e:
        _LOG.exception("transformers or torch not available: %s", e)
        raise

    # Fixed device selection: respect explicit device, otherwise use GPU 0 when available
    device_arg = 0 if (device is None and torch.cuda.is_available()) else (device if device is not None else -1)
    _LOG.debug("Creating translation pipeline %s on device %s", model_name, device_arg)
    pipe = pipeline("translation", model=model_name, tokenizer=model_name, device=device_arg)
    _TRANSLATORS[model_name] = pipe
    return pipe


# ------------------------------- Chunking helpers -------------------------------

def _chunk_text_by_chars_keep_whitespace(text: str, max_chars: int) -> List[str]:
    """Break text into chunks of ≤ max_chars without breaking words."""
    if not text or len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        if end < L:
            last_space = text.rfind(" ", start, end)
            last_newline = text.rfind("\n", start, end)
            cut = max(last_space, last_newline)
            if cut > start:
                end = cut
        chunk = text[start:end] or text[start:min(start + max_chars, L)]
        chunks.append(chunk)
        start = end
    return chunks


# ------------------------------- Safe max_length auto-adjust -------------------------------

def _estimate_tokens_by_chars(s: str) -> int:
    """Rough heuristic: 1 token ≈ 4 chars."""
    return max(1, len(s) // 4)


def _safe_max_for_chunk(chunk: str, pipe: Any, provided_kwargs: dict,
                        safety_ratio: float = 0.9, margin: int = 8) -> dict:
    """
    Dynamically increases max_length/max_new_tokens when input tokens approach safety_ratio * max_length.
    Returns a copy of provided_kwargs with adjusted max_length or max_new_tokens.
    """
    kwargs = dict(provided_kwargs or {})

    try:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None:
            try:
                # some tokenizers expose encode, others require tokenizer.__call__
                if hasattr(tokenizer, "encode"):
                    tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
                else:
                    # fallback to tokenizer() then length of input_ids
                    tok_out = tokenizer(chunk, return_tensors=None)
                    tokens = len(tok_out.get("input_ids", []) or [])
            except Exception:
                tokens = _estimate_tokens_by_chars(chunk)
        else:
            tokens = _estimate_tokens_by_chars(chunk)
    except Exception:
        tokens = _estimate_tokens_by_chars(chunk)

    # determine currently-configured budget (prefer max_new_tokens if present)
    current = int(kwargs.get("max_new_tokens", 0) or kwargs.get("max_length", 512))
    safe_threshold = int(safety_ratio * current)

    if tokens >= safe_threshold:
        needed_max = int(math.ceil(tokens / safety_ratio)) + margin
        needed_max = min(needed_max, 65536)
        if "max_new_tokens" in kwargs and kwargs.get("max_new_tokens"):
            kwargs["max_new_tokens"] = needed_max
        else:
            kwargs["max_length"] = needed_max
        _LOG.debug("[SAFE MAX] Increased max_length → %s (tokens=%s)", needed_max, tokens)

    return kwargs


# ------------------------------- Translation logic -------------------------------

############def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
############    """
############    Translate text while preserving newline runs and spacing.
############    Robust masking approach:
############      - Mask English date substrings with letter-only placeholders before MT.
############      - Translate masked text chunk-wise.
############      - Restore placeholders with Afrikaans date strings (e.g. "21ste van Oktober 2025").
############    This avoids the MT splitting the year into '20' and '25' and avoids numeric post-fixes
############    that previously turned a stray '20' into 'twintig'.
############    """
############    if not text:
############        return ""
############
############    # --- month maps & helper ---
############    months_en_to_af = {
############        "january":"Januarie","february":"Februarie","march":"Maart","april":"April","may":"Mei",
############        "june":"Junie","july":"Julie","august":"Augustus","september":"September","october":"Oktober",
############        "november":"November","december":"Desember",
############        "jan":"Januarie","feb":"Februarie","mar":"Maart","apr":"April","jun":"Junie","jul":"Julie",
############        "aug":"Augustus","sep":"September","oct":"October","oct":"Oktober","nov":"November","dec":"Desember"
############    }
############
############    def af_ordinal(day: int) -> str:
############        if day % 10 == 1 and day % 100 != 11:
############            suf = "ste"
############        else:
############            suf = "de"
############        return f"{day}{suf}"
############
############    # --- permissive date patterns to detect common English forms ---
############    date_patterns = [
############        # "the 21st of October 2025" or "21st of October 2025"
############        re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([A-Za-z]{3,})\s+(\d{4})\b", re.IGNORECASE),
############        # "October 21, 2025" or "Oct 21, 2025"
############        re.compile(r"\b([A-Za-z]{3,})\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b", re.IGNORECASE),
############        # "21 October 2025" or "21 Oct 2025"
############        re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,})\s+(\d{4})\b", re.IGNORECASE),
############    ]
############
############    # Build a map of placeholder -> Afrikaans date string and also create masked text
############    placeholder_map = {}
############    placeholder_counter = 0
############
############    # We'll scan once and build non-overlapping replacements (collect spans first)
############    matches = []
############    for pat_idx, pat in enumerate(date_patterns):
############        for m in pat.finditer(text):
############            matches.append((m.start(), m.end(), pat_idx, m))
############    if not matches:
############        # no dates detected -> fall back to normal behavior (no masking)
############        parts = re.split(r"(\n+)", text)
############        translated_parts: List[str] = []
############        for part in parts:
############            if not part:
############                continue
############            if part.startswith("\n"):
############                translated_parts.append(part)
############                continue
############            chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
############            for chunk in chunks:
############                safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
############                try:
############                    out = pipe(chunk, **safe_kwargs)
############                except TypeError:
############                    out = pipe(chunk)
############                except Exception as e:
############                    _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
############                    translated_parts.append(chunk)
############                    continue
############                if isinstance(out, list) and out:
############                    translated_parts.append(out[0].get("translation_text", ""))
############                elif isinstance(out, dict):
############                    translated_parts.append(out.get("translation_text", ""))
############                else:
############                    translated_parts.append(str(out))
############        return "".join(translated_parts)
############
############    # sort matches and remove overlaps
############    matches.sort(key=lambda t: t[0])
############    non_overlap = []
############    last_end = -1
############    for start, end, pat_idx, m in matches:
############        if start >= last_end:
############            non_overlap.append((start, end, pat_idx, m))
############            last_end = end
############
############    # build masked text
############    out_chunks = []
############    last = 0
############    for start, end, pat_idx, m in non_overlap:
############        out_chunks.append(text[last:start])
############        # compute Afrikaans date for this match
############        try:
############            if pat_idx == 0:
############                day = int(m.group(1)); month_en = m.group(2); year = m.group(3)
############            elif pat_idx == 1:
############                month_en = m.group(1); day = int(m.group(2)); year = m.group(3)
############            else:
############                day = int(m.group(1)); month_en = m.group(2); year = m.group(3)
############        except Exception:
############            # fallback: skip masking this span
############            out_chunks.append(text[start:end])
############            last = end
############            continue
############
############        af_month = months_en_to_af.get(month_en.lower())
############        if not af_month:
############            # unknown month -> don't mask
############            out_chunks.append(text[start:end])
############            last = end
############            continue
############
############        af_date = f"{af_ordinal(day)} van {af_month} {year}"  # e.g. "21ste van Oktober 2025"
############        ph = f"__DATEPH_{placeholder_counter}__"              # safe, letters+underscores only
############        placeholder_counter += 1
############        placeholder_map[ph] = af_date
############        out_chunks.append(ph)
############        last = end
############
############    out_chunks.append(text[last:])
############    masked_text = "".join(out_chunks)
############
############    _LOG.debug("Masked text before MT: %r", masked_text)
############    # Now translate masked_text chunk-wise (same logic as before)
############    parts = re.split(r"(\n+)", masked_text)
############    translated_parts: List[str] = []
############
############    for part in parts:
############        if not part:
############            continue
############        if part.startswith("\n"):
############            translated_parts.append(part)
############            continue
############        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
############        for chunk in chunks:
############            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
############            try:
############                out = pipe(chunk, **safe_kwargs)
############            except TypeError:
############                out = pipe(chunk)
############            except Exception as e:
############                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
############                translated_parts.append(chunk)
############                continue
############            if isinstance(out, list) and out:
############                translated_parts.append(out[0].get("translation_text", ""))
############            elif isinstance(out, dict):
############                translated_parts.append(out.get("translation_text", ""))
############            else:
############                translated_parts.append(str(out))
############
############    joined = "".join(translated_parts)
############
############    # Restore placeholders with Afrikaans date strings (exact replacement)
############    for ph, af_date in placeholder_map.items():
############        joined = joined.replace(ph, af_date)
############
############    # Final cleanup: collapse duplicate spaces
############    joined = re.sub(r"\s{2,}", " ", joined).strip()
############
############    _LOG.debug("Translated + restored (pre post-fix): %r", joined)
############    return joined





##########def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
##########    """Translate text while preserving newline runs and spacing.
##########    Robust approach: pre-convert English date phrases to Afrikaans, translate masked text,
##########    then run post-fix regexes to repair any split-year/month artifacts.
##########    """
##########    if not text:
##########        return ""
##########    
##########    # If the input clearly contains an English full-date sentence, render directly (bypass MT).
##########    direct_date = _render_direct_date_translation(text)
##########    if direct_date is not None:
##########        return direct_date
##########    
##########    # ----- Pre-convert English date phrases -----
##########    months_en_to_af = {
##########        "january": "Januarie", "jan": "Januarie",
##########        "february": "Februarie", "feb": "Februarie",
##########        "march": "Maart", "mar": "Maart",
##########        "april": "April",
##########        "may": "Mei",
##########        "june": "Junie", "jun": "Junie",
##########        "july": "Julie", "jul": "Julie",
##########        "august": "Augustus", "aug": "Augustus",
##########        "september": "September", "sep": "September",
##########        "october": "Oktober", "oct": "Oktober",
##########        "november": "November", "nov": "November",
##########        "december": "Desember", "dec": "Desember"
##########    }
##########
##########    def af_day_ordinal(day_int: int) -> str:
##########        # Afrikaans ordinal heuristic (1,21,31 -> "ste" except 11)
##########        if day_int % 10 == 1 and day_int % 100 != 11:
##########            suf = "ste"
##########        else:
##########            suf = "de"
##########        return f"{day_int}{suf}"
##########
##########    # permissive patterns for common english date strings
##########    pats_pre = [
##########        # "the 21st of October 2025" or "21st of October 2025"
##########        re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([A-Za-z]{3,})\s+(\d{4})\b", re.IGNORECASE),
##########        # "October 21, 2025" or "Oct 21, 2025"
##########        re.compile(r"\b([A-Za-z]{3,})\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b", re.IGNORECASE),
##########        # "21 October 2025" or "21 Oct 2025"
##########        re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,})\s+(\d{4})\b", re.IGNORECASE),
##########    ]
##########
##########    def _preconvert_dates(s: str) -> str:
##########        # apply three patterns sequentially
##########        def repl1(m):
##########            day = int(m.group(1)); month = m.group(2); year = m.group(3)
##########            afm = months_en_to_af.get(month.lower())
##########            if not afm: return m.group(0)
##########            return f"{af_day_ordinal(day)} van {afm} {year}"
##########        def repl2(m):
##########            month = m.group(1); day = int(m.group(2)); year = m.group(3)
##########            afm = months_en_to_af.get(month.lower())
##########            if not afm: return m.group(0)
##########            return f"{af_day_ordinal(day)} van {afm} {year}"
##########        def repl3(m):
##########            day = int(m.group(1)); month = m.group(2); year = m.group(3)
##########            afm = months_en_to_af.get(month.lower())
##########            if not afm: return m.group(0)
##########            return f"{af_day_ordinal(day)} van {afm} {year}"
##########
##########        s = pats_pre[0].sub(repl1, s)
##########        s = pats_pre[1].sub(repl2, s)
##########        s = pats_pre[2].sub(repl3, s)
##########        return s
##########
##########    # apply pre-conversion so translator never sees fragile date substrings
##########    preconverted = _preconvert_dates(text)
##########
##########    # ----- Normal chunked translation on the (preconverted) text -----
##########    parts = re.split(r"(\n+)", preconverted)
##########    translated_parts: List[str] = []
##########
##########    for part in parts:
##########        if not part:
##########            continue
##########        if part.startswith("\n"):
##########            translated_parts.append(part)
##########            continue
##########
##########        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
##########        for chunk in chunks:
##########            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
##########
##########            try:
##########                out = pipe(chunk, **safe_kwargs)
##########            except TypeError:
##########                out = pipe(chunk)
##########            except Exception as e:
##########                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
##########                translated_parts.append(chunk)
##########                continue
##########
##########            if isinstance(out, list) and out:
##########                translated_text = out[0].get("translation_text", "")
##########            elif isinstance(out, dict):
##########                translated_text = out.get("translation_text", "")
##########            else:
##########                translated_text = str(out)
##########
##########            translated_parts.append(translated_text)
##########
##########    joined = "".join(translated_parts)
##########
##########    # ----- Post-fix: robust regex passes to repair split-year/month artifacts -----
##########    # Works case-insensitively and for unicode; targets patterns like:
##########    #   "21ste van 20 Oktober25"  -> "21ste van Oktober 2025"
##########    #   "20 Oktober25"            -> "Oktober 2025"
##########    #   "Oktober25"               -> "Oktober 2025"
##########    flags = re.IGNORECASE | re.UNICODE
##########
##########    def _assemble_year(head: str, tail: str) -> str:
##########        head = (head or "").strip()
##########        tail = (tail or "").strip()
##########        if len(head) == 2 and len(tail) == 2:
##########            return head + tail
##########        if len(tail) == 4:
##########            return tail
##########        if len(head) == 4:
##########            return head
##########        return (tail or head).strip()
##########
##########    # 1) day + 'van' + head(2) + month + tail(2 or 4)
##########    joined = re.sub(
##########        r"(\b\d{1,2}(?:ste|de)?)\s+van\s+(\d{2})\s*([A-Za-z]{3,})\s*(\d{2,4})\b",
##########        lambda m: f"{m.group(1)} van {m.group(3)} {_assemble_year(m.group(2), m.group(4))}",
##########        joined,
##########        flags=flags
##########    )
##########
##########    # 2) 'van' + head + month + tail
##########    joined = re.sub(
##########        r"\bvan\s+(\d{2})\s*([A-Za-z]{3,})\s*(\d{2,4})\b",
##########        lambda m: f"van {m.group(2)} {_assemble_year(m.group(1), m.group(3))}",
##########        joined,
##########        flags=flags
##########    )
##########
##########    # 3) head + month + tail  (no 'van')
##########    joined = re.sub(
##########        r"\b(\d{2})\s*([A-Za-z]{3,})\s*(\d{2,4})\b",
##########        lambda m: f"{m.group(2)} {_assemble_year(m.group(1), m.group(3))}",
##########        joined,
##########        flags=flags
##########    )
##########
##########    # 4) month immediately followed by year without space -> insert a space and expand 2-digit years to 20xx
##########    joined = re.sub(
##########        r"\b([A-Za-z]{3,})(\d{4})\b",
##########        r"\1 \2",
##########        joined,
##########        flags=flags
##########    )
##########    joined = re.sub(
##########        r"\b([A-Za-z]{3,})(\d{2})\b",
##########        lambda m: f"{m.group(1)} 20{m.group(2)}",
##########        joined,
##########        flags=flags
##########    )
##########
##########    # 5) final cleanup: collapse double spaces and trim
##########    joined = re.sub(r"\s{2,}", " ", joined).strip()
##########
##########    _LOG.debug("Final translated (post-fixed): %r", joined)
##########    return joined


########def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
########    """Translate text while preserving newline runs and spacing.
########    Pre-mask English date phrases and restore them to Afrikaans after translation
########    to avoid year-splitting issues from the MT model.
########    """
########    if not text:
########        return ""
########
########    # --- helpers: month mapping and ordinal suffix in Afrikaans ---
########    months_en_to_af = {
########        "january": "Januarie", "february": "Februarie", "march": "Maart",
########        "april": "April", "may": "Mei", "june": "Junie", "july": "Julie",
########        "august": "Augustus", "september": "September", "october": "Oktober",
########        "november": "November", "december": "Desember"
########    }
########
########    def af_day_ordinal(day_int: int) -> str:
########        # Afrikaans ordinal: add "ste" for days ending in 1 except 11, otherwise "de"
########        if day_int % 10 == 1 and day_int % 100 != 11:
########            suf = "ste"
########        else:
########            suf = "de"
########        return f"{day_int}{suf}"
########
########    # --- find and mask English date substrings ---
########    placeholder_map = {}
########    placeholder_counter = 0
########
########    # Patterns to catch common English date forms (made somewhat permissive)
########    patterns = [
########        # "the 21st of October 2025" or "21st of October 2025"
########        re.compile(r"\b(?:the\s+)?(\d{1,2})(st|nd|rd|th)?\s+of\s+([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE),
########        # "October 21, 2025" or "Oct 21, 2025"
########        re.compile(r"\b([A-Za-z]+)\s+(\d{1,2})(st|nd|rd|th)?,\s*(\d{4})\b", re.IGNORECASE),
########        # "21 October 2025" or "21 Oct 2025"
########        re.compile(r"\b(\d{1,2})(st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE),
########    ]
########
########    # We will make a single pass over the text; to avoid overlapping replacements,
########    # build a new string incrementally.
########    def mask_dates(s: str) -> str:
########        nonlocal placeholder_counter
########        # collect all matches with start/end to avoid overlapping issues
########        matches = []
########        for pat in patterns:
########            for m in pat.finditer(s):
########                matches.append((m.start(), m.end(), m, pat))
########        if not matches:
########            return s
########        # sort by start position
########        matches.sort(key=lambda t: t[0])
########
########        out = []
########        last_idx = 0
########        used_spans = set()
########        for start, end, m, pat in matches:
########            # skip overlaps
########            if any(i >= start and i < end for i in used_spans):
########                continue
########            # append preceding text
########            out.append(s[last_idx:start])
########            # generate Afrikaans date for this match
########            try:
########                if pat is patterns[0]:  # "the 21st of October 2025"
########                    day = int(m.group(1))
########                    month_en = m.group(3)
########                    year = m.group(4)
########                elif pat is patterns[1]:  # "October 21, 2025"
########                    month_en = m.group(1)
########                    day = int(m.group(2))
########                    year = m.group(4)
########                else:  # patterns[2] "21 October 2025"
########                    day = int(m.group(1))
########                    month_en = m.group(3)
########                    year = m.group(4)
########            except Exception:
########                # fallback: just keep original substring (don't mask)
########                out.append(s[start:end])
########                last_idx = end
########                # mark span used
########                for i in range(start, end):
########                    used_spans.add(i)
########                continue
########
########            month_key = month_en.lower()
########            af_month = months_en_to_af.get(month_key)
########            if not af_month:
########                # unknown month name: don't mask (preserve original substring)
########                out.append(s[start:end])
########                last_idx = end
########                for i in range(start, end):
########                    used_spans.add(i)
########                continue
########
########            af_date = f"{af_day_ordinal(day)} van {af_month} {year}"
########            placeholder = f"<<<DATE_{placeholder_counter}>>>"
########            placeholder_counter += 1
########            placeholder_map[placeholder] = af_date
########
########            out.append(placeholder)
########            last_idx = end
########            for i in range(start, end):
########                used_spans.add(i)
########
########        out.append(s[last_idx:])
########        return "".join(out)
########
########    masked_text = mask_dates(text)
########
########    # --- proceed with original newline/chunk translation on masked_text ---
########    parts = re.split(r"(\n+)", masked_text)
########    translated_parts: List[str] = []
########
########    for part in parts:
########        if not part:
########            continue
########        if part.startswith("\n"):
########            translated_parts.append(part)
########            continue
########
########        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
########        for chunk in chunks:
########            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
########
########            try:
########                out = pipe(chunk, **safe_kwargs)
########            except TypeError:
########                out = pipe(chunk)
########            except Exception as e:
########                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
########                translated_parts.append(chunk)
########                continue
########
########            if isinstance(out, list) and out:
########                translated_text = out[0].get("translation_text", "")
########            elif isinstance(out, dict):
########                translated_text = out.get("translation_text", "")
########            else:
########                translated_text = str(out)
########
########            translated_parts.append(translated_text)
########
########    joined = "".join(translated_parts)
########
########    # --- restore placeholders with Afrikaans date strings (exact replacement) ---
########    if placeholder_map:
########        # simple replacement loop; placeholders are unique and unlikely to be altered
########        for ph, af_date in placeholder_map.items():
########            joined = joined.replace(ph, af_date)
########
########    return joined




######def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
######    """Translate text while preserving newline runs and spacing."""
######    if not text:
######        return ""
######
######    parts = re.split(r"(\n+)", text)
######    translated_parts: List[str] = []
######
######    def _fix_split_years(s: str) -> str:
######        # 1) Fix patterns like: "21ste van 20 Oktober25" -> "21ste van Oktober 2025"
######        def repl_day_van_split_year(m):
######            day = m.group(1)                     # e.g. "21ste"
######            year_head = m.group(2)               # e.g. "20"
######            month = m.group(3)                   # e.g. "Oktober"
######            year_tail = m.group(4) or ""         # e.g. "25" or "2025"
######
######            # if tail is two digits and head is two digits, join them into a 4-digit year
######            if len(year_head) == 2 and len(year_tail) == 2:
######                year = year_head + year_tail
######            # if tail already looks like 4 digits use it, otherwise try to prefer tail
######            elif len(year_tail) == 4:
######                year = year_tail
######            else:
######                # fallback: prefer tail if present, else use head (handles odd splits)
######                year = (year_tail or year_head).strip()
######
######            return f"{day} van {month} {year}".strip()
######
######        s = re.sub(r"\b(\d{1,2}(?:ste|de)?)\s+van\s+(\d{2})\s*([A-Za-z]+?)(\d{2,4})\b",
######                   repl_day_van_split_year, s)
######
######        # 2) Fix patterns like: "20 Oktober25" -> "Oktober 2025"
######        def repl_head_month_tail(m):
######            head = m.group(1)    # e.g. "20"
######            month = m.group(2)   # e.g. "Oktober"
######            tail = m.group(3)    # e.g. "25"
######            if len(head) == 2 and len(tail) == 2:
######                year = head + tail
######            else:
######                # prefer tail if it's 4-digit, otherwise join cautiously
######                year = tail if len(tail) == 4 else (head + tail if len(head) == 2 else tail)
######            return f"{month} {year}".strip()
######
######        s = re.sub(r"\b(\d{2})\s*([A-Za-z]+?)(\d{2})\b", repl_head_month_tail, s)
######
######        # 3) Ensure there's a space between month and a directly appended 2/4-digit year:
######        #    "Oktober2025" -> "Oktober 2025"
######        s = re.sub(r"([A-Za-z]+)(\d{4})\b", r"\1 \2", s)
######        s = re.sub(r"([A-Za-z]+)(\d{2})\b", r"\1 \2", s)
######
######        return s
######
######    for part in parts:
######        if not part:
######            continue
######        if part.startswith("\n"):
######            translated_parts.append(part)
######            continue
######
######        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
######        for chunk in chunks:
######            # Use tokenizer-aware safe max-length calculation when available
######            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
######
######            try:
######                out = pipe(chunk, **safe_kwargs)
######            except TypeError:
######                # Some pipeline versions expect positional args differently; try calling with only text
######                out = pipe(chunk)
######            except Exception as e:
######                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
######                # fallback: return raw chunk to avoid losing content
######                translated_parts.append(chunk)
######                continue
######
######            if isinstance(out, list) and out:
######                translated_text = out[0].get("translation_text", "")
######            elif isinstance(out, dict):
######                translated_text = out.get("translation_text", "")
######            else:
######                translated_text = str(out)
######
######            # run conservative post-processing fixes for split/misplaced year parts
######            translated_text = _fix_split_years(translated_text)
######
######            translated_parts.append(translated_text)
######
######    return "".join(translated_parts)





####def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
####    """Translate text while preserving newline runs and spacing."""
####    if not text:
####        return ""
####
####    parts = re.split(r"(\n+)", text)
####    translated_parts: List[str] = []
####
####    for part in parts:
####        if not part:
####            continue
####        if part.startswith("\n"):
####            translated_parts.append(part)
####            continue
####
####        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
####        for chunk in chunks:
####            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)
####
####            try:
####                out = pipe(chunk, **safe_kwargs)
####            except TypeError:
####                out = pipe(chunk)
####            except Exception as e:
####                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
####                translated_parts.append(chunk)
####                continue
####
####            if isinstance(out, list) and out:
####                translated_text = out[0].get("translation_text", "")
####            elif isinstance(out, dict):
####                translated_text = out.get("translation_text", "")
####            else:
####                translated_text = str(out)
####
####            # --- Post-fix step for number/date formatting ---
####            translated_text = re.sub(r"\b(\d{1,2})(ste|de)?\s+van\s+(\d{2})\s*([A-Za-z]+)?(\d{2,4})?\b",
####                                     lambda m: f"{m.group(1)}{m.group(2) or ''} van {m.group(4) or ''} {m.group(3)}{m.group(5) or ''}",
####                                     translated_text)
####
####            # Fix misplaced year parts like "20 Oktober25" -> "Oktober 2025"
####            translated_text = re.sub(r"\b(\d{1,2})\s*([A-Za-z]+)\s*(\d{2})\b",
####                                     lambda m: f"{m.group(2)} 20{m.group(3)}" if len(m.group(3)) == 2 else f"{m.group(2)} {m.group(3)}",
####                                     translated_text)
####
####            translated_parts.append(translated_text)
####
####    return "".join(translated_parts)




def _translate_preserve_newlines(text: str, pipe, chunk_chars: int, **pipeline_kwargs) -> str:
    """Translate text while preserving newline runs and spacing."""
    if not text:
        return ""

    parts = re.split(r"(\n+)", text)
    translated_parts: List[str] = []

    for part in parts:
        if not part:
            continue
        if part.startswith("\n"):
            translated_parts.append(part)
            continue

        chunks = _chunk_text_by_chars_keep_whitespace(part, chunk_chars)
        for chunk in chunks:
            # Use tokenizer-aware safe max-length calculation when available
            safe_kwargs = _safe_max_for_chunk(chunk, pipe, pipeline_kwargs)

            try:
                out = pipe(chunk, **safe_kwargs)
            except TypeError:
                # Some pipeline versions expect positional args differently; try calling with only text
                out = pipe(chunk)
            except Exception as e:
                _LOG.exception("Translation pipeline call failed for a chunk: %s", e)
                # fallback: return raw chunk to avoid losing content
                translated_parts.append(chunk)
                continue

            if isinstance(out, list) and out:
                translated_parts.append(out[0].get("translation_text", ""))
            elif isinstance(out, dict):
                translated_parts.append(out.get("translation_text", ""))
            else:
                translated_parts.append(str(out))

    return "".join(translated_parts)


# ------------------------------- Direction detection -------------------------------

def _detect_direction(text: str) -> str:
    """Heuristic: decide whether text looks English (→AF) or Afrikaans (→EN)."""
    if not text:
        return _MODEL_EN_AF
    # quick non-ascii heuristic
    if any(ord(c) > 127 for c in text):
        return _MODEL_AF_EN
    # Afrikaans keywords
    if re.search(r"\b(ek|jy|nie|het|sal|kan|moet|baie|goed|daar|waar)\b", text.lower()):
        return _MODEL_AF_EN
    return _MODEL_EN_AF


# ------------------------------- Public API -------------------------------

def translate(text: str,
              device: Optional[int] = None,
              chunk_chars: int = _DEFAULT_CHUNK_CHARS,
              max_length: Optional[int] = 512,
              **pipeline_kwargs) -> str:
    """
    Translate text between English and Afrikaans (auto-detected direction).
    Automatically adjusts max_length to avoid HF length warnings.
    """
    if not text:
        return ""

    # --- Pre-convert English dates to Afrikaans to prevent MT from splitting years ---
    text = _preconvert_english_dates(text)


    model_name = _detect_direction(text)
    pipe = _get_translator(model_name, device=device)

    call_kwargs = dict(pipeline_kwargs or {})
    if max_length is not None:
        call_kwargs.setdefault("max_length", max_length)

    try:
        return _translate_preserve_newlines(text, pipe, chunk_chars, **call_kwargs)
    except Exception as e:
        _LOG.warning("Primary translation failed (%s). Trying fallback...", e)
        try:
            out = pipe(text, **call_kwargs)
            if isinstance(out, list) and out:
                return out[0].get("translation_text", "")
            elif isinstance(out, dict):
                return out.get("translation_text", "")
            else:
                return str(out)
        except Exception as e2:
            _LOG.exception("Fallback translation failed: %s", e2)
            return text


def translate_batch(texts: List[str],
                    device: Optional[int] = None,
                    chunk_chars: int = _DEFAULT_CHUNK_CHARS,
                    max_length: Optional[int] = 512,
                    **pipeline_kwargs) -> List[str]:
    """Translate multiple strings while preserving order."""
    if not texts:
        return []
    return [translate(t, device=device, chunk_chars=chunk_chars,
                      max_length=max_length, **pipeline_kwargs) for t in texts]


# ------------------------------- Self-test -------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    examples = [
        "Good morning, how are you doing?\nThis is a new line.\n\nThis is a new paragraph with two newlines above.",
        "Waar is die biblioteek?\n  (note: two spaces before parenthesis retained)"
    ]
    for s in examples:
        print("IN:", repr(s))
        print("OUT:", repr(translate(s, max_length=256)))
        print("---")



### en_to_af.py
##"""
##Simple English -> Afrikaans translator module.
##
##Usage (from other modules):
##    from en_to_af import translate, translate_batch
##    text_af = translate("Good morning, how are you?")
##    texts_af = translate_batch(["Hello", "Thank you"])
##"""
##from typing import List, Union
##import torch
##from transformers import pipeline
##
##_MODEL_NAME = "Helsinki-NLP/opus-mt-en-af"
##
##_translator = None
##
##def _get_translator(device: Union[int, None] = None):
##    global _translator
##    if _translator is None:
##        if device is None:
##            device_arg = 0 if torch.cuda.is_available() else -1
##        else:
##            device_arg = device
##        _translator = pipeline(
##            "translation",
##            model=_MODEL_NAME,
##            tokenizer=_MODEL_NAME,
##            device=device_arg,
##            truncation=True,
##        )
##    return _translator
##
##def translate(text: str, device: Union[int, None] = None) -> str:
##    """
##    Translate a single English string to Afrikaans.
##    """
##    if not text:
##        return ""
##    pipe = _get_translator(device)
##    out = pipe(text)
##    return out[0].get("translation_text", "")
##
##def translate_batch(texts: List[str], device: Union[int, None] = None) -> List[str]:
##    """
##    Translate a list of English strings to Afrikaans.
##    """
##    if not texts:
##        return []
##    pipe = _get_translator(device)
##    outs = pipe(texts)
##    return [o.get("translation_text", "") for o in outs]
##
##if __name__ == "__main__":
##    examples = [
##        "Good morning, how are you doing?",
##        "Where is the library?"
##    ]
##    print("EN -> AF examples:")
##    for s in examples:
##        print("EN:", s)
##        print("AF:", translate(s))
##        print("---")
