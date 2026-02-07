#

#!/usr/bin/env python3
"""
Model discovery + loader

- Removes leading "library/" component from model names.
- Finds parameters both as subdirectories and as files inside the model directory.
- Emits "model:parameter" for each parameter found.
- Emits "model" if no parameter found.
- Prunes duplicate model roots so only the "best" entry remains:
    * choose the largest numeric tag (e.g. 8b > 6.7b > 1.5b > 70m)
    * if no numeric tag exists, prefer :latest
    * if neither exists, keep the plain model name
- Produces a Python-ready MODEL_NAMES list.

Usage:
    python models_loader.py                   # will attempt discovery using MODEL_DIR
    python models_loader.py "D:/path/to/registry.ollama.ai"
"""

from pathlib import Path
import sys
import json
import re
from typing import List, Set, Dict, Tuple, Optional

# ---------- CONFIG ----------
DEFAULT_REGISTRY_PATH = Path(r"D:\ollama\models\manifests\registry.ollama.ai")
MAX_DEPTH = 6                 # how deep to search (adjust to your layout)
WRITE_OUTPUT_FILE = False     # if True, writes OUTPUT_FILENAME
OUTPUT_FILENAME = "discovered_model_names.py"

# Optional: models you want to appear first in the final list (edit to match your preferred top ordering)
PRIORITY_ORDER = [
    "minicpm-v",
    "qwen2.5-coder:3b",
    "deepseek-r1:8b",
    "qwen2.5:3b",
    "llama2:7b",
    "zongwei/gemma3-translator:4b",
    "lauchacarro/qwen2.5-translator",
    "0xroyce/NazareAI-Python-Programmer-3B",
    "llama3-chatqa:8b",
    "brxce/stable-diffusion-prompt-generator",
    "granite3.1-moe:1b-instruct-q2_K",
    "granite3.3:2b",
    "qwen2.5-coder:1.5b",
    "granite3.2-vision",
    "Hudson/pythia-instruct:70m",
    "yi-coder:1.5b",
    "deepseek-r1:1.5b",
    "llama3.2:1b",
    "nomic-embed-text",
    "mxbai-embed-large",
    "gemma:2b",
    "llava-phi3",
    "codegemma:2b-code",
    "moondream",
    "llava",
    "deepseek-coder:1.5b",
    "deepseek-coder:6.7b",
    "tinyllama",
    "mistral",
    "stablelm2",
    "qwen:1.8b",
    "starcoder2",
    "dolphin-phi",
    "phi",
    "llama2",
]
# ----------------------------

# Parameter detection regex / helper:
# We'll accept any name that contains digits OR equals 'latest' (case-insensitive).
PARAM_RE = re.compile(r'.*\d.*')  # any name containing a digit

def looks_like_parameter(name: str) -> bool:
    if not name:
        return False
    if name.lower() == "latest":
        return True
    # consider files such as "1.5b", "6.7b", "70m", "1b-instruct-q2_K", etc. valid if they contain a digit
    return bool(PARAM_RE.match(name))

def rel_parts_without_library(registry_root: Path, p: Path) -> List[str]:
    """Return relative parts from registry_root, removing leading 'library' if present."""
    rel = p.relative_to(registry_root)
    parts = list(rel.parts)
    if parts and parts[0].lower() == "library":
        parts = parts[1:]
    return parts

def model_name_from_parts(parts: List[str]) -> str:
    """Join parts with '/' to form model name (preserving namespaces)."""
    return "/".join(parts)

def is_leaf_dir(path: Path) -> bool:
    """True if directory contains no subdirectories (even if it has files)."""
    try:
        for c in path.iterdir():
            if c.is_dir():
                return False
        return True
    except PermissionError:
        return False

def find_parameters_in_model_dir(model_dir: Path) -> List[str]:
    """
    Look for parameters inside model_dir.

    - subdirectories whose name looks like a parameter (contains digit or 'latest')
    - files whose stem looks like a parameter (e.g. '6.7b', '1.5b', 'latest' with or without extension)
    Returns a sorted list of unique parameter strings (no duplicates).
    """
    params = []
    try:
        for child in sorted(model_dir.iterdir(), key=lambda p: p.name.lower()):
            # skip hidden
            if child.name.startswith("."):
                continue
            if child.is_dir():
                if is_leaf_dir(child) and looks_like_parameter(child.name):
                    params.append(child.name)
            elif child.is_file():
                stem = child.stem  # filename without extension
                # skip obvious manifest files that are named like "manifest.json" (no digit)
                if looks_like_parameter(stem):
                    params.append(stem)
    except PermissionError:
        pass
    # deduplicate while preserving order
    seen = set()
    out = []
    for p in params:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# ----------------------------
# New: pruning helpers to pick the best tag per model root
# ----------------------------

NUM_RE = re.compile(r'(\d+(?:\.\d+)?)\s*([bBmM])?')  # capture number and optional unit b/m

def tag_numeric_value(tag: str) -> Optional[float]:
    """
    Return numeric value in 'billions' for tags that include numeric sizes:
      - "6.7b" -> 6.7
      - "4b"   -> 4.0
      - "70m"  -> 0.07
      - "1.5b-instruct-q2_K" -> 1.5
      - "1"    -> 1.0 (we treat bare numbers as 'b')
    Returns None for tags that aren't numeric (e.g. random names), but 'latest' is handled separately.
    """
    if not tag:
        return None
    t = tag.strip().lower()
    if t == "latest":
        return None
    m = NUM_RE.search(t)
    if not m:
        return None
    num_str, unit = m.groups()
    try:
        num = float(num_str)
    except Exception:
        return None
    if unit:
        unit = unit.lower()
        if unit == 'b':
            return num
        if unit == 'm':
            return num / 1000.0
    # no unit — treat as billions (common fall-back)
    return num

def pick_best_tag_for_root(tags: List[str], plain_present: bool) -> Optional[str]:
    """
    Given a list of tags for the same model root (e.g. ['latest','6.7b','1.5b']),
    choose the single best tag to keep:
      - if any numeric tags exist, pick the one with the largest numeric value
      - else if 'latest' exists, pick 'latest'
      - else if plain_present True and no tags, return None (meaning plain model)
      - else if there are non-numeric tags, pick the first (deterministic)
    Returns chosen tag string or None (for plain model).
    """
    # map numeric tags to values
    numeric_candidates: List[Tuple[float, str]] = []
    non_numeric = []
    has_latest = False
    for t in tags:
        if t.lower() == "latest":
            has_latest = True
            continue
        v = tag_numeric_value(t)
        if v is not None:
            numeric_candidates.append((v, t))
        else:
            non_numeric.append(t)
    if numeric_candidates:
        # choose the tag with the max numeric value (if tie, deterministic pick by tag sort)
        numeric_candidates.sort(key=lambda x: (x[0], x[1]))
        best = numeric_candidates[-1][1]
        return best
    if has_latest:
        return "latest"
    if non_numeric:
        return non_numeric[0]
    # no tag, fall back to plain model if present
    if plain_present:
        return None
    return None

def prune_to_best_entries(collected: List[str]) -> List[str]:
    """
    Given collected entries like ["foo:latest","foo:6.7b","foo","bar:3b",...],
    return a pruned list where each base (before ':') appears only once, keeping the best tag (or plain).
    """
    roots: Dict[str, Dict[str, object]] = {}
    # roots[root] = {'tags': set(...), 'plain': bool}
    for entry in collected:
        if ":" in entry:
            root, tag = entry.split(":", 1)
            data = roots.setdefault(root, {"tags": [], "plain": False})
            data["tags"].append(tag)
        else:
            root = entry
            data = roots.setdefault(root, {"tags": [], "plain": False})
            data["plain"] = True

    result = []
    for root, data in roots.items():
        tags = data["tags"]
        plain = data["plain"]
        chosen_tag = pick_best_tag_for_root(tags, plain)
        if chosen_tag is None:
            # plain model chosen
            result.append(root)
        else:
            result.append(f"{root}:{chosen_tag}")
    # Keep deterministic order by sorting (priority later)
    return sorted(result)

# ----------------------------

def discover_models(registry_root: Path, max_depth: int = 6) -> List[str]:
    """
    Discover models under registry_root.

    Returns list of strings like:
      - "owner/model:parameter" (if parameter subdirs/files found)
      - "owner/model" (if no parameters discovered and directory is a leaf)
    """
    registry_root = registry_root.resolve()
    if not registry_root.exists() or not registry_root.is_dir():
        raise FileNotFoundError(f"Registry root not found: {registry_root}")

    collected: List[str] = []
    seen_models: Set[str] = set()

    # Walk directories sorted for stable output
    for path in sorted(registry_root.rglob("*")):
        if not path.is_dir():
            continue
        # skip hidden/system
        if path.name.startswith("."):
            continue
        try:
            rel = path.relative_to(registry_root)
        except Exception:
            continue
        # depth guard
        if len(rel.parts) > max_depth:
            continue

        parts = rel_parts_without_library(registry_root, path)
        if not parts:
            continue  # skip the registry root or "library" root itself

        # Determine parameters inside this directory
        params = find_parameters_in_model_dir(path)
        if params:
            model_root = model_name_from_parts(parts)
            # emit model:parameter for each param
            for p in params:
                entry = f"{model_root}:{p}"
                if entry not in seen_models:
                    collected.append(entry)
                    seen_models.add(entry)
            # ensure we don't later emit the param folder or its files as plain models
            continue

        # If no params found and this is a leaf dir (no subdirectories),
        # treat it as a plain model (unless was already added as param)
        if is_leaf_dir(path):
            model_name = model_name_from_parts(parts)
            if model_name not in seen_models:
                collected.append(model_name)
                seen_models.add(model_name)
            continue

        # Otherwise (intermediate namespace), skip and allow deeper directories to be processed
        continue

    # Prune duplicates: pick best tag per model root
    pruned = prune_to_best_entries(collected)

    # Now sort the pruned list: put PRIORITY_ORDER items first (in that order), then the rest alphabetically
    priority_index = {name: i for i, name in enumerate(PRIORITY_ORDER)}
    def sort_key(name: str):
        return (priority_index.get(name, 10**6), name.lower())
    collected_sorted = sorted(pruned, key=sort_key)
    return collected_sorted

def format_as_python_list(models: List[str], varname: str = "MODEL_NAMES") -> str:
    items = ",\n        ".join(json.dumps(m) for m in models)
    return f"{varname} = [\n        {items}\n    ]\n"

# -----------------------------
# USER-SIDE MODEL LOADING BLOCK
# Keep this in the same general form you had; discovery is attempted first.
# -----------------------------

# Your original MODEL_DIR (keep unchanged)
# MODEL_DIR = "C://Users//ITF//.ollama//models//manifests//registry.ollama.ai//library"
MODEL_DIR = "D://Python_Env//New_Virtual_Env//ollama_models//models//manifests//registry.ollama.ai//library"

# Load model names once
try:
    model_dir_path = Path(MODEL_DIR)
    # If MODEL_DIR points to ".../registry.ollama.ai/library" use the parent as the registry root,
    # otherwise use MODEL_DIR as given.
    registry_root = model_dir_path.parent if model_dir_path.name.lower() == "library" else model_dir_path

    # Attempt discovery using the functions in this same file
    try:
        discovered = discover_models(registry_root, max_depth=MAX_DEPTH)
        if discovered:
            MODEL_NAMES = discovered
        else:
            # If discovery returned empty, fall back to hard-coded list below
            MODEL_NAMES = None
    except FileNotFoundError as e:
        # registry path not found or inaccessible
        print(f"[WARN] Discovery failed: {e}")
        MODEL_NAMES = None
    except Exception as e:
        print(f"[WARN] Discovery encountered an error: {e}")
        MODEL_NAMES = None

    # If discovery didn't run or failed, fall back to the hard-coded list you already had:
    if not MODEL_NAMES:
        MODEL_NAMES = [
            "minicpm-v",
            "vortex/helpingai-9b",
            "openbmb/minicpm-v4:4b",
            "qwen2.5-coder:3b",
            "qwen2.5-coder:1.5b",
            "qwen2.5:3b",
            "qwen:1.8b",
            "deepseek-r1:8b",
            "deepseek-r1:1.5b",
            "deepseek-coder:6.7b",
            "deepseek-coder:1.5b",
            "llama2:7b",
            "llama2-uncensored:latest",
            "zongwei/gemma3-translator:4b",
            "gemma:2b",
            "lauchacarro/qwen2.5-translator",
            "0xroyce/NazareAI-Python-Programmer-3B",
            "llama3-chatqa:8b",
            "llama3.2:1b",
            "mannix/llama3-uncensored",
            "brxce/stable-diffusion-prompt-generator",
            "closex/neuraldaredevil-8b-abliterated",
            "granite3.1-moe:1b-instruct-q2_K",
            "granite3.3:2b",
            "granite3.2-vision",
            "Hudson/pythia-instruct:70m",
            "MeaTLoTioN/Marvin:latest",
            "yi-coder:1.5b",
            "nomic-embed-text",
            "mxbai-embed-large",
            "codegemma:2b-code",
            "gemma:2b",
            "llava-phi3:3.8b",
            "llava",
            "moondream",
            "tinyllama",
            "mistral",
            "stablelm2",
            "starcoder2",
            "dolphin-phi",
            "dolphin-phi:2.7b",
            "dolphin-mistral",
            "phi",
            "qwen2.5-coder:3b",
            "qwen2.5-coder:1.5b",
            "qwen2.5:3b",
            "qwen:1.8b",
            "gemma:2b",
            "codegemma:2b-code",
            "minicpm-v:latest",
            "qwen3-vl:2b",
            "qwen3-vl:4b",
            "llama3.2:1b",
        ]



except FileNotFoundError:
    MODEL_NAMES = []
    print(f"[ERROR] Model directory not found: {MODEL_DIR}")

# -----------------------------
# Your existing exclusion/addition logic (kept as-is)
# -----------------------------





EXCLUSIONS = {
    "thinkbot": [
        "0xroyce/NazareAI-Python-Programmer-3B", "deepseek-r1", "code", "llava", "moondream", "vision", "-embed-", "chatqa",
        "pythia","qwen2.5-coder:3b", "codegemma:2b-code", "minicpm-v:latest", "openbmb/minicpm-v4:4b", "starcoder2", "yi-coder:1.5b", "qwen2.5-coder:1.5b", "deepseek-coder:6.7b", "deepseek-coder:1.5b", "stable", "minicpm-v"
    ],
    "chatbot": [
        "0xroyce", "deepseek-r1", "codegemma", "deepseek-coder", "deepseek-r1:8b", "dolphin-phi", "gemma",
        "granite3.1-moe", "granite3.2-vision", "granite3.3", "llama2", "minicpm-v:latest", "openbmb/minicpm-v4:4b",
        "llama3.2", "llava", "llava-phi3", "mistral", "moondream", "mxbai-embed-large",
        "nomic-embed-text", "qwen2.5-coder", "stablelm2", "starcoder2", "yi-coder",
        "pythia","qwen2.5-coder:3b","codegemma:2b-code", "starcoder2", "yi-coder:1.5b", "qwen2.5-coder:1.5b", "deepseek-coder:6.7b", "deepseek-coder:1.5b", "stable", "minicpm-v"
    ],
    "vision": [
        "0xroyce", "phi", "qwen", "stablelm2", "mistral", "granite3.3", "deepseek-r1", "dolphin-phi", "gemma", "llama"
        "granite3.1-moe", "code", "-embed-","llama2:7b", "llama2-uncensored:latest", "llama3-chatqa:8b", "llama3.2:1b", "mannix/llama3-uncensored", "codegemma:2b-code", "starcoder2", "yi-coder:1.5b", "llama","qwen2.5-coder:3b", "qwen2.5-coder:1.5b", "deepseek-coder:6.7b", "deepseek-coder:1.5b", "pythia"
    ],
    "coding": [
        "llava", "granite3.2-vision", "moondream", "phi", "qwen", "granite3.3", "dolphin-phi",
        "gemma", "granite3.1-moe", "-embed-", "llama", "chatqa", "deepseek-r1", "stable", "minicpm-v", "minicpm-v:latest", "openbmb/minicpm-v4:4b"
    ],
    "rag": [   # only embeddings allowed, exclude everything else
        "0xroyce", "deepseek", "llama", "qwen", "gemma", "granite", "code", "llava",
        "phi", "starcoder", "stablelm", "mistral", "moondream", "chatqa", "minicpm-v:latest", "openbmb/minicpm-v4:4b",
        "pythia", "stable", "minicpm-v","codegemma:2b-code", "starcoder2", "yi-coder:1.5b"
    ]
}

ADDITIONS = {
    "thinkbot": ["deepseek-r1:8b", "deepseek-r1:1.5b", "vortex/helpingai-9b",],
    "chatbot" : ["llama3-chatqa:8b", "zongwei/gemma3-translator:4b"],
    "vision"  : ["llava-phi3", "qwen3-vl:2b", "qwen3-vl:4b"],
    "coding"  : ["deepseek-r1:8b", "deepseek-coder:6.7b", "granite3.1-moe:1b-instruct-q2_K", "codegemma:2b-code", "qwen2.5-coder:3b", "qwen2.5-coder:1.5b",
                   "llama2:7b", "dolphin-phi"],
    "rag"     : ["mxbai-embed-large", "nomic-embed-text"]  # explicit RAG models
}






# --- Put this next to your EXCLUSIONS, ADDITIONS, MODEL_NAMES definitions ---

# token splitter used for "word" matching
_TOKEN_SPLIT_RE = re.compile(r'[/:._\-\s]+')

def _split_model_tokens(model: str) -> List[str]:
    """
    Return normalized tokens for matching.
      - remove tag part (after ':')
      - split on '/', '-', '_', '.', whitespace
      - include the whole lowercased base name as an extra token for loose matches
    Example: "owner/Some-Model:1.5b" -> ['owner', 'some', 'model', 'owner/some-model']
    """
    base = model.split(":", 1)[0].lower()
    parts = [p for p in _TOKEN_SPLIT_RE.split(base) if p]
    # include the full base as a token (owner/some-model) to allow matching 'owner/some-model' exceptions
    parts.append(base)
    return parts

def _matches_exclusion(model: str, exclusion: str) -> bool:
    """
    More precise exclusion matching:
      - If exclusion contains '/' or ':' we treat it as a full name/pattern and use substring match
        (keeps explicit owner/model or model:tag exclusions working).
      - Otherwise treat exclusion as a 'word' and match only whole tokens produced by _split_model_tokens.
    This avoids removing models just because a short substring is present inside a longer token.
    """
    if not exclusion:
        return False
    model_l = model.lower()
    excl_l = exclusion.lower()

    # explicit pattern case: keep substring match (this follows your previous behavior for exact entries)
    if "/" in excl_l or ":" in excl_l:
        return excl_l in model_l

    # word/token match otherwise
    tokens = _split_model_tokens(model)
    return excl_l in tokens

def get_models_by_type(model_type: str, debug: bool = False) -> List[str]:
    """
    Replace for the original get_models_by_type.
      - model_type: one of the keys in EXCLUSIONS (e.g., "thinkbot", "chatbot", "vision", "coding", "rag")
      - debug: when True, prints which models were excluded (useful for troubleshooting)
    Returns the filtered list (with ADDITIONS appended), deduplicated and sorted.
    """
    if model_type not in EXCLUSIONS:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {list(EXCLUSIONS.keys())}")

    excluded_words = EXCLUSIONS[model_type]
    kept = []
    removed = []

    for model in MODEL_NAMES:
        remove = False
        for ex in excluded_words:
            if _matches_exclusion(model, ex):
                remove = True
                removed.append((model, ex))
                break
        if not remove:
            kept.append(model)

    # Add manual inclusions if needed
    additions = ADDITIONS.get(model_type, [])
    kept.extend(additions)

    # Deduplicate while preserving order
    seen = set()
    final = []
    for m in kept:
        if m not in seen:
            seen.add(m)
            final.append(m)

    # Sort alphabetically for clarity (you can change to custom ordering if desired)
    final_sorted = sorted(final)

    if debug:
        print(f"[DEBUG] get_models_by_type('{model_type}') -> kept {len(final_sorted)} models, removed {len(removed)} models")
        if removed:
            print("[DEBUG] Removed samples (model, matched_exclusion):")
            for mod, ex in removed[:80]:
                print("  ", mod, " <-", ex)
            if len(removed) > 80:
                print("  ... (truncated)")

    return final_sorted






##def get_models_by_type(model_type: str):
##    """Return filtered model list for a given model type."""
##    if model_type not in EXCLUSIONS:
##        raise ValueError(f"Invalid model type: {model_type}. Must be one of {list(EXCLUSIONS.keys())}")
##
##    excluded_words = EXCLUSIONS[model_type]
##    models = []
##
##    for model in MODEL_NAMES:
##        if not any(word in model.lower() for word in excluded_words):
##            models.append(model)
##
##    # Add manual inclusions if needed
##    models += ADDITIONS.get(model_type, [])
##
##    # Deduplicate + sort for clarity
##    models = sorted(set(models))
##
##    return models

# -----------------------------
# If run as a script, show discovered / loaded models and optionally write output
# -----------------------------
def _print_loaded_models():
    print(f"# Loaded models ({len(MODEL_NAMES)}):")
    for m in MODEL_NAMES:
        print(" ", m)
    print()
    print("# Python list snippet you can paste into your code:")
    print(format_as_python_list(MODEL_NAMES))

    if WRITE_OUTPUT_FILE:
        out = Path(OUTPUT_FILENAME)
        out.write_text("# Auto-generated model list\n\n" + format_as_python_list(MODEL_NAMES))
        print(f"[INFO] Wrote output to {out.resolve()}")

if __name__ == "__main__":
    # allow passing a registry path override
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            registry_override = Path(arg)
            if registry_override.exists() and registry_override.is_dir():
                try:
                    found = discover_models(registry_override, max_depth=MAX_DEPTH)
                    if found:
                        MODEL_NAMES = found
                    else:
                        print(f"[INFO] No models discovered under: {registry_override}")
                except Exception as e:
                    print(f"[WARN] Could not discover under {registry_override}: {e}")
            else:
                print(f"[WARN] Provided path does not exist or is not a directory: {registry_override}")
        except Exception as e:
            print(f"[WARN] Invalid path argument: {e}")

    _print_loaded_models()


















##import os 
##
####MODEL_DIR = "C://Users//ITF//.ollama//models//manifests//registry.ollama.ai//library"
##MODEL_DIR = "D://Python_Env//New_Virtual_Env//ollama_models//models//manifests//registry.ollama.ai//library"
### Load model names once
##try:
##    ## MODEL_NAMES = os.listdir(MODEL_DIR)
##    MODEL_NAMES = [
##        "minicpm-v",
##        "qwen2.5-coder:3b",
##        "deepseek-r1:8b",
##        "qwen2.5:3b",
##        "llama2:7b",
##        "zongwei/gemma3-translator:4b",
##        "lauchacarro/qwen2.5-translator",
##        "0xroyce/NazareAI-Python-Programmer-3B",
##        "llama3-chatqa:8b",
##        "brxce/stable-diffusion-prompt-generator",
##        "granite3.1-moe:1b-instruct-q2_K",
##        "granite3.3:2b",
##        "qwen2.5-coder:1.5b",
##        "granite3.2-vision",
##        "Hudson/pythia-instruct:70m",
##        "yi-coder:1.5b",
##        "deepseek-r1:1.5b",
##        "llama3.2:1b",
##        "nomic-embed-text",
##        "mxbai-embed-large",
##        "gemma:2b",
##        "llava-phi3",
##        "codegemma:2b-code",
##        "moondream",
##        "llava",
##        "deepseek-coder:1.5b",
##        "deepseek-coder:6.7b",
##        "tinyllama",
##        "mistral",
##        "stablelm2",
##        "qwen:1.8b",
##        "starcoder2",
##        "dolphin-phi",
##        "phi",
##        "llama2"
##    ]
##
##except FileNotFoundError:
##    MODEL_NAMES = []
##    print(f"[ERROR] Model directory not found: {MODEL_DIR}")
##
##
##EXCLUSIONS = {
##    "thinkbot": [
##        "0xroyce/NazareAI-Python-Programmer-3B", "deepseek-r1", "code", "llava", "moondream", "vision", "-embed-", "chatqa", 
##        "pythia", "stable", "minicpm-v"
##    ],
##    "chatbot": [
##        "0xroyce", "deepseek-r1", "codegemma", "deepseek-coder", "deepseek-r1:8b", "dolphin-phi", "gemma",
##        "granite3.1-moe", "granite3.2-vision", "granite3.3", "llama2", 
##        "llama3.2", "llava", "llava-phi3", "mistral", "moondream", "mxbai-embed-large",
##        "nomic-embed-text", "qwen2.5-coder", "stablelm2", "starcoder2", "yi-coder", 
##        "pythia", "stable", "minicpm-v"
##    ],
##    "vision": [
##        "0xroyce", "phi", "qwen", "stablelm2", "mistral", "granite3.3", "deepseek-r1", "dolphin-phi", "gemma",
##        "granite3.1-moe", "code", "-embed-", "llama", 
##        "pythia"
##    ],
##    "coding": [
##        "llava", "granite3.2-vision", "moondream", "phi", "qwen", "granite3.3", "dolphin-phi",
##        "gemma", "granite3.1-moe", "-embed-", "llama", "chatqa", "deepseek-r1", "stable", "minicpm-v"
##    ],
##    "rag": [   # only embeddings allowed, exclude everything else
##        "0xroyce", "deepseek", "llama", "qwen", "gemma", "granite", "code", "llava",
##        "phi", "starcoder", "stablelm", "mistral", "moondream", "chatqa", 
##        "pythia", "stable", "minicpm-v"
##    ]
##}
##
##ADDITIONS = {
##    "thinkbot": ["deepseek-r1:8b", "deepseek-r1:1.5b"],
##    "chatbot" : ["llama3-chatqa:8b", "zongwei/gemma3-translator:4b"],
##    "vision"  : ["llava-phi3"],
##    "coding"  : ["deepseek-r1:8b", "deepseek-coder:6.7b", "granite3.1-moe:1b-instruct-q2_K", "codegemma:2b-code", "qwen2.5-coder:3b", "qwen2.5-coder:1.5b",
##                   "llama2:7b", "dolphin-phi"],
##    "rag"     : ["mxbai-embed-large", "nomic-embed-text"]  # explicit RAG models
##}
##
##
##def get_models_by_type(model_type: str):
##    """Return filtered model list for a given model type."""
##    if model_type not in EXCLUSIONS:
##        raise ValueError(f"Invalid model type: {model_type}. Must be one of {list(EXCLUSIONS.keys())}")
##
##    excluded_words = EXCLUSIONS[model_type]
##    models = []
##
##    for model in MODEL_NAMES:
##        if not any(word in model.lower() for word in excluded_words):
##            models.append(model)
##
##    # Add manual inclusions if needed
##    models += ADDITIONS.get(model_type, [])
##
##    # Deduplicate + sort for clarity
##    models = sorted(set(models))
##
##    return models
##


