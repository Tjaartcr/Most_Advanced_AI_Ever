# scheduled_commands.py
from __future__ import annotations
import re
import os
import json
import uuid
import time
import threading
from typing import List, Optional, Dict, Any, Tuple
import datetime as dt

# try to reuse project speech/listen objects
try:
    from speech import speech
except Exception:
    speech = None
try:
    from listen import listen
except Exception:
    listen = None

# basic helpers
def safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    try:
        return str(val)
    except Exception:
        return ""

import base64
import ast

# -------------------------
# extractor (adapted from your original)
# -------------------------
def extract_text_from_timed_command(query):
    """
    Returns: (message, speaker, score, gender, gender_conf, timestamp)
    """
    if query is None:
        return "", None, None, None, None, None

    def _extract_timestamp(fragment: str):
        if not fragment:
            return None
        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
        if m_date and m_time:
            return f"{m_date.group('date')} {m_time.group('time')}"
        if m_date:
            return m_date.group('date')
        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
        if m_date2:
            return m_date2.group('date')
        return None

    # ---------- dict case ----------
    if isinstance(query, dict):
        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
        speaker_ = query.get("username") or query.get("speaker")
        score_ = query.get("score")
        gender_ = query.get("gender")
        gender_conf_ = query.get("gender_conf")
        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)

    # --- string case ---
    if isinstance(query, str):
        s = query.strip()

        # base64 decode heuristic
        try:
            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
                try:
                    decoded = base64.b64decode(s).decode('utf-8')
                    if decoded:
                        s = decoded.strip()
                except Exception:
                    pass
        except Exception:
            pass

        s = s.strip()

        def _extract_meta(fragment):
            frag = fragment or ""
            frag = str(frag)
            speaker = None
            score = None
            gender = None
            gender_conf = None

            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
            if m:
                speaker = m.group("u").strip()

            if speaker is None:
                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
                if m2:
                    speaker = m2.group("u").strip()

            if speaker is None:
                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
                if m3:
                    maybe_u = m3.group("u").strip()
                    if " " not in maybe_u or len(maybe_u) <= 24:
                        speaker = maybe_u

            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
            if ms:
                score = ms.group("s").strip()
            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
            if mg:
                gender = mg.group("g").strip()
            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
            if mgc:
                gender_conf = mgc.group("gc").strip()

            if speaker and not gender:
                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
                if mg_end:
                    gender = mg_end.group('gword').strip()
                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()

            return speaker, score, gender, gender_conf

        # detect triple fence wrappers
        first_triple = s.find("'''")
        backtick_first = s.find("```")
        wrapper_pos = None
        wrapper_token = None
        if first_triple != -1:
            wrapper_pos = first_triple; wrapper_token = "'''"
        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
            wrapper_pos = backtick_first; wrapper_token = "```"

        if wrapper_pos is not None:
            start = wrapper_pos
            end = s.find(wrapper_token, start + len(wrapper_token))
            if end != -1:
                inner = s[start + len(wrapper_token) : end]
                remainder = s[end + len(wrapper_token) :].strip()
                inner = inner.lstrip()
                inner = re.sub(r"^'+", "", inner)
                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
                speaker, score, gender, gender_conf = _extract_meta(remainder)
                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
                if not speaker:
                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
                    if m_all:
                        speaker = m_all.group("u").strip()
                message = inner.strip()
                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)

        # structured line pattern
        m_struct = re.match(
            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
            s
        )
        if m_struct:
            date = m_struct.group('date'); time_ = m_struct.group('time')
            timestamp = f"{date} {time_}"
            message = m_struct.group('msg').strip()
            user_full = m_struct.group('user').strip()
            gender = None
            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
            if m_gender:
                gender = m_gender.group('g').strip()
                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
            else:
                user_clean = user_full
            speaker = user_clean if user_clean else None
            return message, speaker, None, (gender if gender else None), None, timestamp

        # structured string attempt
        try:
            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
            m = re.search(pattern, s, flags=re.DOTALL)
            if m:
                message_ = m.group('message').strip()
                speaker_ = m.group('username').strip() or None
                score_ = m.group('score') or None
                gender_ = m.group('gender') or None
                gender_conf_ = m.group('gender_conf') or None
                timestamp_ = _extract_timestamp(s)
                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
        except Exception:
            pass

        # username anywhere
        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
        speaker = None; score = None; gender = None; gender_conf = None
        timestamp = _extract_timestamp(s)
        if m_user_any:
            speaker = m_user_any.group("u").strip()
            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
            if ms:
                score = ms.group("s").strip()
            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
            if mg:
                gender = mg.group("g").strip()
            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
            if mgc:
                gender_conf = mgc.group("gc").strip()
            candidate = s.strip()
            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)

        # username at end
        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
        timestamp = _extract_timestamp(s)
        if m_user2:
            speaker = m_user2.group("u").strip()
            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)

        # last-token username heuristic
        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
        timestamp = _extract_timestamp(s)
        if m_user3:
            maybe_u = m_user3.group("u").strip()
            maybe_body = m_user3.group("body").strip()
            if " " not in maybe_u or len(maybe_u) <= 24:
                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)

        # fallback
        return s, None, None, None, None, None

    return str(query).strip(), None, None, None, None, None

# -------------------------
# small words->number helpers (kept minimal and self-contained)
# -------------------------
_UNITS = {
    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19
}
_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}

def words_to_number(phrase: str) -> Optional[int]:
    if phrase is None: return None
    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
    if not words: return None
    total = 0; current = 0; valid = False
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]; valid = True
        elif w in _TENS:
            current += _TENS[w]; valid = True
        elif w == "and":
            continue
        else:
            return None
    return (total + current) if valid else None

# -------------------------
# time parser (improved subset)
# -------------------------
# expanded AM/PM phrases to include multi-word tokens like 'in the morning', 'in the evening'
_AM_WORDS = {"am","a.m.","a.m","a.m.","morning","this morning","in the morning"}
_PM_WORDS = {"pm","p.m.","p.p","pm.","evening","afternoon","tonight","this evening","in the evening","night","this afternoon"}

def _token_to_number(token: str) -> Optional[int]:
    token = safe_str(token).lower()
    if not token: return None
    if re.fullmatch(r"\d+", token):
        try: return int(token)
        except: return None
    if token in _UNITS: return _UNITS[token]
    if token in _TENS: return _TENS[token]
    if "-" in token:
        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
        if all(v is not None for v in vals): return sum(vals)
    return words_to_number(token)

def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
    """
    Detect AM/PM tokens (including multi-word tokens like 'in the morning'/'in the evening')
    and return (cleaned_string, 'am'|'pm'|None).
    It prefers longest matches (so "in the morning" matches before "morning").
    """
    s0 = safe_str(s).lower()
    ampm = None

    # check multi-word tokens first (longest-first)
    multi_am = ["in the morning", "this morning", "morning"]
    multi_pm = ["in the evening", "this evening", "this afternoon", "afternoon", "evening", "tonight", "night"]

    for w in multi_am:
        if re.search(r"\b" + re.escape(w) + r"\b", s0):
            ampm = "am"
            break
    if ampm is None:
        for w in multi_pm:
            if re.search(r"\b" + re.escape(w) + r"\b", s0):
                ampm = "pm"
                break

    # also check short forms like 'am', 'pm', 'a.m.', 'p.m.'
    if ampm is None:
        for w in ("a.m.","am","a.m","p.m.","pm","p.m"):
            if re.search(r"\b" + re.escape(w) + r"\b", s0):
                if 'p' in w:
                    ampm = "pm"
                else:
                    ampm = "am"
                break

    # map 'noon' / 'midnight'
    if re.search(r"\bnoon\b", s0):
        ampm = "pm"
    if re.search(r"\bmidnight\b", s0):
        ampm = "am"

    if ampm:
        # remove a wide range of appearances of the token(s)
        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|in the morning|this morning|morning|in the evening|this evening|evening|afternoon|tonight|night|noon|midnight|this afternoon)\b"
        s0 = re.sub(pattern, " ", s0)
        s0 = re.sub(r'\s+', ' ', s0).strip()
    return s0, ampm

def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
    """
    Robust spoken time -> (hour, minute) parser.
    """
    if spoken is None: return None
    if isinstance(spoken, dt.datetime): return (spoken.hour, spoken.minute)
    if isinstance(spoken, dt.time): return (spoken.hour, spoken.minute)

    s_orig = safe_str(spoken)
    s = s_orig.lower().replace("-", " ").replace(".", " ").replace(",", " ").strip()
    if re.search(r"\bnoon\b", s): return (12, 0)
    if re.search(r"\bmidnight\b", s): return (0, 0)

    s_no_ampm, ampm = _detect_ampm_and_remove(s)

    # explicit 24h with colon or 'h'
    m_colon = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", s_no_ampm, flags=re.I)
    if m_colon:
        try:
            hh = int(m_colon.group(1)) % 24; mm = int(m_colon.group(2)) % 60
            hour = hh; minute = mm
            if ampm == "pm" and hour < 12: hour += 12
            if ampm == "am" and hour == 12: hour = 0
            return (hour, minute)
        except Exception:
            pass

    m_half = re.search(r"\bhalf past ([a-z0-9 ]+)\b", s_no_ampm)
    if m_half:
        token = m_half.group(1).strip(); h = _token_to_number(token)
        if h is not None:
            hour = int(h) % 24; minute = 30
            if ampm == "pm" and hour < 12: hour += 12
            if ampm == "am" and hour == 12: hour = 0
            return (hour, minute)

    m_quarter = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)\b", s_no_ampm)
    if m_quarter:
        typ = m_quarter.group(1); hour_token = m_quarter.group(2).strip(); h = _token_to_number(hour_token)
        if h is not None:
            hour = int(h) % 24
            if typ == "past":
                minute = 15
            else:
                minute = 45; hour = (hour - 1) % 24
            if ampm == "pm" and hour < 12: hour += 12
            if ampm == "am" and hour == 12: hour = 0
            return (hour, minute)

    m_past = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*past\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
    if m_past:
        try:
            mins = int(m_past.group(1))
            htoken = m_past.group(2)
            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
            if h is not None:
                hour = int(h) % 24; minute = mins % 60
                if ampm == "pm" and hour < 12: hour += 12
                if ampm == "am" and hour == 12: hour = 0
                return (hour, minute)
        except Exception:
            pass

    m_to = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*to\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
    if m_to:
        try:
            mins = int(m_to.group(1))
            htoken = m_to.group(2)
            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
            if h is not None:
                hour = (int(h) - 1) % 24; minute = (60 - (mins % 60)) % 60
                if ampm == "pm" and hour < 12: hour += 12
                if ampm == "am" and hour == 12: hour = 0
                return (hour, minute)
        except Exception:
            pass

    # Improved o'clock detection: accept word hours ("one o clock") as well as digits ("1 o'clock")
    m_oclock = re.search(r"\b(?P<h>\d{1,2}|[a-z]+)\s*(?:o['\s]?clock|oclock|o clock)\b", s_no_ampm)
    if m_oclock:
        try:
            h_raw = m_oclock.group('h')
            if re.fullmatch(r"\d+", h_raw):
                hour_val = int(h_raw) % 24
            else:
                hr = _token_to_number(h_raw)
                if hr is None:
                    raise ValueError("invalid hour token")
                hour_val = int(hr) % 24
            minute = 0
            if ampm == "pm" and hour_val < 12:
                hour_val += 12
            if ampm == "am" and hour_val == 12:
                hour_val = 0
            return (hour_val, minute)
        except Exception:
            pass

    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
    if len(tokens) >= 2:
        h_candidate = _token_to_number(tokens[0])
        m_candidate = _token_to_number(tokens[1])
        if h_candidate is not None and m_candidate is not None and 0 <= m_candidate < 60:
            hour = int(h_candidate) % 24; minute = int(m_candidate) % 60
            if ampm == "pm" and hour < 12: hour += 12
            if ampm == "am" and hour == 12: hour = 0
            return (hour, minute)

    if len(tokens) == 1:
        h = _token_to_number(tokens[0])
        if h is not None:
            hour = int(h) % 24; minute = 0
            if ampm == "pm" and hour < 12: hour += 12
            if ampm == "am" and hour == 12: hour = 0
            return (hour, minute)

    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
    if digits_cluster:
        cluster = digits_cluster.group(1)
        try:
            if len(cluster) == 3: h = int(cluster[0]); m = int(cluster[1:])
            else: h = int(cluster[:2]); m = int(cluster[2:])
            if 0 <= h < 24 and 0 <= m < 60:
                hour = h; minute = m
                if ampm == "pm" and hour < 12: hour += 12
                if ampm == "am" and hour == 12: hour = 0
                return (hour, minute)
        except Exception:
            pass

    return None

# -------------------------
# persistence
# -------------------------
SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
os.makedirs(SCHEDULE_DIR, exist_ok=True)
SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
scheduled_events: List[dict] = []

def _load_scheduled_events():
    global scheduled_events
    try:
        if os.path.exists(SCHEDULE_DB):
            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
                scheduled_events = json.load(f)
        else:
            scheduled_events = []
    except Exception as e:
        print("Scheduled load failed:", e); scheduled_events = []

def _save_scheduled_events():
    try:
        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
            json.dump(scheduled_events, f, indent=2, default=str)
    except Exception as e:
        print("Scheduled save failed:", e)

# -------------------------
# schedule management
# -------------------------
def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
    try:
        ev = {
            "id": uuid.uuid4().hex,
            "command": command_text,
            "username": username or "Itf",
            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
            "description": description,
            "fired": False
        }
        scheduled_events.append(ev)
        _save_scheduled_events()
        return ev
    except Exception as e:
        print("add_scheduled_command failed:", e)
        raise

# -------------------------
# parsing helpers (improvements)
# -------------------------
def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
    out = []
    for p in parts:
        p_clean = p.strip()
        if not p_clean:
            continue
        hm = spoken_time_to_hm(p_clean)
        rel_dt = None
        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
        if m_rel:
            num_phrase = m_rel.group(2).strip()
            unit = m_rel.group(3).lower()
            try:
                num = int(num_phrase)
            except:
                num = words_to_number(num_phrase)
            if num is not None:
                now = dt.datetime.now()
                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
                elif unit.startswith("minute"): rel_dt = now + dt.timedelta(minutes=num)
                elif unit.startswith("second"): rel_dt = now + dt.timedelta(seconds=num)
                elif unit.startswith("day"): rel_dt = now + dt.timedelta(days=num)
        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
    return out

def _strip_time_tokens_from_part(part: str) -> str:
    """
    Clean scheduling tokens from a phrase or a log line.
    """
    if not part:
        return part

    orig = part.strip()
    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
    command_candidate = None
    if len(segments) > 1:
        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
        if alpha_segments:
            command_candidate = max(alpha_segments, key=lambda s: len(s))
    s = command_candidate if command_candidate else orig
    s = re.sub(r'\s+', ' ', s).strip()

    # remove explicit "on <date>" (conservative)
    s = re.sub(r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', '', s, flags=re.I)

    # remove "in/after <...> (seconds|minutes|hours|days)" completely
    s = re.sub(r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b', '', s, flags=re.I)

    s = re.sub(r'\s+', ' ', s).strip()

    # If there's an 'at', cut EVERYTHING from 'at' onward
    at_m = re.search(r'\bat\b', s, flags=re.I)
    if at_m:
        s = s[:at_m.start()].rstrip()
        s = re.sub(r'\s+', ' ', s).strip(' ,.')
        return s

    # Handle 'on' occurrences:
    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
    if on_matches:
        first = on_matches[0]
        s = s[: first.end()].rstrip()
        s = re.sub(r'\s+', ' ', s).strip(' ,.')
        return s

    s = re.sub(r'\s+', ' ', s).strip(' ,.')
    return s

# -------------------------
# recurrence & date parsing helpers
# -------------------------
_WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
def _next_weekday_date(weekday_idx: int, start: Optional[dt.date] = None) -> dt.date:
    start = start or dt.date.today()
    days_ahead = (weekday_idx - start.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return start + dt.timedelta(days=days_ahead)

def _parse_base_date_from_part(part: str) -> dt.date:
    txt = safe_str(part).lower()
    today = dt.date.today()
    if "today" in txt:
        return today
    if "tomorrow" in txt:
        return today + dt.timedelta(days=1)
    # 'this evening', 'this morning', 'in the morning', 'in the evening'
    if re.search(r'\b(this|in the)?\s*(evening|morning|afternoon|night)\b', txt):
        return today
    # weekdays
    for wd, idx in _WEEKDAY_MAP.items():
        if re.search(rf'\b{wd}\b', txt):
            if re.search(r'\bthis\b', txt) and idx == today.weekday():
                return today
            return _next_weekday_date(idx, start=today)
    # default
    return today

def _parse_recurrence_rules(part: str) -> Dict[str, Any]:
    txt = safe_str(part).lower()
    rules = {"freq": None, "interval": 1, "count": None, "until": None, "weekdays": None}

    m_every = re.search(r'\bevery\s+(second\s+day|second\s+week|day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', txt)
    if m_every:
        token = m_every.group(1)
        if "second day" in token or "every second day" in txt:
            rules["freq"] = "daily"; rules["interval"] = 2
        elif "second week" in token or "every second week" in txt:
            rules["freq"] = "weekly"; rules["interval"] = 2
        elif token in ("day",):
            rules["freq"] = "daily"; rules["interval"] = 1
        elif token == "week":
            rules["freq"] = "weekly"; rules["interval"] = 1
        elif token == "month":
            rules["freq"] = "monthly"; rules["interval"] = 1
        elif token in _WEEKDAY_MAP:
            rules["freq"] = "weekly"; rules["weekdays"] = [ _WEEKDAY_MAP[token] ]

    # Accept digits or word numbers (e.g., "two")
    m_next = re.search(r'for the next\s+(\d+|[a-z]+)\s+(days?|weeks?|months?)', txt)
    if m_next:
        n_token = m_next.group(1)
        unit = m_next.group(2)
        try:
            n = int(n_token)
        except Exception:
            n = words_to_number(n_token)
        if n is None:
            n = 1
        if unit.startswith("day"):
            # prefer count days
            rules["count"] = n
            if rules["freq"] is None:
                rules["freq"] = "daily"
        elif unit.startswith("week"):
            # schedule daily for next n weeks (common interpretation)
            rules["until"] = (dt.date.today() + dt.timedelta(weeks=n))
            if rules["freq"] is None:
                rules["freq"] = "daily"
        elif unit.startswith("month"):
            rules["until"] = (dt.date.today() + dt.timedelta(days=30 * n))
            if rules["freq"] is None:
                rules["freq"] = "daily"

    if re.search(r'\bfor the month\b', txt) or re.search(r'\bfor a month\b', txt):
        rules["until"] = dt.date.today() + dt.timedelta(days=30)
        if rules["freq"] is None:
            rules["freq"] = "daily"

    if re.search(r'\bthis week\b', txt) and re.search(r'\bevery\b', txt):
        today = dt.date.today()
        end = today + dt.timedelta(days=(6 - today.weekday()))
        rules["until"] = end
        rules["freq"] = "daily"

    return rules

def _generate_recurrence_datetimes(base_dt: dt.datetime, rules: Dict[str, Any]) -> List[dt.datetime]:
    if not rules or not any([rules.get("freq"), rules.get("until"), rules.get("count"), rules.get("weekdays")]):
        return [base_dt]

    out = []
    freq = rules.get("freq")
    interval = max(1, int(rules.get("interval", 1)))
    count = rules.get("count")
    until = rules.get("until")
    weekdays = rules.get("weekdays")

    # If there's an "until" but no explicit freq, default to daily recurrence
    if freq is None and until is not None and weekdays is None:
        freq = "daily"

    if until is None and (freq is not None or weekdays is not None):
        until = (base_dt.date() + dt.timedelta(days=30))  # default one month
        # if we defaulted an 'until' but freq still None, set daily
        if freq is None and weekdays is None:
            freq = "daily"

    if weekdays:
        cur_date = base_dt.date()
        while cur_date <= until:
            if cur_date.weekday() in weekdays:
                out.append(dt.datetime.combine(cur_date, base_dt.time()))
            cur_date = cur_date + dt.timedelta(days=1)
        return out

    if freq == "daily":
        cur = base_dt
        while True:
            if until and cur.date() > until:
                break
            out.append(cur)
            if count and len(out) >= count:
                break
            cur = cur + dt.timedelta(days=interval)
        return out

    if freq == "weekly":
        cur = base_dt
        while True:
            if until and cur.date() > until:
                break
            out.append(cur)
            if count and len(out) >= count:
                break
            cur = cur + dt.timedelta(weeks=interval)
        return out

    if freq == "monthly":
        cur = base_dt
        while True:
            if until and cur.date() > until:
                break
            out.append(cur)
            if count and len(out) >= count:
                break
            cur = cur + dt.timedelta(days=30 * interval)
        return out

    return [base_dt]

# -------------------------
# executor thread
# -------------------------
_SCHEDULER_THREAD = None
_SCHEDULER_LOCK = threading.Lock()

def _execute_event(ev: dict):
    try:
        cmd = ev.get("command", "")
        user = ev.get("username", "Itf")
        if listen is not None and hasattr(listen, "listen"):
            text_msg = cmd
            try:
                if hasattr(listen, "add_text"):
                    listen.add_text(text_msg)
                else:
                    try:
                        listen_queue = getattr(listen, "queue", None)
                        if listen_queue is not None and hasattr(listen_queue, "put"):
                            listen_queue.put(text_msg)
                    except Exception:
                        pass
                if speech is not None and hasattr(speech, "AlfredSpeak"):
                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
                else:
                    print("[scheduled_commands] Executing:", cmd)
            except Exception as e:
                print("Error calling listen.add_text:", e)
                if speech is not None and hasattr(speech, "AlfredSpeak"):
                    speech.AlfredSpeak("Failed to execute timed command.")
        else:
            if speech is not None and hasattr(speech, "AlfredSpeak"):
                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
            else:
                print("(No listen available) Would execute:", cmd)
    except Exception as e:
        print("_execute_event error:", e)

def _scheduler_loop(poll_seconds: int = 15):
    while True:
        try:
            now = dt.datetime.now()
            to_execute: List[dict] = []
            changed = False

            # Collect due events under lock and mark them fired
            with _SCHEDULER_LOCK:
                for ev in list(scheduled_events):
                    try:
                        if ev.get("fired", False):
                            continue
                        dtstart = dt.datetime.fromisoformat(ev["dtstart"])
                        if now >= dtstart:
                            ev["fired"] = True
                            changed = True
                            # copy the event for execution outside the lock
                            to_execute.append(dict(ev))
                    except Exception as e:
                        print("scheduled event inspect error:", e)
                        continue
                if changed:
                    # persist the 'fired' status before executing
                    _save_scheduled_events()

            # Execute the due events outside the lock
            for ev in to_execute:
                try:
                    _execute_event(ev)
                except Exception as e:
                    print("Error executing scheduled event:", e)

            # Cleanup: remove events that have been fired (we already marked & saved them)
            # Use lock while mutating the shared list
            with _SCHEDULER_LOCK:
                before_len = len(scheduled_events)
                # remove items where fired == True
                scheduled_events[:] = [ev for ev in scheduled_events if not ev.get("fired", False)]
                after_len = len(scheduled_events)
                if after_len != before_len:
                    # persist removal
                    _save_scheduled_events()

        except Exception as e:
            print("Scheduler loop error:", e)
        time.sleep(poll_seconds)

def start_scheduler_thread(poll_seconds: int = 15):
    global _SCHEDULER_THREAD
    with _SCHEDULER_LOCK:
        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
            return
        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
        _SCHEDULER_THREAD.start()

# -------------------------
# public API: handle a user utterance and schedule commands if found
# -------------------------
def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
    """
    Parse `text` for time-based commands. If found, schedule them and return a list
    of created event dicts. If no scheduling performed, return None.
    """
    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)

    New_Message = message
    New_speaker = speaker
    New_score = score
    New_gender = gender
    New_gender_conf = gender_conf
    New_timestamp = timestamp

    print(f"New_Message Timed Command Module       : {New_Message}")
    print(f"New_speaker Timed Command Module       : {New_speaker}")
    print(f"New_score Timed Command Module         : {New_score}")
    print(f"New_gender Timed Command Module        : {New_gender}")
    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
    print(f"New_timestamp Timed Command Module     : {New_timestamp}")

    text_clean = safe_str(text)
    lower = text_clean.lower()

    # conservative time indicators (expanded)
    time_indicators = bool(re.search(r"\b(at|o'clock|o clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2}|this (morning|afternoon|evening|night)|in the morning|in the evening|evening|morning|tonight)\b", text_clean, flags=re.I))
    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
    if not time_indicators or reminder_like:
        return None

    parts = _parse_command_sequence_parts(text_clean)
    if not parts:
        return None

    scheduled = []
    for p in parts:
        part_text = p.get("part", "")
        cmd_text = _strip_time_tokens_from_part(part_text) or part_text

        # parse a concrete datetime (prefer rel_dt, then explicit hm+date)
        dtstart = None
        if p.get("rel_dt"):
            dtstart = p["rel_dt"]
        else:
            # determine base date from the whole part (order-insensitive)
            base_date = _parse_base_date_from_part(part_text)

            hm = p.get("hm")
            if hm:
                h, m = hm
                try:
                    cand = dt.datetime.combine(base_date, dt.time(h, m))
                except Exception:
                    cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
                # if cand is in the past and user didn't explicitly say "today", push to next day
                if cand < dt.datetime.now() and not re.search(r'\btoday\b', part_text.lower()):
                    cand = cand + dt.timedelta(days=1)
                dtstart = cand
            else:
                if New_timestamp:
                    try:
                        parsed_ts = dt.datetime.fromisoformat(New_timestamp.replace(" ", "T"))
                        dtstart = parsed_ts
                    except Exception:
                        try:
                            dtstart = dt.datetime.fromisoformat(New_timestamp)
                        except Exception:
                            dtstart = None

        if dtstart is None:
            dtstart = dt.datetime.now() + dt.timedelta(minutes=1)

        # recurrence
        rules = _parse_recurrence_rules(part_text)
        occurrence_datetimes = _generate_recurrence_datetimes(dtstart, rules)

        for occ in occurrence_datetimes:
            try:
                ev = add_scheduled_command(cmd_text, occ, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
                scheduled.append(ev)
            except Exception as e:
                print("Failed to schedule part:", e)
                continue

    if scheduled:
        if speech is not None and hasattr(speech, "AlfredSpeak"):
            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
        else:
            print(f"Scheduled {len(scheduled)} command(s).")
        if gui is not None and hasattr(gui, "log_query"):
            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
        return scheduled
    return None

# initialization
_load_scheduled_events()
start_scheduler_thread()



















### BEST WORKING 2026_02_06__00h30
##
### scheduled_commands.py
##from __future__ import annotations
##import re
##import os
##import json
##import uuid
##import time
##import threading
##from typing import List, Optional, Dict, Any, Tuple
##import datetime as dt
##
### try to reuse project speech/listen objects
##try:
##    from speech import speech
##except Exception:
##    speech = None
##try:
##    from listen import listen
##except Exception:
##    listen = None
##
### basic helpers
##def safe_str(val) -> str:
##    if val is None:
##        return ""
##    if isinstance(val, str):
##        return val.strip()
##    try:
##        return str(val)
##    except Exception:
##        return ""
##
##import base64
##import ast
##
### -------------------------
### extractor (adapted from your original)
### -------------------------
##def extract_text_from_timed_command(query):
##    """
##    Returns: (message, speaker, score, gender, gender_conf, timestamp)
##    """
##    if query is None:
##        return "", None, None, None, None, None
##
##    def _extract_timestamp(fragment: str):
##        if not fragment:
##            return None
##        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
##        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
##        if m_date and m_time:
##            return f"{m_date.group('date')} {m_time.group('time')}"
##        if m_date:
##            return m_date.group('date')
##        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
##        if m_date2:
##            return m_date2.group('date')
##        return None
##
##    # ---------- dict case ----------
##    if isinstance(query, dict):
##        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
##        speaker_ = query.get("username") or query.get("speaker")
##        score_ = query.get("score")
##        gender_ = query.get("gender")
##        gender_conf_ = query.get("gender_conf")
##        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
##        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)
##
##    # --- string case ---
##    if isinstance(query, str):
##        s = query.strip()
##
##        # base64 decode heuristic
##        try:
##            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
##                try:
##                    decoded = base64.b64decode(s).decode('utf-8')
##                    if decoded:
##                        s = decoded.strip()
##                except Exception:
##                    pass
##        except Exception:
##            pass
##
##        s = s.strip()
##
##        def _extract_meta(fragment):
##            frag = fragment or ""
##            frag = str(frag)
##            speaker = None
##            score = None
##            gender = None
##            gender_conf = None
##
##            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
##            if m:
##                speaker = m.group("u").strip()
##
##            if speaker is None:
##                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
##                if m2:
##                    speaker = m2.group("u").strip()
##
##            if speaker is None:
##                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
##                if m3:
##                    maybe_u = m3.group("u").strip()
##                    if " " not in maybe_u or len(maybe_u) <= 24:
##                        speaker = maybe_u
##
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##
##            if speaker and not gender:
##                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
##                if mg_end:
##                    gender = mg_end.group('gword').strip()
##                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
##
##            return speaker, score, gender, gender_conf
##
##        # detect triple fence wrappers
##        first_triple = s.find("'''")
##        backtick_first = s.find("```")
##        wrapper_pos = None
##        wrapper_token = None
##        if first_triple != -1:
##            wrapper_pos = first_triple; wrapper_token = "'''"
##        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
##            wrapper_pos = backtick_first; wrapper_token = "```"
##
##        if wrapper_pos is not None:
##            start = wrapper_pos
##            end = s.find(wrapper_token, start + len(wrapper_token))
##            if end != -1:
##                inner = s[start + len(wrapper_token) : end]
##                remainder = s[end + len(wrapper_token) :].strip()
##                inner = inner.lstrip()
##                inner = re.sub(r"^'+", "", inner)
##                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
##                speaker, score, gender, gender_conf = _extract_meta(remainder)
##                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
##                if not speaker:
##                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
##                    if m_all:
##                        speaker = m_all.group("u").strip()
##                message = inner.strip()
##                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # structured line pattern
##        m_struct = re.match(
##            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
##            s
##        )
##        if m_struct:
##            date = m_struct.group('date'); time_ = m_struct.group('time')
##            timestamp = f"{date} {time_}"
##            message = m_struct.group('msg').strip()
##            user_full = m_struct.group('user').strip()
##            gender = None
##            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
##            if m_gender:
##                gender = m_gender.group('g').strip()
##                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
##            else:
##                user_clean = user_full
##            speaker = user_clean if user_clean else None
##            return message, speaker, None, (gender if gender else None), None, timestamp
##
##        # structured string attempt
##        try:
##            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
##            m = re.search(pattern, s, flags=re.DOTALL)
##            if m:
##                message_ = m.group('message').strip()
##                speaker_ = m.group('username').strip() or None
##                score_ = m.group('score') or None
##                gender_ = m.group('gender') or None
##                gender_conf_ = m.group('gender_conf') or None
##                timestamp_ = _extract_timestamp(s)
##                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
##        except Exception:
##            pass
##
##        # username anywhere
##        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
##        speaker = None; score = None; gender = None; gender_conf = None
##        timestamp = _extract_timestamp(s)
##        if m_user_any:
##            speaker = m_user_any.group("u").strip()
##            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##            candidate = s.strip()
##            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # username at end
##        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
##        timestamp = _extract_timestamp(s)
##        if m_user2:
##            speaker = m_user2.group("u").strip()
##            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
##            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
##
##        # last-token username heuristic
##        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
##        timestamp = _extract_timestamp(s)
##        if m_user3:
##            maybe_u = m_user3.group("u").strip()
##            maybe_body = m_user3.group("body").strip()
##            if " " not in maybe_u or len(maybe_u) <= 24:
##                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
##
##        # fallback
##        return s, None, None, None, None, None
##
##    return str(query).strip(), None, None, None, None, None
##
### -------------------------
### small words->number helpers (kept minimal and self-contained)
### -------------------------
##_UNITS = {
##    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
##    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
##    "seventeen":17,"eighteen":18,"nineteen":19
##}
##_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
##
##def words_to_number(phrase: str) -> Optional[int]:
##    if phrase is None: return None
##    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
##    if not words: return None
##    total = 0; current = 0; valid = False
##    for w in words:
##        if w in _UNITS:
##            current += _UNITS[w]; valid = True
##        elif w in _TENS:
##            current += _TENS[w]; valid = True
##        elif w == "and":
##            continue
##        else:
##            return None
##    return (total + current) if valid else None
##
### -------------------------
### time parser (improved subset)
### -------------------------
### expanded AM/PM phrases to include multi-word tokens like 'in the morning', 'in the evening'
##_AM_WORDS = {"am","a.m.","a.m","a.m.","morning","this morning","in the morning"}
##_PM_WORDS = {"pm","p.m.","p.p","pm.","evening","afternoon","tonight","this evening","in the evening","night","this afternoon"}
##
##def _token_to_number(token: str) -> Optional[int]:
##    token = safe_str(token).lower()
##    if not token: return None
##    if re.fullmatch(r"\d+", token):
##        try: return int(token)
##        except: return None
##    if token in _UNITS: return _UNITS[token]
##    if token in _TENS: return _TENS[token]
##    if "-" in token:
##        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
##        if all(v is not None for v in vals): return sum(vals)
##    return words_to_number(token)
##
##def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
##    """
##    Detect AM/PM tokens (including multi-word tokens like 'in the morning'/'in the evening')
##    and return (cleaned_string, 'am'|'pm'|None).
##    It prefers longest matches (so "in the morning" matches before "morning").
##    """
##    s0 = safe_str(s).lower()
##    ampm = None
##
##    # check multi-word tokens first (longest-first)
##    multi_am = ["in the morning", "this morning", "morning"]
##    multi_pm = ["in the evening", "this evening", "this afternoon", "afternoon", "evening", "tonight", "night"]
##
##    for w in multi_am:
##        if re.search(r"\b" + re.escape(w) + r"\b", s0):
##            ampm = "am"
##            break
##    if ampm is None:
##        for w in multi_pm:
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                ampm = "pm"
##                break
##
##    # also check short forms like 'am', 'pm', 'a.m.', 'p.m.'
##    if ampm is None:
##        for w in ("a.m.","am","a.m","p.m.","pm","p.m"):
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                if 'p' in w:
##                    ampm = "pm"
##                else:
##                    ampm = "am"
##                break
##
##    # map 'noon' / 'midnight'
##    if re.search(r"\bnoon\b", s0):
##        ampm = "pm"
##    if re.search(r"\bmidnight\b", s0):
##        ampm = "am"
##
##    if ampm:
##        # remove a wide range of appearances of the token(s)
##        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|in the morning|this morning|morning|in the evening|this evening|evening|afternoon|tonight|night|noon|midnight|this afternoon)\b"
##        s0 = re.sub(pattern, " ", s0)
##        s0 = re.sub(r'\s+', ' ', s0).strip()
##    return s0, ampm
##
##def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
##    """
##    Robust spoken time -> (hour, minute) parser.
##    """
##    if spoken is None: return None
##    if isinstance(spoken, dt.datetime): return (spoken.hour, spoken.minute)
##    if isinstance(spoken, dt.time): return (spoken.hour, spoken.minute)
##
##    s_orig = safe_str(spoken)
##    s = s_orig.lower().replace("-", " ").replace(".", " ").replace(",", " ").strip()
##    if re.search(r"\bnoon\b", s): return (12, 0)
##    if re.search(r"\bmidnight\b", s): return (0, 0)
##
##    s_no_ampm, ampm = _detect_ampm_and_remove(s)
##
##    # explicit 24h with colon or 'h'
##    m_colon = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", s_no_ampm, flags=re.I)
##    if m_colon:
##        try:
##            hh = int(m_colon.group(1)) % 24; mm = int(m_colon.group(2)) % 60
##            hour = hh; minute = mm
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##        except Exception:
##            pass
##
##    m_half = re.search(r"\bhalf past ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_half:
##        token = m_half.group(1).strip(); h = _token_to_number(token)
##        if h is not None:
##            hour = int(h) % 24; minute = 30
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_quarter = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_quarter:
##        typ = m_quarter.group(1); hour_token = m_quarter.group(2).strip(); h = _token_to_number(hour_token)
##        if h is not None:
##            hour = int(h) % 24
##            if typ == "past":
##                minute = 15
##            else:
##                minute = 45; hour = (hour - 1) % 24
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_past = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*past\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_past:
##        try:
##            mins = int(m_past.group(1))
##            htoken = m_past.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = int(h) % 24; minute = mins % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    m_to = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*to\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_to:
##        try:
##            mins = int(m_to.group(1))
##            htoken = m_to.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = (int(h) - 1) % 24; minute = (60 - (mins % 60)) % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    # Improved o'clock detection: accept word hours ("one o clock") as well as digits ("1 o'clock")
##    m_oclock = re.search(r"\b(?P<h>\d{1,2}|[a-z]+)\s*(?:o['\s]?clock|oclock|o clock)\b", s_no_ampm)
##    if m_oclock:
##        try:
##            h_raw = m_oclock.group('h')
##            if re.fullmatch(r"\d+", h_raw):
##                hour_val = int(h_raw) % 24
##            else:
##                hr = _token_to_number(h_raw)
##                if hr is None:
##                    raise ValueError("invalid hour token")
##                hour_val = int(hr) % 24
##            minute = 0
##            if ampm == "pm" and hour_val < 12:
##                hour_val += 12
##            if ampm == "am" and hour_val == 12:
##                hour_val = 0
##            return (hour_val, minute)
##        except Exception:
##            pass
##
##    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
##    if len(tokens) >= 2:
##        h_candidate = _token_to_number(tokens[0])
##        m_candidate = _token_to_number(tokens[1])
##        if h_candidate is not None and m_candidate is not None and 0 <= m_candidate < 60:
##            hour = int(h_candidate) % 24; minute = int(m_candidate) % 60
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    if len(tokens) == 1:
##        h = _token_to_number(tokens[0])
##        if h is not None:
##            hour = int(h) % 24; minute = 0
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
##    if digits_cluster:
##        cluster = digits_cluster.group(1)
##        try:
##            if len(cluster) == 3: h = int(cluster[0]); m = int(cluster[1:])
##            else: h = int(cluster[:2]); m = int(cluster[2:])
##            if 0 <= h < 24 and 0 <= m < 60:
##                hour = h; minute = m
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    return None
##
### -------------------------
### persistence
### -------------------------
##SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
##os.makedirs(SCHEDULE_DIR, exist_ok=True)
##SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
##scheduled_events: List[dict] = []
##
##def _load_scheduled_events():
##    global scheduled_events
##    try:
##        if os.path.exists(SCHEDULE_DB):
##            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
##                scheduled_events = json.load(f)
##        else:
##            scheduled_events = []
##    except Exception as e:
##        print("Scheduled load failed:", e); scheduled_events = []
##
##def _save_scheduled_events():
##    try:
##        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
##            json.dump(scheduled_events, f, indent=2, default=str)
##    except Exception as e:
##        print("Scheduled save failed:", e)
##
### -------------------------
### schedule management
### -------------------------
##def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
##    try:
##        ev = {
##            "id": uuid.uuid4().hex,
##            "command": command_text,
##            "username": username or "Itf",
##            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
##            "description": description,
##            "fired": False
##        }
##        scheduled_events.append(ev)
##        _save_scheduled_events()
##        return ev
##    except Exception as e:
##        print("add_scheduled_command failed:", e)
##        raise
##
### -------------------------
### parsing helpers (improvements)
### -------------------------
##def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
##    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
##    out = []
##    for p in parts:
##        p_clean = p.strip()
##        if not p_clean:
##            continue
##        hm = spoken_time_to_hm(p_clean)
##        rel_dt = None
##        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
##        if m_rel:
##            num_phrase = m_rel.group(2).strip()
##            unit = m_rel.group(3).lower()
##            try:
##                num = int(num_phrase)
##            except:
##                num = words_to_number(num_phrase)
##            if num is not None:
##                now = dt.datetime.now()
##                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
##                elif unit.startswith("minute"): rel_dt = now + dt.timedelta(minutes=num)
##                elif unit.startswith("second"): rel_dt = now + dt.timedelta(seconds=num)
##                elif unit.startswith("day"): rel_dt = now + dt.timedelta(days=num)
##        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
##    return out
##
##def _strip_time_tokens_from_part(part: str) -> str:
##    """
##    Clean scheduling tokens from a phrase or a log line.
##    """
##    if not part:
##        return part
##
##    orig = part.strip()
##    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
##    command_candidate = None
##    if len(segments) > 1:
##        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
##        if alpha_segments:
##            command_candidate = max(alpha_segments, key=lambda s: len(s))
##    s = command_candidate if command_candidate else orig
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # remove explicit "on <date>" (conservative)
##    s = re.sub(r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', '', s, flags=re.I)
##
##    # remove "in/after <...> (seconds|minutes|hours|days)" completely
##    s = re.sub(r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b', '', s, flags=re.I)
##
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # If there's an 'at', cut EVERYTHING from 'at' onward
##    at_m = re.search(r'\bat\b', s, flags=re.I)
##    if at_m:
##        s = s[:at_m.start()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    # Handle 'on' occurrences:
##    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
##    if on_matches:
##        first = on_matches[0]
##        s = s[: first.end()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    s = re.sub(r'\s+', ' ', s).strip(' ,.')
##    return s
##
### -------------------------
### recurrence & date parsing helpers
### -------------------------
##_WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
##def _next_weekday_date(weekday_idx: int, start: Optional[dt.date] = None) -> dt.date:
##    start = start or dt.date.today()
##    days_ahead = (weekday_idx - start.weekday()) % 7
##    if days_ahead == 0:
##        days_ahead = 7
##    return start + dt.timedelta(days=days_ahead)
##
##def _parse_base_date_from_part(part: str) -> dt.date:
##    txt = safe_str(part).lower()
##    today = dt.date.today()
##    if "today" in txt:
##        return today
##    if "tomorrow" in txt:
##        return today + dt.timedelta(days=1)
##    # 'this evening', 'this morning', 'in the morning', 'in the evening'
##    if re.search(r'\b(this|in the)?\s*(evening|morning|afternoon|night)\b', txt):
##        return today
##    # weekdays
##    for wd, idx in _WEEKDAY_MAP.items():
##        if re.search(rf'\b{wd}\b', txt):
##            if re.search(r'\bthis\b', txt) and idx == today.weekday():
##                return today
##            return _next_weekday_date(idx, start=today)
##    # default
##    return today
##
##def _parse_recurrence_rules(part: str) -> Dict[str, Any]:
##    txt = safe_str(part).lower()
##    rules = {"freq": None, "interval": 1, "count": None, "until": None, "weekdays": None}
##
##    m_every = re.search(r'\bevery\s+(second\s+day|second\s+week|day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', txt)
##    if m_every:
##        token = m_every.group(1)
##        if "second day" in token or "every second day" in txt:
##            rules["freq"] = "daily"; rules["interval"] = 2
##        elif "second week" in token or "every second week" in txt:
##            rules["freq"] = "weekly"; rules["interval"] = 2
##        elif token in ("day",):
##            rules["freq"] = "daily"; rules["interval"] = 1
##        elif token == "week":
##            rules["freq"] = "weekly"; rules["interval"] = 1
##        elif token == "month":
##            rules["freq"] = "monthly"; rules["interval"] = 1
##        elif token in _WEEKDAY_MAP:
##            rules["freq"] = "weekly"; rules["weekdays"] = [ _WEEKDAY_MAP[token] ]
##
##    # Accept digits or word numbers (e.g., "two")
##    m_next = re.search(r'for the next\s+(\d+|[a-z]+)\s+(days?|weeks?|months?)', txt)
##    if m_next:
##        n_token = m_next.group(1)
##        unit = m_next.group(2)
##        try:
##            n = int(n_token)
##        except Exception:
##            n = words_to_number(n_token)
##        if n is None:
##            n = 1
##        if unit.startswith("day"):
##            # prefer count days
##            rules["count"] = n
##            if rules["freq"] is None:
##                rules["freq"] = "daily"
##        elif unit.startswith("week"):
##            # schedule daily for next n weeks (common interpretation)
##            rules["until"] = (dt.date.today() + dt.timedelta(weeks=n))
##            if rules["freq"] is None:
##                rules["freq"] = "daily"
##        elif unit.startswith("month"):
##            rules["until"] = (dt.date.today() + dt.timedelta(days=30 * n))
##            if rules["freq"] is None:
##                rules["freq"] = "daily"
##
##    if re.search(r'\bfor the month\b', txt) or re.search(r'\bfor a month\b', txt):
##        rules["until"] = dt.date.today() + dt.timedelta(days=30)
##        if rules["freq"] is None:
##            rules["freq"] = "daily"
##
##    if re.search(r'\bthis week\b', txt) and re.search(r'\bevery\b', txt):
##        today = dt.date.today()
##        end = today + dt.timedelta(days=(6 - today.weekday()))
##        rules["until"] = end
##        rules["freq"] = "daily"
##
##    return rules
##
##def _generate_recurrence_datetimes(base_dt: dt.datetime, rules: Dict[str, Any]) -> List[dt.datetime]:
##    if not rules or not any([rules.get("freq"), rules.get("until"), rules.get("count"), rules.get("weekdays")]):
##        return [base_dt]
##
##    out = []
##    freq = rules.get("freq")
##    interval = max(1, int(rules.get("interval", 1)))
##    count = rules.get("count")
##    until = rules.get("until")
##    weekdays = rules.get("weekdays")
##
##    # If there's an "until" but no explicit freq, default to daily recurrence
##    if freq is None and until is not None and weekdays is None:
##        freq = "daily"
##
##    if until is None and (freq is not None or weekdays is not None):
##        until = (base_dt.date() + dt.timedelta(days=30))  # default one month
##        # if we defaulted an 'until' but freq still None, set daily
##        if freq is None and weekdays is None:
##            freq = "daily"
##
##    if weekdays:
##        cur_date = base_dt.date()
##        while cur_date <= until:
##            if cur_date.weekday() in weekdays:
##                out.append(dt.datetime.combine(cur_date, base_dt.time()))
##            cur_date = cur_date + dt.timedelta(days=1)
##        return out
##
##    if freq == "daily":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=interval)
##        return out
##
##    if freq == "weekly":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(weeks=interval)
##        return out
##
##    if freq == "monthly":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=30 * interval)
##        return out
##
##    return [base_dt]
##
### -------------------------
### executor thread
### -------------------------
##_SCHEDULER_THREAD = None
##_SCHEDULER_LOCK = threading.Lock()
##
##def _execute_event(ev: dict):
##    try:
##        cmd = ev.get("command", "")
##        user = ev.get("username", "Itf")
##        if listen is not None and hasattr(listen, "listen"):
##            text_msg = cmd
##            try:
##                if hasattr(listen, "add_text"):
##                    listen.add_text(text_msg)
##                else:
##                    try:
##                        listen_queue = getattr(listen, "queue", None)
##                        if listen_queue is not None and hasattr(listen_queue, "put"):
##                            listen_queue.put(text_msg)
##                    except Exception:
##                        pass
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
##                else:
##                    print("[scheduled_commands] Executing:", cmd)
##            except Exception as e:
##                print("Error calling listen.add_text:", e)
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak("Failed to execute timed command.")
##        else:
##            if speech is not None and hasattr(speech, "AlfredSpeak"):
##                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
##            else:
##                print("(No listen available) Would execute:", cmd)
##    except Exception as e:
##        print("_execute_event error:", e)
##
##def _scheduler_loop(poll_seconds: int = 15):
##    while True:
##        try:
##            now = dt.datetime.now()
##            changed = False
##            for ev in scheduled_events:
##                try:
##                    if ev.get("fired", False):
##                        continue
##                    dtstart = dt.datetime.fromisoformat(ev["dtstart"])
##                    if now >= dtstart:
##                        ev["fired"] = True
##                        changed = True
##                        _execute_event(ev)
##                except Exception as e:
##                    print("scheduled event inspect error:", e)
##                    continue
##            if changed:
##                _save_scheduled_events()
##        except Exception as e:
##            print("Scheduler loop error:", e)
##        time.sleep(poll_seconds)
##
##def start_scheduler_thread(poll_seconds: int = 15):
##    global _SCHEDULER_THREAD
##    with _SCHEDULER_LOCK:
##        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
##            return
##        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
##        _SCHEDULER_THREAD.start()
##
### -------------------------
### public API: handle a user utterance and schedule commands if found
### -------------------------
##def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
##    """
##    Parse `text` for time-based commands. If found, schedule them and return a list
##    of created event dicts. If no scheduling performed, return None.
##    """
##    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)
##
##    New_Message = message
##    New_speaker = speaker
##    New_score = score
##    New_gender = gender
##    New_gender_conf = gender_conf
##    New_timestamp = timestamp
##
##    print(f"New_Message Timed Command Module       : {New_Message}")
##    print(f"New_speaker Timed Command Module       : {New_speaker}")
##    print(f"New_score Timed Command Module         : {New_score}")
##    print(f"New_gender Timed Command Module        : {New_gender}")
##    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
##    print(f"New_timestamp Timed Command Module     : {New_timestamp}")
##
##    text_clean = safe_str(text)
##    lower = text_clean.lower()
##
##    # conservative time indicators (expanded)
##    time_indicators = bool(re.search(r"\b(at|o'clock|o clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2}|this (morning|afternoon|evening|night)|in the morning|in the evening|evening|morning|tonight)\b", text_clean, flags=re.I))
##    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
##    if not time_indicators or reminder_like:
##        return None
##
##    parts = _parse_command_sequence_parts(text_clean)
##    if not parts:
##        return None
##
##    scheduled = []
##    for p in parts:
##        part_text = p.get("part", "")
##        cmd_text = _strip_time_tokens_from_part(part_text) or part_text
##
##        # parse a concrete datetime (prefer rel_dt, then explicit hm+date)
##        dtstart = None
##        if p.get("rel_dt"):
##            dtstart = p["rel_dt"]
##        else:
##            # determine base date from the whole part (order-insensitive)
##            base_date = _parse_base_date_from_part(part_text)
##
##            hm = p.get("hm")
##            if hm:
##                h, m = hm
##                try:
##                    cand = dt.datetime.combine(base_date, dt.time(h, m))
##                except Exception:
##                    cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
##                # if cand is in the past and user didn't explicitly say "today", push to next day
##                if cand < dt.datetime.now() and not re.search(r'\btoday\b', part_text.lower()):
##                    cand = cand + dt.timedelta(days=1)
##                dtstart = cand
##            else:
##                if New_timestamp:
##                    try:
##                        parsed_ts = dt.datetime.fromisoformat(New_timestamp.replace(" ", "T"))
##                        dtstart = parsed_ts
##                    except Exception:
##                        try:
##                            dtstart = dt.datetime.fromisoformat(New_timestamp)
##                        except Exception:
##                            dtstart = None
##
##        if dtstart is None:
##            dtstart = dt.datetime.now() + dt.timedelta(minutes=1)
##
##        # recurrence
##        rules = _parse_recurrence_rules(part_text)
##        occurrence_datetimes = _generate_recurrence_datetimes(dtstart, rules)
##
##        for occ in occurrence_datetimes:
##            try:
##                ev = add_scheduled_command(cmd_text, occ, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
##                scheduled.append(ev)
##            except Exception as e:
##                print("Failed to schedule part:", e)
##                continue
##
##    if scheduled:
##        if speech is not None and hasattr(speech, "AlfredSpeak"):
##            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
##        else:
##            print(f"Scheduled {len(scheduled)} command(s).")
##        if gui is not None and hasattr(gui, "log_query"):
##            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
##        return scheduled
##    return None
##
### initialization
##_load_scheduled_events()
##start_scheduler_thread()
##
##


















#####   ALMOST THERE
####
##### scheduled_commands.py
####from __future__ import annotations
####import re
####import os
####import json
####import uuid
####import time
####import threading
####from typing import List, Optional, Dict, Any, Tuple
####import datetime as dt
####
##### try to reuse project speech/listen objects
####try:
####    from speech import speech
####except Exception:
####    speech = None
####try:
####    from listen import listen
####except Exception:
####    listen = None
####
##### basic helpers
####def safe_str(val) -> str:
####    if val is None:
####        return ""
####    if isinstance(val, str):
####        return val.strip()
####    try:
####        return str(val)
####    except Exception:
####        return ""
####
####import base64
####import ast
####
##### -------------------------
##### extractor (adapted from your original)
##### -------------------------
####def extract_text_from_timed_command(query):
####    """
####    Returns: (message, speaker, score, gender, gender_conf, timestamp)
####    """
####    if query is None:
####        return "", None, None, None, None, None
####
####    def _extract_timestamp(fragment: str):
####        if not fragment:
####            return None
####        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
####        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
####        if m_date and m_time:
####            return f"{m_date.group('date')} {m_time.group('time')}"
####        if m_date:
####            return m_date.group('date')
####        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
####        if m_date2:
####            return m_date2.group('date')
####        return None
####
####    # ---------- dict case ----------
####    if isinstance(query, dict):
####        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
####        speaker_ = query.get("username") or query.get("speaker")
####        score_ = query.get("score")
####        gender_ = query.get("gender")
####        gender_conf_ = query.get("gender_conf")
####        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
####        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)
####
####    # --- string case ---
####    if isinstance(query, str):
####        s = query.strip()
####
####        # base64 decode heuristic
####        try:
####            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
####                try:
####                    decoded = base64.b64decode(s).decode('utf-8')
####                    if decoded:
####                        s = decoded.strip()
####                except Exception:
####                    pass
####        except Exception:
####            pass
####
####        s = s.strip()
####
####        def _extract_meta(fragment):
####            frag = fragment or ""
####            frag = str(frag)
####            speaker = None
####            score = None
####            gender = None
####            gender_conf = None
####
####            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
####            if m:
####                speaker = m.group("u").strip()
####
####            if speaker is None:
####                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
####                if m2:
####                    speaker = m2.group("u").strip()
####
####            if speaker is None:
####                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
####                if m3:
####                    maybe_u = m3.group("u").strip()
####                    if " " not in maybe_u or len(maybe_u) <= 24:
####                        speaker = maybe_u
####
####            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if ms:
####                score = ms.group("s").strip()
####            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mg:
####                gender = mg.group("g").strip()
####            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = mgc.group("gc").strip()
####
####            if speaker and not gender:
####                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
####                if mg_end:
####                    gender = mg_end.group('gword').strip()
####                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
####
####            return speaker, score, gender, gender_conf
####
####        # detect triple fence wrappers
####        first_triple = s.find("'''")
####        backtick_first = s.find("```")
####        wrapper_pos = None
####        wrapper_token = None
####        if first_triple != -1:
####            wrapper_pos = first_triple; wrapper_token = "'''"
####        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
####            wrapper_pos = backtick_first; wrapper_token = "```"
####
####        if wrapper_pos is not None:
####            start = wrapper_pos
####            end = s.find(wrapper_token, start + len(wrapper_token))
####            if end != -1:
####                inner = s[start + len(wrapper_token) : end]
####                remainder = s[end + len(wrapper_token) :].strip()
####                inner = inner.lstrip()
####                inner = re.sub(r"^'+", "", inner)
####                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
####                speaker, score, gender, gender_conf = _extract_meta(remainder)
####                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
####                if not speaker:
####                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
####                    if m_all:
####                        speaker = m_all.group("u").strip()
####                message = inner.strip()
####                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # structured line pattern
####        m_struct = re.match(
####            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
####            s
####        )
####        if m_struct:
####            date = m_struct.group('date'); time_ = m_struct.group('time')
####            timestamp = f"{date} {time_}"
####            message = m_struct.group('msg').strip()
####            user_full = m_struct.group('user').strip()
####            gender = None
####            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
####            if m_gender:
####                gender = m_gender.group('g').strip()
####                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
####            else:
####                user_clean = user_full
####            speaker = user_clean if user_clean else None
####            return message, speaker, None, (gender if gender else None), None, timestamp
####
####        # structured string attempt
####        try:
####            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
####            m = re.search(pattern, s, flags=re.DOTALL)
####            if m:
####                message_ = m.group('message').strip()
####                speaker_ = m.group('username').strip() or None
####                score_ = m.group('score') or None
####                gender_ = m.group('gender') or None
####                gender_conf_ = m.group('gender_conf') or None
####                timestamp_ = _extract_timestamp(s)
####                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
####        except Exception:
####            pass
####
####        # username anywhere
####        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
####        speaker = None; score = None; gender = None; gender_conf = None
####        timestamp = _extract_timestamp(s)
####        if m_user_any:
####            speaker = m_user_any.group("u").strip()
####            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
####            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if ms:
####                score = ms.group("s").strip()
####            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mg:
####                gender = mg.group("g").strip()
####            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = mgc.group("gc").strip()
####            candidate = s.strip()
####            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # username at end
####        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
####        timestamp = _extract_timestamp(s)
####        if m_user2:
####            speaker = m_user2.group("u").strip()
####            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
####            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
####
####        # last-token username heuristic
####        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
####        timestamp = _extract_timestamp(s)
####        if m_user3:
####            maybe_u = m_user3.group("u").strip()
####            maybe_body = m_user3.group("body").strip()
####            if " " not in maybe_u or len(maybe_u) <= 24:
####                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
####
####        # fallback
####        return s, None, None, None, None, None
####
####    return str(query).strip(), None, None, None, None, None
####
##### -------------------------
##### small words->number helpers (kept minimal and self-contained)
##### -------------------------
####_UNITS = {
####    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
####    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
####    "seventeen":17,"eighteen":18,"nineteen":19
####}
####_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
####
####def words_to_number(phrase: str) -> Optional[int]:
####    if phrase is None: return None
####    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
####    if not words: return None
####    total = 0; current = 0; valid = False
####    for w in words:
####        if w in _UNITS:
####            current += _UNITS[w]; valid = True
####        elif w in _TENS:
####            current += _TENS[w]; valid = True
####        elif w == "and":
####            continue
####        else:
####            return None
####    return (total + current) if valid else None
####
##### -------------------------
##### time parser (improved subset)
##### -------------------------
##### expanded AM/PM phrases to include multi-word tokens like 'in the morning', 'in the evening'
####_AM_WORDS = {"am","a.m.","a.m","a.m.","morning","this morning","in the morning"}
####_PM_WORDS = {"pm","p.m.","p.p","pm.","evening","afternoon","tonight","this evening","in the evening","night","this afternoon"}
####
####def _token_to_number(token: str) -> Optional[int]:
####    token = safe_str(token).lower()
####    if not token: return None
####    if re.fullmatch(r"\d+", token):
####        try: return int(token)
####        except: return None
####    if token in _UNITS: return _UNITS[token]
####    if token in _TENS: return _TENS[token]
####    if "-" in token:
####        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
####        if all(v is not None for v in vals): return sum(vals)
####    return words_to_number(token)
####
####def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
####    """
####    Detect AM/PM tokens (including multi-word tokens like 'in the morning'/'in the evening')
####    and return (cleaned_string, 'am'|'pm'|None).
####    It prefers longest matches (so "in the morning" matches before "morning").
####    """
####    s0 = safe_str(s).lower()
####    ampm = None
####
####    # check multi-word tokens first (longest-first)
####    multi_am = ["in the morning", "this morning", "morning"]
####    multi_pm = ["in the evening", "this evening", "this afternoon", "afternoon", "evening", "tonight", "night"]
####
####    for w in multi_am:
####        if re.search(r"\b" + re.escape(w) + r"\b", s0):
####            ampm = "am"
####            break
####    if ampm is None:
####        for w in multi_pm:
####            if re.search(r"\b" + re.escape(w) + r"\b", s0):
####                ampm = "pm"
####                break
####
####    # also check short forms like 'am', 'pm', 'a.m.', 'p.m.'
####    if ampm is None:
####        for w in ("a.m.","am","a.m","p.m.","pm","p.m"):
####            if re.search(r"\b" + re.escape(w) + r"\b", s0):
####                if 'p' in w:
####                    ampm = "pm"
####                else:
####                    ampm = "am"
####                break
####
####    # map 'noon' / 'midnight'
####    if re.search(r"\bnoon\b", s0):
####        ampm = "pm"
####    if re.search(r"\bmidnight\b", s0):
####        ampm = "am"
####
####    if ampm:
####        # remove a wide range of appearances of the token(s)
####        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|in the morning|this morning|morning|in the evening|this evening|evening|afternoon|tonight|night|noon|midnight|this afternoon)\b"
####        s0 = re.sub(pattern, " ", s0)
####        s0 = re.sub(r'\s+', ' ', s0).strip()
####    return s0, ampm
####
####def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
####    """
####    Robust spoken time -> (hour, minute) parser.
####    """
####    if spoken is None: return None
####    if isinstance(spoken, dt.datetime): return (spoken.hour, spoken.minute)
####    if isinstance(spoken, dt.time): return (spoken.hour, spoken.minute)
####
####    s_orig = safe_str(spoken)
####    s = s_orig.lower().replace("-", " ").replace(".", " ").replace(",", " ").strip()
####    if re.search(r"\bnoon\b", s): return (12, 0)
####    if re.search(r"\bmidnight\b", s): return (0, 0)
####
####    s_no_ampm, ampm = _detect_ampm_and_remove(s)
####
####    # explicit 24h with colon or 'h'
####    m_colon = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", s_no_ampm, flags=re.I)
####    if m_colon:
####        try:
####            hh = int(m_colon.group(1)) % 24; mm = int(m_colon.group(2)) % 60
####            hour = hh; minute = mm
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####        except Exception:
####            pass
####
####    m_half = re.search(r"\bhalf past ([a-z0-9 ]+)\b", s_no_ampm)
####    if m_half:
####        token = m_half.group(1).strip(); h = _token_to_number(token)
####        if h is not None:
####            hour = int(h) % 24; minute = 30
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####
####    m_quarter = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)\b", s_no_ampm)
####    if m_quarter:
####        typ = m_quarter.group(1); hour_token = m_quarter.group(2).strip(); h = _token_to_number(hour_token)
####        if h is not None:
####            hour = int(h) % 24
####            if typ == "past":
####                minute = 15
####            else:
####                minute = 45; hour = (hour - 1) % 24
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####
####    m_past = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*past\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
####    if m_past:
####        try:
####            mins = int(m_past.group(1))
####            htoken = m_past.group(2)
####            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
####            if h is not None:
####                hour = int(h) % 24; minute = mins % 60
####                if ampm == "pm" and hour < 12: hour += 12
####                if ampm == "am" and hour == 12: hour = 0
####                return (hour, minute)
####        except Exception:
####            pass
####
####    m_to = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*to\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
####    if m_to:
####        try:
####            mins = int(m_to.group(1))
####            htoken = m_to.group(2)
####            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
####            if h is not None:
####                hour = (int(h) - 1) % 24; minute = (60 - (mins % 60)) % 60
####                if ampm == "pm" and hour < 12: hour += 12
####                if ampm == "am" and hour == 12: hour = 0
####                return (hour, minute)
####        except Exception:
####            pass
####
####    m_oclock = re.search(r"\b(\d{1,2})\s*(?:o['\s]?clock|oclock|o clock)\b", s_no_ampm)
####    if m_oclock:
####        try:
####            hour = int(m_oclock.group(1)) % 24; minute = 0
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####        except Exception:
####            pass
####
####    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
####    if len(tokens) >= 2:
####        h_candidate = _token_to_number(tokens[0])
####        m_candidate = _token_to_number(tokens[1])
####        if h_candidate is not None and m_candidate is not None and 0 <= m_candidate < 60:
####            hour = int(h_candidate) % 24; minute = int(m_candidate) % 60
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####
####    if len(tokens) == 1:
####        h = _token_to_number(tokens[0])
####        if h is not None:
####            hour = int(h) % 24; minute = 0
####            if ampm == "pm" and hour < 12: hour += 12
####            if ampm == "am" and hour == 12: hour = 0
####            return (hour, minute)
####
####    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
####    if digits_cluster:
####        cluster = digits_cluster.group(1)
####        try:
####            if len(cluster) == 3: h = int(cluster[0]); m = int(cluster[1:])
####            else: h = int(cluster[:2]); m = int(cluster[2:])
####            if 0 <= h < 24 and 0 <= m < 60:
####                hour = h; minute = m
####                if ampm == "pm" and hour < 12: hour += 12
####                if ampm == "am" and hour == 12: hour = 0
####                return (hour, minute)
####        except Exception:
####            pass
####
####    return None
####
##### -------------------------
##### persistence
##### -------------------------
####SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
####os.makedirs(SCHEDULE_DIR, exist_ok=True)
####SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
####scheduled_events: List[dict] = []
####
####def _load_scheduled_events():
####    global scheduled_events
####    try:
####        if os.path.exists(SCHEDULE_DB):
####            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
####                scheduled_events = json.load(f)
####        else:
####            scheduled_events = []
####    except Exception as e:
####        print("Scheduled load failed:", e); scheduled_events = []
####
####def _save_scheduled_events():
####    try:
####        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
####            json.dump(scheduled_events, f, indent=2, default=str)
####    except Exception as e:
####        print("Scheduled save failed:", e)
####
##### -------------------------
##### schedule management
##### -------------------------
####def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
####    try:
####        ev = {
####            "id": uuid.uuid4().hex,
####            "command": command_text,
####            "username": username or "Itf",
####            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
####            "description": description,
####            "fired": False
####        }
####        scheduled_events.append(ev)
####        _save_scheduled_events()
####        return ev
####    except Exception as e:
####        print("add_scheduled_command failed:", e)
####        raise
####
##### -------------------------
##### parsing helpers (improvements)
##### -------------------------
####def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
####    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
####    out = []
####    for p in parts:
####        p_clean = p.strip()
####        if not p_clean:
####            continue
####        hm = spoken_time_to_hm(p_clean)
####        rel_dt = None
####        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
####        if m_rel:
####            num_phrase = m_rel.group(2).strip()
####            unit = m_rel.group(3).lower()
####            try:
####                num = int(num_phrase)
####            except:
####                num = words_to_number(num_phrase)
####            if num is not None:
####                now = dt.datetime.now()
####                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
####                elif unit.startswith("minute"): rel_dt = now + dt.timedelta(minutes=num)
####                elif unit.startswith("second"): rel_dt = now + dt.timedelta(seconds=num)
####                elif unit.startswith("day"): rel_dt = now + dt.timedelta(days=num)
####        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
####    return out
####
####def _strip_time_tokens_from_part(part: str) -> str:
####    """
####    Clean scheduling tokens from a phrase or a log line.
####    """
####    if not part:
####        return part
####
####    orig = part.strip()
####    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
####    command_candidate = None
####    if len(segments) > 1:
####        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
####        if alpha_segments:
####            command_candidate = max(alpha_segments, key=lambda s: len(s))
####    s = command_candidate if command_candidate else orig
####    s = re.sub(r'\s+', ' ', s).strip()
####
####    # remove explicit "on <date>" (conservative)
####    s = re.sub(r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', '', s, flags=re.I)
####
####    # remove "in/after <...> (seconds|minutes|hours|days)" completely
####    s = re.sub(r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b', '', s, flags=re.I)
####
####    s = re.sub(r'\s+', ' ', s).strip()
####
####    # If there's an 'at', cut EVERYTHING from 'at' onward
####    at_m = re.search(r'\bat\b', s, flags=re.I)
####    if at_m:
####        s = s[:at_m.start()].rstrip()
####        s = re.sub(r'\s+', ' ', s).strip(' ,.')
####        return s
####
####    # Handle 'on' occurrences:
####    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
####    if on_matches:
####        first = on_matches[0]
####        s = s[: first.end()].rstrip()
####        s = re.sub(r'\s+', ' ', s).strip(' ,.')
####        return s
####
####    s = re.sub(r'\s+', ' ', s).strip(' ,.')
####    return s
####
##### -------------------------
##### recurrence & date parsing helpers
##### -------------------------
####_WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
####def _next_weekday_date(weekday_idx: int, start: Optional[dt.date] = None) -> dt.date:
####    start = start or dt.date.today()
####    days_ahead = (weekday_idx - start.weekday()) % 7
####    if days_ahead == 0:
####        days_ahead = 7
####    return start + dt.timedelta(days=days_ahead)
####
####def _parse_base_date_from_part(part: str) -> dt.date:
####    txt = safe_str(part).lower()
####    today = dt.date.today()
####    if "today" in txt:
####        return today
####    if "tomorrow" in txt:
####        return today + dt.timedelta(days=1)
####    # 'this evening', 'this morning', 'in the morning', 'in the evening'
####    if re.search(r'\b(this|in the)?\s*(evening|morning|afternoon|night)\b', txt):
####        return today
####    # weekdays
####    for wd, idx in _WEEKDAY_MAP.items():
####        if re.search(rf'\b{wd}\b', txt):
####            if re.search(r'\bthis\b', txt) and idx == today.weekday():
####                return today
####            return _next_weekday_date(idx, start=today)
####    # default
####    return today
####
####def _parse_recurrence_rules(part: str) -> Dict[str, Any]:
####    txt = safe_str(part).lower()
####    rules = {"freq": None, "interval": 1, "count": None, "until": None, "weekdays": None}
####
####    m_every = re.search(r'\bevery\s+(second\s+day|second\s+week|day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', txt)
####    if m_every:
####        token = m_every.group(1)
####        if "second day" in token or "every second day" in txt:
####            rules["freq"] = "daily"; rules["interval"] = 2
####        elif "second week" in token or "every second week" in txt:
####            rules["freq"] = "weekly"; rules["interval"] = 2
####        elif token in ("day",):
####            rules["freq"] = "daily"; rules["interval"] = 1
####        elif token == "week":
####            rules["freq"] = "weekly"; rules["interval"] = 1
####        elif token == "month":
####            rules["freq"] = "monthly"; rules["interval"] = 1
####        elif token in _WEEKDAY_MAP:
####            rules["freq"] = "weekly"; rules["weekdays"] = [ _WEEKDAY_MAP[token] ]
####
####    # Accept digits or word numbers (e.g., "two")
####    m_next = re.search(r'for the next\s+(\d+|[a-z]+)\s+(days?|weeks?|months?)', txt)
####    if m_next:
####        n_token = m_next.group(1)
####        unit = m_next.group(2)
####        try:
####            n = int(n_token)
####        except Exception:
####            n = words_to_number(n_token)
####        if n is None:
####            n = 1
####        if unit.startswith("day"):
####            # prefer count days
####            rules["count"] = n
####            if rules["freq"] is None:
####                rules["freq"] = "daily"
####        elif unit.startswith("week"):
####            # schedule daily for next n weeks (common interpretation)
####            rules["until"] = (dt.date.today() + dt.timedelta(weeks=n))
####            if rules["freq"] is None:
####                rules["freq"] = "daily"
####        elif unit.startswith("month"):
####            rules["until"] = (dt.date.today() + dt.timedelta(days=30 * n))
####            if rules["freq"] is None:
####                rules["freq"] = "daily"
####
####    if re.search(r'\bfor the month\b', txt) or re.search(r'\bfor a month\b', txt):
####        rules["until"] = dt.date.today() + dt.timedelta(days=30)
####        if rules["freq"] is None:
####            rules["freq"] = "daily"
####
####    if re.search(r'\bthis week\b', txt) and re.search(r'\bevery\b', txt):
####        today = dt.date.today()
####        end = today + dt.timedelta(days=(6 - today.weekday()))
####        rules["until"] = end
####        rules["freq"] = "daily"
####
####    return rules
####
####def _generate_recurrence_datetimes(base_dt: dt.datetime, rules: Dict[str, Any]) -> List[dt.datetime]:
####    if not rules or not any([rules.get("freq"), rules.get("until"), rules.get("count"), rules.get("weekdays")]):
####        return [base_dt]
####
####    out = []
####    freq = rules.get("freq")
####    interval = max(1, int(rules.get("interval", 1)))
####    count = rules.get("count")
####    until = rules.get("until")
####    weekdays = rules.get("weekdays")
####
####    # If there's an "until" but no explicit freq, default to daily recurrence
####    if freq is None and until is not None and weekdays is None:
####        freq = "daily"
####
####    if until is None and (freq is not None or weekdays is not None):
####        until = (base_dt.date() + dt.timedelta(days=30))  # default one month
####        # if we defaulted an 'until' but freq still None, set daily
####        if freq is None and weekdays is None:
####            freq = "daily"
####
####    if weekdays:
####        cur_date = base_dt.date()
####        while cur_date <= until:
####            if cur_date.weekday() in weekdays:
####                out.append(dt.datetime.combine(cur_date, base_dt.time()))
####            cur_date = cur_date + dt.timedelta(days=1)
####        return out
####
####    if freq == "daily":
####        cur = base_dt
####        while True:
####            if until and cur.date() > until:
####                break
####            out.append(cur)
####            if count and len(out) >= count:
####                break
####            cur = cur + dt.timedelta(days=interval)
####        return out
####
####    if freq == "weekly":
####        cur = base_dt
####        while True:
####            if until and cur.date() > until:
####                break
####            out.append(cur)
####            if count and len(out) >= count:
####                break
####            cur = cur + dt.timedelta(weeks=interval)
####        return out
####
####    if freq == "monthly":
####        cur = base_dt
####        while True:
####            if until and cur.date() > until:
####                break
####            out.append(cur)
####            if count and len(out) >= count:
####                break
####            cur = cur + dt.timedelta(days=30 * interval)
####        return out
####
####    return [base_dt]
####
##### -------------------------
##### executor thread
##### -------------------------
####_SCHEDULER_THREAD = None
####_SCHEDULER_LOCK = threading.Lock()
####
####def _execute_event(ev: dict):
####    try:
####        cmd = ev.get("command", "")
####        user = ev.get("username", "Itf")
####        if listen is not None and hasattr(listen, "listen"):
####            text_msg = cmd
####            try:
####                if hasattr(listen, "add_text"):
####                    listen.add_text(text_msg)
####                else:
####                    try:
####                        listen_queue = getattr(listen, "queue", None)
####                        if listen_queue is not None and hasattr(listen_queue, "put"):
####                            listen_queue.put(text_msg)
####                    except Exception:
####                        pass
####                if speech is not None and hasattr(speech, "AlfredSpeak"):
####                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
####                else:
####                    print("[scheduled_commands] Executing:", cmd)
####            except Exception as e:
####                print("Error calling listen.add_text:", e)
####                if speech is not None and hasattr(speech, "AlfredSpeak"):
####                    speech.AlfredSpeak("Failed to execute timed command.")
####        else:
####            if speech is not None and hasattr(speech, "AlfredSpeak"):
####                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
####            else:
####                print("(No listen available) Would execute:", cmd)
####    except Exception as e:
####        print("_execute_event error:", e)
####
####def _scheduler_loop(poll_seconds: int = 15):
####    while True:
####        try:
####            now = dt.datetime.now()
####            changed = False
####            for ev in scheduled_events:
####                try:
####                    if ev.get("fired", False):
####                        continue
####                    dtstart = dt.datetime.fromisoformat(ev["dtstart"])
####                    if now >= dtstart:
####                        ev["fired"] = True
####                        changed = True
####                        _execute_event(ev)
####                except Exception as e:
####                    print("scheduled event inspect error:", e)
####                    continue
####            if changed:
####                _save_scheduled_events()
####        except Exception as e:
####            print("Scheduler loop error:", e)
####        time.sleep(poll_seconds)
####
####def start_scheduler_thread(poll_seconds: int = 15):
####    global _SCHEDULER_THREAD
####    with _SCHEDULER_LOCK:
####        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
####            return
####        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
####        _SCHEDULER_THREAD.start()
####
##### -------------------------
##### public API: handle a user utterance and schedule commands if found
##### -------------------------
####def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
####    """
####    Parse `text` for time-based commands. If found, schedule them and return a list
####    of created event dicts. If no scheduling performed, return None.
####    """
####    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)
####
####    New_Message = message
####    New_speaker = speaker
####    New_score = score
####    New_gender = gender
####    New_gender_conf = gender_conf
####    New_timestamp = timestamp
####
####    print(f"New_Message Timed Command Module       : {New_Message}")
####    print(f"New_speaker Timed Command Module       : {New_speaker}")
####    print(f"New_score Timed Command Module         : {New_score}")
####    print(f"New_gender Timed Command Module        : {New_gender}")
####    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
####    print(f"New_timestamp Timed Command Module     : {New_timestamp}")
####
####    text_clean = safe_str(text)
####    lower = text_clean.lower()
####
####    # conservative time indicators (expanded)
####    time_indicators = bool(re.search(r"\b(at|o'clock|o clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2}|this (morning|afternoon|evening|night)|in the morning|in the evening|evening|morning|tonight)\b", text_clean, flags=re.I))
####    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
####    if not time_indicators or reminder_like:
####        return None
####
####    parts = _parse_command_sequence_parts(text_clean)
####    if not parts:
####        return None
####
####    scheduled = []
####    for p in parts:
####        part_text = p.get("part", "")
####        cmd_text = _strip_time_tokens_from_part(part_text) or part_text
####
####        # parse a concrete datetime (prefer rel_dt, then explicit hm+date)
####        dtstart = None
####        if p.get("rel_dt"):
####            dtstart = p["rel_dt"]
####        else:
####            # determine base date from the whole part (order-insensitive)
####            base_date = _parse_base_date_from_part(part_text)
####
####            hm = p.get("hm")
####            if hm:
####                h, m = hm
####                try:
####                    cand = dt.datetime.combine(base_date, dt.time(h, m))
####                except Exception:
####                    cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
####                # if cand is in the past and user didn't explicitly say "today", push to next day
####                if cand < dt.datetime.now() and not re.search(r'\btoday\b', part_text.lower()):
####                    cand = cand + dt.timedelta(days=1)
####                dtstart = cand
####            else:
####                if New_timestamp:
####                    try:
####                        parsed_ts = dt.datetime.fromisoformat(New_timestamp.replace(" ", "T"))
####                        dtstart = parsed_ts
####                    except Exception:
####                        try:
####                            dtstart = dt.datetime.fromisoformat(New_timestamp)
####                        except Exception:
####                            dtstart = None
####
####        if dtstart is None:
####            dtstart = dt.datetime.now() + dt.timedelta(minutes=1)
####
####        # recurrence
####        rules = _parse_recurrence_rules(part_text)
####        occurrence_datetimes = _generate_recurrence_datetimes(dtstart, rules)
####
####        for occ in occurrence_datetimes:
####            try:
####                ev = add_scheduled_command(cmd_text, occ, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
####                scheduled.append(ev)
####            except Exception as e:
####                print("Failed to schedule part:", e)
####                continue
####
####    if scheduled:
####        if speech is not None and hasattr(speech, "AlfredSpeak"):
####            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
####        else:
####            print(f"Scheduled {len(scheduled)} command(s).")
####        if gui is not None and hasattr(gui, "log_query"):
####            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
####        return scheduled
####    return None
####
##### initialization
####_load_scheduled_events()
####start_scheduler_thread()





























### scheduled_commands.py
##from __future__ import annotations
##import re
##import os
##import json
##import uuid
##import time
##import threading
##from typing import List, Optional, Dict, Any, Tuple
##import datetime as dt
##
### try to reuse project speech/listen objects
##try:
##    from speech import speech
##except Exception:
##    speech = None
##try:
##    from listen import listen
##except Exception:
##    listen = None
##
### basic helpers
##def safe_str(val) -> str:
##    if val is None:
##        return ""
##    if isinstance(val, str):
##        return val.strip()
##    try:
##        return str(val)
##    except Exception:
##        return ""
##
##import base64
##import ast
##
### -------------------------
### extractor (adapted from your original)
### -------------------------
##
##
##
##
##import re
##import base64
##
##
##def extract_text_from_timed_command(query):
##    """
##    Returns: (message, speaker, score, gender, gender_conf, timestamp)
##    """
##    if query is None:
##        return "", None, None, None, None, None
##
##    # helper: extract timestamp from a fragment (YYYY-MM-DD and HH:MM:SS)
##    def _extract_timestamp(fragment: str):
##        if not fragment:
##            return None
##        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
##        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
##        if m_date and m_time:
##            return f"{m_date.group('date')} {m_time.group('time')}"
##        if m_date:
##            return m_date.group('date')
##        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
##        if m_date2:
##            return m_date2.group('date')
##        return None
##
##    def _norm_val(v):
##        if v is None:
##            return None
##        vs = str(v).strip()
##        if vs.lower() in ("none", "null", "nil", ""):
##            return None
##        return vs
##
##    # ---------- dict case ----------
##    if isinstance(query, dict):
##        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
##        speaker_ = query.get("username") or query.get("speaker")
##        score_ = query.get("score")
##        gender_ = query.get("gender")
##        gender_conf_ = query.get("gender_conf")
##        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
##        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), _norm_val(score_), _norm_val(gender_), _norm_val(gender_conf_), (str(timestamp_).strip() if timestamp_ is not None else None)
##
##    # --- string case ---
##    if isinstance(query, str):
##        s = query.strip()
##
##        # base64 decode heuristic
##        try:
##            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
##                try:
##                    decoded = base64.b64decode(s).decode('utf-8')
##                    if decoded:
##                        s = decoded.strip()
##                except Exception:
##                    pass
##        except Exception:
##            pass
##
##        s = s.strip()
##
##        # helper to extract metadata
##        def _extract_meta(fragment):
##            frag = fragment or ""
##            frag = str(frag)
##            speaker = None
##            score = None
##            gender = None
##            gender_conf = None
##
##            m = re.search(r"'username'\s*[:=]\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
##            if m:
##                speaker = m.group("u").strip()
##
##            if speaker is None:
##                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
##                if m2:
##                    speaker = m2.group("u").strip()
##
##            if speaker is None:
##                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
##                if m3:
##                    maybe_u = m3.group("u").strip()
##                    if " " not in maybe_u or len(maybe_u) <= 24:
##                        speaker = maybe_u
##
##            ms = re.search(r"'score'\s*[:=]\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*[:=]\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*[:=]\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##
##            if speaker and not gender:
##                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
##                if mg_end:
##                    gender = mg_end.group('gword').strip()
##                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
##
##            return speaker, _norm_val(score), _norm_val(gender), _norm_val(gender_conf)
##
##        # detect triple fence wrappers
##        first_triple = s.find("'''")
##        backtick_first = s.find("```")
##        wrapper_pos = None
##        wrapper_token = None
##        if first_triple != -1:
##            wrapper_pos = first_triple; wrapper_token = "'''"
##        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
##            wrapper_pos = backtick_first; wrapper_token = "```"
##
##        if wrapper_pos is not None:
##            start = wrapper_pos
##            end = s.find(wrapper_token, start + len(wrapper_token))
##            if end != -1:
##                inner = s[start + len(wrapper_token) : end]
##                remainder = s[end + len(wrapper_token) :].strip()
##                inner = inner.lstrip()
##                inner = re.sub(r"^'+", "", inner)
##                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
##                speaker, score, gender, gender_conf = _extract_meta(remainder)
##                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
##                if not speaker:
##                    m_all = re.search(r"'username'\s*[:=]\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
##                    if m_all:
##                        speaker = m_all.group("u").strip()
##                message = inner.strip()
##                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # --------- IMPROVED: detect date/time anywhere and parse remainder ----------
##        dt_match = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})', s)
##        if dt_match:
##            date = dt_match.group('date')
##            time_ = dt_match.group('time')
##            timestamp = f"{date} {time_}"
##
##            # remainder after the matched date/time
##            rem = s[dt_match.end():].strip()
##            # strip leading separators / labels if present
##            rem = re.sub(r'^[\s:|-]+', '', rem)
##
##            # split remainder into parts separated by " : " (flexible spacing)
##            parts = [p.strip() for p in re.split(r'\s*:\s*', rem) if p is not None]
##
##            message = parts[0] if parts else ""
##            rest = ' : '.join(parts[1:]) if len(parts) > 1 else ""
##
##            # If message still contains explicit metadata patterns at end, strip them
##            message = re.sub(r"\s*(:\s*)?'score'\s*[:=]\s*[^:]+$", "", message, flags=re.IGNORECASE).strip()
##            message = re.sub(r"\s*(:\s*)?'gender'\s*[:=]\s*[^:]+$", "", message, flags=re.IGNORECASE).strip()
##            message = re.sub(r"\s*(:\s*)?'gender_conf'\s*[:=]\s*[^:]+$", "", message, flags=re.IGNORECASE).strip()
##
##            # try to extract explicit score/gender/gender_conf from rest
##            ms = re.search(r"'score'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
##            mg = re.search(r"'gender'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
##            mgc = re.search(r"'gender_conf'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
##
##            score = _norm_val(ms.group('v')) if ms else None
##            gender = _norm_val(mg.group('v')) if mg else None
##            gender_conf = _norm_val(mgc.group('v')) if mgc else None
##
##            # detect username as the last part if it doesn't look like a kv pair
##            username = None
##            if parts and len(parts) > 1:
##                last = parts[-1]
##                if not re.match(r"^['\"]?\w+['\"]?\s*[:=]\s*", last):
##                    username = re.sub(r"^['\"]|['\"]$", "", last).strip()
##
##            return message.strip(), (username if username else None), score, gender, gender_conf, timestamp
##
##        # structured line pattern (fallback older logic)
##        m_struct = re.match(
##            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
##            s
##        )
##        if m_struct:
##            date = m_struct.group('date'); time_ = m_struct.group('time')
##            timestamp = f"{date} {time_}"
##            message = m_struct.group('msg').strip()
##            user_full = m_struct.group('user').strip()
##            gender = None
##            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
##            if m_gender:
##                gender = m_gender.group('g').strip()
##                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
##            else:
##                user_clean = user_full
##            speaker = user_clean if user_clean else None
##            return message, speaker, None, (gender if gender else None), None, timestamp
##
##        # structured string attempt
##        try:
##            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
##            m = re.search(pattern, s, flags=re.DOTALL)
##            if m:
##                message_ = m.group('message').strip()
##                speaker_ = m.group('username').strip() or None
##                score_ = m.group('score') or None
##                gender_ = m.group('gender') or None
##                gender_conf_ = m.group('gender_conf') or None
##                timestamp_ = _extract_timestamp(s)
##                return message_, speaker_, _norm_val(score_), _norm_val(gender_), _norm_val(gender_conf_), timestamp_
##        except Exception:
##            pass
##
##        # username anywhere
##        m_user_any = re.search(r"'username'\s*[:=]\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
##        speaker = None; score = None; gender = None; gender_conf = None
##        timestamp = _extract_timestamp(s)
##        if m_user_any:
##            speaker = m_user_any.group("u").strip()
##            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
##            ms = re.search(r"'score'\s*[:=]\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if ms:
##                score = _norm_val(ms.group("s").strip())
##            mg = re.search(r"'gender'\s*[:=]\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mg:
##                gender = _norm_val(mg.group("g").strip())
##            mgc = re.search(r"'gender_conf'\s*[:=]\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = _norm_val(mgc.group("gc").strip())
##            candidate = s.strip()
##            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # username at end
##        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
##        timestamp = _extract_timestamp(s)
##        if m_user2:
##            speaker = m_user2.group("u").strip()
##            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
##            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
##
##        # last-token username heuristic
##        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
##        timestamp = _extract_timestamp(s)
##        if m_user3:
##            maybe_u = m_user3.group("u").strip()
##            maybe_body = m_user3.group("body").strip()
##            if " " not in maybe_u or len(maybe_u) <= 24:
##                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
##
##        # fallback
##        return s, None, None, None, None, None
##
##    return str(query).strip(), None, None, None, None, None
##
##
##
####def extract_text_from_timed_command(query):
####    """
####    Returns: (message, speaker, score, gender, gender_conf, timestamp)
####    """
####    if query is None:
####        return "", None, None, None, None, None
####
####    # helper: extract timestamp from a fragment (YYYY-MM-DD and HH:MM:SS)
####    def _extract_timestamp(fragment: str):
####        if not fragment:
####            return None
####        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
####        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
####        if m_date and m_time:
####            return f"{m_date.group('date')} {m_time.group('time')}"
####        if m_date:
####            return m_date.group('date')
####        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
####        if m_date2:
####            return m_date2.group('date')
####        return None
####
####    def _norm_val(v):
####        if v is None:
####            return None
####        vs = str(v).strip()
####        if vs.lower() in ("none", "null", "nil", ""):
####            return None
####        return vs
####
####    # ---------- dict case ----------
####    if isinstance(query, dict):
####        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
####        speaker_ = query.get("username") or query.get("speaker")
####        score_ = query.get("score")
####        gender_ = query.get("gender")
####        gender_conf_ = query.get("gender_conf")
####        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
####        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), _norm_val(score_), _norm_val(gender_), _norm_val(gender_conf_), (str(timestamp_).strip() if timestamp_ is not None else None)
####
####    # --- string case ---
####    if isinstance(query, str):
####        s = query.strip()
####
####        # base64 decode heuristic
####        try:
####            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
####                try:
####                    decoded = base64.b64decode(s).decode('utf-8')
####                    if decoded:
####                        s = decoded.strip()
####                except Exception:
####                    pass
####        except Exception:
####            pass
####
####        s = s.strip()
####
####        # helper to extract metadata
####        def _extract_meta(fragment):
####            frag = fragment or ""
####            frag = str(frag)
####            speaker = None
####            score = None
####            gender = None
####            gender_conf = None
####
####            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
####            if m:
####                speaker = m.group("u").strip()
####
####            if speaker is None:
####                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
####                if m2:
####                    speaker = m2.group("u").strip()
####
####            if speaker is None:
####                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
####                if m3:
####                    maybe_u = m3.group("u").strip()
####                    if " " not in maybe_u or len(maybe_u) <= 24:
####                        speaker = maybe_u
####
####            ms = re.search(r"'score'\s*[:=]\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if ms:
####                score = ms.group("s").strip()
####            mg = re.search(r"'gender'\s*[:=]\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mg:
####                gender = mg.group("g").strip()
####            mgc = re.search(r"'gender_conf'\s*[:=]\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = mgc.group("gc").strip()
####
####            if speaker and not gender:
####                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
####                if mg_end:
####                    gender = mg_end.group('gword').strip()
####                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
####
####            return speaker, _norm_val(score), _norm_val(gender), _norm_val(gender_conf)
####
####        # detect triple fence wrappers
####        first_triple = s.find("'''")
####        backtick_first = s.find("```")
####        wrapper_pos = None
####        wrapper_token = None
####        if first_triple != -1:
####            wrapper_pos = first_triple; wrapper_token = "'''"
####        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
####            wrapper_pos = backtick_first; wrapper_token = "```"
####
####        if wrapper_pos is not None:
####            start = wrapper_pos
####            end = s.find(wrapper_token, start + len(wrapper_token))
####            if end != -1:
####                inner = s[start + len(wrapper_token) : end]
####                remainder = s[end + len(wrapper_token) :].strip()
####                inner = inner.lstrip()
####                inner = re.sub(r"^'+", "", inner)
####                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
####                speaker, score, gender, gender_conf = _extract_meta(remainder)
####                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
####                if not speaker:
####                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
####                    if m_all:
####                        speaker = m_all.group("u").strip()
####                message = inner.strip()
####                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # --------- NEW: simple leading date/time with trailing metadata and user ----------
####        # e.g. "2026-01-25 : 22:55:17 : can you ... : 'score':None : 'gender':None : 'gender_conf':None : ITF"
####        m_simple = re.match(
####            r'^\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<rest>.+)$',
####            s,
####            flags=re.DOTALL
####        )
####        if m_simple:
####            date = m_simple.group('date')
####            time_ = m_simple.group('time')
####            timestamp = f"{date} {time_}"
####            message = m_simple.group('msg').strip()
####            rest = m_simple.group('rest').strip()
####
####            # try to extract username as last colon-separated segment if it's not a key:value
####            parts = [p.strip() for p in re.split(r'\s*:\s*', rest) if p.strip() != ""]
####            username = None
####            if parts:
####                last = parts[-1]
####                # If last looks like a kv pair ('score':None etc.) then probably no username at end
####                if not re.match(r"^['\"]?\w+['\"]?\s*[:=]\s*", last):
####                    username = re.sub(r"^['\"]|['\"]$", "", last).strip()
####
####            # extract explicit score/gender/gender_conf
####            ms = re.search(r"'score'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
####            mg = re.search(r"'gender'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
####            mgc = re.search(r"'gender_conf'\s*[:=]\s*(['\"]?)(?P<v>[^'\"\s:]+)\1", rest, flags=re.IGNORECASE)
####
####            score = _norm_val(ms.group('v')) if ms else None
####            gender = _norm_val(mg.group('v')) if mg else None
####            gender_conf = _norm_val(mgc.group('v')) if mgc else None
####
####            return message, (username if username else None), score, gender, gender_conf, timestamp
####
####        # structured line pattern
####        m_struct = re.match(
####            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
####            s
####        )
####        if m_struct:
####            date = m_struct.group('date'); time_ = m_struct.group('time')
####            timestamp = f"{date} {time_}"
####            message = m_struct.group('msg').strip()
####            user_full = m_struct.group('user').strip()
####            gender = None
####            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
####            if m_gender:
####                gender = m_gender.group('g').strip()
####                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
####            else:
####                user_clean = user_full
####            speaker = user_clean if user_clean else None
####            return message, speaker, None, (gender if gender else None), None, timestamp
####
####        # structured string attempt
####        try:
####            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
####            m = re.search(pattern, s, flags=re.DOTALL)
####            if m:
####                message_ = m.group('message').strip()
####                speaker_ = m.group('username').strip() or None
####                score_ = m.group('score') or None
####                gender_ = m.group('gender') or None
####                gender_conf_ = m.group('gender_conf') or None
####                timestamp_ = _extract_timestamp(s)
####                return message_, speaker_, _norm_val(score_), _norm_val(gender_), _norm_val(gender_conf_), timestamp_
####        except Exception:
####            pass
####
####        # username anywhere
####        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
####        speaker = None; score = None; gender = None; gender_conf = None
####        timestamp = _extract_timestamp(s)
####        if m_user_any:
####            speaker = m_user_any.group("u").strip()
####            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
####            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if ms:
####                score = _norm_val(ms.group("s").strip())
####            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mg:
####                gender = _norm_val(mg.group("g").strip())
####            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = _norm_val(mgc.group("gc").strip())
####            candidate = s.strip()
####            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # username at end
####        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
####        timestamp = _extract_timestamp(s)
####        if m_user2:
####            speaker = m_user2.group("u").strip()
####            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
####            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
####
####        # last-token username heuristic
####        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
####        timestamp = _extract_timestamp(s)
####        if m_user3:
####            maybe_u = m_user3.group("u").strip()
####            maybe_body = m_user3.group("body").strip()
####            if " " not in maybe_u or len(maybe_u) <= 24:
####                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
####
####        # fallback
####        return s, None, None, None, None, None
####
####    return str(query).strip(), None, None, None, None, None
##
##
##
##
##
##
####  # LAST WORKIN 2026_01_25__23h00
####
####def extract_text_from_timed_command(query):
####    """
####    Returns: (message, speaker, score, gender, gender_conf, timestamp)
####    """
####    if query is None:
####        return "", None, None, None, None, None
####
####    def _extract_timestamp(fragment: str):
####        if not fragment:
####            return None
####        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
####        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
####        if m_date and m_time:
####            return f"{m_date.group('date')} {m_time.group('time')}"
####        if m_date:
####            return m_date.group('date')
####        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
####        if m_date2:
####            return m_date2.group('date')
####        return None
####
####    # ---------- dict case ----------
####    if isinstance(query, dict):
####        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
####        speaker_ = query.get("username") or query.get("speaker")
####        score_ = query.get("score")
####        gender_ = query.get("gender")
####        gender_conf_ = query.get("gender_conf")
####        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
####        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)
####
####    # --- string case ---
####    if isinstance(query, str):
####        s = query.strip()
####
####        # base64 decode heuristic
####        try:
####            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
####                try:
####                    decoded = base64.b64decode(s).decode('utf-8')
####                    if decoded:
####                        s = decoded.strip()
####                except Exception:
####                    pass
####        except Exception:
####            pass
####
####        s = s.strip()
####
####        def _extract_meta(fragment):
####            frag = fragment or ""
####            frag = str(frag)
####            speaker = None
####            score = None
####            gender = None
####            gender_conf = None
####
####            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
####            if m:
####                speaker = m.group("u").strip()
####
####            if speaker is None:
####                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
####                if m2:
####                    speaker = m2.group("u").strip()
####
####            if speaker is None:
####                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
####                if m3:
####                    maybe_u = m3.group("u").strip()
####                    if " " not in maybe_u or len(maybe_u) <= 24:
####                        speaker = maybe_u
####
####            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if ms:
####                score = ms.group("s").strip()
####            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mg:
####                gender = mg.group("g").strip()
####            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = mgc.group("gc").strip()
####
####            if speaker and not gender:
####                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
####                if mg_end:
####                    gender = mg_end.group('gword').strip()
####                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
####
####            return speaker, score, gender, gender_conf
####
####        # detect triple fence wrappers
####        first_triple = s.find("'''")
####        backtick_first = s.find("```")
####        wrapper_pos = None
####        wrapper_token = None
####        if first_triple != -1:
####            wrapper_pos = first_triple; wrapper_token = "'''"
####        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
####            wrapper_pos = backtick_first; wrapper_token = "```"
####
####        if wrapper_pos is not None:
####            start = wrapper_pos
####            end = s.find(wrapper_token, start + len(wrapper_token))
####            if end != -1:
####                inner = s[start + len(wrapper_token) : end]
####                remainder = s[end + len(wrapper_token) :].strip()
####                inner = inner.lstrip()
####                inner = re.sub(r"^'+", "", inner)
####                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
####                speaker, score, gender, gender_conf = _extract_meta(remainder)
####                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
####                if not speaker:
####                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
####                    if m_all:
####                        speaker = m_all.group("u").strip()
####                message = inner.strip()
####                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # structured line pattern
####        m_struct = re.match(
####            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
####            s
####        )
####        if m_struct:
####            date = m_struct.group('date'); time_ = m_struct.group('time')
####            timestamp = f"{date} {time_}"
####            message = m_struct.group('msg').strip()
####            user_full = m_struct.group('user').strip()
####            gender = None
####            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
####            if m_gender:
####                gender = m_gender.group('g').strip()
####                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
####            else:
####                user_clean = user_full
####            speaker = user_clean if user_clean else None
####            return message, speaker, None, (gender if gender else None), None, timestamp
####
####        # structured string attempt
####        try:
####            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
####            m = re.search(pattern, s, flags=re.DOTALL)
####            if m:
####                message_ = m.group('message').strip()
####                speaker_ = m.group('username').strip() or None
####                score_ = m.group('score') or None
####                gender_ = m.group('gender') or None
####                gender_conf_ = m.group('gender_conf') or None
####                timestamp_ = _extract_timestamp(s)
####                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
####        except Exception:
####            pass
####
####        # username anywhere
####        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
####        speaker = None; score = None; gender = None; gender_conf = None
####        timestamp = _extract_timestamp(s)
####        if m_user_any:
####            speaker = m_user_any.group("u").strip()
####            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
####            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if ms:
####                score = ms.group("s").strip()
####            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mg:
####                gender = mg.group("g").strip()
####            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
####            if mgc:
####                gender_conf = mgc.group("gc").strip()
####            candidate = s.strip()
####            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
####
####        # username at end
####        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
####        timestamp = _extract_timestamp(s)
####        if m_user2:
####            speaker = m_user2.group("u").strip()
####            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
####            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
####
####        # last-token username heuristic
####        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
####        timestamp = _extract_timestamp(s)
####        if m_user3:
####            maybe_u = m_user3.group("u").strip()
####            maybe_body = m_user3.group("body").strip()
####            if " " not in maybe_u or len(maybe_u) <= 24:
####                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
####
####        # fallback
####        return s, None, None, None, None, None
####
####    return str(query).strip(), None, None, None, None, None
##
### -------------------------
### small words->number helpers (kept minimal and self-contained)
### -------------------------
##_UNITS = {
##    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
##    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
##    "seventeen":17,"eighteen":18,"nineteen":19
##}
##_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
##
##def words_to_number(phrase: str) -> Optional[int]:
##    if phrase is None: return None
##    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
##    if not words: return None
##    total = 0; current = 0; valid = False
##    for w in words:
##        if w in _UNITS:
##            current += _UNITS[w]; valid = True
##        elif w in _TENS:
##            current += _TENS[w]; valid = True
##        elif w == "and":
##            continue
##        else:
##            return None
##    return (total + current) if valid else None
##
### -------------------------
### time parser (improved subset)
### -------------------------
### expanded AM/PM phrases to include multi-word tokens like 'in the morning', 'in the evening'
##_AM_WORDS = {"am","a.m.","a.m","a.m.","morning","this morning","in the morning"}
##_PM_WORDS = {"pm","p.m.","p.m","pm.","evening","afternoon","tonight","this evening","in the evening","night","this afternoon"}
##
##def _token_to_number(token: str) -> Optional[int]:
##    token = safe_str(token).lower()
##    if not token: return None
##    if re.fullmatch(r"\d+", token):
##        try: return int(token)
##        except: return None
##    if token in _UNITS: return _UNITS[token]
##    if token in _TENS: return _TENS[token]
##    if "-" in token:
##        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
##        if all(v is not None for v in vals): return sum(vals)
##    return words_to_number(token)
##
##def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
##    """
##    Detect AM/PM tokens (including multi-word tokens like 'in the morning'/'in the evening')
##    and return (cleaned_string, 'am'|'pm'|None).
##    It prefers longest matches (so "in the morning" matches before "morning").
##    """
##    s0 = safe_str(s).lower()
##    ampm = None
##
##    # check multi-word tokens first (longest-first)
##    multi_am = ["in the morning", "this morning", "morning"]
##    multi_pm = ["in the evening", "this evening", "this afternoon", "afternoon", "evening", "tonight", "night"]
##
##    for w in multi_am:
##        if re.search(r"\b" + re.escape(w) + r"\b", s0):
##            ampm = "am"
##            break
##    if ampm is None:
##        for w in multi_pm:
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                ampm = "pm"
##                break
##
##    # also check short forms like 'am', 'pm', 'a.m.', 'p.m.'
##    if ampm is None:
##        for w in ("a.m.","am","a.m","p.m.","pm","p.m"):
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                if 'p' in w:
##                    ampm = "pm"
##                else:
##                    ampm = "am"
##                break
##
##    # map 'noon' / 'midnight'
##    if re.search(r"\bnoon\b", s0):
##        ampm = "pm"
##    if re.search(r"\bmidnight\b", s0):
##        ampm = "am"
##
##    if ampm:
##        # remove a wide range of appearances of the token(s)
##        # build a regex covering common tokens to remove
##        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|in the morning|this morning|morning|in the evening|this evening|evening|afternoon|tonight|night|noon|midnight|this afternoon)\b"
##        s0 = re.sub(pattern, " ", s0)
##        s0 = re.sub(r'\s+', ' ', s0).strip()
##    return s0, ampm
##
##def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
##    """
##    Robust spoken time -> (hour, minute) parser.
##    """
##    if spoken is None: return None
##    if isinstance(spoken, dt.datetime): return (spoken.hour, spoken.minute)
##    if isinstance(spoken, dt.time): return (spoken.hour, spoken.minute)
##
##    s_orig = safe_str(spoken)
##    s = s_orig.lower().replace("-", " ").replace(".", " ").replace(",", " ").strip()
##    if re.search(r"\bnoon\b", s): return (12, 0)
##    if re.search(r"\bmidnight\b", s): return (0, 0)
##
##    s_no_ampm, ampm = _detect_ampm_and_remove(s)
##
##    # explicit 24h with colon or 'h'
##    m_colon = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", s_no_ampm, flags=re.I)
##    if m_colon:
##        try:
##            hh = int(m_colon.group(1)) % 24; mm = int(m_colon.group(2)) % 60
##            hour = hh; minute = mm
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##        except Exception:
##            pass
##
##    m_half = re.search(r"\bhalf past ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_half:
##        token = m_half.group(1).strip(); h = _token_to_number(token)
##        if h is not None:
##            hour = int(h) % 24; minute = 30
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_quarter = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_quarter:
##        typ = m_quarter.group(1); hour_token = m_quarter.group(2).strip(); h = _token_to_number(hour_token)
##        if h is not None:
##            hour = int(h) % 24
##            if typ == "past":
##                minute = 15
##            else:
##                minute = 45; hour = (hour - 1) % 24
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_past = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*past\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_past:
##        try:
##            mins = int(m_past.group(1))
##            htoken = m_past.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = int(h) % 24; minute = mins % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    m_to = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*to\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_to:
##        try:
##            mins = int(m_to.group(1))
##            htoken = m_to.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = (int(h) - 1) % 24; minute = (60 - (mins % 60)) % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    m_oclock = re.search(r"\b(\d{1,2})\s*(?:o['\s]?clock|oclock|o clock)\b", s_no_ampm)
##    if m_oclock:
##        try:
##            hour = int(m_oclock.group(1)) % 24; minute = 0
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##        except Exception:
##            pass
##
##    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
##    if len(tokens) >= 2:
##        h_candidate = _token_to_number(tokens[0])
##        m_candidate = _token_to_number(tokens[1])
##        if h_candidate is not None and m_candidate is not None and 0 <= m_candidate < 60:
##            hour = int(h_candidate) % 24; minute = int(m_candidate) % 60
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    if len(tokens) == 1:
##        h = _token_to_number(tokens[0])
##        if h is not None:
##            hour = int(h) % 24; minute = 0
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
##    if digits_cluster:
##        cluster = digits_cluster.group(1)
##        try:
##            if len(cluster) == 3: h = int(cluster[0]); m = int(cluster[1:])
##            else: h = int(cluster[:2]); m = int(cluster[2:])
##            if 0 <= h < 24 and 0 <= m < 60:
##                hour = h; minute = m
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    return None
##
### -------------------------
### persistence
### -------------------------
##SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
##os.makedirs(SCHEDULE_DIR, exist_ok=True)
##SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
##scheduled_events: List[dict] = []
##
##def _load_scheduled_events():
##    global scheduled_events
##    try:
##        if os.path.exists(SCHEDULE_DB):
##            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
##                scheduled_events = json.load(f)
##        else:
##            scheduled_events = []
##    except Exception as e:
##        print("Scheduled load failed:", e); scheduled_events = []
##
##def _save_scheduled_events():
##    try:
##        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
##            json.dump(scheduled_events, f, indent=2, default=str)
##    except Exception as e:
##        print("Scheduled save failed:", e)
##
### -------------------------
### schedule management
### -------------------------
##def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
##    try:
##        ev = {
##            "id": uuid.uuid4().hex,
##            "command": command_text,
##            "username": username or "Itf",
##            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
##            "description": description,
##            "fired": False
##        }
##        scheduled_events.append(ev)
##        _save_scheduled_events()
##        return ev
##    except Exception as e:
##        print("add_scheduled_command failed:", e)
##        raise
##
### -------------------------
### parsing helpers (improvements)
### -------------------------
##def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
##    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
##    out = []
##    for p in parts:
##        p_clean = p.strip()
##        if not p_clean:
##            continue
##        hm = spoken_time_to_hm(p_clean)
##        rel_dt = None
##        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
##        if m_rel:
##            num_phrase = m_rel.group(2).strip()
##            unit = m_rel.group(3).lower()
##            try:
##                num = int(num_phrase)
##            except:
##                num = words_to_number(num_phrase)
##            if num is not None:
##                now = dt.datetime.now()
##                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
##                elif unit.startswith("minute"): rel_dt = now + dt.timedelta(minutes=num)
##                elif unit.startswith("second"): rel_dt = now + dt.timedelta(seconds=num)
##                elif unit.startswith("day"): rel_dt = now + dt.timedelta(days=num)
##        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
##    return out
##
##def _strip_time_tokens_from_part(part: str) -> str:
##    """
##    Clean scheduling tokens from a phrase or a log line.
##    """
##    if not part:
##        return part
##
##    orig = part.strip()
##    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
##    command_candidate = None
##    if len(segments) > 1:
##        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
##        if alpha_segments:
##            command_candidate = max(alpha_segments, key=lambda s: len(s))
##    s = command_candidate if command_candidate else orig
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # remove explicit "on <date>" (conservative)
##    s = re.sub(r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', '', s, flags=re.I)
##
##    # remove "in/after <...> (seconds|minutes|hours|days)" completely
##    s = re.sub(r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b', '', s, flags=re.I)
##
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # If there's an 'at', cut EVERYTHING from 'at' onward
##    at_m = re.search(r'\bat\b', s, flags=re.I)
##    if at_m:
##        s = s[:at_m.start()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    # Handle 'on' occurrences:
##    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
##    if on_matches:
##        first = on_matches[0]
##        s = s[: first.end()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    s = re.sub(r'\s+', ' ', s).strip(' ,.')
##    return s
##
### -------------------------
### recurrence & date parsing helpers
### -------------------------
##_WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
##def _next_weekday_date(weekday_idx: int, start: Optional[dt.date] = None) -> dt.date:
##    start = start or dt.date.today()
##    days_ahead = (weekday_idx - start.weekday()) % 7
##    if days_ahead == 0:
##        days_ahead = 7
##    return start + dt.timedelta(days=days_ahead)
##
##def _parse_base_date_from_part(part: str) -> dt.date:
##    txt = safe_str(part).lower()
##    today = dt.date.today()
##    if "today" in txt:
##        return today
##    if "tomorrow" in txt:
##        return today + dt.timedelta(days=1)
##    # 'this evening', 'this morning', 'in the morning', 'in the evening'
##    if re.search(r'\b(this|in the)?\s*(evening|morning|afternoon|night)\b', txt):
##        return today
##    # weekdays
##    for wd, idx in _WEEKDAY_MAP.items():
##        if re.search(rf'\b{wd}\b', txt):
##            if re.search(r'\bthis\b', txt) and idx == today.weekday():
##                return today
##            return _next_weekday_date(idx, start=today)
##    # default
##    return today
##
##def _parse_recurrence_rules(part: str) -> Dict[str, Any]:
##    txt = safe_str(part).lower()
##    rules = {"freq": None, "interval": 1, "count": None, "until": None, "weekdays": None}
##
##    m_every = re.search(r'\bevery\s+(second\s+day|second\s+week|day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', txt)
##    if m_every:
##        token = m_every.group(1)
##        if "second day" in token or "every second day" in txt:
##            rules["freq"] = "daily"; rules["interval"] = 2
##        elif "second week" in token or "every second week" in txt:
##            rules["freq"] = "weekly"; rules["interval"] = 2
##        elif token in ("day",):
##            rules["freq"] = "daily"; rules["interval"] = 1
##        elif token == "week":
##            rules["freq"] = "weekly"; rules["interval"] = 1
##        elif token == "month":
##            rules["freq"] = "monthly"; rules["interval"] = 1
##        elif token in _WEEKDAY_MAP:
##            rules["freq"] = "weekly"; rules["weekdays"] = [ _WEEKDAY_MAP[token] ]
##
##    m_next = re.search(r'for the next\s+(\d+)\s+(days?|weeks?|months?)', txt)
##    if m_next:
##        n = int(m_next.group(1))
##        unit = m_next.group(2)
##        if unit.startswith("day"): rules["count"] = n
##        elif unit.startswith("week"): rules["until"] = (dt.date.today() + dt.timedelta(weeks=n))
##        elif unit.startswith("month"):
##            rules["until"] = (dt.date.today() + dt.timedelta(days=30*n))
##
##    if re.search(r'\bfor the month\b', txt) or re.search(r'\bfor a month\b', txt):
##        rules["until"] = dt.date.today() + dt.timedelta(days=30)
##
##    if re.search(r'\bthis week\b', txt) and re.search(r'\bevery\b', txt):
##        today = dt.date.today()
##        end = today + dt.timedelta(days=(6 - today.weekday()))
##        rules["until"] = end
##        rules["freq"] = "daily"
##
##    return rules
##
##def _generate_recurrence_datetimes(base_dt: dt.datetime, rules: Dict[str, Any]) -> List[dt.datetime]:
##    if not rules or not any([rules.get("freq"), rules.get("until"), rules.get("count"), rules.get("weekdays")]):
##        return [base_dt]
##
##    out = []
##    freq = rules.get("freq")
##    interval = max(1, int(rules.get("interval", 1)))
##    count = rules.get("count")
##    until = rules.get("until")
##    weekdays = rules.get("weekdays")
##
##    if until is None and (freq is not None or weekdays is not None):
##        until = (base_dt.date() + dt.timedelta(days=30))  # default one month
##
##    if weekdays:
##        cur_date = base_dt.date()
##        while cur_date <= until:
##            if cur_date.weekday() in weekdays:
##                out.append(dt.datetime.combine(cur_date, base_dt.time()))
##            cur_date = cur_date + dt.timedelta(days=1)
##        return out
##
##    if freq == "daily":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=interval)
##        return out
##
##    if freq == "weekly":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(weeks=interval)
##        return out
##
##    if freq == "monthly":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=30 * interval)
##        return out
##
##    return [base_dt]
##
### -------------------------
### executor thread
### -------------------------
##_SCHEDULER_THREAD = None
##_SCHEDULER_LOCK = threading.Lock()
##
##def _execute_event(ev: dict):
##    try:
##        cmd = ev.get("command", "")
##        user = ev.get("username", "Itf")
##        if listen is not None and hasattr(listen, "listen"):
##            text_msg = cmd
##            try:
##                if hasattr(listen, "add_text"):
##                    listen.add_text(text_msg)
##                else:
##                    try:
##                        listen_queue = getattr(listen, "queue", None)
##                        if listen_queue is not None and hasattr(listen_queue, "put"):
##                            listen_queue.put(text_msg)
##                    except Exception:
##                        pass
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
##                else:
##                    print("[scheduled_commands] Executing:", cmd)
##            except Exception as e:
##                print("Error calling listen.add_text:", e)
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak("Failed to execute timed command.")
##        else:
##            if speech is not None and hasattr(speech, "AlfredSpeak"):
##                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
##            else:
##                print("(No listen available) Would execute:", cmd)
##    except Exception as e:
##        print("_execute_event error:", e)
##
##def _scheduler_loop(poll_seconds: int = 15):
##    while True:
##        try:
##            now = dt.datetime.now()
##            changed = False
##            for ev in scheduled_events:
##                try:
##                    if ev.get("fired", False):
##                        continue
##                    dtstart = dt.datetime.fromisoformat(ev["dtstart"])
##                    if now >= dtstart:
##                        ev["fired"] = True
##                        changed = True
##                        _execute_event(ev)
##                except Exception as e:
##                    print("scheduled event inspect error:", e)
##                    continue
##            if changed:
##                _save_scheduled_events()
##        except Exception as e:
##            print("Scheduler loop error:", e)
##        time.sleep(poll_seconds)
##
##def start_scheduler_thread(poll_seconds: int = 15):
##    global _SCHEDULER_THREAD
##    with _SCHEDULER_LOCK:
##        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
##            return
##        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
##        _SCHEDULER_THREAD.start()
##
### -------------------------
### public API: handle a user utterance and schedule commands if found
### -------------------------
##def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
##    """
##    Parse `text` for time-based commands. If found, schedule them and return a list
##    of created event dicts. If no scheduling performed, return None.
##    """
##    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)
##
##    New_Message = message
##    New_speaker = speaker
##    New_score = score
##    New_gender = gender
##    New_gender_conf = gender_conf
##    New_timestamp = timestamp
##
##    print(f"New_Message Timed Command Module       : {New_Message}")
##    print(f"New_speaker Timed Command Module       : {New_speaker}")
##    print(f"New_score Timed Command Module         : {New_score}")
##    print(f"New_gender Timed Command Module        : {New_gender}")
##    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
##    print(f"New_timestamp Timed Command Module     : {New_timestamp}")
##
##    text_clean = safe_str(text)
##    lower = text_clean.lower()
##
##    # conservative time indicators (expanded)
##    time_indicators = bool(re.search(r"\b(at|o'clock|o clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2}|this (morning|afternoon|evening|night)|in the morning|in the evening|evening|morning|tonight)\b", text_clean, flags=re.I))
##    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
##    if not time_indicators or reminder_like:
##        return None
##
##    parts = _parse_command_sequence_parts(text_clean)
##    if not parts:
##        return None
##
##    scheduled = []
##    for p in parts:
##        part_text = p.get("part", "")
##        cmd_text = _strip_time_tokens_from_part(part_text) or part_text
##
##        # parse a concrete datetime (prefer rel_dt, then explicit hm+date)
##        dtstart = None
##        if p.get("rel_dt"):
##            dtstart = p["rel_dt"]
##        else:
##            # determine base date from the whole part (order-insensitive)
##            base_date = _parse_base_date_from_part(part_text)
##
##            hm = p.get("hm")
##            if hm:
##                h, m = hm
##                try:
##                    cand = dt.datetime.combine(base_date, dt.time(h, m))
##                except Exception:
##                    cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
##                # if cand is in the past and user didn't explicitly say "today", push to next day
##                if cand < dt.datetime.now() and not re.search(r'\btoday\b', part_text.lower()):
##                    cand = cand + dt.timedelta(days=1)
##                dtstart = cand
##            else:
##                if New_timestamp:
##                    try:
##                        parsed_ts = dt.datetime.fromisoformat(New_timestamp.replace(" ", "T"))
##                        dtstart = parsed_ts
##                    except Exception:
##                        try:
##                            dtstart = dt.datetime.fromisoformat(New_timestamp)
##                        except Exception:
##                            dtstart = None
##
##        if dtstart is None:
##            dtstart = dt.datetime.now() + dt.timedelta(minutes=1)
##
##        # recurrence
##        rules = _parse_recurrence_rules(part_text)
##        occurrence_datetimes = _generate_recurrence_datetimes(dtstart, rules)
##
##        for occ in occurrence_datetimes:
##            try:
##                ev = add_scheduled_command(cmd_text, occ, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
##                scheduled.append(ev)
##            except Exception as e:
##                print("Failed to schedule part:", e)
##                continue
##
##    if scheduled:
##        if speech is not None and hasattr(speech, "AlfredSpeak"):
##            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
##        else:
##            print(f"Scheduled {len(scheduled)} command(s).")
##        if gui is not None and hasattr(gui, "log_query"):
##            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
##        return scheduled
##    return None
##
### initialization
##_load_scheduled_events()
##start_scheduler_thread()
##
##































### scheduled_commands.py
##from __future__ import annotations
##import re
##import os
##import json
##import uuid
##import time
##import threading
##from typing import List, Optional, Dict, Any, Tuple
##import datetime as dt
##
### try to reuse project speech/listen objects
##try:
##    from speech import speech
##except Exception:
##    speech = None
##try:
##    from listen import listen
##except Exception:
##    listen = None
##
### basic helpers
##def safe_str(val) -> str:
##    if val is None:
##        return ""
##    if isinstance(val, str):
##        return val.strip()
##    try:
##        return str(val)
##    except Exception:
##        return ""
##
##import re
##import base64
##import ast
##
### -------------------------
### extractor (adapted from your original)
### -------------------------
##def extract_text_from_timed_command(query):
##    """
##    Returns: (message, speaker, score, gender, gender_conf, timestamp)
##    """
##    if query is None:
##        return "", None, None, None, None, None
##
##    # helper: extract timestamp from a fragment (YYYY-MM-DD and HH:MM:SS)
##    def _extract_timestamp(fragment: str):
##        if not fragment:
##            return None
##        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
##        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
##        if m_date and m_time:
##            return f"{m_date.group('date')} {m_time.group('time')}"
##        if m_date:
##            return m_date.group('date')
##        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
##        if m_date2:
##            return m_date2.group('date')
##        return None
##
##    # ---------- dict case ----------
##    if isinstance(query, dict):
##        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
##        speaker_ = query.get("username") or query.get("speaker")
##        score_ = query.get("score")
##        gender_ = query.get("gender")
##        gender_conf_ = query.get("gender_conf")
##        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
##        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)
##
##    # --- string case ---
##    if isinstance(query, str):
##        s = query.strip()
##
##        # base64 decode heuristic
##        try:
##            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
##                try:
##                    decoded = base64.b64decode(s).decode('utf-8')
##                    if decoded:
##                        s = decoded.strip()
##                except Exception:
##                    pass
##        except Exception:
##            pass
##
##        s = s.strip()
##
##        # helper to extract metadata
##        def _extract_meta(fragment):
##            frag = fragment or ""
##            frag = str(frag)
##            speaker = None
##            score = None
##            gender = None
##            gender_conf = None
##
##            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
##            if m:
##                speaker = m.group("u").strip()
##
##            if speaker is None:
##                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
##                if m2:
##                    speaker = m2.group("u").strip()
##
##            if speaker is None:
##                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
##                if m3:
##                    maybe_u = m3.group("u").strip()
##                    if " " not in maybe_u or len(maybe_u) <= 24:
##                        speaker = maybe_u
##
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##
##            if speaker and not gender:
##                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
##                if mg_end:
##                    gender = mg_end.group('gword').strip()
##                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
##
##            return speaker, score, gender, gender_conf
##
##        # detect triple fence wrappers
##        first_triple = s.find("'''")
##        backtick_first = s.find("```")
##        wrapper_pos = None
##        wrapper_token = None
##        if first_triple != -1:
##            wrapper_pos = first_triple; wrapper_token = "'''"
##        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
##            wrapper_pos = backtick_first; wrapper_token = "```"
##
##        if wrapper_pos is not None:
##            start = wrapper_pos
##            end = s.find(wrapper_token, start + len(wrapper_token))
##            if end != -1:
##                inner = s[start + len(wrapper_token) : end]
##                remainder = s[end + len(wrapper_token) :].strip()
##                inner = inner.lstrip()
##                inner = re.sub(r"^'+", "", inner)
##                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
##                speaker, score, gender, gender_conf = _extract_meta(remainder)
##                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
##                if not speaker:
##                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
##                    if m_all:
##                        speaker = m_all.group("u").strip()
##                message = inner.strip()
##                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # structured line pattern
##        m_struct = re.match(
##            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
##            s
##        )
##        if m_struct:
##            date = m_struct.group('date'); time_ = m_struct.group('time')
##            timestamp = f"{date} {time_}"
##            message = m_struct.group('msg').strip()
##            user_full = m_struct.group('user').strip()
##            gender = None
##            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
##            if m_gender:
##                gender = m_gender.group('g').strip()
##                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
##            else:
##                user_clean = user_full
##            speaker = user_clean if user_clean else None
##            return message, speaker, None, (gender if gender else None), None, timestamp
##
##        # structured string attempt
##        try:
##            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
##            m = re.search(pattern, s, flags=re.DOTALL)
##            if m:
##                message_ = m.group('message').strip()
##                speaker_ = m.group('username').strip() or None
##                score_ = m.group('score') or None
##                gender_ = m.group('gender') or None
##                gender_conf_ = m.group('gender_conf') or None
##                timestamp_ = _extract_timestamp(s)
##                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
##        except Exception:
##            pass
##
##        # username anywhere
##        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
##        speaker = None; score = None; gender = None; gender_conf = None
##        timestamp = _extract_timestamp(s)
##        if m_user_any:
##            speaker = m_user_any.group("u").strip()
##            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##            candidate = s.strip()
##            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # username at end
##        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
##        timestamp = _extract_timestamp(s)
##        if m_user2:
##            speaker = m_user2.group("u").strip()
##            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
##            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
##
##        # last-token username heuristic
##        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
##        timestamp = _extract_timestamp(s)
##        if m_user3:
##            maybe_u = m_user3.group("u").strip()
##            maybe_body = m_user3.group("body").strip()
##            if " " not in maybe_u or len(maybe_u) <= 24:
##                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
##
##        # fallback
##        return s, None, None, None, None, None
##
##    return str(query).strip(), None, None, None, None, None
##
### -------------------------
### small words->number helpers (kept minimal and self-contained)
### -------------------------
##_UNITS = {
##    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
##    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
##    "seventeen":17,"eighteen":18,"nineteen":19
##}
##_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
##
##def words_to_number(phrase: str) -> Optional[int]:
##    if phrase is None: return None
##    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
##    if not words: return None
##    total = 0; current = 0; valid = False
##    for w in words:
##        if w in _UNITS:
##            current += _UNITS[w]; valid = True
##        elif w in _TENS:
##            current += _TENS[w]; valid = True
##        elif w == "and":
##            continue
##        else:
##            return None
##    return (total + current) if valid else None
##
### -------------------------
### time parser (improved subset)
### -------------------------
##_AM_WORDS = {"am","a.m.","morning","this morning"}
##_PM_WORDS = {"pm","p.m.","evening","afternoon","tonight","this evening","night","this afternoon"}
##
##def _token_to_number(token: str) -> Optional[int]:
##    token = safe_str(token).lower()
##    if not token: return None
##    if re.fullmatch(r"\d+", token):
##        try: return int(token)
##        except: return None
##    if token in _UNITS: return _UNITS[token]
##    if token in _TENS: return _TENS[token]
##    if "-" in token:
##        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
##        if all(v is not None for v in vals): return sum(vals)
##    return words_to_number(token)
##
##def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
##    s0 = safe_str(s).lower()
##    ampm = None
##    for w in _AM_WORDS:
##        if re.search(r"\b" + re.escape(w) + r"\b", s0):
##            ampm = "am"; break
##    if ampm is None:
##        for w in _PM_WORDS:
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                ampm = "pm"; break
##    if re.search(r"\bnoon\b", s0): ampm = "pm"
##    if re.search(r"\bmidnight\b", s0): ampm = "am"
##    if ampm:
##        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|morning|afternoon|evening|night|in the morning|in the evening|this morning|this evening|tonight|noon|midnight|this afternoon)\b"
##        s0 = re.sub(pattern, " ", s0)
##        s0 = re.sub(r'\s+', ' ', s0).strip()
##    return s0, ampm
##
##def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
##    """
##    Robust spoken time -> (hour, minute) parser.
##    """
##    if spoken is None: return None
##    if isinstance(spoken, dt.datetime): return (spoken.hour, spoken.minute)
##    if isinstance(spoken, dt.time): return (spoken.hour, spoken.minute)
##
##    s_orig = safe_str(spoken)
##    s = s_orig.lower().replace("-", " ").replace(".", " ").replace(",", " ").strip()
##    if re.search(r"\bnoon\b", s): return (12, 0)
##    if re.search(r"\bmidnight\b", s): return (0, 0)
##
##    s_no_ampm, ampm = _detect_ampm_and_remove(s)
##
##    # explicit 24h with colon or 'h'
##    m_colon = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", s_no_ampm, flags=re.I)
##    if m_colon:
##        try:
##            hh = int(m_colon.group(1)) % 24; mm = int(m_colon.group(2)) % 60
##            hour = hh; minute = mm
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##        except Exception:
##            pass
##
##    m_half = re.search(r"\bhalf past ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_half:
##        token = m_half.group(1).strip(); h = _token_to_number(token)
##        if h is not None:
##            hour = int(h) % 24; minute = 30
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_quarter = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)\b", s_no_ampm)
##    if m_quarter:
##        typ = m_quarter.group(1); hour_token = m_quarter.group(2).strip(); h = _token_to_number(hour_token)
##        if h is not None:
##            hour = int(h) % 24
##            if typ == "past":
##                minute = 15
##            else:
##                minute = 45; hour = (hour - 1) % 24
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    m_past = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*past\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_past:
##        try:
##            mins = int(m_past.group(1))
##            htoken = m_past.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = int(h) % 24; minute = mins % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    m_to = re.search(r"\b(\d{1,2})\s*(?:minutes?|mins?)?\s*to\s+(\d{1,2}|[a-z]+)\b", s_no_ampm)
##    if m_to:
##        try:
##            mins = int(m_to.group(1))
##            htoken = m_to.group(2)
##            h = _token_to_number(htoken) if not re.fullmatch(r"\d+", htoken) else int(htoken)
##            if h is not None:
##                hour = (int(h) - 1) % 24; minute = (60 - (mins % 60)) % 60
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    m_oclock = re.search(r"\b(\d{1,2})\s*(?:o['\s]?clock|oclock|o clock)\b", s_no_ampm)
##    if m_oclock:
##        try:
##            hour = int(m_oclock.group(1)) % 24; minute = 0
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##        except Exception:
##            pass
##
##    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
##    if len(tokens) >= 2:
##        h_candidate = _token_to_number(tokens[0])
##        m_candidate = _token_to_number(tokens[1])
##        if h_candidate is not None and m_candidate is not None and 0 <= m_candidate < 60:
##            hour = int(h_candidate) % 24; minute = int(m_candidate) % 60
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    if len(tokens) == 1:
##        h = _token_to_number(tokens[0])
##        if h is not None:
##            hour = int(h) % 24; minute = 0
##            if ampm == "pm" and hour < 12: hour += 12
##            if ampm == "am" and hour == 12: hour = 0
##            return (hour, minute)
##
##    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
##    if digits_cluster:
##        cluster = digits_cluster.group(1)
##        try:
##            if len(cluster) == 3: h = int(cluster[0]); m = int(cluster[1:])
##            else: h = int(cluster[:2]); m = int(cluster[2:])
##            if 0 <= h < 24 and 0 <= m < 60:
##                hour = h; minute = m
##                if ampm == "pm" and hour < 12: hour += 12
##                if ampm == "am" and hour == 12: hour = 0
##                return (hour, minute)
##        except Exception:
##            pass
##
##    return None
##
### -------------------------
### persistence
### -------------------------
##SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
##os.makedirs(SCHEDULE_DIR, exist_ok=True)
##SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
##scheduled_events: List[dict] = []
##
##def _load_scheduled_events():
##    global scheduled_events
##    try:
##        if os.path.exists(SCHEDULE_DB):
##            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
##                scheduled_events = json.load(f)
##        else:
##            scheduled_events = []
##    except Exception as e:
##        print("Scheduled load failed:", e); scheduled_events = []
##
##def _save_scheduled_events():
##    try:
##        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
##            json.dump(scheduled_events, f, indent=2, default=str)
##    except Exception as e:
##        print("Scheduled save failed:", e)
##
### -------------------------
### schedule management
### -------------------------
##def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
##    try:
##        ev = {
##            "id": uuid.uuid4().hex,
##            "command": command_text,
##            "username": username or "Itf",
##            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
##            "description": description,
##            "fired": False
##        }
##        scheduled_events.append(ev)
##        _save_scheduled_events()
##        return ev
##    except Exception as e:
##        print("add_scheduled_command failed:", e)
##        raise
##
### -------------------------
### parsing helpers (improvements)
### -------------------------
##def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
##    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
##    out = []
##    for p in parts:
##        p_clean = p.strip()
##        if not p_clean:
##            continue
##        hm = spoken_time_to_hm(p_clean)
##        rel_dt = None
##        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
##        if m_rel:
##            num_phrase = m_rel.group(2).strip()
##            unit = m_rel.group(3).lower()
##            try:
##                num = int(num_phrase)
##            except:
##                num = words_to_number(num_phrase)
##            if num is not None:
##                now = dt.datetime.now()
##                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
##                elif unit.startswith("minute"): rel_dt = now + dt.timedelta(minutes=num)
##                elif unit.startswith("second"): rel_dt = now + dt.timedelta(seconds=num)
##                elif unit.startswith("day"): rel_dt = now + dt.timedelta(days=num)
##        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
##    return out
##
##def _strip_time_tokens_from_part(part: str) -> str:
##    """
##    Clean scheduling tokens from a phrase or a log line.
##    """
##    if not part:
##        return part
##
##    orig = part.strip()
##    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
##    command_candidate = None
##    if len(segments) > 1:
##        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
##        if alpha_segments:
##            command_candidate = max(alpha_segments, key=lambda s: len(s))
##    s = command_candidate if command_candidate else orig
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # remove explicit "on <date>" (conservative)
##    s = re.sub(r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', '', s, flags=re.I)
##
##    # remove "in/after <...> (seconds|minutes|hours|days)" completely
##    s = re.sub(r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b', '', s, flags=re.I)
##
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # If there's an 'at', cut EVERYTHING from 'at' onward
##    at_m = re.search(r'\bat\b', s, flags=re.I)
##    if at_m:
##        s = s[:at_m.start()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    # Handle 'on' occurrences:
##    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
##    if on_matches:
##        first = on_matches[0]
##        s = s[: first.end()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    s = re.sub(r'\s+', ' ', s).strip(' ,.')
##    return s
##
### -------------------------
### recurrence & date parsing helpers
### -------------------------
##_WEEKDAY_MAP = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
##def _next_weekday_date(weekday_idx: int, start: Optional[dt.date] = None) -> dt.date:
##    start = start or dt.date.today()
##    days_ahead = (weekday_idx - start.weekday()) % 7
##    if days_ahead == 0:
##        days_ahead = 7
##    return start + dt.timedelta(days=days_ahead)
##
##def _parse_base_date_from_part(part: str) -> dt.date:
##    txt = safe_str(part).lower()
##    today = dt.date.today()
##    if "today" in txt:
##        return today
##    if "tomorrow" in txt:
##        return today + dt.timedelta(days=1)
##    # 'this evening', 'this morning' -> use today
##    if re.search(r'\bthis\b.*\b(evening|morning|afternoon|night)\b', txt):
##        return today
##    # weekdays
##    for wd, idx in _WEEKDAY_MAP.items():
##        if re.search(rf'\b{wd}\b', txt):
##            # if user said e.g., "every monday" we may want next monday (including today? choose next)
##            # If today is the same weekday and user included "this", schedule today, else next occurrence.
##            if re.search(r'\bthis\b', txt) and idx == today.weekday():
##                return today
##            return _next_weekday_date(idx, start=today)
##    # default: today
##    return today
##
##def _parse_recurrence_rules(part: str) -> Dict[str, Any]:
##    """
##    Detect recurrence intent and parameters.
##    Returns dict like:
##      {"freq":"daily"|"weekly"|"monthly"|"weekday":<int>|None, "interval":1, "count":<int> or None, "until": date or None}
##    """
##    txt = safe_str(part).lower()
##    rules = {"freq": None, "interval": 1, "count": None, "until": None, "weekdays": None}
##
##    # "every X"
##    m_every = re.search(r'\bevery\s+(second\s+day|second\s+week|day|week|month|day of the week|day this week|day of the month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|day)\b', txt)
##    if m_every:
##        token = m_every.group(1)
##        if "second day" in token or "every second day" in txt:
##            rules["freq"] = "daily"; rules["interval"] = 2
##        elif "second week" in token or "every second week" in txt:
##            rules["freq"] = "weekly"; rules["interval"] = 2
##        elif token in ("day", "day this week", "day of the week"):
##            rules["freq"] = "daily"; rules["interval"] = 1
##        elif token == "week":
##            rules["freq"] = "weekly"; rules["interval"] = 1
##        elif token == "month" or "month" in txt and "every month" in txt:
##            rules["freq"] = "monthly"; rules["interval"] = 1
##        elif token in _WEEKDAY_MAP:
##            rules["freq"] = "weekly"; rules["weekdays"] = [ _WEEKDAY_MAP[token] ]
##    # "for the next N weeks/days/months"
##    m_next = re.search(r'for the next\s+(\d+)\s+(days?|weeks?|months?)', txt)
##    if m_next:
##        n = int(m_next.group(1))
##        unit = m_next.group(2)
##        if unit.startswith("day"): rules["count"] = n
##        elif unit.startswith("week"): rules["count"] = n * 7  # interpret count as days (we'll use until date)
##        elif unit.startswith("month"):
##            # convert to until date later
##            rules["until"] = (dt.date.today() + dt.timedelta(days=30*n))
##    # "for the month" meaning next 30 days
##    if re.search(r'\bfor the month\b', txt) or re.search(r'\bfor a month\b', txt):
##        rules["until"] = dt.date.today() + dt.timedelta(days=30)
##    # "for every day this week" -> schedule remaining days in the week (until Sunday)
##    if re.search(r'\bthis week\b', txt) and re.search(r'\bevery\b', txt):
##        # compute until end of week (Sunday)
##        today = dt.date.today()
##        end = today + dt.timedelta(days=(6 - today.weekday()))
##        rules["until"] = end
##        rules["freq"] = "daily"
##    return rules
##
##def _generate_recurrence_datetimes(base_dt: dt.datetime, rules: Dict[str, Any]) -> List[dt.datetime]:
##    """
##    Given a base datetime and rules, generate a list of datetimes for scheduling.
##    Conservative defaults:
##      - if rules empty => return [base_dt]
##      - if freq daily/weekly/monthly with until or count generate accordingly
##      - default duration: 30 days (1 month) if freq present but no until/count
##    """
##    if not rules or not any([rules.get("freq"), rules.get("until"), rules.get("count"), rules.get("weekdays")]):
##        return [base_dt]
##
##    out = []
##    freq = rules.get("freq")
##    interval = max(1, int(rules.get("interval", 1)))
##    count = rules.get("count")
##    until = rules.get("until")
##    weekdays = rules.get("weekdays")
##
##    # compute until date default
##    if until is None and (freq is not None or weekdays is not None):
##        until = (base_dt.date() + dt.timedelta(days=30))  # default one month
##
##    current = base_dt
##    # If weekdays specified (like every Monday), iterate weekly and pick matching weekdays
##    if weekdays:
##        # generate occurrences from base_dt until 'until' by stepping day by day and selecting matching weekdays
##        cur_date = base_dt.date()
##        while cur_date <= until:
##            if cur_date.weekday() in weekdays:
##                out.append(dt.datetime.combine(cur_date, base_dt.time()))
##            cur_date = cur_date + dt.timedelta(days=1)
##        return out
##
##    if freq == "daily":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=interval)
##        return out
##
##    if freq == "weekly":
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(weeks=interval)
##        return out
##
##    if freq == "monthly":
##        # monthly stepping naive: add interval*30 days (approx)
##        cur = base_dt
##        while True:
##            if until and cur.date() > until:
##                break
##            out.append(cur)
##            if count and len(out) >= count:
##                break
##            cur = cur + dt.timedelta(days=30 * interval)
##        return out
##
##    # fallback
##    return [base_dt]
##
### -------------------------
### executor thread
### -------------------------
##_SCHEDULER_THREAD = None
##_SCHEDULER_LOCK = threading.Lock()
##
##def _execute_event(ev: dict):
##    try:
##        cmd = ev.get("command", "")
##        user = ev.get("username", "Itf")
##        if listen is not None and hasattr(listen, "listen"):
##            text_msg = cmd
##            try:
##                # add_text if available, else call listen.add_text if that exists
##                if hasattr(listen, "add_text"):
##                    listen.add_text(text_msg)
##                else:
##                    # fallback: attempt to push into a queue if present
##                    try:
##                        listen_queue = getattr(listen, "queue", None)
##                        if listen_queue is not None and hasattr(listen_queue, "put"):
##                            listen_queue.put(text_msg)
##                    except Exception:
##                        pass
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
##                else:
##                    print("[scheduled_commands] Executing:", cmd)
##            except Exception as e:
##                print("Error calling listen.add_text:", e)
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak("Failed to execute timed command.")
##        else:
##            if speech is not None and hasattr(speech, "AlfredSpeak"):
##                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
##            else:
##                print("(No listen available) Would execute:", cmd)
##    except Exception as e:
##        print("_execute_event error:", e)
##
##def _scheduler_loop(poll_seconds: int = 15):
##    while True:
##        try:
##            now = dt.datetime.now()
##            changed = False
##            for ev in scheduled_events:
##                try:
##                    if ev.get("fired", False):
##                        continue
##                    dtstart = dt.datetime.fromisoformat(ev["dtstart"])
##                    if now >= dtstart:
##                        ev["fired"] = True
##                        changed = True
##                        _execute_event(ev)
##                except Exception as e:
##                    print("scheduled event inspect error:", e)
##                    continue
##            if changed:
##                _save_scheduled_events()
##        except Exception as e:
##            print("Scheduler loop error:", e)
##        time.sleep(poll_seconds)
##
##def start_scheduler_thread(poll_seconds: int = 15):
##    global _SCHEDULER_THREAD
##    with _SCHEDULER_LOCK:
##        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
##            return
##        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
##        _SCHEDULER_THREAD.start()
##
### -------------------------
### public API: handle a user utterance and schedule commands if found
### -------------------------
##def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
##    """
##    Parse `text` for time-based commands. If found, schedule them and return a list
##    of created event dicts. If no scheduling performed, return None.
##    """
##    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)
##
##    New_Message = message
##    New_speaker = speaker
##    New_score = score
##    New_gender = gender
##    New_gender_conf = gender_conf
##    New_timestamp = timestamp
##
##    print(f"New_Message Timed Command Module       : {New_Message}")
##    print(f"New_speaker Timed Command Module       : {New_speaker}")
##    print(f"New_score Timed Command Module         : {New_score}")
##    print(f"New_gender Timed Command Module        : {New_gender}")
##    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
##    print(f"New_timestamp Timed Command Module     : {New_timestamp}")
##
##    text_clean = safe_str(text)
##    lower = text_clean.lower()
##
##    # conservative time indicators (expanded)
##    time_indicators = bool(re.search(r"\b(at|o'clock|o clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2}|this (morning|afternoon|evening|night)|evening|morning|tonight)\b", text_clean, flags=re.I))
##    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
##    if not time_indicators or reminder_like:
##        return None
##
##    parts = _parse_command_sequence_parts(text_clean)
##    if not parts:
##        return None
##
##    scheduled = []
##    for p in parts:
##        part_text = p.get("part", "")
##        cmd_text = _strip_time_tokens_from_part(part_text) or part_text
##
##        # parse a concrete datetime (prefer rel_dt, then explicit hm+date)
##        dtstart = None
##        if p.get("rel_dt"):
##            dtstart = p["rel_dt"]
##        else:
##            # determine base date (today/tomorrow/weekday) from the whole part (so order doesn't matter)
##            base_date = _parse_base_date_from_part(part_text)
##
##            hm = p.get("hm")
##            if hm:
##                h, m = hm
##                try:
##                    cand = dt.datetime.combine(base_date, dt.time(h, m))
##                except Exception:
##                    cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
##                # if cand is in the past (for same-day) push to next day unless user explicitly said today
##                if cand < dt.datetime.now():
##                    if re.search(r'\btoday\b', part_text.lower()):
##                        # keep as-is even if past (user explicitly said today)
##                        pass
##                    else:
##                        cand = cand + dt.timedelta(days=1)
##                dtstart = cand
##            else:
##                # try to detect relative textual time inside part (e.g., 'in 10 minutes' already handled above)
##                # fallback: if there's a timestamp extracted earlier, use it
##                if New_timestamp:
##                    try:
##                        parsed_ts = dt.datetime.fromisoformat(New_timestamp.replace(" ", "T"))
##                        dtstart = parsed_ts
##                    except Exception:
##                        try:
##                            dtstart = dt.datetime.fromisoformat(New_timestamp)
##                        except Exception:
##                            dtstart = None
##
##        # fallback minimal dtstart
##        if dtstart is None:
##            # schedule 1 minute from now
##            dtstart = dt.datetime.now() + dt.timedelta(minutes=1)
##
##        # parse recurrence rules from the part text (allow "every ..." and "for the next N ..." variants)
##        rules = _parse_recurrence_rules(part_text)
##        occurrence_datetimes = _generate_recurrence_datetimes(dtstart, rules)
##
##        # create scheduled events for each datetime
##        for occ in occurrence_datetimes:
##            try:
##                ev = add_scheduled_command(cmd_text, occ, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
##                scheduled.append(ev)
##            except Exception as e:
##                print("Failed to schedule part:", e)
##                continue
##
##    if scheduled:
##        if speech is not None and hasattr(speech, "AlfredSpeak"):
##            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
##        else:
##            print(f"Scheduled {len(scheduled)} command(s).")
##        if gui is not None and hasattr(gui, "log_query"):
##            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
##        return scheduled
##    return None
##
### initialization
##_load_scheduled_events()
##start_scheduler_thread()




















##  # BEST SO FAR 2026_01_25__22h00
##
### scheduled_commands.py
##from __future__ import annotations
##import re
##import os
##import json
##import uuid
##import time
##import threading
##from typing import List, Optional, Dict, Any, Tuple
##import datetime as dt
##
### try to reuse project speech/listen objects
##try:
##    from speech import speech
##except Exception:
##    speech = None
##try:
##    from listen import listen
##except Exception:
##    listen = None
##
### basic helpers
##def safe_str(val) -> str:
##    if val is None:
##        return ""
##    if isinstance(val, str):
##        return val.strip()
##    try:
##        return str(val)
##    except Exception:
##        return ""
##
##import re
##import base64
##
##def extract_text_from_timed_command(query):
##    """
##    Returns: (message, speaker, score, gender, gender_conf, timestamp)
##
##    Behavior:
##      - dict: uses keys text/query/message/q and username/speaker and looks for timestamp/time/date keys
##      - strings:
##        * will detect and extract triple-single-quote wrappers '''...''' (or ``` fences)
##        * if wrapper present: returns inner content (cleaned) as message and parses remainder for username/score/gender/gender_conf/timestamp
##        * if no wrapper: tries safe structured-string regex for 'message':'...':'username':'name' patterns
##        * also tries to detect structured lines like:
##            "<tag> : 2026-01-22 : 18:44:34 : message : Username (Female)"
##          and will extract timestamp (YYYY-MM-DD HH:MM:SS), message, username, and gender.
##        * falls back to returning raw string (and any username if parseable)
##      - tries base64 decode if the input looks like base64
##    """
##    if query is None:
##        return "", None, None, None, None, None
##
##    import re, base64
##
##    # helper: extract timestamp from a fragment (YYYY-MM-DD and HH:MM:SS)
##    def _extract_timestamp(fragment: str):
##        if not fragment:
##            return None
##        # try to find date and time nearby
##        m_date = re.search(r'(?P<date>\d{4}-\d{2}-\d{2})', fragment)
##        m_time = re.search(r'(?P<time>\d{2}:\d{2}:\d{2})', fragment)
##        if m_date and m_time:
##            return f"{m_date.group('date')} {m_time.group('time')}"
##        if m_date:
##            return m_date.group('date')
##        # Try alternative common formats like DD/MM/YYYY or YYYY/MM/DD (optional)
##        m_date2 = re.search(r'(?P<date>\d{2}/\d{2}/\d{4})', fragment)
##        if m_date2:
##            return m_date2.group('date')
##        return None
##
##    # ---------- dict case (preferred structured format) ----------
##    if isinstance(query, dict):
##        text_ = query.get("text") or query.get("query") or query.get("message") or query.get("q") or ""
##        speaker_ = query.get("username") or query.get("speaker")
##        score_ = query.get("score")
##        gender_ = query.get("gender")
##        gender_conf_ = query.get("gender_conf")
##        timestamp_ = query.get("timestamp") or query.get("time") or query.get("date") or None
##
##        # normalize to str
##        return str(text_).strip(), (str(speaker_).strip() if speaker_ is not None else None), score_, gender_, gender_conf_, (str(timestamp_).strip() if timestamp_ is not None else None)
##
##    # --- string case ---
##    if isinstance(query, str):
##        s = query.strip()
##
##        # 1) Try base64 decode heuristic (optional)
##        try:
##            if len(s) > 50 and re.fullmatch(r'[A-Za-z0-9+/=\s]+', s) and '\n' not in s:
##                try:
##                    decoded = base64.b64decode(s).decode('utf-8')
##                    if decoded:
##                        s = decoded.strip()
##                except Exception:
##                    pass
##        except Exception:
##            pass
##
##        # normalize repeated leading spaces/newlines
##        s = s.strip()
##
##        # Helper to extract metadata (username/score/gender/gender_conf) from a text fragment
##        def _extract_meta(fragment):
##            frag = fragment or ""
##            frag = str(frag)
##            speaker = None
##            score = None
##            gender = None
##            gender_conf = None
##
##            # 1) explicit 'username':Name
##            m = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", frag)
##            if m:
##                speaker = m.group("u").strip()
##
##            # 2) username: Name or user: Name (end-of-string preferred)
##            if speaker is None:
##                m2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", frag, flags=re.IGNORECASE)
##                if m2:
##                    speaker = m2.group("u").strip()
##
##            # 3) last-token " : Name" heuristic (conservative)
##            if speaker is None:
##                m3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", frag)
##                if m3:
##                    maybe_u = m3.group("u").strip()
##                    if " " not in maybe_u or len(maybe_u) <= 24:
##                        speaker = maybe_u
##
##            # 4) numeric/word tokens for score/gender/gender_conf if present
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", frag, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##
##            # Also capture "(Female)" style gender at end of a username token
##            if speaker and not gender:
##                mg_end = re.search(r'\((?P<gword>Male|Female|Other|M|F)\)\s*$', speaker, flags=re.IGNORECASE)
##                if mg_end:
##                    gender = mg_end.group('gword').strip()
##                    # remove parenthetical from speaker
##                    speaker = re.sub(r'\s*\([^\)]*\)\s*$', '', speaker).strip()
##
##            return speaker, score, gender, gender_conf
##
##        # 2) Detect triple-single-quote wrapper '''...'''
##        first_triple = s.find("'''")
##        backtick_first = s.find("```")
##        # choose wrapper if present earliest
##        wrapper_pos = None
##        wrapper_token = None
##        if first_triple != -1:
##            wrapper_pos = first_triple
##            wrapper_token = "'''"
##        if backtick_first != -1 and (wrapper_pos is None or backtick_first < wrapper_pos):
##            wrapper_pos = backtick_first
##            wrapper_token = "```"
##
##        if wrapper_pos is not None:
##            start = wrapper_pos
##            end = s.find(wrapper_token, start + len(wrapper_token))
##            if end != -1:
##                inner = s[start + len(wrapper_token) : end]
##                remainder = s[end + len(wrapper_token) :].strip()
##
##                # if inner begins with stray ' or stray message marker, remove it
##                inner = inner.lstrip()
##                inner = re.sub(r"^'+", "", inner)  # remove extra leading single quotes
##                inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*|message\s*:)\s*", "", inner, flags=re.IGNORECASE)
##
##                # extract meta from remainder (after the closing wrapper)
##                speaker, score, gender, gender_conf = _extract_meta(remainder)
##
##                # attempt to extract timestamp from remainder
##                timestamp = _extract_timestamp(remainder) or _extract_timestamp(s)
##
##                # if not found in remainder, maybe metadata sits after additional separators like " : 'username':Name"
##                if not speaker:
##                    # search whole original string after wrapper for 'username' anywhere
##                    m_all = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s[end+len(wrapper_token):])
##                    if m_all:
##                        speaker = m_all.group("u").strip()
##
##                # final message is inner (trim)
##                message = inner.strip()
##
##                return message, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##            # if no closing wrapper found, fall through to normal parsing
##
##        # 2.5) Try to match the very specific structured line:
##        # "<tag> : YYYY-MM-DD : HH:MM:SS : message : Username (Gender)"
##        m_struct = re.match(
##            r'^(?P<prefix>[^:]+?)\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*:\s*(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<msg>.*?)\s*:\s*(?P<user>.+)$',
##            s
##        )
##        if m_struct:
##            date = m_struct.group('date')
##            time_ = m_struct.group('time')
##            timestamp = f"{date} {time_}"
##            message = m_struct.group('msg').strip()
##            user_full = m_struct.group('user').strip()
##
##            # extract gender if present in parentheses at end
##            gender = None
##            m_gender = re.search(r'\((?P<g>[^)]+)\)\s*$', user_full)
##            if m_gender:
##                gender = m_gender.group('g').strip()
##                user_clean = re.sub(r'\s*\([^)]+\)\s*$', '', user_full).strip()
##            else:
##                user_clean = user_full
##
##            speaker = user_clean if user_clean else None
##            return message, speaker, None, (gender if gender else None), None, timestamp
##
##        # 3) Try structured-string parsing with quoted keys (safer)
##        try:
##            pattern = r"'message'\s*:\s*'(?P<message>.*?)'\s*:\s*'username'\s*:\s*'(?P<username>[^']*)'(?:\s*:\s*'score'\s*:\s*'(?P<score>[^']*)')?(?:\s*:\s*'gender'\s*:\s*'(?P<gender>[^']*)')?(?:\s*:\s*'gender_conf'\s*:\s*'(?P<gender_conf>[^']*)')?"
##            m = re.search(pattern, s, flags=re.DOTALL)
##            if m:
##                message_ = m.group('message').strip()
##                speaker_ = m.group('username').strip() or None
##                score_ = m.group('score') or None
##                gender_ = m.group('gender') or None
##                gender_conf_ = m.group('gender_conf') or None
##                timestamp_ = _extract_timestamp(s)
##                return message_, speaker_, score_, gender_, gender_conf_, timestamp_
##        except Exception:
##            pass
##
##        # 4) Try to extract "'username':Name" anywhere (and remove it from message candidate)
##        m_user_any = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
##        speaker = None
##        score = None
##        gender = None
##        gender_conf = None
##        timestamp = _extract_timestamp(s)
##        if m_user_any:
##            speaker = m_user_any.group("u").strip()
##            # remove the matched token from s
##            s = (s[: m_user_any.start()] + s[m_user_any.end():]).strip(" :\n\t ")
##
##            # also attempt score/gender after removing username token
##            ms = re.search(r"'score'\s*:\s*['\"]?(?P<s>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if ms:
##                score = ms.group("s").strip()
##            mg = re.search(r"'gender'\s*:\s*['\"]?(?P<g>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mg:
##                gender = mg.group("g").strip()
##            mgc = re.search(r"'gender_conf'\s*:\s*['\"]?(?P<gc>[^'\"\s:]+)['\"]?", s, flags=re.IGNORECASE)
##            if mgc:
##                gender_conf = mgc.group("gc").strip()
##
##            # remaining s is candidate message
##            candidate = s.strip()
##            return candidate, (speaker if speaker else None), (score if score else None), (gender if gender else None), (gender_conf if gender_conf else None), (timestamp if timestamp else None)
##
##        # 5) Try "username: Name" or "user: Name" at end
##        m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", s, flags=re.IGNORECASE)
##        timestamp = _extract_timestamp(s)
##        if m_user2:
##            speaker = m_user2.group("u").strip()
##            message_candidate = re.sub(r"(?:[:\s]*\busername\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)|(?:[:\s]*\buser\b\s*[:=]\s*['\"]?.+?['\"]?\s*$)", "", s, flags=re.IGNORECASE).strip(" :")
##            return message_candidate, (speaker if speaker else None), None, None, None, (timestamp if timestamp else None)
##
##        # 6) Last-token username heuristic "body : Name"
##        m_user3 = re.match(r"^(?P<body>.*\S)\s*:\s*(?P<u>[^\n:]{1,48})\s*$", s)
##        timestamp = _extract_timestamp(s)
##        if m_user3:
##            maybe_u = m_user3.group("u").strip()
##            maybe_body = m_user3.group("body").strip()
##            if " " not in maybe_u or len(maybe_u) <= 24:
##                return maybe_body, maybe_u, None, None, None, (timestamp if timestamp else None)
##
##        # 7) fallback: return raw string (no username)
##        return s, None, None, None, None, None
##
##    # --- fallback for other types ---
##    return str(query).strip(), None, None, None, None, None
##
##
### small words->number helpers (kept minimal and self-contained)
##_UNITS = {
##    "zero":0,"oh":0,"o":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
##    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
##    "seventeen":17,"eighteen":18,"nineteen":19
##}
##_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
##
##def words_to_number(phrase: str) -> Optional[int]:
##    if phrase is None: return None
##    words = re.findall(r"[a-z]+", safe_str(phrase).lower())
##    if not words: return None
##    total = 0; current = 0; valid = False
##    for w in words:
##        if w in _UNITS:
##            current += _UNITS[w]; valid = True
##        elif w in _TENS:
##            current += _TENS[w]; valid = True
##        elif w == "and":
##            continue
##        else:
##            return None
##    return (total + current) if valid else None
##
### time parser (a compact subset; mirrors your reminders logic)
##_AM_WORDS = {"am","a.m.","morning","this morning"}
##_PM_WORDS = {"pm","p.m.","evening","afternoon","tonight","this evening","night"}
##
##def _token_to_number(token: str) -> Optional[int]:
##    token = safe_str(token).lower()
##    if not token: return None
##    if re.fullmatch(r"\d+", token):
##        try: return int(token)
##        except: return None
##    if token in _UNITS: return _UNITS[token]
##    if token in _TENS: return _TENS[token]
##    if "-" in token:
##        parts = token.split("-"); vals = [_token_to_number(p) for p in parts]
##        if all(v is not None for v in vals): return sum(vals)
##    return words_to_number(token)
##
##def _detect_ampm_and_remove(s: str) -> Tuple[str, Optional[str]]:
##    s0 = safe_str(s).lower()
##    ampm = None
##    for w in _AM_WORDS:
##        if re.search(r"\b" + re.escape(w) + r"\b", s0):
##            ampm = "am"; break
##    if ampm is None:
##        for w in _PM_WORDS:
##            if re.search(r"\b" + re.escape(w) + r"\b", s0):
##                ampm = "pm"; break
##    if re.search(r"\bnoon\b", s0): ampm = "pm"
##    if re.search(r"\bmidnight\b", s0): ampm = "am"
##    if ampm:
##        pattern = r"\b(a\.?m\.?|p\.?m\.?|am|pm|morning|afternoon|evening|night|noon|midnight|this morning|this evening|tonight)\b"
##        s0 = re.sub(pattern, " ", s0)
##        s0 = re.sub(r"\s+", " ", s0).strip()
##    return s0, ampm
##
##def spoken_time_to_hm(spoken) -> Optional[Tuple[int,int]]:
##    if spoken is None: return None
##    s = safe_str(spoken).lower().replace("-", " ").replace(".", " ").replace(",", " ")
##    if re.search(r"\bnoon\b", s): return (12,0)
##    if re.search(r"\bmidnight\b", s): return (0,0)
##    s = re.sub(r"\b(o'clock)\b", "", s)
##    s = re.sub(r"\s+", " ", s).strip()
##    s_no_ampm, ampm = _detect_ampm_and_remove(s)
##    m = re.search(r"\bhalf past ([a-z0-9 ]+)$", s_no_ampm)
##    if m:
##        hour_token = m.group(1).strip(); h = _token_to_number(hour_token)
##        if h is None: return None
##        hour = int(h)%24; minute = 30
##        if ampm=="pm" and hour<12: hour+=12
##        if ampm=="am" and hour==12: hour=0
##        return (hour, minute)
##    m = re.search(r"\bquarter (past|to) ([a-z0-9 ]+)$", s_no_ampm)
##    if m:
##        typ = m.group(1); hour_token = m.group(2).strip(); h = _token_to_number(hour_token)
##        if h is None: return None
##        hour = int(h)%24
##        if typ=="past": minute=15
##        else: minute=45; hour=(hour-1)%24
##        if ampm=="pm" and hour<12: hour+=12
##        if ampm=="am" and hour==12: hour=0
##        return (hour, minute)
##    digits_cluster = re.search(r"\b(\d{3,4})\b", s_no_ampm)
##    if digits_cluster:
##        cluster = digits_cluster.group(1)
##        try:
##            if len(cluster)==3: h=int(cluster[0]); m=int(cluster[1:])
##            else: h=int(cluster[:2]); m=int(cluster[2:])
##            if 0<=h<24 and 0<=m<60:
##                if ampm=="pm" and h<12: h+=12
##                if ampm=="am" and h==12: h=0
##                return (h,m)
##        except: pass
##    tokens = re.findall(r"[a-z]+|\d+", s_no_ampm.lower())
##    num_list: List[int] = []
##    for t in tokens:
##        v = _token_to_number(t)
##        if v is not None: num_list.append(v)
##    if len(num_list) >= 2:
##        hour = int(num_list[0])%24; minute=int(num_list[1])%60
##        if ampm=="pm" and hour<12: hour+=12
##        if ampm=="am" and hour==12: hour=0
##        return (hour, minute)
##    if len(num_list) == 1:
##        hour=int(num_list[0])%24
##        if ampm=="pm" and hour<12: hour+=12
##        if ampm=="am" and hour==12: hour=0
##        return (hour, 0)
##    return None
##
### persistence
##SCHEDULE_DIR = os.path.join(os.path.expanduser("~"), ".alfred_scheduled_commands")
##os.makedirs(SCHEDULE_DIR, exist_ok=True)
##SCHEDULE_DB = os.path.join(SCHEDULE_DIR, "commands.json")
##scheduled_events: List[dict] = []
##
##def _load_scheduled_events():
##    global scheduled_events
##    try:
##        if os.path.exists(SCHEDULE_DB):
##            with open(SCHEDULE_DB, "r", encoding="utf-8") as f:
##                scheduled_events = json.load(f)
##        else:
##            scheduled_events = []
##    except Exception as e:
##        print("Scheduled load failed:", e); scheduled_events = []
##
##def _save_scheduled_events():
##    try:
##        with open(SCHEDULE_DB, "w", encoding="utf-8") as f:
##            json.dump(scheduled_events, f, indent=2, default=str)
##    except Exception as e:
##        print("Scheduled save failed:", e)
##
### schedule management
##def add_scheduled_command(command_text: str, dtstart: dt.datetime, username: Optional[str] = None, description: str = "") -> dict:
##    try:
##        ev = {
##            "id": uuid.uuid4().hex,
##            "command": command_text,
##            "username": username or "Itf",
##            "dtstart": dtstart.replace(second=0, microsecond=0).isoformat(),
##            "description": description,
##            "fired": False
##        }
##        scheduled_events.append(ev)
##        _save_scheduled_events()
##        return ev
##    except Exception as e:
##        print("add_scheduled_command failed:", e)
##        raise
##
### parsing helpers
##def _parse_command_sequence_parts(text: str) -> List[Dict[str, Any]]:
##    parts = re.split(r'\band then\b|\bafter that\b|\bthen\b', text, flags=re.I)
##    out = []
##    for p in parts:
##        p_clean = p.strip()
##        if not p_clean:
##            continue
##        hm = spoken_time_to_hm(p_clean)
##        rel_dt = None
##        m_rel = re.search(r"\b(in|after)\s+([a-z0-9\s-]+)\s+(seconds?|minutes?|hours?|days?)\b", p_clean, flags=re.I)
##        if m_rel:
##            num_phrase = m_rel.group(2).strip()
##            unit = m_rel.group(3).lower()
##            try:
##                num = int(num_phrase)
##            except:
##                num = words_to_number(num_phrase)
##            if num is not None:
##                now = dt.datetime.now()
##                if unit.startswith("hour"): rel_dt = now + dt.timedelta(hours=num)
##                elif unit.startswith("minute"):
##                    rel_dt = now + dt.timedelta(minutes=num)
##                elif unit.startswith("second"):
##                    rel_dt = now + dt.timedelta(seconds=num)
##                elif unit.startswith("day"):
##                    rel_dt = now + dt.timedelta(days=num)
##        out.append({"part": p_clean, "hm": hm, "rel_dt": rel_dt})
##    return out
##
##
##def _strip_time_tokens_from_part(part: str) -> str:
##    """
##    Clean scheduling tokens from a phrase or a log line.
##
##    Behaviour:
##      - If input looks like a log line with multiple " : " segments, pick the
##        most likely command segment (the longest segment that contains letters).
##      - Remove explicit date-like "on <date>" patterns.
##      - Remove "in/after <...> (seconds|minutes|hours|days)" cleanly (so "on in 5 minutes"
##        becomes just "on").
##      - If an 'at' exists, cut everything from 'at' onward.
##      - Count 'on' words:
##         * 0 -> return cleaned text (no further split)
##         * 1 -> return up to and INCLUDING that single 'on'
##         * 2+ -> return up to and INCLUDING the FIRST 'on'
##      - Normalize whitespace and strip trailing punctuation.
##    """
##    if not part:
##        return part
##
##    orig = part.strip()
##
##    # If given a full log line with " : " separators, pick the most likely command
##    # segment (the longest segment that contains letters).
##    segments = [seg.strip() for seg in re.split(r'\s*:\s*', orig) if seg.strip() != ""]
##    command_candidate = None
##    if len(segments) > 1:
##        alpha_segments = [seg for seg in segments if re.search(r'[A-Za-z]', seg)]
##        if alpha_segments:
##            # choose the longest alpha-containing segment (most likely the command)
##            command_candidate = max(alpha_segments, key=lambda s: len(s))
##    if command_candidate:
##        s = command_candidate
##    else:
##        s = orig
##
##    # Normalize whitespace early
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # 1) remove explicit "on <date>" (kept conservative to avoid overmatching)
##    s = re.sub(
##        r'\bat\s+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b',
##        '',
##        s,
##        flags=re.I
##    )
##
##    # 2) remove "in/after <...> (seconds|minutes|hours|days)" *completely*
##    #    Use non-greedy capture to avoid swallowing unrelated text.
##    s = re.sub(
##        r'\b(?:in|after)\s+[0-9a-z\s\-]+?\s+(?:seconds?|minutes?|hours?|days?)\b',
##        '',
##        s,
##        flags=re.I
##    )
##
##    # collapse whitespace again after removals
##    s = re.sub(r'\s+', ' ', s).strip()
##
##    # 3) If there's an 'at', cut EVERYTHING from 'at' onward (user rule)
##    at_m = re.search(r'\bat\b', s, flags=re.I)
##    if at_m:
##        s = s[:at_m.start()].rstrip()
##        s = re.sub(r'\s+', ' ', s).strip(' ,.')
##        return s
##
##    # 4) Handle 'on' occurrences according to your corrected rule:
##    #    - if 0 'on' -> leave as is
##    #    - if 1 'on'  -> keep up to and including that 'on'
##    #    - if >=2 'on' -> keep up to and including the FIRST 'on'
##    on_matches = list(re.finditer(r'\bon\b', s, flags=re.I))
##    if on_matches:
##        if len(on_matches) >= 2:
##            first = on_matches[0]
##            s = s[: first.end()].rstrip()
##            s = re.sub(r'\s+', ' ', s).strip(' ,.')
##            return s
##        else:
##            m = on_matches[0]
##            s = s[: m.end()].rstrip()
##            s = re.sub(r'\s+', ' ', s).strip(' ,.')
##            return s
##
##    # 5) final cleanup (nothing matched above)
##    s = re.sub(r'\s+', ' ', s).strip(' ,.')
##    return s
##
### executor thread
##_SCHEDULER_THREAD = None
##_SCHEDULER_LOCK = threading.Lock()
##
##def _execute_event(ev: dict):
##    try:
##        cmd = ev.get("command", "")
##        user = ev.get("username", "Itf")
##        if listen is not None and hasattr(listen, "listen"):
####            text_msg = {'username': user, 'query': cmd}
####            text_msg = (f"username:{user} : query:{cmd}")
##            text_msg = cmd
##            try:
##                listen.add_text(text_msg)
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak(f"Executing timed command: {cmd}")
##                else:
##                    print("[scheduled_commands] Executing:", cmd)
##            except Exception as e:
##                print("Error calling listen.listen:", e)
##                if speech is not None and hasattr(speech, "AlfredSpeak"):
##                    speech.AlfredSpeak("Failed to execute timed command.")
##        else:
##            # fallback: announce but do not execute
##            if speech is not None and hasattr(speech, "AlfredSpeak"):
##                speech.AlfredSpeak(f"(No listen available) Would execute: {cmd}")
##            else:
##                print("(No listen available) Would execute:", cmd)
##    except Exception as e:
##        print("_execute_event error:", e)
##
##def _scheduler_loop(poll_seconds: int = 15):
##    while True:
##        try:
##            now = dt.datetime.now()
##            changed = False
##            for ev in scheduled_events:
##                try:
##                    if ev.get("fired", False):
##                        continue
##                    dtstart = dt.datetime.fromisoformat(ev["dtstart"])
##                    if now >= dtstart:
##                        # mark fired and execute
##                        ev["fired"] = True
##                        changed = True
##                        _execute_event(ev)
##                except Exception as e:
##                    print("scheduled event inspect error:", e)
##                    continue
##            if changed:
##                _save_scheduled_events()
##        except Exception as e:
##            print("Scheduler loop error:", e)
##        time.sleep(poll_seconds)
##
##def start_scheduler_thread(poll_seconds: int = 15):
##    global _SCHEDULER_THREAD
##    with _SCHEDULER_LOCK:
##        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
##            return
##        _SCHEDULER_THREAD = threading.Thread(target=_scheduler_loop, kwargs={"poll_seconds": poll_seconds}, daemon=True)
##        _SCHEDULER_THREAD.start()
##
### public API: handle a user utterance and schedule commands if found
##def handle_command_text(text: str, gui=None) -> Optional[List[dict]]:
##    """
##    Parse `text` for time-based commands. If found, schedule them and return a list
##    of created event dicts. If no scheduling performed, return None.
##    """
##    message, speaker, score, gender, gender_conf, timestamp = extract_text_from_timed_command(text)
##
##    New_Message = message
##    New_speaker = speaker
##    New_score = score
##    New_gender = gender
##    New_gender_conf = gender_conf
##    New_timestamp = timestamp
##
##    print(f"New_Message Timed Command Module       : {New_Message}")
##    print(f"New_speaker Timed Command Module       : {New_speaker}")
##    print(f"New_score Timed Command Module         : {New_score}")
##    print(f"New_gender Timed Command Module        : {New_gender}")
##    print(f"New_gender_conf Timed Command Module   : {New_gender_conf}")
##    print(f"New_timestamp Timed Command Module     : {New_gender_conf}")
##    
##    text_clean = safe_str(text)
##    lower = text_clean.lower()
##
##    # time indicators regex: keep simple and conservative
##    time_indicators = bool(re.search(r"\b(at|o'clock|half past|quarter past|quarter to|in \d+ (minutes|hours|seconds)|tomorrow|today|noon|midnight|\d{1,2}:\d{2})\b", text_clean, flags=re.I))
##    # Do not treat explicit reminders as scheduled commands here:
##    reminder_like = any(k in lower for k in ("remind me","set a reminder","set reminder","create a reminder"))
##    if not time_indicators or reminder_like:
##        return None
##
##    parts = _parse_command_sequence_parts(text_clean)
##    if not parts:
##        return None
##
##    scheduled = []
##    prev_dt = None
##    for p in parts:
##        part_text = p.get("part", "")
##        cmd_text = _strip_time_tokens_from_part(part_text) or part_text
##        dtstart = None
##        if p.get("rel_dt"):
##            dtstart = p["rel_dt"]
##        elif p.get("hm"):
##            h, m = p["hm"]
##            cand = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
##            if cand < dt.datetime.now():
##                cand = cand + dt.timedelta(days=1)
##            dtstart = cand
##        else:
##            # fallback: schedule 1 minute after previous or 1 minute from now
##            if prev_dt:
##                dtstart = prev_dt + dt.timedelta(minutes=1)
##            else:
##                dtstart = dt.datetime.now() + dt.timedelta(minutes=1)
##        prev_dt = dtstart
##        try:
##            ev = add_scheduled_command(cmd_text, dtstart, username=(None if gui is None else getattr(gui, "current_user", "Itf")), description="scheduled command")
##            scheduled.append(ev)
##        except Exception as e:
##            print("Failed to schedule:", e)
##            continue
##
##    if scheduled:
##        if speech is not None and hasattr(speech, "AlfredSpeak"):
##            speech.AlfredSpeak(f"Scheduled {len(scheduled)} command(s).")
##        else:
##            print(f"Scheduled {len(scheduled)} command(s).")
##        if gui is not None and hasattr(gui, "log_query"):
##            gui.log_query(f"Scheduled commands: {[ev.get('command') for ev in scheduled]}")
##        return scheduled
##    return None
##
### initialization
##_load_scheduled_events()
##start_scheduler_thread()
