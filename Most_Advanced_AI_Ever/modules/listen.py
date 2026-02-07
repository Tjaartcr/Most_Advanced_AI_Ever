# listen.py (fixed + wake-word improvements)
# Minimal, necessary fixes to make the module load & the non-blocking whisper worker work,
# added energy-spike check (already present) and improved wake-word command capture:
# - Waits for wake words to start command capture.
# - Stops recording when stop phrases are detected ("ok", "thank you", "that's all", "thanks")
#   allowing up to 2 extra words after the stop phrase, or after 5 seconds of silence.
# Changes are intentionally small and local; all other logic preserved.

import re
import os
import sys
import time
import json
import queue
import collections
import string
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Optional, Callable
from multiprocessing import Pipe, Process

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
import vosk
import whisper
import serial

import concurrent.futures
from collections import deque

from speech import speech
from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH

# optional global for UI usage
try:
    from af_to_en import translate
except Exception:
    translate = lambda x: x

# ------------------ Speaker-ID module (module import + robust wrapper) ------------------
speaker_name = ""
home_name = "Home"
combo_name = ""

# import module object directly (avoid local shadowed names)
try:
    import Speech_Identification_User as spkmod
except Exception as e:
    # If import fails, create a tiny shim so listen.py doesn't crash.
    print(f"[DEBUG LISTEN][speech_id] ERROR importing Speech_Identification_User: {e}")
    class _Shim:
        pass
    spkmod = _Shim()
    spkmod.DB_FILE = Path(__file__).resolve().parent / "voice_db.json"
    spkmod.load_db = lambda: {}
    spkmod.identify_from_file = lambda p: (None, None)
    spkmod.identify_user = lambda e: (None, None)
    spkmod.get_embedding = lambda p: None

# persistent log so pythonw services show diagnostics
_LOG_PATH = Path(__file__).resolve().parent / "speech_id_runtime.log"
def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        try:
            print(f"[DEBUG LISTEN][{ts}] {msg}")
        except Exception:
            pass


def identify_safely_from_file(
    tmp_wav_path: str,
    precomputed_transcript: str = None,
    precomputed_mfcc=None
):
    """
    Robust identification wrapper.
    Returns (name_or_None, score_or_None, gender_or_None, gender_confidence_or_None).
    Can reuse precomputed transcript or MFCC features if provided.
    """
    _log(f"--- identify called for: {tmp_wav_path} ---")
    try:
        _log(f"spkmod.__file__ = {getattr(spkmod, '__file__', 'UNKNOWN')}")
        _log(f"spkmod.DB_FILE = {getattr(spkmod, 'DB_FILE', 'UNKNOWN')}")
        _log(f"python executable = {sys.executable}")
        _log(f"cwd = {Path.cwd()}")
    except Exception as e:
        _log(f"minor logging error: {e}")

    # 1) Load DB if possible
    try:
        if hasattr(spkmod, "load_db") and callable(spkmod.load_db):
            db = spkmod.load_db() or {}
            _log(f"spkmod.load_db() keys = {list(db.keys())}")
        else:
            _log("spkmod.load_db() not available")
            db = {}
    except Exception as e:
        _log(f"spkmod.load_db() raised: {e}\n{traceback.format_exc()}")
        db = {}

    # 2) Fallback DB candidates
    if not db:
        candidates = []
        try:
            dbfile = getattr(spkmod, "DB_FILE", None)
            if dbfile:
                candidates.append(Path(dbfile))
        except Exception:
            pass
        if getattr(spkmod, "__file__", None):
            candidates.append(Path(spkmod.__file__).resolve().parent / "voice_db.json")
        candidates.append(Path(__file__).resolve().parent / "voice_db.json")
        candidates.append(Path.cwd() / "voice_db.json")

        for c in candidates:
            try:
                if not c or not c.exists():
                    _log(f"candidate not found: {c}")
                    continue
                txt = c.read_text(encoding="utf-8")
                j = json.loads(txt) if txt else {}
                _log(f"loaded candidate DB {c} keys={list(j.keys())}")
                if j:
                    db = j
                    break
            except Exception as e:
                _log(f"failed reading candidate {c}: {e}")

    name, score = None, None
    # 3) Try primary identify
    try:
        if hasattr(spkmod, "identify_from_file") and callable(spkmod.identify_from_file):
            name, score = spkmod.identify_from_file(tmp_wav_path)
            _log(f"identify_from_file -> {name}, {score}")
    except Exception as e:
        _log(f"identify_from_file failed: {e}\n{traceback.format_exc()}")

    # 4) Fallback embedding
    if name is None:
        try:
            if hasattr(spkmod, "get_embedding") and callable(spkmod.get_embedding) and \
               hasattr(spkmod, "identify_user") and callable(spkmod.identify_user):
                emb = spkmod.get_embedding(tmp_wav_path)
                if emb is not None:
                    name, score = spkmod.identify_user(emb)
                    _log(f"fallback identify_user -> {name}, {score}")
                    print(f"fallback identify_user -> {name}, {score}")
                else:
                    _log("get_embedding returned None")
        except Exception as e:
            _log(f"fallback get_embedding/identify_user error: {e}\n{traceback.format_exc()}")

    # ------------------------
    # Run gender classification
    # ------------------------
    try:
        from whisper_gender import extract_mfcc, scaler, clf

        # If caller gave us precomputed MFCC, reuse it
        if precomputed_mfcc is not None:
            mfcc = precomputed_mfcc
        else:
            mfcc = extract_mfcc(tmp_wav_path)

        mfcc = mfcc.reshape(1, -1)
        mfcc_scaled = scaler.transform(mfcc)
        gender_label = clf.predict(mfcc_scaled)[0]
        gender_prob = clf.predict_proba(mfcc_scaled)[0]

        if gender_label == 0:
            gender_str = "Male"

        if gender_label == 1:
            gender_str = "Female"

        if gender_label >= 2:
            gender_str = "Monkey"

##
##
####        gender_str = "Male" if gender_label == 0 else "Female"
##        gender_str = ("Male" if gender_label == 0 else
##                    "Female" if gender_label == 1) else "unknown"
##            
        
        gender_conf = float(gender_prob[gender_label])
        _log(f"gender -> {gender_str} ({gender_conf:.2f})")
    except Exception as e:
        _log(f"gender classification failed: {e}")
        gender_str, gender_conf = None, None

    return name, score, gender_str, gender_conf


# -----------------------
# Keyword lists
# -----------------------
CODING_KEYWORDS = [
    "check", "fix", "python", "arduino", "c++", "a python", "an arduino", "a c++",
    "java", "python code", "arduino code", "c++ code", "javascript", "typescript",
    "go", "rust", "c#", "swift", "kotlin", "ruby", "php", "perl", "matlab", "scala",
    "haskell", "fortran", "lua", "dart", "shell", "bash", "code", "a code", "code for",
    "powershell"
]

RAG_KEYWORDS = [
    "rag", "this document", "this text", "this pdf", "this word", "this excel",
    "this story", "this list", "this rag document", "this rag text", "this rag pdf",
    "this rag word", "this rag excel", "this rag story", "this rag list"
]


# -----------------------
# Helpers
# -----------------------
def _needs_wrap(msg: str) -> bool:
    if not msg:
        return False
    lower = msg.lower()
    for kw in CODING_KEYWORDS + RAG_KEYWORDS:
        if kw and kw in lower:
            return True
    return False


def _already_wrapped(msg: str) -> bool:
    if not msg:
        return False
    s = msg.strip()
    if (s.startswith("'''") and s.endswith("'''")) or (s.startswith('"""') and s.endswith('"""')):
        return True
    if s.startswith("```") and s.endswith("```"):
        return True
    return False


def _wrap_message_only(msg: str) -> str:
    """Place triple-single quotes on their own lines around the message."""
    if not msg:
        return msg
    if _already_wrapped(msg):
        return msg
    return "'''\n" + msg.rstrip() + "\n'''"


def _normalize_iso_timestamp(ts: str):
    """Normalize common ISO timestamps to 'YYYY-MM-DD HH:MM:SS' (best-effort)."""
    if not ts:
        return None
    s = str(ts).strip()
    # Try simple ISO with 'T'
    if "T" in s:
        # remove timezone Z or offset, and fractional seconds
        try:
            # chop off timezone info
            core = re.split(r"[Z+-]", s, maxsplit=1)[0]
            # remove fractional seconds if present
            core = re.sub(r"\.\d+", "", core)
            from datetime import datetime
            dt = datetime.fromisoformat(core)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # fallback: replace T with space and drop fractions/zone
            core = s
            core = re.sub(r"\.\d+", "", core)
            core = core.replace("T", " ")
            core = re.sub(r"Z$", "", core)
            return core.strip()
    return s


def _format_final(message, username, score=None, gender=None, gender_conf=None):
    """Return final single-line pipeline string. Keep message wrapper intact."""
    # Normalize whitespace inside message unless wrapped: preserve internal newlines when wrapped.
    m = str(message)
    print(f"[DEBUG LISTEN _format_final]extracted listen message :  {m}")

    s = m.strip()
    if s.startswith("'''") and s.endswith("'''"):
        inner = s[3:-3]
        # collapse many blank lines but keep line breaks for code readability
        inner = re.sub(r"\n{3,}", "\n\n", inner)
        inner = inner.strip()
        message_out = "'''{}'''".format(inner)
    else:
        message_out = re.sub(r"\s*\n+\s*", " ", m)
        message_out = re.sub(r"\s{2,}", " ", message_out).strip()

    username_clean = str(username).strip() if username is not None else "ITF"
    return f"{message_out} : 'username':{username_clean} : 'score':{score} : 'gender':{gender} : 'gender_conf':{gender_conf}"


# -----------------------
# Extractor for incoming event / text
# -----------------------
def extract_message_username_timestamp_from_event(event, fallback_user="ITF"):
    """
    event: dict or str. If dict and contains 'payload', payload wins.
    Returns: (final_string, message, username, timestamp) where final_string is the pipeline-ready string.
    """
    payload = None
    outer_username = None
    outer_timestamp = None

    if isinstance(event, dict):
        outer_username = event.get("username") or event.get("user")
        outer_timestamp = event.get("timestamp") or event.get("time")
        payload = event.get("payload") or event
    else:
        payload = event

    message_raw = ""
    username = None
    timestamp = None
    score = None
    gender = None
    gender_conf = None

    if isinstance(payload, dict):
        message_raw = (payload.get("query") or payload.get("description") or payload.get("text")
                       or payload.get("message") or payload.get("q") or "")
        username = payload.get("username") or payload.get("user") or outer_username
        if not username:
            try:
                cu = getattr(event, "current_user", None)
                if cu is not None:
                    username = str(cu)
            except Exception:
                username = None
        score = payload.get("score")
        gender = payload.get("gender")
        gender_conf = payload.get("gender_conf")
        timestamp = payload.get("timestamp") or payload.get("time") or payload.get("date")
    else:
        if isinstance(payload, str):
            message_raw = payload
        else:
            message_raw = str(payload)

    if not message_raw and isinstance(event, dict):
        message_raw = event.get("query") or event.get("description") or event.get("text") or event.get("message") or message_raw

    message_raw = str(message_raw).strip()

    if not timestamp:
        timestamp = outer_timestamp
    timestamp = _normalize_iso_timestamp(timestamp) if timestamp else None

    if not username:
        username = fallback_user

    m_user_embedded = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", message_raw)
    if m_user_embedded:
        extracted_u = m_user_embedded.group("u").strip()
        if extracted_u and (not username or username == fallback_user):
            username = extracted_u

    m_user_simple = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", message_raw, flags=re.IGNORECASE)
    if m_user_simple:
        extracted_u = m_user_simple.group("u").strip()
        if extracted_u and (not username or username == fallback_user):
            username = extracted_u

    message_to_queue = message_raw
    if _needs_wrap(message_raw) and not _already_wrapped(message_raw):
        message_to_queue = _wrap_message_only(message_raw)

    print(f"[DEBUG LISTEN]extracted listen message_to_queue :  {message_to_queue}")

    final = _format_final(message_to_queue, username or fallback_user, score, gender, gender_conf)

    print(f"[DEBUG LISTEN]extracted listen message_raw :  {message_raw}")
    print(f"[DEBUG LISTEN]extracted listen username :  {username}")
    print(f"[DEBUG LISTEN]extracted listen timestamp :  {timestamp}")
    print(f"[DEBUG LISTEN]extracted listen score :  {score}")
    print(f"[DEBUG LISTEN]extracted listen gender :  {gender}")
    print(f"[DEBUG LISTEN]extracted listen gender_conf :  {gender_conf}")
    print(f"[DEBUG LISTEN]extracted listen final :  {final}")

    return message_raw, message_to_queue, username or fallback_user, timestamp


def convert_to_superscript(expression):
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    def replace_power(match):
        base = match.group(1)
        power = ''.join(superscript_map.get(d, d) for d in match.group(2))
        return base + power
    return re.sub(r'([a-zA-Z])(\d+)', replace_power, expression)


def prepare_text_for_tts(text):
    superscript_to_words = {
        '⁰': ' to the power of 0', '¹': ' to the power of 1', '²': ' squared',
        '³': ' cubed', '⁴': ' to the power of 4', '⁵': ' to the power of 5',
        '⁶': ' to the power of 6', '⁷': ' to the power of 7',
        '⁸': ' to the power of 8', '⁹': ' to the power of 9'
    }
    for sup, spoken in superscript_to_words.items():
        text = text.replace(sup, spoken)
    text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 minus \2', text)
    text = re.sub(r'(?<![\w])-(\d+)', r'negative \1', text)
    text = re.sub(r'(?<![\w])-([a-zA-Z])', r'negative \1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.replace(" negative ", " -").replace(" positive ", " +").strip()


def is_math_expression(text):
    return bool(re.search(
        r'\d+[a-zA-Z]?\d*|[+\-*/^=]|\b(plus|minus|times|square|multiply|divide)\b',
        text
    ))


# ---------------- ListenModule ----------------
class ListenModule:
    def __init__(self):
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"❌ Vosk model not found! Expected path: {VOSK_MODEL_PATH}")
        print(f"[DEBUG LISTEN]✅ Loading Vosk model from {VOSK_MODEL_PATH}...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)

        print("[DEBUG LISTEN]✅ Loading Whisper model (this may take a moment)...")
        # Uncomment the exact model variant you want (base vs base.en) if available
        # self.whisper_model = whisper.load_model("base.en")
        self.whisper_model = whisper.load_model("base")

        self.samplerate = 16000
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        self.bluetooth = None
        self.timeout_set = 3.05
        try:
            self.bluetooth = serial.Serial(
                SERIAL_PORT_BLUETOOTH,
                BAUDRATE_BLUETOOTH,
                timeout=self.timeout_set
            )
            print(f"[DEBUG LISTEN]✅ Bluetooth is now Connected to PORT {SERIAL_PORT_BLUETOOTH}")
        except serial.SerialException:
            print("[DEBUG LISTEN]❌ OH NO !!! Bluetooth connection failed.")

        # Wake-word set
        self.wake_words = {"hi", "alfred", "hey", "hello", "yo", "hallo"}

        self.use_whisper_listen = False
        self.use_vosk_listen = False
        self.mobile_speech_enabled = False
        self.recording_hold = False
        self.stop_listening = False
        self.wake_word_on_off = False

        self.parent_conn = None
        self.listen_process = None

        # Transcription executor/queue (lazy-created)
        self._transcribe_executor = None
        self._transcribe_result_queue = None

        # optional current user used by listen_text/add_text
        self.current_user = None

    def set_recording_hold(self, enabled: bool):
        self.recording_hold = enabled
        time.sleep(0.5)

    def set_stop_listen_on(self, enabled: bool = True):
        self.stop_listening = True
        print("[DEBUG LISTEN]🔔 Listening Has Stopped")

    def set_stop_listen_off(self, enabled: bool = False):
        self.stop_listening = False
        print("[DEBUG LISTEN]🔔 Listening Resumed")

    def set_wake_word_on(self, enabled: bool = True):
        self.wake_word_on_off = True
        print("[DEBUG LISTEN]🔔 Listen Wake Word On")

    def set_wake_word_off(self, enabled: bool = False):
        self.wake_word_on_off = False
        print("[DEBUG LISTEN]🔔 Listen Wake Word Off \n 🔔 Listening Resumed")

    def add_text(self, text):
        """
        Accepts dict or string, extracts message and username and enqueues a pipeline-ready single-line string.
        """
        fallback_user = getattr(self, "current_user", "ITF") or "ITF"

        if not text:
            return

        if isinstance(text, dict):
            final, message_wrapped, username, timestamp = extract_message_username_timestamp_from_event(text, fallback_user)
            self.text_queue.put(final)
            print(f"[DEBUG LISTEN]Text added to queue: {final}")
            return

        if isinstance(text, str):
            final, message_wrapped, username, timestamp = extract_message_username_timestamp_from_event(text, fallback_user)
            self.text_queue.put(final)
            print(f"[DEBUG LISTEN]Text added to queue: {final}")
            return

        s = str(text)
        final, message_wrapped, username, timestamp = extract_message_username_timestamp_from_event(s, fallback_user)
        self.text_queue.put(final)
        print(f"[DEBUG LISTEN]Text added to queue: {final}")

    def listen_text(self, text):
        """
        Normalize incoming text (dict or raw string) into a single string:
        "<message> : 'username':<username> : 'score':<score> : 'gender':<gender> : 'gender_conf':<gender_conf>"
        """
        CODING_KEYWORDS_LOCAL = CODING_KEYWORDS
        RAG_KEYWORDS_LOCAL = RAG_KEYWORDS
        Default_Name = "ITF"

        def _needs_wrap_local(msg: str) -> bool:
            if not msg:
                return False
            lower = msg.lower()
            for kw in CODING_KEYWORDS_LOCAL + RAG_KEYWORDS_LOCAL:
                if kw and kw in lower:
                    return True
            return False

        def _already_wrapped_local(msg: str) -> bool:
            if not msg:
                return False
            s = msg.strip()
            return (s.startswith("'''") and s.endswith("'''")) or (s.startswith('"""') and s.endswith('"""'))

        def _normalize_none(x):
            if x is None:
                return None
            xs = str(x).strip()
            if xs.lower() in ("none", "null", ""):
                return None
            return xs

        def _format(message_part, username_val, score_val=None, gender_val=None, gender_conf_val=None):
            m_str = str(message_part)
            stripped = m_str.strip()
            if stripped.startswith("'''") and stripped.endswith("'''"):
                inner = stripped[3:-3]
                inner = re.sub(r"\s*\n+\s*", " ", inner)
                inner = re.sub(r"\s{2,}", " ", inner).strip()
                m_out = "'''{}'''".format(inner)
            else:
                m_out = re.sub(r"\s*\n+\s*", " ", m_str)
                m_out = re.sub(r"\s{2,}", " ", m_out).strip()
            username_clean = str(username_val).strip() if username_val is not None else Default_Name
            return f"{m_out} : 'username':{username_clean} : 'score':{score_val} : 'gender':{gender_val} : 'gender_conf':{gender_conf_val}"

        if not text:
            print("[DEBUG LISTEN]No text received.")
            return None

        message_raw = ""
        username = None
        score = None
        gender = None
        gender_conf = None

        if isinstance(text, dict):
            payload = text.get("payload") or text
            message_raw = payload.get("query") or payload.get("message") or payload.get("text") or ""
            username = payload.get("username") or payload.get("user") or text.get("username") or text.get("user") or getattr(self, "current_user", None) or Default_Name
            score = _normalize_none(payload.get("score"))
            gender = _normalize_none(payload.get("gender"))
            gender_conf = _normalize_none(payload.get("gender_conf"))
            message_raw = str(message_raw).strip()
            message_part = message_raw
            if not _already_wrapped_local(message_part) and _needs_wrap_local(message_part):
                message_part = f"'''\n{message_part}\n'''"
            print("[DEBUG LISTEN]Received message (dict): 'message':%s : 'username':%s : 'score':%s : 'gender':%s : 'gender_conf':%s" % (message_part, username, score, gender, gender_conf))
            return _format(message_part, username, score, gender, gender_conf)

        s = str(text).strip()
        prefix = "text received to listen is"
        if s.lower().startswith(prefix):
            try:
                s_after = s[len(prefix):].strip()
                if s_after:
                    s = s_after
            except Exception:
                pass

        first_triple = s.find("'''")
        if first_triple != -1:
            second_triple = s.find("'''", first_triple + 3)
            if second_triple != -1:
                inner = s[first_triple + 3:second_triple]
                remainder = s[second_triple + 3 :].strip()
                if inner.lstrip().startswith("'message':") or inner.lstrip().startswith('"message":'):
                    inner = re.sub(r"^\s*(?:'message'\s*:\s*|\"message\"\s*:\s*)", "", inner).strip()
                message_raw = inner.strip()
                username = getattr(self, "current_user", None) or Default_Name
                m_user = re.search(r"'username'\s*:\s*['\"]?(?P<u>[^'\"\s:]+)['\"]?", remainder)
                if m_user:
                    username = m_user.group("u").strip()
                else:
                    m_user2 = re.search(r"(?:\busername\b|\buser\b)\s*[:=]\s*['\"]?(?P<u>[^'\"\n]+)['\"]?\s*$", remainder, flags=re.IGNORECASE)
                    if m_user2:
                        username = m_user2.group("u").strip()
                    else:
                        m_user3 = re.search(r":\s*'username'\s*:\s*['\"]?(?P<u>[^'\"\s:]+)['\"]?", s)
                        if m_user3:
                            username = m_user3.group("u").strip()
                score_m = re.search(r"(?:'score'|\"score\"|score)\s*:\s*([^:\n]+)", remainder)
                score = _normalize_none(score_m.group(1)) if score_m else None
                gender_m = re.search(r"(?:'gender'|\"gender\"|gender)\s*:\s*([^:\n]+)", remainder)
                gender = _normalize_none(gender_m.group(1)) if gender_m else None
                gconf_m = re.search(r"(?:'gender_conf'|\"gender_conf\"|gender_conf)\s*:\s*([^:\n]+)", remainder)
                gender_conf = _normalize_none(gconf_m.group(1)) if gconf_m else None
                message_part = message_raw
                if not _already_wrapped_local(message_part) and _needs_wrap_local(message_part):
                    message_part = f"'''\n{message_part}\n'''"
                    print("[DEBUG LISTEN][LISTEN] Auto-wrapped inner message in ''' because coding/RAG keyword detected (wrapper present).")
                print(f"[DEBUG LISTEN][PARSED LISTEN FINAL] 'message':{message_part} : 'username':{username} : 'score':{score} : 'gender':{gender} : 'gender_conf':{gender_conf}")
                return _format(message_part, username, score, gender, gender_conf)

        username = getattr(self, "current_user", None) or Default_Name

        strict_pattern = re.compile(r"(?si)"
                                    r"(?:(?:'message'|\"message\"|message)\s*:\s*)?(?P<msg>.*?)\s*:\s*"
                                    r"(?:'username'|\"username\"|username)\s*:\s*(?P<user>[^:\n]+)\s*:\s*"
                                    r"(?:'score'|\"score\"|score)\s*:\s*(?P<score>[^:\n]+)\s*:\s*"
                                    r"(?:'gender'|\"gender\"|gender)\s*:\s*(?P<gender>[^:\n]+)\s*:\s*"
                                    r"(?:'gender_conf'|\"gender_conf\"|gender_conf)\s*:\s*(?P<gconf>[^\n]+)\s*$",
                                    flags=re.DOTALL)
        m_strict = strict_pattern.search(s)
        if m_strict:
            message_raw = m_strict.group("msg").strip()
            username = m_strict.group("user").strip() or username
            score = _normalize_none(m_strict.group("score"))
            gender = _normalize_none(m_strict.group("gender"))
            gender_conf = _normalize_none(m_strict.group("gconf"))
        else:
            m_user_any = re.search(r"(?i)(?:'username'|\"username\"|username)\s*:\s*['\"]?(?P<u>[^'\"\n:]+)['\"]?", s)
            if m_user_any:
                username = m_user_any.group("u").strip() or username
                message_raw = s[: m_user_any.start()].strip(" :\n\t")
                after = s[m_user_any.end():].strip(" :\n\t ")
                score_m = re.search(r"(?i)(?:'score'|\"score\"|score)\s*:\s*([^:\n]+)", after)
                score = _normalize_none(score_m.group(1)) if score_m else None
                gender_m = re.search(r"(?i)(?:'gender'|\"gender\"|gender)\s*:\s*([^:\n]+)", after)
                gender = _normalize_none(gender_m.group(1)) if gender_m else None
                gconf_m = re.search(r"(?i)(?:'gender_conf'|\"gender_conf\"|gender_conf)\s*:\s*([^:\n]+)", after)
                gender_conf = _normalize_none(gconf_m.group(1)) if gconf_m else None
            else:
                message_raw = s.strip()
                username = username or Default_Name

        message_part = message_raw
        if not _already_wrapped_local(message_part) and _needs_wrap_local(message_part):
            message_part = f"'''\n{message_part}\n'''"
            print("[DEBUG LISTEN][LISTEN] Auto-wrapped string message in ''' because coding/RAG keyword detected.")

        print(f"[DEBUG LISTEN][PARSED LISTEN RAW] 'message':{message_part} : 'username':{username} : 'score':{score} : 'gender':{gender} : 'gender_conf':{gender_conf}")
        return _format(message_part, username, score, gender, gender_conf)


    def set_mobile_speech(self, enabled: bool):
        self.mobile_speech_enabled = enabled
        print(f"[DEBUG LISTEN]mobile_speech_enabled : {self.mobile_speech_enabled}")
        if enabled:
            speech.AlfredSpeak(
                "Remember to start the home automation application on your Android device and connect via Bluetooth."
            )

    def set_listen_vosk(self, enabled: bool):
        self.use_vosk_listen = enabled
        print("[DEBUG LISTEN]🎙️ Speech Recognizer set to: Vosk")

    def set_listen_whisper(self, enabled: bool):
        self.use_whisper_listen = enabled
        print("[DEBUG LISTEN]🎙️ Speech Recognizer set to: Whisper")

    def send_bluetooth(self, message: str):
        if self.mobile_speech_enabled and self.bluetooth:
            try:
                self.bluetooth.write(message.encode('utf-8'))
                print(f"[DEBUG LISTEN]📡 Sent to Device via Bluetooth: {message}")
            except serial.SerialException as e:
                print(f"[DEBUG LISTEN]❌ Bluetooth error: {e}")
        else:
            print("[DEBUG LISTEN]Bluetooth speech output not enabled or no device connected.")

    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"[DEBUG LISTEN]❌ Audio callback error: {status}")
        self.audio_queue.put(bytes(indata))

    # ---------------- VOSK-based listening ----------------
    def listen_vosk(self):
        recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)
        frames = []

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000,
                               dtype="int16", channels=1, callback=self.callback):
            while True:
                data = self.audio_queue.get()
                frames.append(data)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    print(f"[DEBUG LISTEN]📝 Vosk recognized: {text}")

                    # Check text length before stopping
                    if 10 <= len(text) <= 150:
                        speaker_name_local, speaker_score = None, None
                        gender_label, gender_conf = None, None
                        tmp_wav = None

                        try:
                            # Save audio so both speaker ID and gender can reuse it
                            pcm = b"".join(frames)
                            audio_int16 = np.frombuffer(pcm, dtype=np.int16)
                            audio_float = audio_int16.astype(np.float32) / 32768.0
                            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                            sf.write(tmp_wav.name, audio_float, self.samplerate)

                            # --- Speaker identification ---
                            speaker_name_local, speaker_score, gender_label, gender_conf = identify_safely_from_file(tmp_wav.name)

                            # --- Gender classification (single MFCC extraction) ---
                            from whisper_gender import extract_mfcc, scaler, clf
                            try:
                                mfcc = extract_mfcc(tmp_wav.name).reshape(1, -1)
                                mfcc_scaled = scaler.transform(mfcc)
                                gender_pred = clf.predict(mfcc_scaled)[0]
                                gender_prob = clf.predict_proba(mfcc_scaled)[0]
                                gender_label = "Male" if gender_pred == 0 else "Female"
                                gender_conf = gender_prob[gender_pred]
                            except Exception as e:
                                print(f"[DEBUG LISTEN]❌ Gender classification error (Vosk): {e}")
                                gender_label, gender_conf = None, None

                        except Exception as e:
                            print(f"[DEBUG LISTEN]❌ Speaker/Gender ID error (Vosk): {e}")
                            speaker_name_local, speaker_score = None, None
                            gender_label, gender_conf = None, None
                        finally:
                            frames.clear()

                        combo_name_local = f"{speaker_name_local} {home_name}" if speaker_name_local else None

                        # Return final structured result
                        return {
                            "text": text,
                            "speaker": combo_name_local,
                            "score": speaker_score,
                            "gender": gender_label,
                            "gender_conf": gender_conf,
                        }
                    else:
                        print(f"[DEBUG LISTEN]⚠️ Ignored result (length {len(text)})")
                        return {"text": "", "speaker": None, "score": None,
                                "gender": None, "gender_conf": None}

    # ---------------- VOSK: record button listen (interactive) ----------------
    def record_button_listen_vosk(self):
        recognizer = vosk.KaldiRecognizer(self.vosk_model, self.samplerate)
        frames = []
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.callback
        ):
            while True:
                data = self.audio_queue.get()
                frames.append(data)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    print(f"[DEBUG LISTEN]📝 Vosk recognized (button): {text}")

                    speaker_name_local, speaker_score = None, None
                    gender_label, gender_conf = None, None
                    tmp_wav = None

                    try:
                        pcm = b"".join(frames)
                        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32768.0
                        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        sf.write(tmp_wav.name, audio_float, self.samplerate)

                        speaker_name_local, speaker_score = identify_safely_from_file(tmp_wav.name)
                        if speaker_name_local:
                            print(f"[DEBUG LISTEN]🔊 Speaker identified: {speaker_name_local} (score={speaker_score})")
                        else:
                            print("[DEBUG LISTEN]⚠️ Speaker not recognized")

                        from whisper_gender import extract_mfcc, scaler, clf
                        try:
                            mfcc = extract_mfcc(tmp_wav.name).reshape(1, -1)
                            mfcc_scaled = scaler.transform(mfcc)
                            gender_pred = clf.predict(mfcc_scaled)[0]
                            gender_prob = clf.predict_proba(mfcc_scaled)[0]
                            gender_label = "Male" if gender_pred == 0 else "Female"
                            gender_conf = gender_prob[gender_pred]
                            print(f"[DEBUG LISTEN]⚥ Gender: {gender_label} ({gender_conf})")
                        except Exception as e:
                            print(f"[DEBUG LISTEN]❌ Gender classification error (Vosk button): {e}")
                            gender_label, gender_conf = None, None

                    except Exception as e:
                        print(f"[DEBUG LISTEN]❌ Speaker/Gender ID error (Vosk button): {e}")
                        speaker_name_local, speaker_score = None, None
                        gender_label, gender_conf = None, None
                    finally:
                        frames.clear()

                    clean_text = self.listen_text(text)
                    combo_name_local = f"{speaker_name_local} {home_name}" if speaker_name_local else None

                    return {
                        "text": clean_text,
                        "speaker": combo_name_local,
                        "score": speaker_score,
                        "gender": gender_label,
                        "gender_conf": gender_conf
                    }

    # --- Helper: ensure executor created lazily on the instance ---
    def _ensure_transcribe_executor(self, max_workers: int = 1):
        if not hasattr(self, "_transcribe_executor") or self._transcribe_executor is None:
            self._transcribe_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        if not hasattr(self, "_transcribe_result_queue") or self._transcribe_result_queue is None:
            self._transcribe_result_queue = queue.Queue()

    # --- Helper: process raw transcription same as your main pipeline ---
    def _process_transcription_text(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        text_no_dup_sentences = self.remove_duplicate_sentences(raw_text)
        cleaned_text = self.remove_extreme_repetition(text_no_dup_sentences)
        chunk_output = re.sub(r'[!?;:]', '', cleaned_text.lower())
        chunk_output = chunk_output.replace(
            "i'm going to go ahead and get a little bit of a little bit of a little bit", ""
        )
        chunk_output = chunk_output.replace("i'm going to go ahead and get a little bit of a", "")
        cleaned = re.sub(r'\s+', ' ', chunk_output).strip()

        # Remove immediate repeated words
        words = cleaned.split()
        unique_words = []
        for w in words:
            if not unique_words or w != unique_words[-1]:
                unique_words.append(w)
        final_result = " ".join(unique_words).strip()

        # Enforce length same as before
        if len(final_result) > 80:
            final_result = final_result[:80]

        return final_result if final_result else None

    # --- Background transcription worker ---
    def _transcribe_worker(self, pcm_float_array: np.ndarray, total_pcm_bytes: Optional[bytearray] = None):
        """
        Runs in background thread. Returns the same dict structure as old listen_whisper.
        """
        result_dict = None
        tmp_wav = None
        try:
            # write temp wav (float32 array expected)
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(tmp_wav.name, pcm_float_array, self.samplerate)

            # Transcribe with Whisper (be cautious about model thread-safety)
            result = self.whisper_model.transcribe(tmp_wav.name, language="en", fp16=False)
            raw_text = result.get("text", "").strip()
            if not raw_text:
                return None

            final_result = self._process_transcription_text(raw_text)
            if not final_result or len(final_result.split()) < 2:
                return None

            # Speaker/gender processing (if you captured PCM bytes)
            speaker_name_local, speaker_score = None, None
            gender_str, gender_conf = None, None
            try:
                if total_pcm_bytes:
                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    combined_int16 = np.frombuffer(bytes(total_pcm_bytes), dtype=np.int16)
                    combined_float = combined_int16.astype(np.float32) / 32768.0
                    sf.write(tmp_wav2.name, combined_float, self.samplerate)

                    from whisper_gender import extract_mfcc, scaler, clf
                    precomputed_mfcc = extract_mfcc(tmp_wav2.name)

                    speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
                        tmp_wav2.name,
                        precomputed_transcript=final_result,
                        precomputed_mfcc=precomputed_mfcc
                    )

                    try:
                        os.unlink(tmp_wav2.name)
                    except Exception:
                        pass
            except Exception as e:
                # keep going even if speaker/gender fails
                print("[DEBUG LISTEN]❌ Speaker ID / Gender (background) error:", e)
                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None

            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None

            result_dict = {
                "text": final_result,
                "speaker": combo_name_local,
                "score": speaker_score,
                "gender": gender_str,
                "gender_conf": gender_conf
            }

        except Exception as e:
            print("[DEBUG LISTEN]❌ Background transcription error:", e)
            result_dict = None
        finally:
            try:
                if tmp_wav:
                    os.unlink(tmp_wav.name)
            except Exception:
                pass

        # push result onto instance queue for consumers if desired
        try:
            if not hasattr(self, "_transcribe_result_queue") or self._transcribe_result_queue is None:
                self._transcribe_result_queue = queue.Queue()
            self._transcribe_result_queue.put(result_dict)
        except Exception:
            pass

        return result_dict

    # ---------------- VAD-based recording (updated with energy-spike detection) ----------------
    def record_until_silence(self,
                             frame_duration_ms=30,
                             vad_mode=3,
                             silence_duration_ms=1500,
                             max_record_duration_s=20,
                             vad_attenuation=0.3,
                             min_vad_rms=250,
                             debug=False,
                             energy_spike_factor=1.8,
                             ambient_window_ms=500):
        """
        Returns: (audio_float32_array, stopped_by_silence_bool)

        New behavior (small additive change):
          - maintain a short ambient energy buffer (ambient_window_ms)
          - require an energy spike: current_frame_rms > ambient_mean * energy_spike_factor
            in addition to VAD and min_vad_rms for start-of-speech detection.
          - This reduces false triggers from steady background voices/noise.
        """
        if self.stop_listening:
            return np.array([], dtype=np.float32), False

        vad = webrtcvad.Vad(vad_mode)
        frame_size = int(self.samplerate * frame_duration_ms / 1000)
        bytes_per_frame = frame_size * 2  # int16 => 2 bytes/sample
        voiced_frames = []
        got_speech = False
        start_time = time.time()

        # how many consecutive silent frames needed to conclude silence
        required_silent_frames = max(1, int(silence_duration_ms / frame_duration_ms))

        consec_silent = 0
        stopped_by_silence = False

        # ambient buffer holds recent non-speech RMS values to estimate background energy
        ambient_buf_len = max(1, int(max(1, ambient_window_ms / max(1, frame_duration_ms))))
        ambient_buffer = deque(maxlen=ambient_buf_len)

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=frame_size,
                               dtype='int16', channels=1) as stream:
            while True:
                # safety timeout
                if time.time() - start_time > max_record_duration_s:
                    if debug:
                        print("[DEBUG LISTEN]⚠️ Max record duration reached, stopping.")
                    break

                try:
                    frame_bytes, overflowed = stream.read(frame_size)
                except Exception as e:
                    if debug:
                        print("[DEBUG LISTEN]read error:", e)
                    continue

                # sanity check
                if overflowed or len(frame_bytes) < bytes_per_frame:
                    if debug:
                        print("[DEBUG LISTEN] underrun/overflow or partial frame; skipping")
                    continue

                # convert to int16 array (original level)
                try:
                    arr = np.frombuffer(frame_bytes, dtype=np.int16)
                except Exception as e:
                    if debug:
                        print("[DEBUG LISTEN]frombuffer error:", e)
                    # treat as non-speech
                    vad_result = False
                    rms = 0.0
                else:
                    # create attenuated copy for VAD & RMS checks
                    if vad_attenuation is None or vad_attenuation == 1.0:
                        atten_arr = arr
                    else:
                        atten = arr.astype(np.float32) * float(vad_attenuation)
                        atten = np.clip(atten, -32768, 32767)
                        atten_arr = atten.astype(np.int16)

                    atten_bytes = atten_arr.tobytes()

                    # compute RMS on attenuated frame (int16 scale)
                    rms = float(np.sqrt(np.mean(atten_arr.astype(np.float32) ** 2)))

                    # call VAD on attenuated bytes
                    try:
                        vad_result = vad.is_speech(atten_bytes, self.samplerate)
                    except Exception:
                        vad_result = False

                # update ambient buffer when we are not currently in a speech segment
                if not got_speech:
                    # Add the rms to ambient buffer for background estimation only for non-speech frames
                    # We treat frame as non-speech for ambient estimation if VAD says not speech OR rms is low.
                    # This keeps ambient estimate robust when there is background conversation but not a clear spike.
                    if not vad_result or (min_vad_rms is not None and rms < float(min_vad_rms)):
                        ambient_buffer.append(rms)

                # compute ambient mean (guard against empty)
                try:
                    ambient_mean = float(np.mean(list(ambient_buffer))) if ambient_buffer else 0.0
                except Exception:
                    ambient_mean = 0.0

                # final per-frame "is_speech" decision:
                # require BOTH VAD and RMS >= threshold (if min_vad_rms set) AND (if ambient available) rms spike
                if min_vad_rms is not None:
                    if len(ambient_buffer) >= 3:
                        # we have some ambient history: require a spike compared to ambient
                        is_speech = vad_result and (rms >= float(min_vad_rms)) and (rms > ambient_mean * float(energy_spike_factor))
                    else:
                        # not enough ambient history yet, be slightly more permissive (use VAD + absolute RMS)
                        is_speech = vad_result and (rms >= float(min_vad_rms))
                else:
                    if len(ambient_buffer) >= 3:
                        is_speech = vad_result and (rms > ambient_mean * float(energy_spike_factor))
                    else:
                        is_speech = vad_result

                if debug:
                    print(f"[DEBUG LISTEN]VAD={vad_result} RMS={rms:.1f} ambient_mean={ambient_mean:.1f} "
                          f"factor={energy_spike_factor} -> is_speech={is_speech} got_speech={got_speech} "
                          f"consec_silent={consec_silent} ambient_buf_len={len(ambient_buffer)}")

                # keep original frame bytes for recording
                if not got_speech:
                    if is_speech:
                        got_speech = True
                        voiced_frames.append(frame_bytes)
                        consec_silent = 0
                        if debug:
                            print("[DEBUG LISTEN]>>> speech started (energy spike + VAD detected)")
                    else:
                        # still waiting for initial speech
                        continue
                else:
                    # already in a speech segment
                    voiced_frames.append(frame_bytes)
                    if is_speech:
                        consec_silent = 0
                    else:
                        consec_silent += 1

                    if consec_silent >= required_silent_frames:
                        if debug:
                            print(f"[DEBUG LISTEN]Silence detected ({consec_silent} frames). stopping.")
                        stopped_by_silence = True
                        break

        # if nothing recorded, return empty array and False
        if not voiced_frames:
            if debug:
                print("[DEBUG LISTEN]No voiced frames recorded.")
            return np.array([], dtype=np.float32), False

        pcm_data = b"".join(voiced_frames)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        # return normalized float32 (same as before) plus stopped flag
        return audio_int16.astype(np.float32) / 32768.0, stopped_by_silence

    def remove_consecutive_duplicate_words(self, sentence: str) -> str:
        words = sentence.split()
        cleaned = []
        prev_norm = None
        for w in words:
            w_norm = w.strip(string.punctuation).lower()
            if prev_norm is None or w_norm != prev_norm:
                cleaned.append(w)
            prev_norm = w_norm
        return ' '.join(cleaned)

    def remove_duplicate_sentences(self, text: str) -> str:
        sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
        seen = set()
        unique_sentences = []
        for s in sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            cleaned = self.remove_consecutive_duplicate_words(s_stripped)
            norm = cleaned.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique_sentences.append(cleaned)
        return ' '.join(unique_sentences)

    def remove_extreme_repetition(self, text: str, max_repeats: int = 3) -> str:
        tokens = text.split()
        cleaned = []
        i = 0
        while i < len(tokens):
            for size in range(1, 6):
                if i + size * max_repeats > len(tokens):
                    continue
                chunk = tokens[i:i + size]
                repeated = all(
                    tokens[i + r * size:i + (r + 1) * size] == chunk
                    for r in range(1, max_repeats)
                )
                if repeated:
                    i += size * max_repeats
                    break
            else:
                cleaned.append(tokens[i])
                i += 1
        return ' '.join(cleaned)

    # ---------------- Whisper-based listening (chunked + 150-char stop) ----------------
    def listen_whisper(self, blocking: bool = True, callback: Optional[Callable] = None, *, max_workers: int = 1):
        """
        Modified Whisper listener:

        - By default behaves exactly like your original listen_whisper (blocking=True).
        - If blocking=False, it will return quickly AFTER submitting the audio to a background transcription
          worker. When the transcription completes, if `callback` is provided it will be called with the result dict.
          Also, results are placed into self._transcribe_result_queue for polling.

        Args:
          blocking: if True, wait for transcription and return dict (old behavior). If False, offload transcription.
          callback: optional function(result_dict) called in background when transcription finishes.
          max_workers: number of background workers (default 1). Increase only if you know the model safely supports concurrency.
        """
##        try:
##            speech.AlfredSpeak("Listening...")
##        except Exception:
##            # If speech module unavailable, continue silently
##            pass
##
##        if self.stop_listening:
##            return None

        print("[DEBUG LISTEN]🎤 Starting single-shot VAD recording for Whisper transcription...")

        # Tunables
        per_chunk_max_s = 20
        vad_mode = 3
        silence_duration_ms = 1200
        min_rms = 1e-4
        min_words_required = 2

        # Record exactly once (single utterance)
        try:
            audio, stopped_by_silence = self.record_until_silence(
                frame_duration_ms=30,
                vad_mode=vad_mode,
                silence_duration_ms=silence_duration_ms,
                max_record_duration_s=per_chunk_max_s
            )
        except TypeError:
            try:
                audio, stopped_by_silence, _ = self.record_until_silence(
                    frame_duration_ms=30,
                    vad_mode=vad_mode,
                    silence_duration_ms=silence_duration_ms,
                    max_record_duration_s=per_chunk_max_s
                )
            except Exception as e:
                print("[DEBUG LISTEN]❌ record_until_silence failed:", e)
                return None
        except Exception as e:
            print("[DEBUG LISTEN]❌ record_until_silence error:", e)
            return None

        if audio is None or getattr(audio, "size", 0) == 0:
            print("[DEBUG LISTEN]⚠️ No audio captured for Whisper (single-shot).")
            return None

        # Quick energy/RMS check
        try:
            audio_f32 = audio.astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(audio_f32))))
        except Exception:
            rms = 0.0

        if rms < min_rms:
            print(f"[DEBUG LISTEN]🔇 Recorded audio RMS too low ({rms:.6f}), skipping transcription.")
            return None

        # Prepare total_pcm_bytes for speaker ID later (same as before)
        total_pcm_bytes = bytearray()
        try:
            audio_int16 = (audio_f32 * 32768.0).astype(np.int16)
            total_pcm_bytes.extend(audio_int16.tobytes())
        except Exception as e:
            print("[DEBUG LISTEN]⚠️ Error converting audio for speaker ID:", e)

        # If blocking, reuse your previous flow (write tmp wav and call model synchronously)
        if blocking:
            tmp_wav = None
            try:
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tmp_wav.name, audio, self.samplerate)
            except Exception as e:
                print("[DEBUG LISTEN]❌ Could not write temp WAV for transcription:", e)
                if tmp_wav:
                    try: os.unlink(tmp_wav.name)
                    except Exception: pass
                return None

            try:
##                try:
##                    speech.AlfredSpeak("Stopped listening...")
##                except Exception:
##                    pass
                print("[DEBUG LISTEN]🎤 Transcribing (single-shot) with Whisper...")
                result = self.whisper_model.transcribe(tmp_wav.name, language="en", fp16=False)
            except Exception as e:
                print("[DEBUG LISTEN]❌ Whisper transcription error:", e)
                try:
                    if tmp_wav: os.unlink(tmp_wav.name)
                except Exception:
                    pass
                return None
            finally:
                try:
                    if tmp_wav:
                        os.unlink(tmp_wav.name)
                except Exception:
                    pass

            raw_text = result.get("text", "").strip()
            print(f"[DEBUG LISTEN]📝 Whisper raw recognized: {raw_text!r}")

            if not raw_text:
                print("[DEBUG LISTEN]⚠️ Whisper returned empty text.")
                return None

            final_result = self._process_transcription_text(raw_text)
            if not final_result or len(final_result.split()) < min_words_required:
                print("[DEBUG LISTEN]🔎 Transcribed result too short after cleaning, ignoring.")
                return None



            # Speaker / Gender step (fixed, separated logic)
            speaker_name_local, speaker_score = None, None
            gender_str, gender_conf = None, None

            try:
                if total_pcm_bytes:
                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

                    combined_int16 = np.frombuffer(bytes(total_pcm_bytes), dtype=np.int16)
                    combined_float = combined_int16.astype(np.float32) / 32768.0

                    sf.write(tmp_wav2.name, combined_float, self.samplerate)

                    # ---------- Speaker identification ----------
                    try:
                        from Speech_Identification_User import identify_from_file
                        speaker_name_local, speaker_score = identify_from_file(tmp_wav2.name)
                    except Exception as e:
                        print("[DEBUG LISTEN]❌ Speaker ID error:", e)
                        speaker_name_local, speaker_score = None, None

                    # ---------- Gender classification (single MFCC pass) ----------
                    try:
                        from whisper_gender import extract_mfcc, scaler, clf

                        mfcc = extract_mfcc(tmp_wav2.name).reshape(1, -1)
                        mfcc_scaled = scaler.transform(mfcc)

                        gender_pred = clf.predict(mfcc_scaled)[0]
                        gender_prob = clf.predict_proba(mfcc_scaled)[0]

                        gender_str = "Male" if gender_pred == 0 else "Female"
                        gender_conf = float(gender_prob[gender_pred])

                    except Exception as e:
                        print("[DEBUG LISTEN]❌ Gender classification error:", e)
                        gender_str, gender_conf = None, None

                    try:
                        os.unlink(tmp_wav2.name)
                    except Exception:
                        pass

            except Exception as e:
                print("[DEBUG LISTEN]❌ Speaker/Gender pipeline error:", e)
                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None

            combo_name_local = (
                f"{speaker_name_local} ({gender_str})"
                if speaker_name_local and gender_str
                else speaker_name_local
            )

            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")







##            # --- Speaker & Gender classification (drop-in replacement) ---
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##
##            try:
##                if total_pcm_bytes:
##                    # normalize raw bytes input (accept list of chunks or bytes)
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    # interpret as int16 PCM and convert to float32 in [-1,1]
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # downmix stereo -> mono if interleaved stereo detected (heuristic)
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (
##                                abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6
##                            ):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    # resample if needed (target 16000)
##                    TARGET_SR = 16000
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        write_sr = self.samplerate
##
##                    # write a safe temp wav for downstream extraction/identification
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()  # ensure writable on Windows
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype="PCM_16")
##
##                    # ------------------------ Gender (exact working pipeline) ------------------------
##                    try:
##                        from whisper_gender import extract_mfcc, scaler, clf
##                    except Exception as e:
##                        print("[DEBUG LISTEN]❌ Could not import whisper_gender:", e)
##                        extract_mfcc, scaler, clf = None, None, None
##
##                    try:
##                        if extract_mfcc is not None:
##                            # exact flow you provided (reshape to (1, -1))
##                            mfcc = extract_mfcc(tmp_wav2.name)
##                            mfcc = np.asarray(mfcc).reshape(1, -1)
##                            mfcc_scaled = scaler.transform(mfcc)
##                            gender_pred = clf.predict(mfcc_scaled)[0]
##                            gender_prob = clf.predict_proba(mfcc_scaled)[0]
##                            gender_label = "Male" if int(gender_pred) == 0 else "Female"
##                            gender_conf = float(gender_prob[int(gender_pred)]) if gender_prob is not None else None
##                            gender_str = gender_label
##                        else:
##                            gender_str, gender_conf = None, None
##                    except Exception as e:
##                        print(f"[DEBUG LISTEN]❌ Gender classification error: {e}")
##                        gender_str, gender_conf = None, None
##
##                    # ------------------------ Speaker identification (SpeechBrain primary) ------------------------
##                    sb_used = False
##                    try:
##                        try:
##                            import Speech_Identification_User as sb_id_mod  # ensure module path matches your project
##                            sb_available = True
##                        except Exception:
##                            sb_id_mod = None
##                            sb_available = False
##
##                        need_id = True
##                        if sb_available:
##                            try:
##                                sb_name, sb_score = sb_id_mod.identify_from_file(tmp_wav2.name)
##                                print(f"[DEBUG LISTEN]ℹ️ SpeechBrain identify_from_file -> {sb_name}, {sb_score}")
##                                if sb_name:
##                                    speaker_name_local = str(sb_name)
##                                    speaker_score = float(sb_score) if sb_score is not None else None
##                                    sb_used = True
##                                    need_id = False
##                            except Exception as e_sb:
##                                print("[DEBUG LISTEN]⚠️ SpeechBrain identification failed:", e_sb)
##                                sb_used = False
##                                need_id = True
##                    except Exception:
##                        sb_used = False
##                        need_id = True
##
##                    # ------------------------ Fallback: your existing wrapper ------------------------
##                    if need_id:
##                        try:
##                            speaker_name_local, speaker_score, gender_from_wrapper, gender_conf_from_wrapper = identify_safely_from_file(
##                                tmp_wav2.name,
##                                precomputed_transcript=final_result,
##                                precomputed_mfcc=(mfcc if 'mfcc' in locals() else None)
##                            )
##                            if gender_str is None and gender_from_wrapper is not None:
##                                gender_str = gender_from_wrapper
##                                gender_conf = gender_conf_from_wrapper
##                        except Exception as e_idwrap:
##                            print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                            speaker_name_local, speaker_score = None, None
##
##                    # ------------------------ Last-resort: segment voting (if helper exists) ------------------------
##                    still_need = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                    if still_need and ('identify_speaker_by_segment_voting' in globals()):
##                        try:
##                            raw_label, avg_conf = identify_speaker_by_segment_voting(tmp_wav2.name, segment_seconds=2.0, min_confidence=0.05)
##                            if raw_label is not None:
##                                # try to map classifier label to readable name via whisper_gender clf.classes_ or fallback naming
##                                mapped_name = None
##                                try:
##                                    if 'clf' in locals() and hasattr(clf, "classes_"):
##                                        classes = list(clf.classes_)
##                                        if raw_label in classes:
##                                            mapped_name = str(raw_label)
##                                        else:
##                                            # if raw_label is numeric index, try indexing
##                                            try:
##                                                idx = int(raw_label)
##                                                if 0 <= idx < len(classes):
##                                                    mapped_name = str(classes[idx])
##                                            except Exception:
##                                                mapped_name = None
##                                except Exception:
##                                    mapped_name = None
##
##                                if mapped_name is None:
##                                    mapped_name = f"Speaker_{raw_label}"
##
##                                if (not speaker_name_local) or (speaker_score is None) or (avg_conf is not None and avg_conf > (speaker_score or 0.0)):
##                                    speaker_name_local = str(mapped_name)
##                                    speaker_score = float(avg_conf) if avg_conf is not None else None
##                                    print(f"[DEBUG LISTEN]🔁 Segment-voting fallback used: raw_label={raw_label} -> name={speaker_name_local} (score={speaker_score})")
##                        except Exception as e:
##                            print("[DEBUG LISTEN]❌ Segment-voting error:", e)
##
##                    # ------------------------------------------------------------------------------
##
##                    # cleanup tmp wav
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")












##            # ---------- Speaker / Gender block (drop-in replacement) ----------
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##
##            try:
##                if total_pcm_bytes:
##                    # normalize raw bytes input
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # downmix stereo -> mono if appears interleaved
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (
##                                abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6
##                            ):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    # resample if needed
##                    TARGET_SR = 16000
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        write_sr = self.samplerate
##
##                    # write temp wav for MFCC extraction / other ID attempts
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    # load whisper_gender pieces (gender classifier) and optionally SpeechBrain module
##                    from whisper_gender import extract_mfcc, scaler, clf
##                    try:
##                        import Speech_Identification_User as sb_id_mod  # your SpeechBrain module
##                        sb_available = True
##                    except Exception:
##                        sb_id_mod = None
##                        sb_available = False
##
##                    # --- extract MFCC once (can be 1D or 2D) ---
##                    precomputed_mfcc = None
##                    try:
##                        precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##                    except Exception as e_mfcc:
##                        print("[DEBUG LISTEN]⚠️ extract_mfcc failed:", e_mfcc)
##                        precomputed_mfcc = None
##
##                    # --- Gender classification (exact flow you provided) ---
##                    try:
##                        if precomputed_mfcc is not None:
##                            mfcc = np.asarray(precomputed_mfcc).reshape(1, -1)
##                            mfcc_scaled = scaler.transform(mfcc)
##                            gender_pred = clf.predict(mfcc_scaled)[0]
##                            gender_prob = clf.predict_proba(mfcc_scaled)[0]
##                            gender_label = "Male" if int(gender_pred) == 0 else "Female"
##                            gender_conf = float(gender_prob[int(gender_pred)]) if gender_prob is not None else None
##                            gender_str = gender_label
##                        else:
##                            gender_str, gender_conf = None, None
##                    except Exception as e:
##                        print(f"[DEBUG LISTEN]❌ Gender classification error (Vosk): {e}")
##                        gender_str, gender_conf = None, None
##
##                    # --- Primary identification (your existing wrapper) ---
##                    try:
##                        speaker_name_local, speaker_score, gender_from_wrapper, gender_conf_from_wrapper = identify_safely_from_file(
##                            tmp_wav2.name,
##                            precomputed_transcript=final_result,
##                            precomputed_mfcc=precomputed_mfcc
##                        )
##                        if gender_str is None and gender_from_wrapper is not None:
##                            gender_str = gender_from_wrapper
##                            gender_conf = gender_conf_from_wrapper
##                    except Exception as e_idwrap:
##                        print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                        speaker_name_local, speaker_score = None, None
##
##                    # --- SECONDARY fallback: SpeechBrain embeddings-based identification (preferred if available) ---
##                    try:
##                        need_fallback = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if need_fallback and sb_available:
##                            try:
##                                # identify_from_file returns (name, score) or (None, score)
##                                sb_name, sb_score = sb_id_mod.identify_from_file(tmp_wav2.name)
##                                print(f"[DEBUG LISTEN]ℹ️ SpeechBrain fallback result: {sb_name}, {sb_score}")
##                                if sb_name:
##                                    # accept if sb_score better than current or if no current name
##                                    if (not speaker_name_local) or (speaker_score is None) or (sb_score is not None and sb_score > (speaker_score or 0.0)):
##                                        speaker_name_local = str(sb_name)
##                                        speaker_score = float(sb_score) if sb_score is not None else None
##                                        print(f"[DEBUG LISTEN]🔁 SpeechBrain fallback used: {speaker_name_local} (score={speaker_score})")
##                            except Exception as e_sb:
##                                print("[DEBUG LISTEN]⚠️ SpeechBrain identification failed:", e_sb)
##                    except Exception as e_fb:
##                        print("[DEBUG LISTEN]⚠️ Fallback ID step error:", e_fb)
##
##                    # --- LAST RESORT: segment voting fallback (only if still no good ID) ---
##                    try:
##                        still_need = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if still_need:
##                            # call your existing segment voting helper if defined; otherwise skip
##                            if 'identify_speaker_by_segment_voting' in globals():
##                                raw_label, avg_conf = identify_speaker_by_segment_voting(tmp_wav2.name, segment_seconds=2.0, min_confidence=0.05)
##                                if raw_label is not None:
##                                    # map raw_label->name using previous mapping helper if present, else use raw_label
##                                    if '_map_label_to_name' in globals():
##                                        mapped = _map_label_to_name(raw_label, clf)
##                                    else:
##                                        mapped = f"Speaker_{raw_label}"
##                                    if (not speaker_name_local) or (speaker_score is None) or (avg_conf is not None and avg_conf > (speaker_score or 0.0)):
##                                        speaker_name_local = str(mapped)
##                                        speaker_score = float(avg_conf) if avg_conf is not None else None
##                                        print(f"[DEBUG LISTEN]🔁 Segment-voting fallback used: raw_label={raw_label} -> name={speaker_name_local} (score={speaker_score})")
##                            else:
##                                print("[DEBUG LISTEN]ℹ️ Segment voting not available (helper not defined).")
##                    except Exception as e_seg:
##                        print("[DEBUG LISTEN]⚠️ segment voting fallback error:", e_seg)
##
##                    # cleanup
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")
##            # --------------------------------------------------------------------




##            # Speaker/gender step (drop-in replacement - uses your exact MFCC->scaler->clf flow for gender)
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##
##            try:
##                if total_pcm_bytes:
##                    # normalize raw bytes input
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # downmix stereo -> mono if appears interleaved
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    # resample to target if needed
##                    TARGET_SR = 16000
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        write_sr = self.samplerate
##
##                    # write safe temp wav for MFCC extraction
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    # load extractor + model pieces (assumes whisper_gender exposes these names)
##                    from whisper_gender import extract_mfcc, scaler, clf
##
##                    # --- extract MFCC once (can be 1D or 2D) ---
##                    precomputed_mfcc = None
##                    try:
##                        precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##                    except Exception as e_mfcc:
##                        print("[DEBUG LISTEN]⚠️ extract_mfcc failed:", e_mfcc)
##                        precomputed_mfcc = None
##
##                    # --- Gender classification (single MFCC extraction) ---
##                    try:
##                        if precomputed_mfcc is not None:
##                            mfcc = np.asarray(precomputed_mfcc).reshape(1, -1)          # reshape to (1, -1) exactly like your working snippet
##                            mfcc_scaled = scaler.transform(mfcc)
##                            gender_pred = clf.predict(mfcc_scaled)[0]
##                            gender_prob = clf.predict_proba(mfcc_scaled)[0]
##                            gender_label = "Male" if int(gender_pred) == 0 else "Female"
##                            gender_conf = float(gender_prob[int(gender_pred)])         # same indexing as your snippet
##                            gender_str = gender_label
##                        else:
##                            gender_str, gender_conf = None, None
##                    except Exception as e:
##                        print(f"[DEBUG LISTEN]❌ Gender classification error (Vosk): {e}")
##                        gender_str, gender_conf = None, None
##
##                    # --- Primary identification (existing wrapper you used) ---
##                    try:
##                        speaker_name_local, speaker_score, gender_from_wrapper, gender_conf_from_wrapper = identify_safely_from_file(
##                            tmp_wav2.name,
##                            precomputed_transcript=final_result,
##                            precomputed_mfcc=precomputed_mfcc
##                        )
##                        # prefer gender we just computed, but keep wrapper gender if ours is None
##                        if gender_str is None and gender_from_wrapper is not None:
##                            gender_str = gender_from_wrapper
##                            gender_conf = gender_conf_from_wrapper
##                    except Exception as e_idwrap:
##                        print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                        speaker_name_local, speaker_score = None, None
##
##                    # --- Fallback ID step: if wrapper returned no/low-confidence ID, try MFCC->scaler->clf approach (flattened) ---
##                    try:
##                        need_fallback = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if need_fallback and precomputed_mfcc is not None:
##                            mfcc_arr = np.asarray(precomputed_mfcc)
##                            mfcc_flat = mfcc_arr.reshape(1, -1)  # handle both 1D (n_mfcc,) and 2D (n_frames,n_mfcc)
##
##                            # apply scaler if available (re-use same scaler variable; if speaker-ID uses different scaler you should swap it here)
##                            try:
##                                mfcc_scaled_id = scaler.transform(mfcc_flat)
##                            except Exception as e_scale:
##                                print("[DEBUG LISTEN]⚠️ scaler.transform failed for ID fallback:", e_scale)
##                                mfcc_scaled_id = mfcc_flat
##
##                            # classifier predict for ID (robust handling)
##                            try:
##                                pred_id = clf.predict(mfcc_scaled_id)[0]
##                            except Exception as e_clfpred:
##                                print("[DEBUG LISTEN]⚠️ clf.predict failed for ID fallback:", e_clfpred)
##                                pred_id = None
##
##                            fallback_label, fallback_score = None, None
##                            if pred_id is not None:
##                                # treat pred as label by default
##                                fallback_label = pred_id
##                                # try to get probability
##                                try:
##                                    if hasattr(clf, "predict_proba"):
##                                        probs_id = clf.predict_proba(mfcc_scaled_id)[0]
##                                        if hasattr(clf, "classes_"):
##                                            try:
##                                                classes = np.asarray(clf.classes_)
##                                                matches = np.where(classes == pred_id)[0]
##                                                if matches.size > 0:
##                                                    idx = int(matches[0])
##                                                    fallback_score = float(probs_id[idx])
##                                                else:
##                                                    # maybe pred is numeric index
##                                                    try:
##                                                        idx = int(pred_id)
##                                                        fallback_score = float(probs_id[idx])
##                                                    except Exception:
##                                                        fallback_score = float(np.max(probs_id))
##                                            except Exception:
##                                                fallback_score = float(np.max(probs_id))
##                                        else:
##                                            fallback_score = float(np.max(probs_id))
##                                    else:
##                                        fallback_score = 1.0
##                                except Exception as e_prob:
##                                    print("[DEBUG LISTEN]⚠️ computing fallback probability failed:", e_prob)
##                                    fallback_score = None
##
##                            if fallback_label:
##                                if (not speaker_name_local) or (speaker_score is None) or (fallback_score is None) or (fallback_score > (speaker_score or 0.0)):
##                                    speaker_name_local = str(fallback_label)
##                                    speaker_score = float(fallback_score) if fallback_score is not None else None
##                                    print(f"[DEBUG LISTEN]🔁 Fallback speaker ID used: {speaker_name_local} (score={speaker_score})")
##                    except Exception as e_fb:
##                        print("[DEBUG LISTEN]⚠️ Fallback ID step error:", e_fb)
##
##                    # cleanup temp file
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")







##            # Speaker/gender step (drop-in replacement) - robust shape handling + robust fallback for ID
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##            try:
##                if total_pcm_bytes:
##                    # normalize raw bytes input
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # downmix stereo -> mono if appears interleaved
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    TARGET_SR = 16000
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        write_sr = self.samplerate
##
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    # load model pieces
##                    from whisper_gender import extract_mfcc, scaler, clf
##
##                    precomputed_mfcc = None
##                    try:
##                        precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##                    except Exception as e_mfcc:
##                        print("[DEBUG LISTEN]⚠️ extract_mfcc failed:", e_mfcc)
##
##                    # Primary identification (existing wrapper you used)
##                    try:
##                        speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
##                            tmp_wav2.name,
##                            precomputed_transcript=final_result,
##                            precomputed_mfcc=precomputed_mfcc
##                        )
##                    except Exception as e_idwrap:
##                        print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                        speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##                    # Fallback: use the same MFCC -> scaler -> clf flow you use for gender (handles 1D or 2D MFCC)
##                    try:
##                        need_fallback = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if need_fallback and precomputed_mfcc is not None:
##                            mfcc_arr = np.asarray(precomputed_mfcc)
##
##                            # Accept either 1D (n_mfcc,) or 2D (n_frames, n_mfcc) and flatten to (1, -1)
##                            if mfcc_arr.ndim == 1:
##                                mfcc_flat = mfcc_arr.reshape(1, -1)
##                            elif mfcc_arr.ndim == 2:
##                                mfcc_flat = mfcc_arr.reshape(1, -1)
##                            else:
##                                print("[DEBUG LISTEN]⚠️ precomputed_mfcc has unexpected ndim:", mfcc_arr.ndim)
##                                mfcc_flat = None
##
##                            if mfcc_flat is not None:
##                                print(f"[DEBUG LISTEN]ℹ️ MFCC flat shape for fallback: {mfcc_flat.shape}")
##
##                                # apply scaler if available
##                                try:
##                                    if scaler is not None:
##                                        mfcc_scaled = scaler.transform(mfcc_flat)
##                                    else:
##                                        mfcc_scaled = mfcc_flat
##                                except Exception as e_scale:
##                                    print("[DEBUG LISTEN]⚠️ scaler.transform failed for ID fallback:", e_scale)
##                                    mfcc_scaled = mfcc_flat
##
##                                # classifier predict (robust handling whether predict returns label or numeric index)
##                                try:
##                                    pred = clf.predict(mfcc_scaled)[0]
##                                except Exception as e_clfpred:
##                                    print("[DEBUG LISTEN]⚠️ clf.predict failed for ID fallback:", e_clfpred)
##                                    pred = None
##
##                                fallback_label, fallback_score = None, None
##                                try:
##                                    if pred is not None:
##                                        # default: treat pred as the label directly (most sklearn classifiers return labels)
##                                        fallback_label = pred
##
##                                        # if we have predict_proba, get the corresponding probability robustly
##                                        if hasattr(clf, "predict_proba"):
##                                            probs = clf.predict_proba(mfcc_scaled)[0]
##                                            # try to find index of pred in classes_
##                                            if hasattr(clf, "classes_"):
##                                                try:
##                                                    classes = np.asarray(clf.classes_)
##                                                    matches = np.where(classes == pred)[0]
##                                                    if matches.size > 0:
##                                                        idx = int(matches[0])
##                                                        fallback_score = float(probs[idx])
##                                                    else:
##                                                        # maybe pred is numeric index
##                                                        try:
##                                                            idx = int(pred)
##                                                            fallback_score = float(probs[idx])
##                                                        except Exception:
##                                                            fallback_score = float(np.max(probs))
##                                                except Exception:
##                                                    fallback_score = float(np.max(probs))
##                                            else:
##                                                fallback_score = float(np.max(probs))
##                                        else:
##                                            fallback_score = 1.0
##                                    else:
##                                        fallback_label, fallback_score = None, None
##                                except Exception as e_prob:
##                                    print("[DEBUG LISTEN]⚠️ computing fallback probability failed:", e_prob)
##                                    fallback_label, fallback_score = fallback_label, fallback_score
##
##                                # accept fallback if it seems better than wrapper result
##                                if fallback_label:
##                                    if (not speaker_name_local) or (speaker_score is None) or (fallback_score is None) or (fallback_score > (speaker_score or 0.0)):
##                                        speaker_name_local = str(fallback_label)
##                                        speaker_score = float(fallback_score) if fallback_score is not None else None
##                                        print(f"[DEBUG LISTEN]🔁 Fallback speaker ID used: {speaker_name_local} (score={speaker_score})")
##                            else:
##                                print("[DEBUG LISTEN]⚠️ cannot compute fallback: mfcc_flat is None")
##                    except Exception as e_fb:
##                        print("[DEBUG LISTEN]⚠️ Fallback ID step error:", e_fb)
##
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")
##





##            # Speaker/gender step (same steps as original) - use MFCC reshape fallback for speaker ID
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##            try:
##                if total_pcm_bytes:
##                    # normalize raw bytes input
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # downmix stereo -> mono if appears interleaved
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    TARGET_SR = 16000
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        write_sr = self.samplerate
##
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    # load model pieces
##                    from whisper_gender import extract_mfcc, scaler, clf
##
##                    precomputed_mfcc = None
##                    try:
##                        precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##                    except Exception as e_mfcc:
##                        print("[DEBUG LISTEN]⚠️ extract_mfcc failed:", e_mfcc)
##
##                    # Primary identification (existing wrapper you used)
##                    try:
##                        speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
##                            tmp_wav2.name,
##                            precomputed_transcript=final_result,
##                            precomputed_mfcc=precomputed_mfcc
##                        )
##                    except Exception as e_idwrap:
##                        print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                        speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##                    # Fallback: use same MFCC -> scaler -> clf flow as your working gender code
##                    try:
##                        need_fallback = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if need_fallback and precomputed_mfcc is not None:
##                            mfcc_arr = np.asarray(precomputed_mfcc)
##
##                            # expected shape (n_frames, n_mfcc) -> flatten to (1, -1) as your gender snippet does
##                            if mfcc_arr.ndim == 2 and mfcc_arr.size > 0:
##                                mfcc_flat = mfcc_arr.reshape(1, -1)
##                                print(f"[DEBUG LISTEN]ℹ️ MFCC flat shape for fallback: {mfcc_flat.shape}")
##
##                                # apply scaler if available (same approach as in your gender code)
##                                try:
##                                    if scaler is not None:
##                                        mfcc_scaled = scaler.transform(mfcc_flat)
##                                    else:
##                                        mfcc_scaled = mfcc_flat
##                                except Exception as e_scale:
##                                    print("[DEBUG LISTEN]⚠️ scaler.transform failed for ID fallback:", e_scale)
##                                    mfcc_scaled = mfcc_flat  # fallback
##
##                                # classifier predict
##                                try:
##                                    pred = clf.predict(mfcc_scaled)[0]
##                                    # get human-readable label: if clf.classes_ exists, prefer that mapping
##                                    if hasattr(clf, "classes_"):
##                                        fallback_label = clf.classes_[int(pred)]
##                                    else:
##                                        fallback_label = pred
##
##                                    # get probability/confidence if available
##                                    if hasattr(clf, "predict_proba"):
##                                        probs = clf.predict_proba(mfcc_scaled)[0]
##                                        try:
##                                            fallback_score = float(probs[int(pred)])
##                                        except Exception:
##                                            # if pred is not an index into probs, fallback to max
##                                            fallback_score = float(np.max(probs))
##                                    else:
##                                        fallback_score = 1.0
##                                except Exception as e_clf:
##                                    print("[DEBUG LISTEN]⚠️ clf prediction failed for ID fallback:", e_clf)
##                                    fallback_label, fallback_score = None, None
##
##                                # accept fallback if better than wrapper result
##                                if fallback_label:
##                                    if (not speaker_name_local) or (speaker_score is None) or (fallback_score > (speaker_score or 0.0)):
##                                        speaker_name_local = str(fallback_label)
##                                        speaker_score = float(fallback_score) if fallback_score is not None else None
##                                        print(f"[DEBUG LISTEN]🔁 Fallback speaker ID used: {speaker_name_local} (score={speaker_score})")
##                            else:
##                                print("[DEBUG LISTEN]⚠️ precomputed_mfcc shape unexpected for fallback:", mfcc_arr.shape)
##                    except Exception as e_fb:
##                        print("[DEBUG LISTEN]⚠️ Fallback ID step error:", e_fb)
##
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")





##            # Speaker/gender step (same steps as original) - improved with fallback speaker-ID using scaler+clf
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##            try:
##                if total_pcm_bytes:
##                    # make raw bytes robustly whether total_pcm_bytes is a list of chunks or already bytes
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    # interpret as int16 PCM
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##
##                    # convert to float32 in -1..1 range
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # ensure mono (if interleaved stereo, downmix to mono)
##                    if combined_float.size % 2 == 0:
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6):
##                                combined_float = (a + b) / 2.0
##                        except Exception:
##                            pass
##
##                    # target sample rate expected by the gender/classifier (adjust if your classifier expects a different rate)
##                    TARGET_SR = 16000
##
##                    # resample if needed
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        # fallback: if resampling fails, continue with original samplerate
##                        write_sr = self.samplerate
##
##                    # write safe temp wav for MFCC extraction
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()  # allow sf.write to open it on all platforms
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    # import model pieces (assumes whisper_gender exposes these)
##                    from whisper_gender import extract_mfcc, scaler, clf
##
##                    # extract MFCC from the resampled file
##                    precomputed_mfcc = None
##                    try:
##                        precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##                    except Exception as e_mfcc:
##                        print("[DEBUG LISTEN]⚠️ extract_mfcc failed:", e_mfcc)
##
##                    # Primary identification (existing wrapper you used)
##                    try:
##                        speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
##                            tmp_wav2.name,
##                            precomputed_transcript=final_result,
##                            precomputed_mfcc=precomputed_mfcc
##                        )
##                    except Exception as e_idwrap:
##                        print("[DEBUG LISTEN]⚠️ identify_safely_from_file wrapper failed:", e_idwrap)
##                        speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##                    # Fallback: if wrapper gave no ID or low confidence, try direct clf+scaler prediction
##                    try:
##                        need_fallback = (not speaker_name_local) or (speaker_score is None) or (speaker_score < 0.60)
##                        if need_fallback and precomputed_mfcc is not None:
##                            # Aggregate MFCC to a fixed-length feature vector (mean + std across time axis)
##                            # precomputed_mfcc expected shape: (n_frames, n_mfcc)
##                            mfcc = np.asarray(precomputed_mfcc)
##                            if mfcc.ndim == 2 and mfcc.shape[0] >= 1:
##                                feat_mean = mfcc.mean(axis=0)
##                                feat_std = mfcc.std(axis=0)
##                                feat = np.concatenate([feat_mean, feat_std])
##
##                                # scaler may be None if not provided; check and transform if available
##                                X = feat.reshape(1, -1)
##                                try:
##                                    if scaler is not None:
##                                        X = scaler.transform(X)
##                                except Exception as e_scale:
##                                    print("[DEBUG LISTEN]⚠️ scaler.transform failed:", e_scale)
##
##                                # predict probabilities if classifier supports it, else predict label
##                                try:
##                                    if hasattr(clf, "predict_proba"):
##                                        probs = clf.predict_proba(X)[0]
##                                        idx = int(np.argmax(probs))
##                                        fallback_label = clf.classes_[idx]
##                                        fallback_score = float(probs[idx])
##                                    else:
##                                        fallback_label = clf.predict(X)[0]
##                                        fallback_score = 1.0  # unknown calibration
##                                except Exception as e_clf:
##                                    print("[DEBUG LISTEN]⚠️ clf prediction failed:", e_clf)
##                                    fallback_label, fallback_score = None, None
##
##                                if fallback_label:
##                                    # Use fallback only if it seems better than wrapper result
##                                    if (not speaker_name_local) or (speaker_score is None) or (fallback_score > (speaker_score or 0.0)):
##                                        speaker_name_local = str(fallback_label)
##                                        speaker_score = float(fallback_score) if fallback_score is not None else None
##                                        print(f"[DEBUG LISTEN]🔁 Fallback speaker ID used: {speaker_name_local} (score={speaker_score})")
##                            else:
##                                print("[DEBUG LISTEN]⚠️ precomputed_mfcc shape unexpected:", mfcc.shape)
##                    except Exception as e_fb:
##                        print("[DEBUG LISTEN]⚠️ Fallback ID step error:", e_fb)
##
##                    # clean up temp file
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")
##            print(f"[DEBUG LISTEN]🔍 ID debug -> name: {speaker_name_local}, score: {speaker_score}, gender: {gender_str}, gender_conf: {gender_conf}")



##            # Speaker/gender step (same steps as original)
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##            try:
##                if total_pcm_bytes:
##                    # make raw bytes robustly whether total_pcm_bytes is a list of chunks or already bytes
##                    if isinstance(total_pcm_bytes, (list, tuple)):
##                        raw_bytes = b"".join(total_pcm_bytes)
##                    elif isinstance(total_pcm_bytes, bytes):
##                        raw_bytes = total_pcm_bytes
##                    else:
##                        # fallback: try to coerce
##                        raw_bytes = bytes(total_pcm_bytes)
##
##                    # interpret as int16 PCM
##                    combined_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
##
##                    # convert to float32 in -1..1 range
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##
##                    # ensure mono (if interleaved stereo/audio with 2 channels, downmix to mono)
##                    # only do this if length is divisible by 2 and the signal looks interleaved
##                    if combined_float.size % 2 == 0:
##                        # crude heuristic: if alternating samples have different statistics, it's likely stereo interleaved
##                        try:
##                            a = combined_float[0::2]
##                            b = combined_float[1::2]
##                            if (np.std(a) > 0 and np.std(b) > 0) and (abs(np.mean(a) - np.mean(b)) > 1e-6 or abs(np.std(a) - np.std(b)) > 1e-6):
##                                combined_float = (a + b) / 2.0  # downmix stereo -> mono
##                            # otherwise leave as-is (likely mono but even length)
##                        except Exception:
##                            pass
##
##                    # target sample rate expected by the gender classifier (adjust if your classifier expects a different rate)
##                    TARGET_SR = 16000
##
##                    # resample if needed
##                    try:
##                        if self.samplerate != TARGET_SR:
##                            import math
##                            from scipy.signal import resample_poly
##
##                            # reduce up/down by gcd to keep arrays small
##                            g = math.gcd(self.samplerate, TARGET_SR)
##                            up = TARGET_SR // g
##                            down = self.samplerate // g
##                            combined_float = resample_poly(combined_float, up, down)
##                            write_sr = TARGET_SR
##                        else:
##                            write_sr = self.samplerate
##                    except Exception:
##                        # scipy not available or resampling failed -> write with original samplerate (best-effort)
##                        write_sr = self.samplerate
##
##                    # write safe temp wav for MFCC extraction
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    tmp_wav2.close()  # close so sf.write can open it on all platforms
##                    sf.write(tmp_wav2.name, combined_float, write_sr, subtype='PCM_16')
##
##                    from whisper_gender import extract_mfcc, scaler, clf
##                    # extract MFCC from the resampled file (extract_mfcc should expect the TARGET_SR)
##                    precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##
##                    speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
##                        tmp_wav2.name,
##                        precomputed_transcript=final_result,
##                        precomputed_mfcc=precomputed_mfcc
##                    )
##
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")


##            # Speaker/gender step (same steps as original)
##            speaker_name_local, speaker_score = None, None
##            gender_str, gender_conf = None, None
##            try:
##                if total_pcm_bytes:
##                    tmp_wav2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
##                    combined_int16 = np.frombuffer(bytes(total_pcm_bytes), dtype=np.int16)
##                    combined_float = combined_int16.astype(np.float32) / 32768.0
##                    sf.write(tmp_wav2.name, combined_float, self.samplerate)
##
##                    from whisper_gender import extract_mfcc, scaler, clf
##                    precomputed_mfcc = extract_mfcc(tmp_wav2.name)
##
##                    speaker_name_local, speaker_score, gender_str, gender_conf = identify_safely_from_file(
##                        tmp_wav2.name,
##                        precomputed_transcript=final_result,
##                        precomputed_mfcc=precomputed_mfcc
##                    )
##
##                    try:
##                        os.unlink(tmp_wav2.name)
##                    except Exception:
##                        pass
##            except Exception as e:
##                print("[DEBUG LISTEN]❌ Speaker ID / Gender error:", e)
##                speaker_name_local, speaker_score, gender_str, gender_conf = None, None, None, None
##
##            combo_name_local = f"{speaker_name_local} ({gender_str})" if speaker_name_local else None
##
##            print(f"[DEBUG LISTEN]📝 Final processed output (len={len(final_result)}): {final_result}")

            return {
                "text": final_result,
                "speaker": combo_name_local,
                "score": speaker_score,
                "gender": gender_str,
                "gender_conf": gender_conf
            }

        # Non-blocking path: submit background worker and return immediately
        else:
            # ensure executor and queue exist
            self._ensure_transcribe_executor(max_workers=max_workers)

            # convert provided audio (float32) to numpy float32 array for background task (Whisper expects float32 WAV)
            pcm_float_array = audio.astype(np.float32)

            # Submit background task with PCM float array and total pcm bytes for speaker/gender
            future = self._transcribe_executor.submit(self._transcribe_worker, pcm_float_array, total_pcm_bytes)

            # If callback supplied, schedule a waiter thread to call it once future done
            if callback is not None:
                def _call_cb(fut):
                    try:
                        res = fut.result()
                        try:
                            callback(res)
                        except Exception as e:
                            print("[DEBUG LISTEN]❌ callback error:", e)
                    except Exception as e:
                        print("[DEBUG LISTEN]❌ background future error:", e)
                future.add_done_callback(_call_cb)

            # Return a lightweight token to indicate submission. Use result queue to poll results if needed.
            return {"status": "submitted", "future": future}

    # ----------------- other features (wake/continuous) follow ---------------
    def listen_bluetooth(self):
        if self.bluetooth:
            try:
                return self.bluetooth.readline().decode("utf-8").strip()
            except serial.SerialException:
                return None
        return None

    def listen_bluetooth_only(self):
        return self.listen_bluetooth() or "No input"

    def start_continuous_listening(self, use_whisper=True):
        self.set_listen_whisper(use_whisper)
        self.set_listen_vosk(not use_whisper)
        self.parent_conn, child_conn = Pipe()
        self.listen_process = Process(
            target=continuous_listen_worker,
            args=(use_whisper, child_conn),
            daemon=True
        )
        self.listen_process.start()
        print("[DEBUG LISTEN]🚀 Background listener process started.")

    def get_result_from_process(self):
        if self.parent_conn and self.parent_conn.poll():
            return self.parent_conn.recv()
        return None

    def stop_listening_process(self):
        self.set_stop_listen_on(True)
        if self.listen_process and self.listen_process.is_alive():
            self.listen_process.terminate()
            print("[DEBUG LISTEN]🛑 Background listener process terminated.")

    # The wake/listen workflows (wake_and_listen_vosk, wake_and_listen_whisper, etc.)
    # are present in original file and mostly unchanged; we only add stop-phrase & silence-timeout behavior.

    def wake_and_listen_vosk(self):
        """
        Continuously listens with Vosk for a wake word.
        When detected, replies then captures command chunks (calls listen_vosk) and returns aggregated command.
        """
        stop_phrases = [r"\bok\b", r"\bthank you\b", r"\bthanks\b", r"\bthat's all\b", r"\bthats all\b"]
        stop_re = re.compile("|".join(stop_phrases), flags=re.IGNORECASE)
        trailing_strip_re = re.compile(r"(?:\b(ok|thank you|thanks|that's all|thats all)\b(?:\s+\S+){0,2})\s*$", flags=re.IGNORECASE)

        while True:
            wake = self.listen_vosk()
            if not wake:
                continue

            wake_text = wake.get("text", "").lower().strip() if isinstance(wake, dict) else str(wake).lower().strip()
            speaker_name_local = wake.get("speaker") if isinstance(wake, dict) else None
            speaker_score = wake.get("score") if isinstance(wake, dict) else None

            # --- Gender detection for wake word ---
            try:
                from whisper_gender import extract_mfcc, scaler, clf
                wake_gender, wake_gender_conf = None, None
                if isinstance(wake, dict) and "audio_file" in wake:
                    mfcc = extract_mfcc(wake["audio_file"]).reshape(1, -1)
                    mfcc_scaled = scaler.transform(mfcc)
                    gender_label = clf.predict(mfcc_scaled)[0]
                    gender_prob = clf.predict_proba(mfcc_scaled)[0]
                    wake_gender = "Male" if gender_label == 0 else "Female"
                    wake_gender_conf = gender_prob[gender_label]
            except Exception as e:
                print("[DEBUG LISTEN]❌ Wake-word gender classification error:", e)
                wake_gender, wake_gender_conf = None, None

            print(f"[DEBUG LISTEN]🛎️ Wake-word check (Vosk): {wake_text}")
            if not wake_text:
                continue

            clean_text = self.remove_extreme_repetition(self.remove_duplicate_sentences(wake_text))
            clean_text = re.sub(r'[.,!?;:]', '', clean_text).strip()
            print(f"[DEBUG LISTEN]🧽 Cleaned wake phrase (Vosk): {clean_text}")

            if any(word in clean_text.lower() for word in self.wake_words):
                try:
                    speech.AlfredSpeak(f"Yes, {speaker_name_local or 'sir'}")
                except Exception:
                    pass
                time.sleep(1)
                print("[DEBUG LISTEN]🔔 Wake word detected, awaiting command…")

                cumulative_text = ""
                cmd_speaker = speaker_name_local
                cmd_score = speaker_score
                cmd_gender = wake_gender
                cmd_gender_conf = wake_gender_conf

                max_total_seconds = 30
                max_chunks = 6
                start_time = time.time()
                last_chunk_time = time.time()
                chunks = 0

                command_mfcc = None  # precompute MFCC once
                stop_detected = False

                while True:
                    if time.time() - start_time > max_total_seconds:
                        print("[DEBUG LISTEN]⚠️ Command capture timeout reached.")
                        break
                    # silence timeout: stop if more than 5 seconds since last chunk
                    if (time.time() - last_chunk_time) > 5:
                        print("[DEBUG LISTEN]⚠️ Silence timeout (5s) reached during command capture.")
                        break
                    if chunks >= max_chunks:
                        print("[DEBUG LISTEN]⚠️ Max command chunks reached.")
                        break

                    cmd = self.listen_vosk()
                    if not cmd:
                        # if no chunk returned, wait a bit and continue (handled by silence check above)
                        time.sleep(0.1)
                        continue

                    cmd_text = cmd.get("text", "").strip() if isinstance(cmd, dict) else str(cmd).strip()
                    chunk_speaker = cmd.get("speaker") if isinstance(cmd, dict) else None
                    chunk_score = cmd.get("score") if isinstance(cmd, dict) else None
                    audio_file = cmd.get("audio_file") if isinstance(cmd, dict) else None

                    if not cmd_text:
                        continue

                    last_chunk_time = time.time()

                    if chunk_speaker:
                        cmd_speaker = chunk_speaker
                    if chunk_score:
                        cmd_score = chunk_score

                    # --- Gender detection for command chunk (best-effort) ---
                    try:
                        if audio_file and command_mfcc is None:
                            from whisper_gender import extract_mfcc, scaler, clf
                            command_mfcc = extract_mfcc(audio_file).reshape(1, -1)
                            mfcc_scaled = scaler.transform(command_mfcc)
                            gender_label = clf.predict(command_mfcc)[0]
                            gender_prob = clf.predict_proba(command_mfcc)[0]
                            cmd_gender = "Male" if gender_label == 0 else "Female"
                            cmd_gender_conf = gender_prob[gender_label]
                    except Exception as e:
                        print("[DEBUG LISTEN]❌ Command gender classification error:", e)

                    # append chunk
                    if cumulative_text:
                        cumulative_text += " " + cmd_text
                    else:
                        cumulative_text = cmd_text
                    chunks += 1
                    print(f"[DEBUG LISTEN]📝 Vosk chunk: {cmd_text} (cumulative_len={len(cumulative_text)})")

                    # check for stop phrase in the cumulative_text or latest chunk
                    if stop_re.search(cmd_text) or stop_re.search(cumulative_text):
                        print("[DEBUG LISTEN]🔚 Stop phrase detected in command chunk.")
                        stop_detected = True
                        # allow loop to continue until silence or quick extra words, but we will break immediately:
                        break

                    if len(cumulative_text) >= 150:
                        cumulative_text = cumulative_text[:150]
                        print("[DEBUG LISTEN]🔒 Reached 150-character cap; stopping capture.")
                        break

                if not cumulative_text:
                    print("[DEBUG LISTEN]⚠️ No transcribed command after capture.")
                    return None

                # Strip trailing stop-phrase plus up to 2 extra words
                cleaned_cmd = cumulative_text.strip()
                cleaned_cmd = trailing_strip_re.sub("", cleaned_cmd).strip()

                cleaned_cmd = self.remove_extreme_repetition(self.remove_duplicate_sentences(cleaned_cmd))
                final_output = re.sub(r'[!?:]', '', cleaned_cmd.lower()).strip()
                final_output = final_output.replace(
                    "i'm going to go ahead and get a little bit of a little bit of a little bit", ""
                )
                final_output = final_output.replace(
                    "i'm going to go ahead and get a little bit of a", ""
                )
                cleaned = re.sub(r'\s+', ' ', final_output).strip()
                words = cleaned.split()
                unique_words = []
                for w in words:
                    if not unique_words or w != unique_words[-1]:
                        unique_words.append(w)
                result = " ".join(unique_words)
                if len(result) > 150:
                    result = result[:150]

                combo_name_local = f"{cmd_speaker} ({cmd_gender})" if cmd_speaker else None

                print(f"[DEBUG LISTEN]🎯 Final Vosk command output (len={len(result)}): {result}")

                return {
                    "text": result,
                    "speaker": combo_name_local,
                    "score": cmd_score,
                    "gender": cmd_gender,
                    "gender_conf": cmd_gender_conf
                }

    def wake_and_listen_whisper(self,
                                min_words: int = 2,
                                max_words: int = 5,
                                vad_mode: int = 3,
                                silence_duration_ms: int = 1500,
                                max_record_duration_s: int = 15):
        """
        One-shot wake-and-command listener (Whisper):
        - Listen for a wake phrase (single VAD capture + one Whisper transcription).
        - If a wake word is detected, say "Yes, <speaker>" then record one command utterance
          (single VAD capture) and transcribe it once.
        - Clean/dedupe result, run speaker+gender on the same recorded command audio,
          and return a dict or None if nothing useful captured.

        Improvements:
        - Allows user to say stop phrases ("ok", "thank you", "thanks", "that's all") to end the command.
        - Allows up to 2 extra words after stop phrase (stripped off).
        - Also stops capture if silence of 5 seconds occurs during command capture.
        """
        stop_phrases = [r"\bok\b", r"\bthank you\b", r"\bthanks\b", r"\bthat's all\b", r"\bthats all\b"]
        stop_re = re.compile("|".join(stop_phrases), flags=re.IGNORECASE)
        trailing_strip_re = re.compile(r"(?:\b(ok|thank you|thanks|that's all|thats all)\b(?:\s+\S+){0,2})\s*$", flags=re.IGNORECASE)

        def vad_record(max_s=None):
            return self.record_until_silence(
                frame_duration_ms=30,
                vad_mode=vad_mode,
                silence_duration_ms=silence_duration_ms,
                max_record_duration_s=(max_s if max_s is not None else max_record_duration_s)
            )

        print("[DEBUG LISTEN]🎤 Waiting for wake phrase (one-shot mode).")
        while True:
            # --- Wake phrase capture (single VAD capture + single Whisper transcribe) ---
            try:
                wake_audio, _ = vad_record()
            except TypeError:
                try:
                    wake_audio, _ = vad_record()
                except Exception as e:
                    print("[DEBUG LISTEN]❌ record_until_silence (wake) failed:", e)
                    continue
            except Exception as e:
                print("[DEBUG LISTEN]❌ record_until_silence (wake) error:", e)
                continue

            if wake_audio is None or getattr(wake_audio, "size", 0) == 0:
                print("[DEBUG LISTEN]⚠️ No audio captured for wake-word check.")
                continue

            tmp_wav = None
            try:
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tmp_wav.name, wake_audio, self.samplerate)
            except Exception as e:
                print("[DEBUG LISTEN]❌ Could not write temp wake WAV:", e)
                try:
                    if tmp_wav:
                        os.unlink(tmp_wav.name)
                except Exception:
                    pass
                continue

            # optional speaker ID from wake phrase
            precomputed_mfcc = None
            speaker_name_local = None
            try:
                res = identify_safely_from_file(tmp_wav.name)
                if isinstance(res, (tuple, list)):
                    if len(res) >= 5:
                        speaker_name_local, speaker_score, gender_str, gender_conf, precomputed_mfcc = res[:5]
                    elif len(res) == 4:
                        speaker_name_local, speaker_score, gender_str, gender_conf = res
                    else:
                        speaker_name_local = res[0] if len(res) > 0 else None
            except Exception as e:
                print("[DEBUG LISTEN]⚠️ Wake speaker ID error (non-fatal):", e)

            # --- Transcribe wake phrase once ---
            try:
                result = self.whisper_model.transcribe(tmp_wav.name, language="en", fp16=False)
            except Exception as e:
                print(f"[DEBUG LISTEN]❌ Whisper transcription error (wake check): {e}")
                try:
                    os.unlink(tmp_wav.name)
                except Exception:
                    pass
                continue

            # remove temp wake file
            try:
                if tmp_wav:
                    os.unlink(tmp_wav.name)
            except Exception:
                pass
            tmp_wav = None

            raw_text = result.get("text", "").strip().lower()
            print(f"[DEBUG LISTEN]🛎️ Wake-word check: {raw_text!r}")
            if not raw_text:
                continue

            words = raw_text.split()
            if not (min_words <= len(words) <= max_words):
                continue

            clean_text = self.remove_extreme_repetition(self.remove_duplicate_sentences(raw_text)).lower().strip()
            clean_text = re.sub(r'[.!?;:]', '', clean_text)

            print(f"[DEBUG LISTEN]🧽 Cleaned wake phrase: {clean_text!r}")

            if not any(word in clean_text for word in self.wake_words):
                continue

            try:
                speech.AlfredSpeak(f"Yes, {speaker_name_local or 'sir'}")
            except Exception:
                pass

            time.sleep(1)
            print("[DEBUG LISTEN]🔔 Wake word detected. Recording single-shot command...")

            # Now capture command in chunks (similar to your previous flow but with stop-phrase & silence timeout)
            cumulative_text = ""
            total_pcm = bytearray()
            final_speaker = None
            final_score = None
            cmd_gender_str = None
            cmd_gender_conf = None
            start_time = time.time()
            last_chunk_time = time.time()
            max_total_seconds = 30
            max_chunks = 6
            chunks = 0
            stop_detected = False

            while True:
                # break by overall timeout
                if time.time() - start_time > max_total_seconds:
                    print("[DEBUG LISTEN]⚠️ Command capture timeout reached.")
                    break
                # break by silence of 5 seconds since last chunk
                if (time.time() - last_chunk_time) > 5:
                    print("[DEBUG LISTEN]⚠️ Silence timeout (5s) reached during command capture.")
                    break
                if chunks >= max_chunks:
                    print("[DEBUG LISTEN]⚠️ Max command chunks reached.")
                    break

                try:
                    cmd_audio, stopped_by_silence = vad_record(max_s=min(10, max_record_duration_s))
                except Exception as e:
                    print("[DEBUG LISTEN]❌ record_until_silence (command) error:", e)
                    break

                if cmd_audio is None or getattr(cmd_audio, "size", 0) == 0:
                    # no audio this iteration; small sleep to allow silence timeout to trigger
                    time.sleep(0.05)
                    continue

                last_chunk_time = time.time()
                chunks += 1

                try:
                    cmd_f32 = cmd_audio.astype(np.float32)
                    rms = float(np.sqrt(np.mean(np.square(cmd_f32))))
                except Exception:
                    rms = 0.0

                if rms < 1e-4:
                    print(f"[DEBUG LISTEN]🔇 Command audio RMS too low ({rms:.6f}), ignoring chunk.")
                    continue

                # save pcm for speaker/gender later
                try:
                    cmd_int16 = (cmd_f32 * 32768.0).astype(np.int16)
                    total_pcm.extend(cmd_int16.tobytes())
                except Exception:
                    pass

                # transcribe this chunk quickly with Whisper (small one-shot)
                tmp_chunk_wav = None
                chunk_text = ""
                try:
                    tmp_chunk_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(tmp_chunk_wav.name, cmd_f32, self.samplerate)
                    res_chunk = self.whisper_model.transcribe(tmp_chunk_wav.name, language="en", fp16=False)
                    chunk_text = res_chunk.get("text", "").strip()
                except Exception as e:
                    print("[DEBUG LISTEN]❌ Whisper chunk transcription error (non-fatal):", e)
                finally:
                    try:
                        if tmp_chunk_wav:
                            os.unlink(tmp_chunk_wav.name)
                    except Exception:
                        pass

                if chunk_text:
                    print(f"[DEBUG LISTEN]📝 Chunk recognized: {chunk_text!r}")
                    if cumulative_text:
                        cumulative_text += " " + chunk_text
                    else:
                        cumulative_text = chunk_text

                # speaker/gender quick attempt from this chunk if not available yet
                if total_pcm and final_speaker is None:
                    try:
                        tmp_w = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        combined_int16 = np.frombuffer(bytes(total_pcm), dtype=np.int16)
                        combined_float = combined_int16.astype(np.float32) / 32768.0
                        sf.write(tmp_w.name, combined_float, self.samplerate)
                        res_id = identify_safely_from_file(tmp_w.name)
                        if isinstance(res_id, (tuple, list)):
                            if len(res_id) >= 4:
                                final_speaker, final_score, cmd_gender_str, cmd_gender_conf = res_id[:4]
                            elif len(res_id) == 3:
                                final_speaker, final_score, cmd_gender_str = res_id
                            elif len(res_id) == 1:
                                final_speaker = res_id[0]
                        try:
                            os.unlink(tmp_w.name)
                        except Exception:
                            pass
                    except Exception as e:
                        print("[DEBUG LISTEN]❌ Speaker ID (chunk) error (non-fatal):", e)

                # look for stop phrase in latest chunk or cumulative text
                if stop_re.search(chunk_text) or stop_re.search(cumulative_text):
                    print("[DEBUG LISTEN]🔚 Stop phrase detected; ending capture.")
                    stop_detected = True
                    # allow loop to continue to let silence timeout or break immediately (we break now)
                    break

                # if chunk indicated silence completed, optionally break
                if stopped_by_silence:
                    print("[DEBUG LISTEN]🔕 Chunk ended by silence; checking whether to continue or finish.")
                    # if we have some captured text, allow finalization
                    if cumulative_text:
                        break
                    else:
                        continue

                # if total cumulative too long, stop
                if len(cumulative_text) >= 150:
                    cumulative_text = cumulative_text[:150]
                    print("[DEBUG LISTEN]🔒 Reached 150-character cap; stopping capture.")
                    break

            # If nothing captured
            if not cumulative_text:
                print("[DEBUG LISTEN]⚠️ No transcribed command after capture.")
                return None

            # Strip trailing stop-phrase plus up to 2 extra words
            cleaned_cmd = cumulative_text.strip()
            cleaned_cmd = trailing_strip_re.sub("", cleaned_cmd).strip()

            final_result = self._process_transcription_text(cleaned_cmd)
            if not final_result:
                return None

            combo_name_local = final_speaker or speaker_name_local
            if combo_name_local and cmd_gender_str:
                combo_name_local = f"{combo_name_local} ({cmd_gender_str})"
            elif combo_name_local:
                combo_name_local = f"{combo_name_local}"

            print(f"[DEBUG LISTEN]🎯 Final command output (len={len(final_result)}): {final_result!r}")

            return {
                "text": final_result,
                "speaker": combo_name_local,
                "score": final_score,
                "gender": cmd_gender_str,
                "gender_conf": cmd_gender_conf
            }

    # Main listen entrypoint used by continuous worker or callers
    def listen(self):
        try:
            text = self.text_queue.get_nowait()
            print(f"[DEBUG LISTEN]\n Text received to LISTEN is {text}")
            if text:
                return self.listen_text(text)
        except queue.Empty:
            pass

        if not self.wake_word_on_off:
            bt = self.listen_bluetooth()
            if bt:
                return bt

            if self.stop_listening:
                return None

            if self.use_whisper_listen:
                return self.listen_whisper()
            else:
                return self.listen_vosk()
        else:
            # wake-word flows (Whisper wake) - default to whisper wake flow
            return self.wake_and_listen_whisper()


# Module-level worker used by start_continuous_listening
def continuous_listen_worker(use_whisper, conn):
    listener = ListenModule()
    listener.set_listen_whisper(use_whisper)
    listener.set_listen_vosk(not use_whisper)
    while not listener.stop_listening:
        try:
            result = listener.listen()
            if result:
                conn.send(result)
        except Exception as e:
            try:
                conn.send(f"❌ Error: {e}")
            except Exception:
                pass


# Singleton instance
listen = ListenModule()

print("[DEBUG LISTEN]speech module initialization end")
