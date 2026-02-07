

# GUI.py — drop-in replacement (rewritten to fix poll_speech_state crash + safer TTS state handling)
import sys
import os
import signal
import datetime
import threading
import queue
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
import tkinter.ttk as ttk
from tkinter import messagebox
import cv2
import ssl
import socketio, requests, ssl, urllib3, getpass
import requests
import urllib3
import time
from collections import deque   # << added
import re, textwrap

###
import os
import io
import time
import uuid
import shutil
import base64
import mimetypes
import traceback
import getpass

from tkinter import filedialog, messagebox
import chardet
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# embeddings & vector DB libs (optional fallbacks)
try:
    import ollama
except Exception:
    ollama = None

# chromadb may not be installed in all envs; if not, user will need to install it.
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except Exception:
    chromadb = None

# fallback embedding model
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# keep per-user raw text/chunks as a fallback (used by "this file" queries)
USER_FILES = {}   # mapping username -> {"text": str, "chunks": [...], "uploaded_ids": [...], "embeddings": [...]}





def _chunk_text(text, max_tokens=5000, max_chars=4000):
    """
    Simple character-based chunker. Returns list of text chunks.
    - max_chars default covers long docs — tune as needed for your embed model.
    """
    if not text:
        return []

    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        # try to cut at newline or sentence boundary for nicer chunks
        if end < L:
            # find last newline before end
            nl = text.rfind("\n", start, end)
            if nl > start + 200:  # not too small
                end = nl
            else:
                # fallback to last space
                sp = text.rfind(" ", start, end)
                if sp > start + 50:
                    end = sp
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _get_embeddings_for_chunks(chunks):
    """
    Robust embedding helper tries:
      1) ollama.embeddings (if ollama module available)
      2) sentence-transformers (all-MiniLM-L6-v2)
      3) openai (if OPENAI_API_KEY is set)
    Returns: (embeddings_list, source_str)
    Raises RuntimeError with a clear message if all fail.
    """
    if not chunks:
        return [], 'none'

    # 1) Ollama
    try:
        import ollama
        embeddings = []
        for c in chunks:
            try:
                # try common signatures
                resp = None
                try:
                    resp = ollama.embeddings(model="mxbai-embed-large", input=c)
                except Exception:
                    resp = ollama.embeddings("mxbai-embed-large", c)
                # extract vector
                vec = None
                if isinstance(resp, dict):
                    vec = resp.get('embedding') or resp.get('embeddings') or (resp.get('data') and resp['data'][0].get('embedding'))
                if vec is None and isinstance(resp, (list, tuple)) and len(resp) > 0:
                    vec = resp[0]
                if vec is None:
                    raise RuntimeError("No vector in ollama response")
                embeddings.append(list(vec))
            except Exception as e:
                raise RuntimeError(f"Ollama embedding failed for a chunk: {e}")
        return embeddings, 'ollama-mxbai'
    except Exception as e:
        # don't hard-fail; record and try next
        ollama_err = str(e)

    # 2) sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')  # auto-downloads model on first run
        vects = model.encode(chunks, show_progress_bar=False)
        embeddings = [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vects]
        return embeddings, 'sentence-transformers'
    except Exception as e:
        st_err = str(e)

    # 3) OpenAI (if API key present)
    try:
        import openai
        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
        if key:
            openai.api_key = key
            embeddings = []
            for c in chunks:
                resp = openai.Embedding.create(model="text-embedding-3-small", input=c)
                vec = resp['data'][0]['embedding']
                embeddings.append(vec)
            return embeddings, 'openai-text-embedding-3-small'
    except Exception as e:
        openai_err = str(e)

    # If we get here, all providers failed — raise helpful error with details
    msg = (
        "No embedding provider available. Tried:\n"
        f" - ollama: {ollama_err if 'ollama_err' in locals() else 'not attempted'}\n"
        f" - sentence-transformers: {st_err if 'st_err' in locals() else 'not attempted'}\n"
        f" - openai: {openai_err if 'openai_err' in locals() else 'not attempted'}\n\n"
        "Resolve by either:\n"
        "  * Installing sentence-transformers: pip install sentence-transformers\n"
        "  * Installing/starting Ollama and ensuring mxbai-embed-large is available\n"
        "  * Setting OPENAI_API_KEY and installing openai: pip install openai\n"
    )
    raise RuntimeError(msg)






##def _chunk_text(text, max_tokens=5000, max_chars=4000):
##    """
##    Simple character-based chunker. Returns list of text chunks.
##    - max_chars default covers long docs — tune as needed for your embed model.
##    """
##    if not text:
##        return []
##
##    chunks = []
##    start = 0
##    L = len(text)
##    while start < L:
##        end = start + max_chars
##        # try to cut at newline or sentence boundary for nicer chunks
##        if end < L:
##            # find last newline before end
##            nl = text.rfind("\n", start, end)
##            if nl > start + 200:  # not too small
##                end = nl
##            else:
##                # fallback to last space
##                sp = text.rfind(" ", start, end)
##                if sp > start + 50:
##                    end = sp
##        chunks.append(text[start:end].strip())
##        start = end
##    return [c for c in chunks if c]
##
##
##def _get_embeddings_for_chunks(chunks):
##    """
##    Robust embedding helper tries:
##      1) ollama.embeddings (if ollama module available)
##      2) sentence-transformers (all-MiniLM-L6-v2)
##      3) openai (if OPENAI_API_KEY is set)
##    Returns: (embeddings_list, source_str)
##    Raises RuntimeError with a clear message if all fail.
##    """
##    if not chunks:
##        return [], 'none'
##
##    # 1) Ollama
##    try:
##        import ollama
##        embeddings = []
##        for c in chunks:
##            try:
##                # try common signatures
##                resp = None
##                try:
##                    resp = ollama.embeddings(model="mxbai-embed-large", input=c)
##                except Exception:
##                    resp = ollama.embeddings("mxbai-embed-large", c)
##                # extract vector
##                vec = None
##                if isinstance(resp, dict):
##                    vec = resp.get('embedding') or resp.get('embeddings') or (resp.get('data') and resp['data'][0].get('embedding'))
##                if vec is None and isinstance(resp, (list, tuple)) and len(resp) > 0:
##                    vec = resp[0]
##                if vec is None:
##                    raise RuntimeError("No vector in ollama response")
##                embeddings.append(list(vec))
##            except Exception as e:
##                raise RuntimeError(f"Ollama embedding failed for a chunk: {e}")
##        return embeddings, 'ollama-mxbai'
##    except Exception as e:
##        # don't hard-fail; record and try next
##        ollama_err = str(e)
##
##    # 2) sentence-transformers
##    try:
##        from sentence_transformers import SentenceTransformer
##        model = SentenceTransformer('all-MiniLM-L6-v2')  # auto-downloads model on first run
##        vects = model.encode(chunks, show_progress_bar=False)
##        embeddings = [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vects]
##        return embeddings, 'sentence-transformers'
##    except Exception as e:
##        st_err = str(e)
##
##    # 3) OpenAI (if API key present)
##    try:
##        import openai
##        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
##        if key:
##            openai.api_key = key
##            embeddings = []
##            for c in chunks:
##                resp = openai.Embedding.create(model="text-embedding-3-small", input=c)
##                vec = resp['data'][0]['embedding']
##                embeddings.append(vec)
##            return embeddings, 'openai-text-embedding-3-small'
##    except Exception as e:
##        openai_err = str(e)
##
##    # If we get here, all providers failed — raise helpful error with details
##    msg = (
##        "No embedding provider available. Tried:\n"
##        f" - ollama: {ollama_err if 'ollama_err' in locals() else 'not attempted'}\n"
##        f" - sentence-transformers: {st_err if 'st_err' in locals() else 'not attempted'}\n"
##        f" - openai: {openai_err if 'openai_err' in locals() else 'not attempted'}\n\n"
##        "Resolve by either:\n"
##        "  * Installing sentence-transformers: pip install sentence-transformers\n"
##        "  * Installing/starting Ollama and ensuring mxbai-embed-large is available\n"
##        "  * Setting OPENAI_API_KEY and installing openai: pip install openai\n"
##    )
##    raise RuntimeError(msg)



# Disable warnings about self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Requests session (no SSL verify)
session = requests.Session()
session.verify = False

# Socket.IO clients
sio_mobile = socketio.Client(http_session=session, reconnection=True)
try:
    sio_mobile.connect('https://localhost:5000')
except Exception as e:
    print("Could not connect sio_mobile:", e)
sio = socketio.Client()
try:
    sio.connect('http://localhost:5001')
except Exception as e:
    print("Could not connect sio:", e)

# Add modules dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))
from speech import speech
from arduino_com import arduino_com
from memory import memory
from Repeat_Last import repeat
from listen import listen

from model_selector import get_models_by_type

My_Selected_Thinkbot_model_list = get_models_by_type("thinkbot")
print(f"My_Selected_Thinkbot_model_list : {My_Selected_Thinkbot_model_list}")

My_Selected_Chatbot_model_list = get_models_by_type("chatbot")
print(f"My_Selected_Chatbot_model_list : {My_Selected_Chatbot_model_list}")

My_Selected_Visionbot_model_list = get_models_by_type("vision")
print(f"My_Selected_Visionbot_model_list : {My_Selected_Visionbot_model_list}")

My_Selected_Codingbot_model_list = get_models_by_type("coding")
print(f"My_Selected_Codingbot_model_list : {My_Selected_Codingbot_model_list}")


def strip_markdown(md: str) -> str:
    if not md:
        return ""
    text = md
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'\n+', '\n', text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


##self.model_used = ""
user_model_used = None

import os
import sys
import ctypes
from ctypes import wintypes
import subprocess
import platform
import shutdown_helpers as sh

# ----------------------------
# Helpers to close Windows console windows (safe & fallback)
# ----------------------------

def _close_console_windows_via_enum():
    """
    Enumerate top-level windows, find windows with class "ConsoleWindowClass"
    and send WM_CLOSE. This closes visible console windows gracefully.
    Windows-only.
    """
    if platform.system() != "Windows":
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    GetClassName = user32.GetClassNameW
    GetWindowText = user32.GetWindowTextW
    IsWindowVisible = user32.IsWindowVisible
    PostMessage = user32.PostMessageW

    WM_CLOSE = 0x0010
    buf = ctypes.create_unicode_buffer(512)

    closed_any = False

    def _cb(hwnd, lParam):
        try:
            if not IsWindowVisible(hwnd):
                return True  # continue
            # class name
            buf.value = "\0" * 512
            GetClassName(hwnd, buf, 512)
            cls = buf.value
            if not cls:
                return True
            if cls == "ConsoleWindowClass":
                # optionally get title
                tbuf = ctypes.create_unicode_buffer(512)
                GetWindowText(hwnd, tbuf, 512)
                title = tbuf.value
                # send close
                PostMessage(hwnd, WM_CLOSE, 0, 0)
                nonlocal closed_any  # Python 3.8+; if older, remove and rely on outer variable via list
                closed_any = True
        except Exception:
            pass
        return True

    # work around for Python versions without nonlocal in nested callbacks: use a container
    closed_container = {"closed": False}
    def _cb2(hwnd, lParam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            # class name
            buf.value = "\0" * 512
            GetClassName(hwnd, buf, 512)
            cls = buf.value
            if cls == "ConsoleWindowClass":
                tbuf = ctypes.create_unicode_buffer(512)
                GetWindowText(hwnd, tbuf, 512)
                # close gracefully
                PostMessage(hwnd, WM_CLOSE, 0, 0)
                closed_container["closed"] = True
        except Exception:
            pass
        return True

    # call enum with _cb2 (more robust for various Python versions)
    EnumWindows(EnumWindowsProc(_cb2), 0)
    return bool(closed_container["closed"])


def _kill_all_cmd_windows_force():
    """
    Aggressive fallback: use taskkill to forcibly kill cmd.exe processes.
    WARNING: this kills ALL cmd.exe processes system-wide.
    Windows-only.
    """
    if platform.system() != "Windows":
        return False
    try:
        # /F = force, /IM = image name
        subprocess.run(["taskkill", "/F", "/IM", "cmd.exe"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def close_all_console_windows_graceful_or_force():
    """
    Try graceful close first (WM_CLOSE). If nothing was closed and you explicitly
    want, fall back to taskkill to ensure windows are closed.
    """
    if platform.system() != "Windows":
        return False

    try:
        closed = _close_console_windows_via_enum()
        if closed:
            return True
    except Exception:
        pass

    # nothing closed by graceful method -> fallback to taskkill
    try:
        _kill_all_cmd_windows_force()
        return True
    except Exception:
        return False

def get_chroma_client(persist_directory=None):
    """
    Version-tolerant chroma client creator that disables telemetry / monkeypatches capture
    when telemetry APIs have incompatible signatures.
    Returns (client, used_path) or (None, path) on failure.
    """
    import os, traceback
    # Ensure telemetry is disabled at process level before chromadb import
    os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
    CHROMA_PATH = persist_directory or os.path.join(os.getcwd(), "chroma_db")

    try:
        import chromadb
    except Exception as e:
        print(f"[get_chroma_client] chromadb import failed: {e}")
        return None, CHROMA_PATH

    # defensive: try to turn telemetry off on module if attribute exists
    try:
        # common boolean flag
        if hasattr(chromadb, "telemetry_enabled"):
            try:
                chromadb.telemetry_enabled = False
            except Exception:
                pass

        # some versions expose a telemetry module/object
        telemetry = getattr(chromadb, "telemetry", None)
        if telemetry is not None:
            # try a few possible attributes and replace them with safe no-ops
            for name in ("capture", "capture_event", "send_event", "capture_exception"):
                if hasattr(telemetry, name):
                    try:
                        setattr(telemetry, name, lambda *a, **kw: None)
                        print(f"[get_chroma_client] Patched chromadb.telemetry.{name} -> no-op")
                    except Exception:
                        # if we can't set attribute, ignore
                        pass
            # also try top-level convenience
            try:
                telemetry_enabled_attr = getattr(telemetry, "enabled", None)
                if telemetry_enabled_attr is not None:
                    try:
                        telemetry.enabled = False
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        print("[get_chroma_client] telemetry monkeypatch failed:\n", traceback.format_exc())

    # Try to construct client using new PersistentClient first
    try:
        PersistentClient = getattr(chromadb, "PersistentClient", None)
        if persist_directory and PersistentClient is not None:
            try:
                client = PersistentClient(path=CHROMA_PATH)
                print(f"[get_chroma_client] Using chromadb.PersistentClient(path={CHROMA_PATH})")
                globals()['chromadb'] = chromadb
                return client, CHROMA_PATH
            except Exception as e:
                print(f"[get_chroma_client] PersistentClient init failed: {e}")

        # Try high-level Client() call (works for many versions)
        try:
            client = chromadb.Client()
            print("[get_chroma_client] Using chromadb.Client() fallback")
            globals()['chromadb'] = chromadb
            return client, CHROMA_PATH
        except Exception as e:
            print(f"[get_chroma_client] chromadb.Client() failed: {e}")

        # Legacy Settings() fallback
        try:
            from chromadb.config import Settings as ChromaSettings
            chroma_settings = ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH)
            client = chromadb.Client(chroma_settings)
            print(f"[get_chroma_client] Using legacy chromadb.Client(Settings) at {CHROMA_PATH}")
            globals()['chromadb'] = chromadb
            return client, CHROMA_PATH
        except Exception as e:
            print(f"[get_chroma_client] legacy Client(Settings) fallback failed: {e}")

    except Exception:
        print("[get_chroma_client] Unexpected error creating chroma client:\n", traceback.format_exc())

    return None, CHROMA_PATH

#####


import os
import shutil
import time
import gc
import platform
import ctypes
import ctypes.wintypes
import traceback

def _schedule_delete_on_reboot(path):
    """Schedule a path for deletion on next reboot (Windows). Returns True if scheduled."""
    try:
        if platform.system().lower() != "windows":
            return False
        MoveFileEx = ctypes.windll.kernel32.MoveFileExW
        MoveFileEx.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD]
        MoveFileEx.restype = ctypes.wintypes.BOOL
        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4
        ok = MoveFileEx(path, None, MOVEFILE_DELAY_UNTIL_REBOOT)
        return bool(ok)
    except Exception:
        return False

def _try_close_chroma_client(client, logger_fn=None):
    """
    Best-effort close/persist/shutdown for chroma client.
    logger_fn(msg) optional for logging (e.g. self.log_message).
    """
    def _log(msg):
        try:
            if logger_fn:
                logger_fn(msg)
            else:
                print(msg)
        except Exception:
            pass

    if client is None:
        return

    methods = ("persist", "flush", "close", "shutdown", "stop", "terminate", "disconnect")
    for m in methods:
        try:
            fn = getattr(client, m, None)
            if callable(fn):
                try:
                    fn()
                    _log(f"[RAG] Called client.{m}()")
                except Exception as e:
                    _log(f"[RAG] client.{m}() -> {e}")
        except Exception:
            pass

    # Try nested attributes that might be holding the resource
    for attr in ("_client", "client", "_server", "server", "connection", "engine"):
        try:
            nested = getattr(client, attr, None)
            if nested is None:
                continue
            for m in methods:
                try:
                    fn = getattr(nested, m, None)
                    if callable(fn):
                        try:
                            fn()
                            _log(f"[RAG] Called client.{attr}.{m}()")
                        except Exception as e:
                            _log(f"[RAG] client.{attr}.{m}() -> {e}")
                except Exception:
                    pass
            # try direct shutdown if present
            try:
                if hasattr(nested, "shutdown") and callable(nested.shutdown):
                    try:
                        nested.shutdown()
                        _log(f"[RAG] Called {attr}.shutdown()")
                    except Exception as e:
                        _log(f"[RAG] {attr}.shutdown() -> {e}")
            except Exception:
                pass
        except Exception:
            pass

    # force GC and give OS a moment to release handles
    try:
        gc.collect()
        time.sleep(0.15)
    except Exception:
        pass

def _list_locking_processes(path):
    """
    If psutil is available, return a list of (pid, name) that have handles open to `path`.
    If psutil not available or no info, returns [].
    """
    try:
        import psutil
    except Exception:
        return []
    lockers = []
    try:
        for p in psutil.process_iter(["pid", "name", "open_files"]):
            try:
                of = p.info.get("open_files") or []
                for f in of:
                    if f and os.path.abspath(f.path) == os.path.abspath(path):
                        lockers.append((p.info["pid"], p.info["name"]))
                        break
            except Exception:
                continue
    except Exception:
        pass
    return lockers


class gui:

    def __init__(self):
        # ---- Root window ----
        self.root = tk.Tk()
        self.root.title("Alfred Voice Assistant GUI")
        self.root.geometry("800x900")
        self.root.configure(bg="lightblue")

        # Make root expand its single child (main_frame)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # ---- Main frame (grid container) ----
        self.main_frame = tk.Frame(self.root, bg="lightblue")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        # Allow 6 rows to expand: header, dropdowns, middle, input, log, responses

        # Configure row weights
        weights = [0, 0, 3, 0, 2, 3]
        for i, w in enumerate(weights):
            self.main_frame.rowconfigure(i, weight=w)
        self.main_frame.columnconfigure(0, weight=1)

        # ---- Internal state ----
        self.log_queue   = queue.Queue()
        self.query_queue = queue.Queue()

        self.user_deque     = deque()   # holds usernames in query order
        self.response_deque = deque()   # holds responses in arrival order
        self.user_queue  = queue.Queue()
        self.response_queue  = queue.Queue()

        self.muted = False

        # model vars
        self.thinkbot_model = tk.StringVar(value=My_Selected_Thinkbot_model_list[13])
        self.chatbot_model  = tk.StringVar(value=My_Selected_Chatbot_model_list[10])
        self.vision_model   = tk.StringVar(value=My_Selected_Visionbot_model_list[16])
        self.coding_model   = tk.StringVar(value=My_Selected_Codingbot_model_list[6])
        self.model_used = None
        self.last_model_used_by_user = {}

        self.current_user = getpass.getuser().capitalize()
        self.last_query = ""
        self.models = {
            "thinkbot": self.thinkbot_model.get(),
            "chatbot": self.chatbot_model.get(),
            "vision": self.vision_model.get(),
            "coding": self.coding_model.get()
        }

        # ---- Initial WebUI events ----
        try:
            sio.emit('gui_event', {'type':'log','payload':"System booted", 'username':self.current_user})
            sio_mobile.emit('gui_event', {'type':'log','payload':"System booted", 'username':self.current_user})
            sio.emit('gui_event', {'type':'setting','payload':{'use_whisper':True}, 'username':self.current_user})
            sio_mobile.emit('gui_event', {'type':'setting','payload':{'use_whisper':True}, 'username':self.current_user})
        except Exception:
            pass

        # ---- Speech settings vars ----
        self.use_bluetooth_speech = tk.BooleanVar(value=False)
        self.use_vosk             = tk.BooleanVar(value=True)
        self.enter_submits        = tk.BooleanVar(value=True)
        self.use_whisper          = tk.BooleanVar(value=False)
        self.manual_record_only   = tk.BooleanVar(value=False)
        self.wake_word_call       = tk.BooleanVar(value=False)

        self.use_whisper.trace_add('write', lambda *a: self.toggle_whisper_listen())
        self.use_vosk.trace_add('write',     lambda *a: self.toggle_vosk_listen())
        self.manual_record_only.trace_add('write', lambda *a: self.toggle_manual_record_only())
        self.wake_word_call.trace_add('write', lambda *a: self.toggle_wake_word())
        self.use_bluetooth_speech.trace_add('write', lambda *a: self.toggle_bluetooth_speech())

        # ---- Build UI (these create self.input_text etc.) ----
        self.setup_menu()
        self.setup_header()
        self.setup_dropdowns()
        self.setup_middle()
        self.setup_input_panel()   # IMPORTANT: creates self.input_text
        self.setup_log()

        # ---- Initialize speech-state attributes to avoid poll errors ----
        self._prev_playing = False
        self._prev_paused = False
        self._prev_suppress = False

        # ---- Robust focus routine so input_text receives keyboard focus automatically ----
        # We'll try multiple times because some window managers prevent immediate focus stealing.
        def _try_focus_attempts(max_attempts=12, delay_ms=120):
            attempts = {'n': 0}  # closure mutable

            def _attempt():
                try:
                    attempts['n'] += 1
                    # raise window and attempt to force focus to input_text
                    try:
                        self.root.lift()
                    except Exception:
                        pass
                    try:
                        # Temporarily set topmost so focus_force is more reliable
                        self.root.attributes('-topmost', True)
                    except Exception:
                        pass
                    try:
                        self.root.focus_force()
                    except Exception:
                        pass
                    try:
                        # Prefer focus_force then focus_set on the input widget
                        self.input_text.focus_force()
                        self.input_text.focus_set()
                        # ensure insertion cursor is visible in the widget
                        try:
                            self.input_text.mark_set("insert", "1.0")
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # If focus succeeded, clear topmost and stop
                    try:
                        cur = self.root.focus_get()
                        if cur is self.input_text:
                            try:
                                self.root.attributes('-topmost', False)
                            except Exception:
                                pass
                            return  # success
                    except Exception:
                        pass

                    # not focused yet — if we haven't exceeded attempts, try again
                    if attempts['n'] < max_attempts:
                        self.root.after(delay_ms, _attempt)
                    else:
                        # final attempt to clear topmost if set
                        try:
                            self.root.attributes('-topmost', False)
                        except Exception:
                            pass
                except Exception as e:
                    # ignore failures — don't raise during init
                    print("focus attempt error:", e)

            # schedule 1st attempt at idle (after window maps)
            try:
                self.root.after_idle(_attempt)
            except Exception:
                # fallback scheduling
                self.root.after(50, _attempt)

            # also bind events to try again if the window is mapped/visible later
            try:
                self.root.bind("<Map>", lambda e: self.root.after(50, _attempt))
                self.root.bind("<Visibility>", lambda e: self.root.after(50, _attempt))
            except Exception:
                pass

        # Kick off the focus attempts (input_text exists by now)
        try:
            _try_focus_attempts()
        except Exception as e:
            print("Failed to start focus attempts:", e)

        # ---- Focus, polling & threads ----
        try:
            self.root.after(50, lambda: self.input_text.focus_set())
        except Exception:
            pass

        self.update_log_widget()
        self.update_clock()

        # start polling speech state (this will run periodically)
        try:
            self.poll_speech_state()
        except Exception as e:
            print("poll_speech_state startup failed:", e)

        # Record button wiring and initial states
        try:
            self.record_button.config(text="🎙 Hold to Record ")
            self.record_button.bind("<ButtonPress-1>",   self._on_record_press)
            self.record_button.bind("<ButtonRelease-1>", self._on_record_release)
        except Exception:
            pass
        self._recording = False

        # Start webui listener thread
        threading.Thread(target=self._start_webui_listener, daemon=True).start()


    def setup_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Load File",                command=self.load_file)
        fm.add_command(label="Load Main Responses",      command=self.load_main_responses)
        fm.add_command(label="Load Assistant Responses", command=self.load_assistant_responses)
        fm.add_separator()
        fm.add_command(label="Exit",                     command=self.exit_program)
        mb.add_cascade(label="File", menu=fm)

    def setup_header(self):
        f = tk.Frame(self.main_frame, bg="lightblue")
        f.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.clock_label = tk.Label(f, text="", font=("Helvetica",12), bg="lightblue", fg="navy")
        self.clock_label.pack(side=tk.LEFT, padx=5)
        self.llm_status_label = tk.Label(f, text="LLM: Idle", font=("Helvetica",12), bg="lightblue", fg="darkgreen")
        self.llm_status_label.pack(side=tk.RIGHT, padx=5)

    def setup_dropdowns(self):
        f = tk.Frame(self.main_frame, bg="lightblue")
        f.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        f.columnconfigure(0, weight=1)
        self.dropdown_frame = f
        self.thinkbot_model = tk.StringVar(value=My_Selected_Thinkbot_model_list[13])
        self.chatbot_model  = tk.StringVar(value=My_Selected_Chatbot_model_list[10])
        self.vision_model   = tk.StringVar(value=My_Selected_Visionbot_model_list[16])
        self.coding_model   = tk.StringVar(value=My_Selected_Codingbot_model_list[6])
        self.dropdown_items = []
        for label_text, var, opts in [
            ("Thinkbot AI Model:", self.thinkbot_model, My_Selected_Thinkbot_model_list),
            ("Chat AI Model:",    self.chatbot_model,  My_Selected_Chatbot_model_list),
            ("Vision AI Model:",  self.vision_model,   My_Selected_Visionbot_model_list),
            ("Coding Model:",     self.coding_model,   My_Selected_Codingbot_model_list),
        ]:
            item = tk.Frame(f, bg="lightblue")
            tk.Label(item, text=label_text, font=("Helvetica",10), bg="lightblue").pack(side=tk.LEFT)
            cb = ttk.Combobox(item, textvariable=var, values=opts, state="readonly", font=("Helvetica",10))
            cb.pack(side=tk.LEFT, padx=(2,0))
            var.trace_add("write",
                lambda *a, k=label_text.split()[0].lower()+"_model", v=var:
                    self.emit_dropdown_change(k, v.get())
            )
            self.dropdown_items.append(item)
        self._current_dd_cols = None  
        f.bind("<Configure>", self._reflow_dropdowns)

    def _reflow_dropdowns(self, event):
        f = self.dropdown_frame
        width = event.width
        new_cols = 4 if width >= 800 else 2
        if new_cols == self._current_dd_cols:
            return
        self._current_dd_cols = new_cols
        for w in self.dropdown_items:
            w.grid_forget()
        for idx, item in enumerate(self.dropdown_items):
            row = idx // new_cols
            col = idx %  new_cols
            item.grid(row=row, column=col, sticky="w", padx=5, pady=2)
        for c in range(new_cols):
            f.columnconfigure(c, weight=1)
        for c in range(new_cols, 4):
            f.columnconfigure(c, weight=0)

    def setup_middle(self):
        f = tk.Frame(self.main_frame, bg="lightblue")
        f.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        f.columnconfigure(0, weight=1)

    def setup_input_panel(self):
        f = tk.Frame(self.main_frame, bg="lightblue")
        f.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        f.columnconfigure(0, weight=1)
        self.input_frame = f
        items = [
            ("Bluetooth Output", self.use_bluetooth_speech, self.toggle_bluetooth_speech),
            ("Listen Vosk",  self.use_vosk,             self.toggle_vosk_listen),
            ("Listen Whisper",   self.use_whisper,         self.toggle_whisper_listen),
            ("Manual Record Only",    self.manual_record_only,  self.toggle_manual_record_only),
            ("Wake Word On/Off",    self.wake_word_call,  self.toggle_wake_word),
            ("Enter key sends query",           self.enter_submits,       None),
      ]
        self.check_items = []
        for text, var, cmd in items:
            sub = tk.Frame(f, bg="lightblue")
            cb = tk.Checkbutton(sub, text=text, variable=var, command=cmd,
                                bg="lightblue", font=("Helvetica",10))
            cb.pack(anchor="w")
            self.check_items.append(sub)
        self._make_input_widgets(f)
        self._current_cb_cols = None
        f.bind("<Configure>", self._reflow_checkboxes)




    def _make_input_widgets(self, f):
        tk.Label(f, text="Input Text", font=("Helvetica",12,"bold"), bg="lightblue")\
            .grid(row=99, column=0, columnspan=2, sticky="w", pady=(10,0))

        self.input_text = ScrolledText(f, wrap=tk.WORD, height=4, font=("Helvetica",10))
        self.input_text.grid(row=100, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Cursor auto-focus here on startup
        self.input_text.focus_set()

        # Make the input area expand nicely
        f.rowconfigure(100, weight=1)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        self.input_text.bind("<Return>", self.on_enter_pressed)

        # Button frame — we'll use grid so buttons expand/shrink with the window
        btnf = tk.Frame(f, bg="lightblue")
        btnf.grid(row=101, column=0, columnspan=2, sticky="ew", pady=5, padx=5)

        # Make each column in btnf expand equally
        btnf.columnconfigure(0, weight=1)
        btnf.columnconfigure(1, weight=1)
        btnf.columnconfigure(2, weight=1)
        btnf.columnconfigure(3, weight=1)
        btnf.columnconfigure(4, weight=1)
        btnf.columnconfigure(5, weight=1)
        btnf.columnconfigure(6, weight=1)

        # Smaller, consistent button font
        btn_font = ("Helvetica", 9)

        # Record button — placed in column 0
        self.record_button = tk.Button(
            btnf, text="🎙️ Hold to Speak", command=self._on_record_press,
            bg="darkgreen", font=btn_font
        )
        self.record_button.grid(row=0, column=0, sticky="ew", padx=3)

        # Send text button — column 1
        tk.Button(
            btnf, text="Send Text", command=self.send_text,
            bg="darkblue", fg="white", font=btn_font
        ).grid(row=0, column=1, sticky="ew", padx=3)

        # Listen last — column 2
        tk.Button(
            btnf, text="Listen Last", command=self.listen_last,
            bg="purple", fg="white", font=btn_font
        ).grid(row=0, column=2, sticky="ew", padx=3)

        # Start / Pause / Stop buttons — columns 3,4,5
        self.start_button = tk.Button(
            btnf, text="▶️ Start", command=self._tk_start_speech,
            bg="#2e7d32", fg="white", font=btn_font
        )
        self.start_button.grid(row=0, column=3, sticky="ew", padx=3)

        self.pause_button = tk.Button(
            btnf, text="⏸️ Pause", command=self._tk_pause_speech,
            bg="#f9a825", fg="black", font=btn_font
        )
        self.pause_button.grid(row=0, column=4, sticky="ew", padx=3)

        self.stop_button = tk.Button(
            btnf, text="⏹️ Stop", command=self._tk_stop_speech,
            bg="#c62828", fg="white", font=btn_font
        )
        self.stop_button.grid(row=0, column=5, sticky="ew", padx=3)

        # Optional spacer / filler in column 6 for breathing room (keeps layout tidy)
        tk.Label(btnf, text="", bg="lightblue").grid(row=0, column=6, sticky="ew")

        # Start with Pause/Stop disabled until speech starts
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

        # Bindings for record button press/release (keep behaviour)
        self.record_button.bind("<ButtonPress-1>", self._on_record_press)
        self.record_button.bind("<ButtonRelease-1>", self._on_record_release)

        # Make sure the btnf expands horizontally when the parent resizes
        f.rowconfigure(101, weight=0)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)


        # --- Reset RAG button frame (centered below main buttons) ---
        ragf = tk.Frame(f, bg="lightblue")
        ragf.grid(row=102, column=0, columnspan=2, sticky="ew", pady=(2,10), padx=5)

        # Make left and right columns expand so the middle column stays centered
        ragf.columnconfigure(0, weight=1)
        ragf.columnconfigure(1, weight=0)
        ragf.columnconfigure(2, weight=1)

        # Use the same small button font
        btn_font = ("Helvetica", 9)

        # Centered Reset RAG button in column 1
        reset_rag_btn = tk.Button(
            ragf,
            text="🗑️ Reset RAG",
            command=lambda: self.reset_rag_local(),
            bg="#b71c1c",
            fg="white",
            font=btn_font,
            relief=tk.RAISED
        )
        reset_rag_btn.grid(row=0, column=1, sticky="ew", padx=6)

        # --- Auto-start the speech controller so Start needn't be pressed first ---
        try:
            # schedule a short-delayed call so everything is constructed first
            self.root.after(100, lambda: getattr(speech, 'set_tk_start_speech', lambda: None)())
            self.log_message("Speech controller auto-started at GUI init")
        except Exception as e:
            self.log_message(f"Auto-start of speech controller failed: {e}")

            
    def _reflow_checkboxes(self, event):
        f = self.input_frame
        width = event.width
        new_cols = 2 if width >= 500 else 1
        if new_cols == self._current_cb_cols:
            return
        self._current_cb_cols = new_cols
        for w in self.check_items:
            w.grid_forget()
        for idx, item in enumerate(self.check_items):
            r = idx // new_cols
            c = idx %  new_cols
            item.grid(row=r, column=c, sticky="w", padx=5, pady=2)
        for c in range(new_cols):
            f.columnconfigure(c, weight=1)
        for c in range(new_cols, 2):
            f.columnconfigure(c, weight=0)

    def setup_log(self):
        f = tk.Frame(self.main_frame, bg="lightblue")
        f.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        tk.Label(f, text="Log", font=("Helvetica",12,"bold"), bg="lightblue").pack(anchor="w", padx=5, pady=2)
        self.log_widget = ScrolledText(f, wrap=tk.WORD, font=("Helvetica",10))
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_main_responses(self):
        path = filedialog.askopenfilename(title="Select Main Responses File",
                                          filetypes=(("Text files","*.txt"),("All files","*.*")))
        if path:
            try:
                with open(path,'r',encoding='utf-8') as f: txt=f.read()
                widget = self.response_notebook.nametowidget(self.response_notebook.tabs()[0]).winfo_children()[0]
                widget.delete("1.0", tk.END); widget.insert(tk.END, txt)
                self.log_message(f"Loaded Main Responses: {path}")
            except Exception as e:
                self.log_message(f"Failed to load Main Responses: {e}")

    def load_assistant_responses(self):
        path = filedialog.askopenfilename(title="Select Assistant Responses File",
                                          filetypes=(("Text files","*.txt"),("All files","*.*")))
        if path:
            try:
                with open(path,'r',encoding='utf-8') as f: txt=f.read()
                widget = self.response_notebook.nametowidget(self.response_notebook.tabs()[1]).winfo_children()[0]
                widget.delete("1.0", tk.END); widget.insert(tk.END, txt)
                self.log_message(f"Loaded Assistant Responses: {path}")
            except Exception as e:
                self.log_message(f"Failed to load Assistant Responses: {e}")




#############################################################################################
#############################################################################################




    def clear_rag_for_user(self):
        """Delete RAG documents that belong to this user. If that fails, remove full chroma dir.
           Also clear in-memory user file store and GUI preview so context truly disappears.
        """
        try:
            user = getattr(self, "username", getattr(self, "current_user", "unknown"))
        except Exception:
            user = "unknown"

        # Try to get client & chroma_dir from helper(s)
        client = None
        chroma_dir = os.path.join(getattr(self, "BASE_DIR", os.getcwd()), "chroma_db")
        try:
            # Prefer self._get_chroma_client if available
            if hasattr(self, "_get_chroma_client"):
                try:
                    client, chroma_dir = self._get_chroma_client()
                except Exception:
                    client = None
            if client is None:
                # global helper fallback
                try:
                    client, chroma_dir = get_chroma_client(persist_directory=getattr(self, "BASE_DIR", os.getcwd()))
                except Exception:
                    client = None
        except Exception:
            client = None

        # Immediately clear in-memory caches so UI stops showing context
        try:
            if "USER_FILES" in globals():
                USER_FILES.pop(user, None)
            if "RAG_STORE" in globals():
                RAG_STORE.pop(user, None)
        except Exception as e:
            self.log_message(f"Warning: failed to clear in-memory RAG caches: {e}")

        # Update GUI state immediately
        try: self.rag_enabled = False
        except Exception: pass
        try: self.uploaded_doc_ids = []
        except Exception: pass
        try:
            self.loaded_file_box.delete("1.0", "end")
            self.loaded_file_box.insert("1.0", "[Local RAG cleared — normal LLM mode restored]")
        except Exception:
            pass

        # If no Chroma client is available, try to remove directory immediately
        if client is None:
            try:
                if os.path.exists(chroma_dir):
                    shutil.rmtree(chroma_dir)
                    os.makedirs(chroma_dir, exist_ok=True)
                self.log_message("Chroma persistence directory removed (no client available).")
            except Exception as e:
                # Try scheduling deletion on reboot if we are on Windows
                scheduled = _schedule_delete_on_reboot(chroma_dir)
                if scheduled:
                    self.log_message(f"Chroma dir scheduled for deletion on reboot: {chroma_dir}")
                else:
                    self.log_message(f"Failed to remove Chroma dir (no client): {e}")
            return

        # Attempt per-user deletion first (safer)
        try:
            coll = None
            try:
                if hasattr(client, "get_or_create_collection"):
                    coll = client.get_or_create_collection(name="docs")
                else:
                    try:
                        coll = client.get_collection("docs")
                    except Exception:
                        coll = None
            except Exception:
                coll = None

            if coll is not None:
                ids = []
                try:
                    data = coll.get(include=["ids"])
                    if isinstance(data, dict):
                        ids = data.get("ids", [])
                    elif isinstance(data, list):
                        ids = data
                except Exception:
                    try:
                        data = coll.get()
                        if isinstance(data, dict):
                            ids = data.get("ids", []) or []
                    except Exception:
                        ids = []

                to_delete = [i for i in ids if str(i).startswith(f"{user}_")]
                if to_delete:
                    try:
                        try:
                            coll.delete(ids=to_delete)
                        except Exception:
                            # fallback: delete collection entirely then recreate if supported
                            try:
                                if hasattr(client, "delete_collection"):
                                    client.delete_collection(name="docs")
                                    if hasattr(client, "get_or_create_collection"):
                                        client.get_or_create_collection(name="docs")
                            except Exception:
                                pass

                        # persist best-effort
                        try:
                            if hasattr(client, "persist"):
                                client.persist()
                        except Exception:
                            pass

                        self.log_message(f"Deleted {len(to_delete)} RAG document(s) for user {user}.")
                        return
                    except Exception as e:
                        self.log_message(f"Failed to delete user docs by id: {e}")
        except Exception as e:
            self.log_message(f"Error while attempting per-user deletion: {e}")

        # If we reach here, attempt full directory removal (robust retry)
        try:
            if os.path.exists(chroma_dir):
                _try_close_chroma_client(client, logger_fn=getattr(self, "log_message", None))
                gc.collect()
                time.sleep(0.2)

                last_err = None
                for attempt in range(8):
                    try:
                        shutil.rmtree(chroma_dir)
                        os.makedirs(chroma_dir, exist_ok=True)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.2 + 0.05 * attempt)

                if last_err is not None:
                    # Try to provide diagnostic info about which process holds the file (psutil optional)
                    sqlite_file = os.path.join(chroma_dir, "chroma.sqlite3")
                    lockers = _list_locking_processes(sqlite_file)
                    if lockers:
                        locker_info = ", ".join(f"{pid}:{name}" for pid, name in lockers)
                        self.log_message(f"Could not remove Chroma dir; file locked by processes: {locker_info}")
                    else:
                        self.log_message(f"Failed to remove Chroma dir: {last_err}")

                    # On Windows schedule deletion on reboot
                    if platform.system().lower() == "windows":
                        scheduled = _schedule_delete_on_reboot(chroma_dir)
                        if scheduled:
                            self.log_message(f"Chroma dir scheduled for deletion on reboot: {chroma_dir}")
                        else:
                            self.log_message("Could not schedule deletion on reboot.")
                else:
                    self.log_message("Chroma persistence directory removed (fallback).")
        except Exception as e:
            self.log_message(f"Failed to remove Chroma directory: {e}\n{traceback.format_exc()}")


    def reset_rag_local(self):
        """
        Confirm and delete local Chroma persistence directory, returning app to plain LLM behaviour.
        Also clears in-memory caches and GUI state.
        """
        try:
            ok = messagebox.askyesno("Reset RAG", "This will permanently delete the local RAG database (chroma_db). Continue?")
        except Exception:
            ok = True
        if not ok:
            return

        base_dir = getattr(self, "BASE_DIR", os.getcwd())
        chroma_dir = os.path.join(base_dir, "chroma_db")

        # Try to get and close a client gracefully
        client = None
        try:
            if hasattr(self, "_get_chroma_client"):
                try:
                    client, _ = self._get_chroma_client()
                except Exception:
                    client = None
            if client is None:
                try:
                    client, _ = get_chroma_client(persist_directory=chroma_dir)
                except Exception:
                    client = None
        except Exception:
            client = None

        try:
            if client is not None:
                _try_close_chroma_client(client, logger_fn=getattr(self, "log_message", None))
                gc.collect()
                time.sleep(0.2)
        except Exception:
            pass

        # Attempt robust deletion with retries and schedule on reboot if necessary
        try:
            if os.path.exists(chroma_dir):
                last_err = None
                for attempt in range(12):
                    try:
                        shutil.rmtree(chroma_dir)
                        os.makedirs(chroma_dir, exist_ok=True)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.2 + 0.05 * attempt)

                if last_err is not None:
                    # If still failing, show lockers (if available) and schedule deletion on reboot on Windows
                    sqlite_file = os.path.join(chroma_dir, "chroma.sqlite3")
                    lockers = _list_locking_processes(sqlite_file)
                    if lockers:
                        locker_info = ", ".join(f"{pid}:{name}" for pid, name in lockers)
                        self.log_message(f"Could not remove Chroma dir; file locked by processes: {locker_info}")
                    else:
                        self.log_message(f"Failed to remove Chroma dir: {last_err}")

                    if platform.system().lower() == "windows":
                        scheduled = _schedule_delete_on_reboot(chroma_dir)
                        if scheduled:
                            self.log_message(f"Chroma dir scheduled for deletion on reboot: {chroma_dir}")
                        else:
                            self.log_message("Could not schedule deletion on reboot.")
                else:
                    self.log_message("Chroma persistence directory removed successfully.")
            else:
                self.log_message("No chroma_dir found; nothing to remove.")
        except Exception as e:
            self.log_message(f"Failed to remove Chroma dir: {e}\n{traceback.format_exc()}")

        # Clear in-memory caches and GUI state
        try:
            if "USER_FILES" in globals():
                USER_FILES.clear()
            if "RAG_STORE" in globals():
                RAG_STORE.clear()
        except Exception as e:
            self.log_message(f"Warning: failed to clear in-memory caches: {e}")

        try: self.rag_enabled = False
        except Exception: pass
        try: self.uploaded_doc_ids = []
        except Exception: pass
        try:
            self.loaded_file_box.delete("1.0", "end")
            self.loaded_file_box.insert("1.0", "[Local RAG cleared — normal LLM mode restored]")
        except Exception:
            pass

        self.log_message("Local RAG cleared successfully.")


    def _on_reset_rag_clicked(self):
        if not messagebox.askyesno("Reset RAG", "This will remove stored RAG embeddings for your user and return the assistant to the normal LLM. Continue?"):
            self.log_message("Reset RAG cancelled.")
            return
        # call the local clearing logic
        clear_rag_for_user(self)


    def load_file(self):
        """
        Open file dialog, extract text, chunk, embed, store into local Chroma.
        Uses get_chroma_client() for consistent client creation.
        This version also removes occurrences of "..", "___", and "---" from the
        extracted text before chunking/embedding.
        """
        path = filedialog.askopenfilename(title="Select a File", filetypes=(("All files","*.*"),))
        if not path:
            return

        self.current_user = getpass.getuser().capitalize()
        filename = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            self.log_message(f"Failed to read file: {e}")
            return

        # --- extract text (PDF or text)
        mime_type, _ = mimetypes.guess_type(filename)
        text = ""
        try:
            if (mime_type == "application/pdf") or filename.lower().endswith(".pdf"):
                if PyPDF2 is None:
                    self.log_message("PyPDF2 not installed — cannot extract PDF text.")
                    text = ""
                else:
                    try:
                        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                        pages = []
                        for p in reader.pages:
                            try:
                                pt = p.extract_text() or ""
                            except Exception:
                                pt = ""
                            pages.append(pt)
                        text = "\n".join(pages).strip()
                    except Exception as e:
                        self.log_message(f"PDF extraction failed: {e}")
                        text = ""
            else:
                detect = chardet.detect(file_bytes[:4096] if len(file_bytes) > 4096 else file_bytes)
                enc = detect.get("encoding") or "utf-8"
                try:
                    text = file_bytes.decode(enc, errors="replace")
                except Exception:
                    text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            self.log_message(f"File parsing failed: {e}\n{traceback.format_exc()}")
            text = ""

        if not text:
            try:
                text = file_bytes.decode("latin-1", errors="replace")
            except Exception:
                text = ""

        # --- CLEANUP: remove filler sequences (dot leaders, ellipses, underscores, dashes), and quotes/asterisks
        try:
            import re
            # 1) Remove sequences of dots, including spaced dots: "...." or ". . . ."
            text = re.sub(r'(?:\.\s*){2,}', ' ', text)


            # 2) Remove repeated Unicode ellipsis characters (U+2026) like "……" or single ones
            text = re.sub(r'…+', ' ', text)


            # 3) Remove long runs of underscores or dashes (keeps single ones intact)
            text = re.sub(r'(_{2,}|-{2,}|—{2,})', ' ', text)


            # 4) Remove double quotes and asterisks entirely (they are filler here)
            text = re.sub(r'[\"*]', '', text)


            # 5) Collapse runs of spaces/tabs but preserve single newlines and paragraph breaks.
            text = re.sub(r'[ \t]{2,}', ' ', text) # collapse repeated spaces/tabs
            text = re.sub(r'\n{3,}', '\n\n', text) # collapse 3+ newlines -> 2
            # optionally collapse trailing spaces on each line
            text = re.sub(r'[ \t]+(?=\n)', '', text)


            # 6) Trim leading/trailing whitespace
            text = text.strip()
        except Exception:
            # non-fatal; leave original text
            pass

        # preview
        try:
            preview = text[:3000] + ("\n\n[truncated]" if len(text) > 2000 else "")
            self.loaded_file_box.delete("1.0", "end")
            self.loaded_file_box.insert("1.0", preview)
        except Exception:
            pass

        # make sure we store raw text in USER_FILES for query-time fallback
        try:
            USER_FILES.setdefault(self.current_user, {})
            USER_FILES[self.current_user]["text"] = text
        except Exception:
            # non-fatal
            pass

        # chunk - uses your _chunk_text implementation
        chunks = _chunk_text(text, max_chars=2000)
        if not chunks:
            self.log_message("No text extracted — nothing to add to RAG.")
            return

        # get embeddings
        try:
            embeddings, emb_source = _get_embeddings_for_chunks(chunks)
        except Exception as e:
            self.log_message(f"Embedding generation failed: {e}")
            return

        # Normalize embeddings to plain Python lists of floats and validate dimensions
        try:
            if not embeddings:
                self.log_message("No embeddings produced for chunks; aborting.")
                return
            # convert numpy/tuple -> list, ensure floats
            normalized = []
            for v in embeddings:
                if hasattr(v, "tolist"):
                    pv = v.tolist()
                elif isinstance(v, (tuple,)):
                    pv = list(v)
                elif isinstance(v, list):
                    pv = v
                else:
                    pv = v  # assume list-like
                try:
                    pv2 = [float(x) for x in pv]
                except Exception:
                    # if conversion fails, keep as-is and let chroma raise
                    pv2 = pv
                normalized.append(pv2)
            embeddings = normalized
            # ensure uniform dim
            dims = set(len(v) for v in embeddings)
            if len(dims) != 1:
                self.log_message(f"Embeddings have mixed dimensions: {dims}. Aborting upload.")
                USER_FILES[self.current_user]["chunks"] = chunks
                USER_FILES[self.current_user]["embeddings"] = embeddings
                return
            emb_dim = dims.pop()
            self.log_message(f"Embeddings produced with dimension: {emb_dim} (source={emb_source})")
        except Exception as e:
            self.log_message(f"Failed to normalize embeddings: {e}\n{traceback.format_exc()}")
            USER_FILES[self.current_user]["chunks"] = chunks
            USER_FILES[self.current_user]["embeddings"] = embeddings
            return

        # get chroma client (shared helper)
        client, chroma_dir = get_chroma_client(persist_directory=getattr(self, "BASE_DIR", os.getcwd()))
        if client is None:
            self.log_message("chromadb client unavailable — storing only in-memory fallback.")
            # still keep chunks in memory for "this file" flow
            USER_FILES[self.current_user]["chunks"] = chunks
            USER_FILES[self.current_user]["embeddings"] = embeddings
            return

        # ensure collection in a version tolerant way
        try:
            if hasattr(client, "get_or_create_collection"):
                collection = client.get_or_create_collection(name="docs")
            else:
                try:
                    collection = client.get_collection("docs")
                except Exception:
                    try:
                        collection = client.create_collection("docs")
                    except Exception:
                        collection = None
        except Exception as e:
            self.log_message(f"Failed to get/create collection: {e}")
            collection = None

        if collection is None:
            self.log_message("Could not obtain collection object; storing in-memory fallback and returning.")
            USER_FILES[self.current_user]["chunks"] = chunks
            USER_FILES[self.current_user]["embeddings"] = embeddings
            return

        # prepare ids/metadatas/docs
        ids = []
        docs = []
        metas = []
        ts = int(time.time())
        for i, chunk in enumerate(chunks):
            doc_id = f"{self.current_user}_{filename}_{ts}_{i}_{uuid.uuid4().hex[:6]}"
            ids.append(doc_id)
            docs.append(chunk)
            metas.append({"source_filename": filename, "username": self.current_user, "chunk_index": i})

        # add (try several arg shapes). If a dimension mismatch occurs, attempt to delete+recreate collection and retry.
        try:
            try:
                collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            except Exception as e:
                msg = str(e) + "\n" + traceback.format_exc()
                # detect typical chromadb dimension error text
                if ("dimension" in msg.lower() and "got" in msg.lower()) or ("expecting embedding" in msg.lower()) or ("got " in msg.lower() and "embedding" in msg.lower()):
                    self.log_message("Embedding dimension mismatch detected when adding to Chroma. Attempting to recreate 'docs' collection to accept new embedding size and retry.")
                    try:
                        # attempt to delete collection safely
                        try:
                            if hasattr(client, "delete_collection"):
                                client.delete_collection(name="docs")
                                self.log_message("Deleted existing 'docs' collection to resolve embedding-dimension mismatch.")
                            else:
                                # older clients may not have delete_collection; try best-effort fallbacks
                                try:
                                    client.delete_collection("docs")
                                    self.log_message("Deleted existing 'docs' collection (legacy API).")
                                except Exception:
                                    pass
                        except Exception as de:
                            self.log_message(f"Could not delete existing collection via client API: {de}")

                        # recreate collection (without an embedding_function so it accepts supplied vectors)
                        try:
                            if hasattr(client, "get_or_create_collection"):
                                collection = client.get_or_create_collection(name="docs")
                            else:
                                collection = client.create_collection("docs")
                            self.log_message("Recreated 'docs' collection. Retrying add().")
                            collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
                        except Exception as e2:
                            # If retry fails, re-raise to outer except handler
                            raise RuntimeError(f"Recreate-and-add attempt failed: {e2}\n{traceback.format_exc()}")
                    except Exception as e3:
                        raise
                else:
                    # not a dimension-related error; re-raise
                    raise
        except Exception as e:
            # final fallback: store in-memory and log full traceback
            self.log_message(f"Failed to add chunks to collection: {e}\n{traceback.format_exc()}")
            # store in-memory so queries still work
            try:
                USER_FILES[self.current_user]["chunks"] = chunks
                USER_FILES[self.current_user]["embeddings"] = embeddings
            except Exception:
                pass
            return

        # attempt to persist (best-effort)
        try:
            if hasattr(client, "persist"):
                client.persist()
            elif hasattr(collection, "persist"):
                collection.persist()
        except Exception:
            pass

        # store in-memory quick-access copy so "this file" queries never fail
        try:
            USER_FILES[self.current_user]["chunks"] = chunks
            USER_FILES[self.current_user]["embeddings"] = embeddings
            USER_FILES[self.current_user]["uploaded_ids"] = ids
        except Exception:
            pass

        # mark rag enabled and store ids for deletion
        self.rag_enabled = True
        existing = getattr(self, "uploaded_doc_ids", [])
        existing.extend(ids)
        self.uploaded_doc_ids = existing

        self.log_message(f"Stored {len(ids)} chunks from '{filename}' into Chroma (emb_source={emb_source}).")
        try:
            self.loaded_file_box.delete("1.0", "end")
            self.loaded_file_box.insert("1.0", f"[Uploaded {filename} to local RAG ({len(ids)} chunks).]")
        except Exception:
            pass


    def get_chroma_client(persist_directory=None):
        """
        Return (client, path). This tries PersistentClient, then Client(), then Client(Settings).
        It also disables telemetry where possible to avoid noisy exceptions.
        """
        import os, traceback
        os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
        CHROMA_PATH = persist_directory or os.path.join(os.getcwd(), "chroma_db")

        try:
            import chromadb
        except Exception as e:
            print(f"[get_chroma_client] chromadb import failed: {e}")
            return None, CHROMA_PATH

        # Try to disable telemetry on module
        try:
            if hasattr(chromadb, "telemetry_enabled"):
                try:
                    chromadb.telemetry_enabled = False
                except Exception:
                    pass
            telemetry = getattr(chromadb, "telemetry", None)
            if telemetry is not None:
                for name in ("capture", "capture_event", "send_event", "capture_exception"):
                    if hasattr(telemetry, name):
                        try:
                            setattr(telemetry, name, lambda *a, **kw: None)
                        except Exception:
                            pass
        except Exception:
            pass

        # Try new PersistentClient
        try:
            PersistentClient = getattr(chromadb, "PersistentClient", None)
            if PersistentClient is not None:
                try:
                    client = PersistentClient(path=CHROMA_PATH)
                    print(f"[get_chroma_client] Using chromadb.PersistentClient(path={CHROMA_PATH})")
                    globals()['chromadb'] = chromadb
                    return client, CHROMA_PATH
                except Exception as e:
                    print(f"[get_chroma_client] PersistentClient init failed: {e}")

        except Exception:
            pass

        # Try high-level Client()
        try:
            client = chromadb.Client()
            print("[get_chroma_client] Using chromadb.Client() fallback")
            globals()['chromadb'] = chromadb
            return client, CHROMA_PATH
        except Exception as e:
            print(f"[get_chroma_client] chromadb.Client() failed: {e}")

        # Legacy Settings fallback
        try:
            from chromadb.config import Settings as ChromaSettings
            chroma_settings = ChromaSettings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PATH)
            client = chromadb.Client(chroma_settings)
            print(f"[get_chroma_client] Using legacy chromadb.Client(Settings) at {CHROMA_PATH}")
            globals()['chromadb'] = chromadb
            return client, CHROMA_PATH
        except Exception as e:
            print(f"[get_chroma_client] legacy Client(Settings) fallback failed: {e}")

        return None, CHROMA_PATH


    def retrieve_rag_context(self, query_text, top_k=20):
        """
        Retrieve most relevant RAG chunks for the given query.
        Works with both new and legacy Chroma clients. Falls back to in-memory USER_FILES.
        """
        try:
            import numpy as np
        except Exception:
            np = None

        try:
            base_dir = getattr(self, "BASE_DIR", os.getcwd())
            chroma_dir = os.path.join(base_dir, "chroma_db")

            # 1) Try to use chroma if available
            client, used_path = get_chroma_client(persist_directory=chroma_dir)
            if client is not None:
                # version-tolerant collection retrieval
                try:
                    if hasattr(client, "get_or_create_collection"):
                        coll = client.get_or_create_collection(name="docs")
                    else:
                        try:
                            coll = client.get_collection("docs")
                        except Exception:
                            coll = client.create_collection("docs") if hasattr(client, "create_collection") else None
                except Exception as e:
                    self.log_message(f"[retrieve_rag_context] failed to get collection: {e}")
                    coll = None

                if coll is not None:
                    # embed query (prefer sentence-transformers here for consistency)
                    q_emb = None
                    try:
                        from sentence_transformers import SentenceTransformer
                        st_model = SentenceTransformer('all-MiniLM-L6-v2')
                        q_vecs = st_model.encode([query_text])
                        q_emb = q_vecs[0].tolist() if hasattr(q_vecs[0], "tolist") else list(q_vecs[0])
                    except Exception as e:
                        # try ollama if sentence-transformers not available
                        try:
                            if ollama is not None and hasattr(ollama, "embeddings"):
                                q_resp = ollama.embeddings(model="mxbai-embed-large", prompt=query_text)
                                q_emb = q_resp.get("embedding") if isinstance(q_resp, dict) else q_resp
                        except Exception:
                            q_emb = None

                    if q_emb is not None:
                        # try multiple query shapes
                        results = None
                        try:
                            results = coll.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas', 'distances'])
                        except Exception:
                            try:
                                results = coll.query(query_embeddings=[q_emb], n_results=top_k)
                            except Exception:
                                try:
                                    results = coll.query(embeddings=[q_emb], n_results=top_k)
                                except Exception as e:
                                    self.log_message(f"[retrieve_rag_context] coll.query failed: {e}")
                                    results = None

                        # normalize documents
                        docs = []
                        if isinstance(results, dict):
                            docs_field = results.get('documents') or []
                            if isinstance(docs_field, list):
                                if len(docs_field) > 0 and isinstance(docs_field[0], list):
                                    docs = docs_field[0]
                                else:
                                    docs = docs_field
                        elif isinstance(results, list):
                            docs = results
                        # filter strings
                        docs = [d for d in (docs or []) if isinstance(d, str) and d.strip()]
                        if docs:
                            # join top_k into a single context string
                            context = "\n\n".join(docs[:top_k])
                            preview = (context[:2000] + '...') if len(context) > 2000 else context
                            self.log_message(f"[retrieve_rag_context] returning {len(docs[:top_k])} docs, preview:\n{preview}")
                            return context
                        else:
                            self.log_message("[retrieve_rag_context] Chroma returned no text documents; falling back to in-memory store.")
                    else:
                        self.log_message("[retrieve_rag_context] Query embedding failed; falling back to in-memory store.")
                else:
                    self.log_message("[retrieve_rag_context] No collection object available; falling back to in-memory store.")
            else:
                self.log_message("[retrieve_rag_context] chroma client unavailable; using in-memory store.")

            # 2) Fallback: search in-memory USER_FILES for user
            user_files = globals().get("USER_FILES", {})
            u = self.current_user if hasattr(self, "current_user") else None
            if not u:
                # attempt to find a username in other known places
                u = getattr(self, "username", None)
            if u and u in user_files and user_files[u].get("chunks"):
                # simple heuristic: substring match ranking
                local_chunks = user_files[u]["chunks"]
                matches = []
                qlow = (query_text or "").lower()
                for c in local_chunks:
                    score = 0
                    cl = c.lower()
                    if qlow in cl:
                        score += 10
                    # word intersection
                    words = set(re.findall(r"\w+", qlow))
                    if words:
                        inter = sum(1 for w in words if w in cl)
                        score += inter
                    matches.append((score, c))
                matches.sort(reverse=True, key=lambda x: x[0])
                chosen = [m[1] for m in matches if m[0] > 0][:top_k]
                if not chosen:
                    chosen = local_chunks[:top_k]
                context = "\n\n".join(chosen)
                self.log_message(f"[retrieve_rag_context] returning {len(chosen)} in-memory chunks")
                return context

            # 3) final fallback: use rag_embeddings module against saved text if available
            try:
                if u and u in user_files and user_files[u].get("text"):
                    from rag_embeddings import file_embeddings
                    fe = file_embeddings()
                    fe.build_store(user_files[u]["text"])
                    context_chunks = fe.retrieve(query=query_text, top_k=top_k)
                    if context_chunks:
                        return "\n\n".join(context_chunks)
            except Exception as e:
                self.log_message(f"[retrieve_rag_context] final rag_embeddings fallback failed: {e}")

            # nothing available
            self.log_message("[retrieve_rag_context] no RAG context available")
            return ""
        except Exception as e:
            self.log_message(f"[retrieve_rag_context] unexpected error: {e}\n{traceback.format_exc()}")
            return ""


    def exit_program(self):
        
        import shutdown_helpers as sh
        # 1) release GUI-local resources
        if hasattr(self, "vision_module"):
            try:
                self.vision_module.release()
            except Exception:
                pass

        # 2) signal background threads/subprocesses to stop
        try:
            sh.initiate_shutdown(grace_period=2.0, wait_thread_timeout=3.0)
            self.log_message("Shutdown initiated (threads/subprocesses notified).")
        except Exception as e:
            self.log_message(f"Error initiating shutdown: {e}")

        # 3) close any tracked console windows / tracked subprocesses (best-effort)
        try:
            closed = sh.close_all_console_windows_graceful_or_force()
            if closed:
                self.log_message("Closed tracked console windows on exit.")
            else:
                self.log_message("No tracked console windows to close.")
        except Exception as e:
            self.log_message(f"Error closing console windows: {e}")

        # 4) normal GUI teardown
        try:
            self.root.destroy()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 5) speak a goodbye (best-effort)
        try:
            # nonblocking path would be better if available; use AlfredSpeak best-effort
            speech.AlfredSpeak('Good bye. Alfred Signing off…')
        except Exception:
            pass

        # 6) final ensure process exit
        try:
            time.sleep(0.15)
        except Exception:
            pass
        try:
            os._exit(0)
        except Exception:
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass


    def gui_extract_text_from_query(self, resp):
        import re

        username = getattr(self, "current_user", "ITF")
        timestamp = None
        model_used = None
        message = ""

        if not resp:
            return None, None, "", username

        if isinstance(resp, dict):
            date = resp.get("date") or resp.get("timestamp_date")
            time = resp.get("time") or resp.get("timestamp_time")
            if date and time:
                timestamp = f"{date} {time}"
            elif resp.get("timestamp"):
                timestamp = str(resp.get("timestamp")).strip()

            model_used = resp.get("model") or resp.get("model_used") \
                         or resp.get("AI_Assistant_Running") or resp.get("assistant")
            if model_used is not None:
                model_used = str(model_used).strip()

            message = str(resp.get("text") or resp.get("query") or resp.get("response") or resp.get("message") or "").strip()
            username = str(resp.get("username") or resp.get("user") or username)
            message = re.sub(r'\s*\n+\s*', ' ', message).strip()
            message = re.sub(r'\s{2,}', ' ', message)
            return timestamp, model_used, message, username

        if isinstance(resp, str):
            s = resp.replace('\r', '').strip()
            s = re.sub(r'\n+\s*$', '', s)
            m_user = re.search(r":\s*'username':\s*(?P<u>[^\n:]+)\s*$", s)
            if m_user:
                username = m_user.group("u").strip()
                s = s[:m_user.start()].rstrip(" :\n\t")
            else:
                m_user2 = re.search(r":\s*(?P<u>[A-Za-z0-9_\-]+)\s*$", s)
                if m_user2:
                    username = m_user2.group("u").strip()
                    s = s[:m_user2.start()].rstrip(" :\n\t")
            s_no_at = re.sub(r'^\s*At\s+', '', s, flags=re.IGNORECASE)
            m_ts = re.match(
                r"^(?P<date>\d{4}-\d{2}-\d{2})\s*(?:[:]\s*|\s+)(?P<time>\d{2}:\d{2}:\d{2})\s*:\s*(?P<rest>.*)$",
                s_no_at,
                flags=re.DOTALL
            )
            if m_ts:
                timestamp = f"{m_ts.group('date')} {m_ts.group('time')}"
                rest = m_ts.group("rest")
            else:
                rest = s
            m_sep = re.search(r'\s+:\s+', rest)
            if m_sep:
                model_used = rest[:m_sep.start()].strip()
                message_part = rest[m_sep.end():].strip()
            else:
                model_used = None
                message_part = rest
            message_part = re.sub(r"^\s*(I replied:|I said:)\s*", "", message_part, flags=re.IGNORECASE)
            message = re.sub(r'\s*\n+\s*', ' ', message_part).strip()
            message = re.sub(r'\s{2,}', ' ', message)
            return timestamp, model_used, message, username
        message = str(resp).strip()
        message = re.sub(r'\s*\n+\s*', ' ', message).strip()
        message = re.sub(r'\s{2,}', ' ', message)
        return None, None, message, username


    def log_response(self, resp):
        global user_model_used
        print(f"GUI my_received_response  : {resp}")
        timestamp, self.model_used, message, username = self.gui_extract_text_from_query(resp)
        print(f"GUI gui_extract_text_from_query  timestamp : {timestamp!r}")
        print(f"GUI gui_extract_text_from_query  message : {message!r}")
        user_timestamp = timestamp
        user_model_used = self.model_used
        user_for_this = username
        response_for_user = message
        print(f"GUI gui_extract_text_from_query  username: {username!r}")
        try:
            user_model_used = user_model_used.replace(":","-")
        except Exception:
            user_model_used = str(user_model_used)
        print(f"GUI RESPONSE Model before : {user_model_used}")
        self.log_queue.put(resp)
        print(f"GUI RESPONSE Message 'resp' : {resp}")
        print(f"GUI RESPONSE Message 'message' : {message}")
        print(f"GUI RESPONSE Model Used : {user_model_used}")
        print("DEBUG — Emitting paired item:")
        print(f"DEBUG — User: {user_for_this}")
        print(f"DEBUG — Response: {response_for_user}")
        self.model_used = user_model_used
        try:
            sio_mobile.emit('gui_event', {'type':'log','payload':resp, 'username': user_for_this})
            sio.emit('gui_event', {'type':'log', 'payload': resp, 'username': user_for_this})
        except Exception:
            pass


    def log_message(self, msg):
        print(f"\n user_model_used for MESSAGE : {self.model_used} \n")
        current_model = None
        try:
            username = getattr(self, 'current_user', None)
            if username and hasattr(self, 'last_model_used_by_user'):
                current_model = self.last_model_used_by_user.get(username)
                if current_model:
                    print(f"DEBUG: found model for user {username} in last_model_used_by_user: {current_model}")
        except Exception as e:
            print("DEBUG: error checking per-user map:", e)
        if not current_model:
            current_model = getattr(self, 'last_model_used', None) or getattr(self, 'model_used', None)
            if current_model:
                print(f"DEBUG: using instance model field: {current_model}")
        if not current_model and isinstance(msg, str):
            import re
            m = re.search(r":\s*([^:\n]+)\s*:\s*[^:]+$", msg)
            if m:
                candidate = m.group(1).strip().replace(":", "-")
                if candidate and not candidate.lower().startswith("unknown"):
                    current_model = candidate
                    print(f"DEBUG: extracted model from msg text: {current_model}")
        if not current_model:
            current_model = "unknown"
            print("DEBUG: no model found by any method — falling back to 'unknown'")
        print(f"\n user_model_used for MESSAGE : {current_model} \n")
        my_received_response = msg
        print(f"\n my_received_response : {my_received_response} \n")
        self.log_queue.put(msg)
        print(f"Message : {msg}")
        payload = {
            'type': 'resp',
            'username': self.current_user,
            'query': self.last_query,
            'response': msg,
            'thinkbot_model': self.model_used
        }
        print(f"payload from GUI : {payload}")
        try:
            sio.emit('gui_event', {'payload': payload, 'model':self.model_used, 'username': self.current_user})
            sio_mobile.emit('gui_event', {'payload': payload, 'username': self.current_user})
        except Exception:
            pass

    def log_query(self, msg):
        my_received_query = msg
        print(f"\n [QUERY GUI] my_received_query  : {my_received_query} \n")
        self.query_queue.put(msg)
        self.log_message(msg)

    def update_log_widget(self):
        try:
            while True:
                m = self.log_queue.get_nowait()
                timestamp = datetime.datetime.now().astimezone().strftime('%H:%M:%S')
                self.log_widget.insert(
                    tk.END,
                    f"{timestamp} - {m} \n"
                )
                self.log_widget.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(200, self.update_log_widget)

    def update_clock(self):
        self.clock_label.config(text=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_clock)

    def toggle_whisper_listen(self):
        v = self.use_whisper.get()
        if v:
            self.use_vosk.set(False)
            self.use_bluetooth_speech.set(False)
        listen.set_listen_whisper(v)
        listen.set_listen_vosk(False)
        self.log_message(f"🎙️ Whisper {'enabled' if v else 'disabled'}")
        try:
            sio.emit('gui_event', {'type':'setting','payload':{'use_whisper':v, 'username': self.current_user}})
        except Exception:
            pass

    def toggle_vosk_listen(self):
        v = self.use_vosk.get()
        if v:
            self.use_whisper.set(False)
            self.use_bluetooth_speech.set(False)
        listen.set_listen_vosk(v)
        self.log_message(f"🎙️ Vosk {'enabled' if v else 'disabled'}")
        try:
            sio.emit('gui_event', {'type':'setting','payload':{'use_vosk':v, 'username': self.current_user}})
        except Exception:
            pass

    def toggle_bluetooth_speech(self):
        v = self.use_bluetooth_speech.get()
        if v:
            listen.set_mobile_speech(v)
            self.log_message(f"📱 Bluetooth Speech {'enabled' if v else 'disabled'}")
            try:
                sio.emit('gui_event', {'type':'setting','payload':{'use_bluetooth_speech':v, 'username': self.current_user}}) 
            except Exception:
                pass

    def toggle_wake_word(self):
        v = self.wake_word_call.get()
        print(f"wake word v : {v}")
        if v:
            self.use_whisper.set(True)
            self.use_vosk.set(False)
            listen.set_wake_word_on(v)
            speech.set_wake_word_on(v)
            self.log_message(f"📱 Wake Word {'On' if v else 'Off'}")
            try:
                sio.emit('gui_event', {'type':'setting','payload':{'wake_word_call':v, 'username': self.current_user}})
            except Exception:
                pass
        else:
            listen.set_wake_word_off(v)
            speech.set_wake_word_off(v)
            self.use_whisper.set(True)

    def _start_webui_listener(self):
      
        @sio.on('state_update')
        @sio_mobile.on('state_update')
        def on_state_update(data):
            global current_user
            if data.get('type') == 'login_ack':
                self.current_user = data.get('username')
                self.log_message(f"🔑 Logged in as {self.current_user}")
                return  
            if data['type']=='query':
                t = data.get('type')
                payload = data.get('payload')
                Name_Received_Query = data.get('username')
                print(f"[DEBUG GUI] The Name_Received_Query is {Name_Received_Query}")
                self.current_user = Name_Received_Query
                print(f"[DEBUG GUI] The QUERY self.current_user is {self.current_user}")
                if Name_Received_Query:
                    if not self.user_deque or self.user_deque[-1] != Name_Received_Query:
                        self.user_deque.append(Name_Received_Query)
                        print("DEBUG — appended username to user_deque:", list(self.user_deque))
                if t == 'query':
                    if isinstance(payload, dict):
                        text = payload.get('query', '')
                    else:
                        text = str(payload)

                    query = text.lower()
                    print (f"[DEBUG GUI] QUERY FROM FRONTEND : {query}")
                    query_send_from_GUI = f"'message':{query} : 'username':{self.current_user} : 'score':None : 'gender':None : 'gender_conf':None "
                    print (f"[DEBUG GUI] QUERY FROM FRONTEND query_send_from_GUI : {query_send_from_GUI}")
                    time.sleep(0.5)

                    self.query_queue.put(f"[WebUI] {query_send_from_GUI}")
                    listen.add_text(query_send_from_GUI)

            if data['type']=='setting':
                s = data['payload']
                def set_if_diff(v, val):
                    def _set():
                        if v.get() != val:
                            v.set(val)
                    self.root.after(0, _set)
                for var_name in ('use_whisper','use_vosk','use_bluetooth_speech','enter_submits'):
                    if var_name in s:
                        set_if_diff(getattr(self, var_name), s[var_name])
                for key,var in [('thinkbot_model', self.thinkbot_model),
                                ('chatbot_model',  self.chatbot_model),
                                ('vision_model',   self.vision_model),
                                ('coding_model',   self.coding_model)]:
                    if key in s and var.get()!=s[key]:
                        self.root.after(0, var.set, s[key])

        @sio.on('gui_event')
        def on_gui_event(data):
            event_type = data.get('type')
            username = data.get('username')
            if event_type == 'login' and username:
                self.current_user = username
                self.log_message(f"🔑 Logged in as: {username}")
                print(f"[DEBUG GUI] The Username is {self.current_user}")
            elif event_type == 'login_ack' and username:
                self.current_user = username
                self.log_message(f"🔑 Logged in as: {username}")
                print(f"[DEBUG GUI] The Username is {self.current_user}")

    def toggle_manual_record_only(self):
        en = self.manual_record_only.get()
        if en:
            self.use_whisper.set(False)
            self.use_vosk.set(False)
            listen.set_listen_whisper(False)
            listen.set_listen_vosk(False)
            listen.set_stop_listen_on(True)
            self.log_message("🔒 Manual‐Record‐Only mode ENABLED")
        if not en:
            listen.set_stop_listen_off(False)
            self.log_message("🔓 Manual‐Record‐Only mode DISABLED")
            self._on_record_release(event=None)

    def _on_record_press(self, event=None):
        self.record_button.config(bg="red", activebackground="red", text="🔴 Recording...")
        if self.manual_record_only.get():
            listen.set_recording_hold(True)
            threading.Thread(target=self._record_vosk, daemon=True).start()
        else:
            self.log_message(f"This is for 'Manual Record Only' Please Select 'Manual Record Only' for this button to work...")

    def _record_whisper(self):
        try:
            self._on_record_release(event=None)
            transcript = listen.listen_whisper()
            if transcript:
                self.log_message("Transcribed: " + transcript)
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert(tk.END, transcript)
                return transcript
            else:
                self.log_message("No voice input detected.")
        except Exception as e:
            self.log_message(f"Voice recording error: {e}")

    def _record_vosk(self):
        try:
            transcript = listen.record_button_listen_vosk()
            self._on_record_release(event=None)
            if transcript:
                self.log_message("Transcribed: " + transcript)
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert(tk.END, transcript)
                return transcript
            else:
                self.log_message("No voice input detected.")
        except Exception as e:
            self.log_message(f"Voice recording error: {e}")


    def _on_record_release(self, event=None):
        listen.set_recording_hold(False)
        self.record_button.config(bg="darkgreen", text="🎙️ Hold to Speak")
        self.log_message("🟢 Recording stopped")


    def send_text(self):
        self.current_user = "Itf"
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            self.last_query = text
            self.models = {
                "thinkbot": self.thinkbot_model.get(),
                "chatbot": self.chatbot_model.get(),
                "vision": self.vision_model.get(),
                "coding": self.coding_model.get()
            }

            # Retrieve RAG context first
            context = self.retrieve_rag_context(text)
            if context:
                full_query = f"{text}\n\n[Context from knowledge base:]\n{context}"
            else:
                full_query = text

            text_msg = {
                'username': self.current_user,
                'query': full_query,
            }

            self.log_query(f"Text to send from GUI: {text_msg}")
            listen.add_text(text_msg)
            self.input_text.delete("1.0", tk.END)
        else:
            self.log_message("No text entered.")


    def on_enter_pressed(self, event):
        if self.enter_submits.get():
            self.send_text()
            return "break"


    def listen_last(self):
        """Play the last stored response but DO NOT start listening while TTS plays.
        This method suppresses automatic listening (main loop should observe speech._suppress_auto_listen)
        and uses listen.set_stop_listen_on(True) to prevent the listen module from starting."""
        self.log_message("Listening to last message received...")

        # obtain the last stored response (support repeat.get_last() or repeat())
        try:
            if hasattr(repeat, "get_last"):
                stored = repeat.get_last()
            else:
                try:
                    stored = repeat()
                except Exception:
                    stored = None
        except Exception:
            try:
                stored = repeat()
            except Exception:
                stored = None

        # extract text
        text = None
        if isinstance(stored, dict):
            text = stored.get("response") or stored.get("text") or stored.get("message")
        elif isinstance(stored, str):
            text = stored
        elif stored is not None:
            try:
                text = str(stored)
            except Exception:
                text = None

        if not text:
            self.log_message("No last message available.")
            return

        # log it
        self.log_query("Repeated: " + text)

        # Best-effort: suppress main-loop auto-listen and tell listen module to stop auto-starting
        try:
            prev_suppress = bool(getattr(speech, "_suppress_auto_listen", False))
            speech._suppress_auto_listen = True
        except Exception:
            prev_suppress = False
        try:
            # If your listen module exposes these controls use them; store previous value to restore
            prev_listen_stop = False
            if hasattr(listen, "set_stop_listen_on") and hasattr(listen, "set_stop_listen_off"):
                # assume there's internal state; we can't reliably read the current flag API, so just store None and restore by calling "off"
                listen.set_stop_listen_on(True)
                prev_listen_stop = True
            else:
                # If listen has a direct flag, try to set it
                if hasattr(listen, "_stop_listen_flag"):
                    prev_listen_stop = bool(getattr(listen, "_stop_listen_flag", False))
                    setattr(listen, "_stop_listen_flag", True)
        except Exception:
            prev_listen_stop = False

        # Ensure TTS controller is in started/resumable mode so Pause/Stop buttons work
        try:
            self.root.after(0, lambda: getattr(speech, 'set_tk_start_speech', lambda: None)())
        except Exception:
            pass

        # Update GUI buttons immediately
        try:
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
        except Exception:
            pass

        # Play in background thread to avoid blocking GUI
        def _play_and_restore():
            try:
                # Primary TTS call (blocking). Let speech module handle fallbacks internally.
                try:
                    speech.AlfredSpeak(text)
                except Exception as e:
                    # fallback to onboard if available
                    try:
                        self.log_message(f"Primary TTS failed ({e}), falling back to onboard.")
                    except Exception:
                        pass
                    try:
                        speech.AlfredSpeak_Onboard(text)
                    except Exception as e2:
                        try:
                            self.log_message(f"Both TTS methods failed: {e2}")
                        except Exception:
                            pass
            finally:
                # Always restore listening flags and GUI buttons on GUI thread
                def _restore_ui_and_listen():
                    try:
                        # restore buttons
                        self.start_button.config(state=tk.NORMAL)
                        self.pause_button.config(state=tk.DISABLED)
                        self.stop_button.config(state=tk.DISABLED)
                        self.log_message("Finished repeating last message.")
                    except Exception:
                        pass

                    try:
                        # restore speech suppression flag
                        try:
                            speech._suppress_auto_listen = bool(prev_suppress)
                        except Exception:
                            try:
                                # if attribute didn't exist before, delete it
                                if not prev_suppress and hasattr(speech, "_suppress_auto_listen"):
                                    delattr(speech, "_suppress_auto_listen")
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        # restore listen module state
                        if prev_listen_stop and hasattr(listen, "set_stop_listen_off"):
                            try:
                                listen.set_stop_listen_off(False)
                            except Exception:
                                # best effort to clear
                                pass
                        else:
                            # If we set an internal flag, try to restore it
                            try:
                                if hasattr(listen, "_stop_listen_flag"):
                                    setattr(listen, "_stop_listen_flag", False)
                            except Exception:
                                pass
                    except Exception:
                        pass

                try:
                    self.root.after(0, _restore_ui_and_listen)
                except Exception:
                    _restore_ui_and_listen()

        t = threading.Thread(target=_play_and_restore, daemon=True)
        t.start()


    def emit_dropdown_change(self, key, value):
        if getattr(self, f'_last_{key}', None)==value:
            return
        setattr(self, f'_last_{key}', value)
        try:
            sio.emit('gui_event', {'type':'setting','payload':{key:value, 'username': self.current_user}})
            sio_mobile.emit('gui_event', {'type':'setting','payload':{key:value, 'username': self.current_user}})
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


    # ----------------- TK speech button handlers -----------------

    def _tk_start_speech(self):
        """Called when user presses the GUI Start Speech button.
        Resume TTS and allow automatic listening again (clear suppression)."""
        try:
            # clear suppression so main() will allow listen.listen() again
            try:
                speech._suppress_auto_listen = False
            except Exception:
                pass

            # call speech module to resume/start
            self.root.after(0, lambda: getattr(speech, 'set_tk_start_speech', lambda: None)())

            self.log_message("▶️ Start Speech pressed")
            # update button states
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
        except Exception as e:
            self.log_message(f"Error calling start speech: {e}")


    def _tk_pause_speech(self):
        """Request pause/halt. While paused we SUPPRESS automatic listening (main() will skip listen())."""
        try:
            # Set suppression: while paused we do NOT want the mic to auto-start.
            try:
                speech._suppress_auto_listen = True
            except Exception:
                pass

            # Request pause in the speech module (it will attempt to pause playback)
            self.root.after(0, lambda: getattr(speech, 'set_tk_pause_speech', lambda: None)())

            self.log_message("⏸️ Pause Speech pressed")
            # allow Start (so user can resume), keep Stop enabled
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        except Exception as e:
            self.log_message(f"Error calling pause speech: {e}")


    def _tk_stop_speech(self):
        """Stop completely: call speech.set_tk_stop_speech() and allow listening again (clear suppression)."""
        try:
            # Stop should clear suppression so system can listen again
            try:
                speech._suppress_auto_listen = False
            except Exception:
                pass

            self.root.after(0, lambda: getattr(speech, 'set_tk_stop_speech', lambda: None)())
            # also ask speech module to attempt immediate stop
            try:
                getattr(speech, 'stop_current', lambda: None)()
            except Exception:
                pass

            self.log_message("⏹️ Stop Speech pressed")
            # reset buttons
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
        except Exception as e:
            self.log_message(f"Error calling stop speech: {e}")


    # ----------------- Speech state poller -----------------
    def poll_speech_state(self):
        """
        Periodically poll the speech module for playing/paused/stopped state,
        update GUI buttons and set speech._suppress_auto_listen to avoid the main loop
        calling listen.listen() while TTS is active or paused.
        """
        try:
            # determine playing state
            playing = False
            try:
                if callable(getattr(speech, "is_speaking", None)):
                    playing = bool(speech.is_speaking())
                else:
                    # fallback: check if a player exists or a flag set
                    playing = bool(getattr(speech, "_player", None) is not None and not getattr(speech, "_pause_requested", False))
            except Exception:
                playing = False

            # determine paused state
            paused = False
            try:
                if callable(getattr(speech, "is_paused", None)):
                    paused = bool(speech.is_paused())
                else:
                    # fallback to internal pause-request flag
                    paused = bool(getattr(speech, "_pause_requested", False) or getattr(speech, "_paused_file", None) is not None)
            except Exception:
                paused = False

            # compute suppress auto-listen: while playing OR paused we want to suppress
            suppress = playing or paused

            # If attribute exists on speech, set it so main() can read it
            try:
                setattr(speech, "_suppress_auto_listen", bool(suppress))
            except Exception:
                pass

            # Edge detection: transitions
            if playing and not self._prev_playing:
                # started playing
                self.log_message("🔊 Speech started (detected)")
                # Update button states
                try:
                    self.start_button.config(state=tk.DISABLED)
                    self.pause_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.NORMAL)
                except Exception:
                    pass

            if paused and not self._prev_paused:
                # newly paused
                self.log_message("⏸️ Speech paused (detected)")
                try:
                    self.start_button.config(state=tk.NORMAL)   # allow resume
                    self.pause_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                except Exception:
                    pass

            if (not playing) and (not paused) and (self._prev_playing or self._prev_paused):
                # finished/stopped
                self.log_message("⏹️ Speech stopped/finished (detected)")
                try:
                    self.start_button.config(state=tk.NORMAL)
                    self.pause_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.DISABLED)
                except Exception:
                    pass

            # update prev flags for next poll
            self._prev_playing = bool(playing)
            self._prev_paused = bool(paused)
            self._prev_suppress = bool(suppress)

        except Exception as e:
            # Keep the poller resilient — log but don't stop scheduling
            try:
                self.log_queue.put(f"poll_speech_state error: {e}")
            except Exception:
                pass

        # run again
        try:
            self.root.after(150, self.poll_speech_state)
        except Exception:
            # last-resort: schedule with time.sleep in a thread if mainloop is gone (unlikely)
            threading.Timer(0.15, self.poll_speech_state).start()


if __name__ == "__main__":
    gui().run()



