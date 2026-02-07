# assistant_standby_wait.py
"""
Minimal, non-auto-running Alfred standby/scan module.

- Uses your existing modules:
    - speech: expects `speech.AlfredSpeak(text)` (callable)
    - listen: expects `listen.listen()` (callable)
    - arduino_com: expects `arduino.send_arduino(cmd)` OR `arduino.send(cmd)`
    - Alfred_config: optional, for CHEST_CAMERA_INPUT

- Does NOT run on import. Call start_service(...) to create and run components.
- All major runtime knobs exposed via set_* helpers.

Keep this file minimal — most heavy-lifting is delegated to your modules.
"""
import threading
import json
import time
import traceback
import re
import cv2
import base64
import requests
import numpy as np
import os
import math
import random
from collections import Counter
from math import sqrt

# --- user modules (expected to exist) ---
try:
    from speech import speech as external_speech
except Exception:
    external_speech = None

# allow either module or callable named listen
try:
    from listen import listen as external_listen
except Exception:
    try:
        import listen as _listen_mod
        external_listen = getattr(_listen_mod, "listen", _listen_mod)
    except Exception:
        external_listen = None

try:
    from arduino_com import arduino as external_arduino
except Exception:
    external_arduino = None

# Alfred_config for camera input (optional)
try:
    import Alfred_config
    CAMERA_INPUT_DEFAULT = getattr(Alfred_config, "CHEST_CAMERA_INPUT", None)
except Exception:
    CAMERA_INPUT_DEFAULT = None

# ---------------- CONFIG ----------------
FRAME_W, FRAME_H = 640, 480

VISION_MODEL = "qwen3-vl:2b"
LANG_MODEL = "dolphin-phi"
OLLAMA_URL = "http://localhost:11434/api/generate"

VISION_INTERVAL = 0.3
MEMORY_FILE = "people_memory.json"

# Head mapping (kept so head controller can be used unchanged)
HEAD_CMD_X = "X"
HEAD_CMD_Y = "Y"
SWAP_AXES = False
INVERT_X = False
INVERT_Y = True
HEAD_X_MIN = 0
HEAD_X_MAX = 640
HEAD_Y_MIN = 0
HEAD_Y_MAX = 480
SMOOTH = 0.18
SEND_THRESHOLD_PIX = 3

# USER-SUPPLIED SCAN SEQUENCE (kept as requested)
SCAN_SEQUENCE = [
    ("p", 2), ("f", 2), ("q", 3), ("e", 3), ("M", 3),
    ("f", 3), ("q", 3), ("w", 3), ("f", 3), ("e", 3)
]

# Memory limits (adjustable via set_memory_limits())
MEMORY_MAX_PEOPLE = 40
MEMORY_MAX_CONVERSATIONS_PER_PERSON = 120
MEMORY_MAX_TOPICS_PER_PERSON = 40

# Idle micro-motion toggles and defaults (adjustable via set_enable_* functions)
ENABLE_BREATHING = True
BREATH_PERIOD = 10.0
BREATH_AMPLITUDE_X = 6.0
BREATH_AMPLITUDE_Y = 4.0

ENABLE_JITTER = True
JITTER_AMPLITUDE = 5
JITTER_DURATION = 3.0
JITTER_FREQ = 0.12

# ---------------- Helpers: tiny LLM/vision wrappers ----------------
def llm_generate(prompt, timeout=20):
    """Simple wrapper to call local Ollama-like endpoint. Returns string or ''."""
    try:
        payload = {"model": LANG_MODEL, "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        d = r.json() if r is not None else {}
        return d.get("response", "") if isinstance(d, dict) else str(d)
    except Exception:
        return ""

def scene_objects_from_frame(frame):
    """Return list of short object names detected by LLM from an image frame. Non-fatal on errors."""
    if frame is None:
        return []
    try:
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        b64 = base64.b64encode(buf).decode("utf-8")
        prompt = "List distinct visible objects in this image as a comma-separated list (short words)."
        payload = {"model": VISION_MODEL, "prompt": prompt, "images": [b64], "stream": False}
        r = requests.post(OLLAMA_URL, json=payload, timeout=18)
        j = r.json() if r is not None else {}
        text = j.get("response", "") if isinstance(j, dict) else str(j)
        items = re.split(r"[,\n]+", text)
        items = [i.strip().lower() for i in items if i.strip()]
        if not items:
            # cheap fallback
            matches = re.findall(r"\b(chair|table|door|window|sofa|phone|bag|book|cup|bottle|tv|lamp|plant|desk)\b", text.lower())
            items = list(set(matches))
        return items
    except Exception:
        return []

# ---------------- PersonMemory (keeps caps) ----------------
class PersonMemory:
    def __init__(self, filename=MEMORY_FILE):
        self.filename = filename
        self.people = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.people = json.load(f)
        except Exception:
            self.people = []
        # enforce limits and coerce fields
        self._ensure_limits()
        self._save()

    def _save(self):
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.people, f, indent=2)
        except Exception:
            pass

    def _ensure_limits(self):
        global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
        # per-person caps
        for p in self.people:
            if "conversation_history" in p and isinstance(p["conversation_history"], list):
                if MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
                    p["conversation_history"] = p["conversation_history"][-MEMORY_MAX_CONVERSATIONS_PER_PERSON:]
            if "mentioned_topics" in p and isinstance(p["mentioned_topics"], list):
                if MEMORY_MAX_TOPICS_PER_PERSON is not None:
                    p["mentioned_topics"] = p["mentioned_topics"][-MEMORY_MAX_TOPICS_PER_PERSON:]
        # total people cap: remove oldest by last_interaction
        if MEMORY_MAX_PEOPLE is not None and len(self.people) > MEMORY_MAX_PEOPLE:
            # ensure sortable last_interaction
            def last_inter(p):
                v = p.get("last_interaction", 0)
                try:
                    return float(v)
                except Exception:
                    return 0.0
            self.people.sort(key=last_inter)
            while len(self.people) > MEMORY_MAX_PEOPLE:
                self.people.pop(0)

    def add_person(self, name, enc=None):
        rec = {
            "name": name,
            "enc_type": None,
            "enc": enc,
            "last_seen": time.time(),
            "last_interaction": time.time(),
            "mentioned_topics": [],
            "conversation_history": []
        }
        self.people.append(rec)
        self._ensure_limits()
        self._save()

    def append_conversation(self, name, text):
        for p in self.people:
            if p.get("name") == name:
                p.setdefault("conversation_history", []).append({"ts": time.time(), "text": text})
                p["last_interaction"] = time.time()
                p["last_seen"] = time.time()
                # trim
                if MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
                    p["conversation_history"] = p["conversation_history"][-MEMORY_MAX_CONVERSATIONS_PER_PERSON:]
                self._save()
                return

    def add_mentioned_topic(self, name, topic):
        for p in self.people:
            if p.get("name") == name:
                lst = p.setdefault("mentioned_topics", [])
                if topic not in lst:
                    lst.append(topic)
                    if MEMORY_MAX_TOPICS_PER_PERSON is not None:
                        p["mentioned_topics"] = lst[-MEMORY_MAX_TOPICS_PER_PERSON:]
                    self._save()
                return

    def update_last_seen(self, name):
        for p in self.people:
            if p.get("name") == name:
                p["last_seen"] = time.time()
                p["last_interaction"] = time.time()
                self._save()
                return

    def get_person(self, name):
        for p in self.people:
            if p.get("name") == name:
                return p
        return None

# ---------------- HeadController (sends arduino commands only) ----------------
class HeadController:
    def __init__(self):
        self.last_x = int(FRAME_W // 2)
        self.last_y = int(FRAME_H // 2)
        # breathing/jitter threads
        self._breath_thread = None
        self._breath_stop = threading.Event()
        self._jitter_thread = None
        self._jitter_stop = threading.Event()

    def _send_cmd(self, val, cmd):
        """Prefer user's arduino API if present. Fall back to no-op."""
        msg = f"{cmd}{int(val)}"
        # Try standard interface name
        try:
            if external_arduino is not None:
                # try both send_arduino or send
                if hasattr(external_arduino, "send_arduino"):
                    external_arduino.send_arduino(msg)
                    return
                if hasattr(external_arduino, "send"):
                    external_arduino.send(msg)
                    return
                # fallback: try callable
                if callable(external_arduino):
                    external_arduino(msg)
                    return
        except Exception:
            # swallow; keep module resilient
            pass
        # no arduino available -> no-op

    def force_move(self, tx, ty, send_both=True):
        txi = int(max(HEAD_X_MIN, min(HEAD_X_MAX, tx)))
        tyi = int(max(HEAD_Y_MIN, min(HEAD_Y_MAX, ty)))
        if send_both:
            self._send_cmd(txi, HEAD_CMD_X)
            time.sleep(0.02)
            self._send_cmd(tyi, HEAD_CMD_Y)
        else:
            self._send_cmd(txi, HEAD_CMD_X)
        self.last_x = txi
        self.last_y = tyi

    # breathing
    def start_breathing(self, amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD):
        if not ENABLE_BREATHING:
            return
        if self._breath_thread and self._breath_thread.is_alive():
            return
        self._breath_stop.clear()
        def _loop():
            base_x = self.last_x
            base_y = self.last_y
            t0 = time.time()
            while not self._breath_stop.is_set():
                t = time.time() - t0
                offx = int(amplitude_x * math.sin(2 * math.pi * t / (period + 1e-9)))
                offy = int(amplitude_y * math.sin(2 * math.pi * t / (period + 1e-9) + math.pi/4))
                tx = max(HEAD_X_MIN, min(HEAD_X_MAX, base_x + offx))
                ty = max(HEAD_Y_MIN, min(HEAD_Y_MAX, base_y + offy))
                try:
                    self._send_cmd(tx, HEAD_CMD_X)
                    time.sleep(0.02)
                    self._send_cmd(ty, HEAD_CMD_Y)
                except Exception:
                    pass
                time.sleep(max(0.08, period/80.0))
            # restore
            try:
                self._send_cmd(self.last_x, HEAD_CMD_X)
                time.sleep(0.02)
                self._send_cmd(self.last_y, HEAD_CMD_Y)
            except Exception:
                pass
        self._breath_thread = threading.Thread(target=_loop, daemon=True)
        self._breath_thread.start()

    def stop_breathing(self):
        if self._breath_thread:
            self._breath_stop.set()
            self._breath_thread = None

    # jitter
    def start_jitter(self, duration=JITTER_DURATION, amplitude=JITTER_AMPLITUDE):
        if not ENABLE_JITTER:
            return
        if self._jitter_thread and self._jitter_thread.is_alive():
            return
        self._jitter_stop.clear()
        def _loop():
            start = time.time()
            base_x = self.last_x
            base_y = self.last_y
            while time.time() - start < duration and not self._jitter_stop.is_set():
                rx = int(random.uniform(-amplitude, amplitude))
                ry = int(random.uniform(-amplitude//2, amplitude//2))
                tx = max(HEAD_X_MIN, min(HEAD_X_MAX, base_x + rx))
                ty = max(HEAD_Y_MIN, min(HEAD_Y_MAX, base_y + ry))
                try:
                    self._send_cmd(tx, HEAD_CMD_X)
                    time.sleep(0.01)
                    self._send_cmd(ty, HEAD_CMD_Y)
                except Exception:
                    pass
                time.sleep(max(0.02, JITTER_FREQ))
            # restore
            try:
                self._send_cmd(base_x, HEAD_CMD_X)
                time.sleep(0.01)
                self._send_cmd(base_y, HEAD_CMD_Y)
            except Exception:
                pass
        self._jitter_thread = threading.Thread(target=_loop, daemon=True)
        self._jitter_thread.start()

    def stop_jitter(self):
        if self._jitter_thread:
            self._jitter_stop.set()
            self._jitter_thread = None

# ---------------- CameraManager (minimal) ----------------
class CameraManager:
    def __init__(self, input_url, size=(FRAME_W, FRAME_H)):
        self.input_url = input_url
        self.w, self.h = size
        self.lock = threading.Lock()
        self.cap = None

    def _open(self):
        try:
            cap = cv2.VideoCapture(self.input_url)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            return cap
        except Exception:
            return None

    def get_frame(self):
        with self.lock:
            if self.cap is None:
                self.cap = self._open()
            if self.cap is None:
                return None
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    return None
                if (frame.shape[1], frame.shape[0]) != (self.w, self.h):
                    frame = cv2.resize(frame, (self.w, self.h))
                return frame
            except Exception:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                return None

    def close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

# ---------------- Vision worker ----------------
class VisionWorker(threading.Thread):
    def __init__(self, cam_mgr, interval=VISION_INTERVAL, model=VISION_MODEL):
        super().__init__(daemon=True)
        self.cam = cam_mgr
        self.interval = interval
        self.model = model
        self.running = False
        self.lock = threading.Lock()
        self.latest_text = ""
        self.latest_persons = 0
        self.latest_x = FRAME_W//2
        self.latest_y = FRAME_H//2
        self.latest_frame = None

    def run(self):
        self.running = True
        while self.running:
            frame = self.cam.get_frame()
            if frame is None:
                with self.lock:
                    self.latest_text = "[no camera]"
                    self.latest_persons = 0
                    self.latest_x = FRAME_W//2
                    self.latest_y = FRAME_H//2
                    self.latest_frame = None
                time.sleep(self.interval)
                continue
            try:
                items = scene_objects_from_frame(frame)
                txt = ", ".join(items[:6]) if items else ""
            except Exception:
                txt = ""
            persons = 1 if re.search(r"\b(person|people|human)\b", (txt or "").lower()) else 0
            tx = FRAME_W//2; ty = FRAME_H//2
            t = (txt or "").lower()
            if "left" in t: tx = int(FRAME_W * 0.25)
            if "right" in t: tx = int(FRAME_W * 0.75)
            if "up" in t: ty = int(FRAME_H * 0.25)
            if "down" in t: ty = int(FRAME_H * 0.75)
            with self.lock:
                self.latest_text = txt
                self.latest_persons = persons
                self.latest_x = tx
                self.latest_y = ty
                self.latest_frame = frame.copy()
            time.sleep(self.interval)

    def get_latest(self):
        with self.lock:
            return (self.latest_text, self.latest_persons, self.latest_x, self.latest_y,
                    (self.latest_frame.copy() if self.latest_frame is not None else None))

    def stop(self):
        self.running = False

# ---------------- Conversation manager (minimal) ----------------
class ConversationManager:
    """
    Minimal conversation manager that uses your external modules:
    - speech.AlfredSpeak(text)
    - listen.listen()
    - arduino.send_arduino(cmd) or arduino.send(cmd)
    It does not try to reimplement your speech/listen or arduino code.
    """
    def __init__(self, cam_mgr, vision_worker, head_ctrl, memory):
        self.cam = cam_mgr
        self.vision = vision_worker
        self.head = head_ctrl
        self.memory = memory
        self.state = "idle"
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    # wrapper helpers to call external modules
    def _speak(self, text, wait=True):
        try:
            if external_speech and hasattr(external_speech, "AlfredSpeak"):
                external_speech.AlfredSpeak(text)
                return True
            return False
        except Exception:
            return False

    def _listen(self, timeout=None):
        try:
            if external_listen is None:
                return None
            if hasattr(external_listen, "listen"):
                return external_listen.listen()
            elif callable(external_listen):
                return external_listen()
            else:
                return None
        except Exception:
            return None

    def _send_arduino(self, cmd):
        try:
            if external_arduino is None:
                return
            if hasattr(external_arduino, "send_arduino"):
                external_arduino.send_arduino(cmd)
                return
            if hasattr(external_arduino, "send"):
                external_arduino.send(cmd)
                return
            if callable(external_arduino):
                external_arduino(cmd)
                return
        except Exception:
            pass

    def _scan_for_object(self, obj):
        """Use SCAN_SEQUENCE and external_arduino to move and inspect vision frames."""
        if not obj:
            return False
        obj = obj.lower().strip()
        # announce
        self._speak(f"Okay — I will look around for the {obj}.", wait=False)
        found = False
        for cmd, delay in SCAN_SEQUENCE:
            # send command then wait a bit
            try:
                # send via serial wrapper
                self._send_arduino(cmd)
                # give servo/time to move
                t0 = time.time()
                while time.time() - t0 < delay:
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)
                # quick vision check
                vtext, persons, tx, ty, frame = self.vision.get_latest()
                items = scene_objects_from_frame(frame) if frame is not None else []
                if any(obj == it or obj in it for it in items) or obj in (vtext or "").lower():
                    # focus
                    if self.head:
                        self.head.force_move(tx, ty)
                    self._speak(f"Found the {obj}.", wait=True)
                    found = True
                    break
            except Exception:
                continue
        if not found:
            self._speak(f"Sorry, I couldn't find a {obj}.", wait=True)
        return found

    def _loop(self):
        # Very small placeholder loop: listen for simple "look at X" commands from external_listen.
        while not self._stop.is_set():
            try:
                # we poll vision for an interesting object; if found announce (non-invasive)
                vtext, persons, tx, ty, frame = self.vision.get_latest()
                # If someone asks externally (main app) we expect main app to call into this manager
                # For demonstration, sleep to avoid busy-loop
                time.sleep(1.0)
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)

# ---------------- Module-level start/stop API ----------------
MODULE_STATE = {"running": False}
MODULE_COMPONENTS = {}

def start_service(background=True, use_camera_input=None,
                  enable_breathing=ENABLE_BREATHING, enable_jitter=ENABLE_JITTER):
    """
    Create components and start vision and conversation manager.
    - use_camera_input: str or None. If None, will use Alfred_config.CHEST_CAMERA_INPUT if available.
    - Returns True on success.
    """
    global MODULE_STATE, MODULE_COMPONENTS, ENABLE_BREATHING, ENABLE_JITTER

    if MODULE_STATE["running"]:
        return False

    cam_input = use_camera_input if use_camera_input is not None else CAMERA_INPUT_DEFAULT
    if cam_input is None:
        raise RuntimeError("No camera input provided. Set Alfred_config.CHEST_CAMERA_INPUT or pass use_camera_input.")

    ENABLE_BREATHING = bool(enable_breathing)
    ENABLE_JITTER = bool(enable_jitter)

    cam = CameraManager(cam_input, size=(FRAME_W, FRAME_H))
    vision = VisionWorker(cam, interval=VISION_INTERVAL, model=VISION_MODEL)
    head = HeadController()
    memory = PersonMemory()

    # start vision worker
    vision.start()

    # start breathing if requested
    if ENABLE_BREATHING:
        head.start_breathing(amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD)

    conv = ConversationManager(cam, vision, head, memory)
    conv.start()

    MODULE_COMPONENTS.update({
        "cam": cam,
        "vision": vision,
        "head": head,
        "memory": memory,
        "conv": conv
    })
    MODULE_STATE["running"] = True
    return True

def stop_service():
    """Stop running components started by start_service()."""
    global MODULE_STATE, MODULE_COMPONENTS
    if not MODULE_STATE["running"]:
        return False
    comp = MODULE_COMPONENTS
    try:
        if comp.get("conv"):
            comp["conv"].stop()
    except Exception:
        pass
    try:
        if comp.get("vision"):
            comp["vision"].stop()
    except Exception:
        pass
    try:
        if comp.get("cam"):
            comp["cam"].close()
    except Exception:
        pass
    try:
        if comp.get("head"):
            comp["head"].stop_breathing()
            comp["head"].stop_jitter()
    except Exception:
        pass
    MODULE_COMPONENTS.clear()
    MODULE_STATE["running"] = False
    return True

# ---------------- runtime control helpers ----------------
def set_memory_limits(max_people=None, max_conversations=None, max_topics=None):
    global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
    if max_people is not None:
        MEMORY_MAX_PEOPLE = int(max_people)
    if max_conversations is not None:
        MEMORY_MAX_CONVERSATIONS_PER_PERSON = int(max_conversations)
    if max_topics is not None:
        MEMORY_MAX_TOPICS_PER_PERSON = int(max_topics)
    # enforce now if running
    mem = MODULE_COMPONENTS.get("memory")
    if mem:
        mem._ensure_limits()
        mem._save()

def set_enable_breathing(flag: bool):
    global ENABLE_BREATHING
    ENABLE_BREATHING = bool(flag)
    head = MODULE_COMPONENTS.get("head")
    if head:
        if ENABLE_BREATHING:
            head.start_breathing(amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD)
        else:
            head.stop_breathing()

def set_enable_jitter(flag: bool):
    global ENABLE_JITTER
    ENABLE_JITTER = bool(flag)
    head = MODULE_COMPONENTS.get("head")
    if head and not ENABLE_JITTER:
        head.stop_jitter()

# no auto-run on import
__all__ = [
    "start_service", "stop_service", "set_memory_limits",
    "set_enable_breathing", "set_enable_jitter", "MODULE_STATE", "MODULE_COMPONENTS"
]



















######"""
###### NOT SO GOOD
######alfred_module_interface.py
######
######Minimal, non-auto-running Alfred module that uses your existing modules:
######- speech (expects `speech` instance with AlfredSpeak(...) and control methods)
######- listen (expects callable/listen.listen())
######- arduino_com (expects `arduino` with send_arduino(cmd))
######- Alfred_config (optional, for CHEST_CAMERA_INPUT)
######
######This file avoids running anything on import. Call start_service(...) to start.
######"""
######
######import threading
######import json
######import time
######import traceback
######import re
######import cv2
######import base64
######import requests
######import numpy as np
######import os
######from collections import Counter
######from math import sqrt
######import math
######import random
######
####### --- user modules (expected to exist) ---
######try:
######    from speech import speech as external_speech
######except Exception:
######    external_speech = None
######
######try:
######    from listen import listen as external_listen
######except Exception:
######    # allow either module or callable named listen
######    try:
######        import listen as _listen_mod
######        external_listen = getattr(_listen_mod, "listen", _listen_mod)
######    except Exception:
######        external_listen = None
######
######try:
######    from arduino_com import arduino as external_arduino
######except Exception:
######    external_arduino = None
######
######try:
######    import Alfred_config
######    CAMERA_INPUT = getattr(Alfred_config, "CHEST_CAMERA_INPUT", None)
######except Exception:
######    Alfred_config = None
######    CAMERA_INPUT = None
######
####### ---------------- CONFIG ----------------
######FRAME_W, FRAME_H = 640, 480
######
######VISION_MODEL = "qwen3-vl:2b"
######LANG_MODEL = "dolphin-phi"
######OLLAMA_URL = "http://localhost:11434/api/generate"
######
######VISION_INTERVAL = 0.3
######
######MEMORY_FILE = "people_memory.json"
######
####### Head mapping (kept so head controller can be used unchanged)
######HEAD_CMD_X = "X"
######HEAD_CMD_Y = "Y"
######SWAP_AXES = False
######INVERT_X = False
######INVERT_Y = True
######HEAD_X_MIN = 0
######HEAD_X_MAX = 640
######HEAD_Y_MIN = 0
######HEAD_Y_MAX = 480
######
######SMOOTH = 0.18
######SEND_THRESHOLD_PIX = 3
######
####### USER-SUPPLIED SCAN SEQUENCE (kept as requested)
######SCAN_SEQUENCE = [
######    ("p", 2), ("f", 2), ("q", 3), ("e", 3),
######    ("M", 3), ("f", 3), ("q", 3), ("w", 3),
######    ("f", 3), ("e", 3)
######]
######
####### Memory limits (adjustable via set_memory_limits())
######MEMORY_MAX_PEOPLE = 40
######MEMORY_MAX_CONVERSATIONS_PER_PERSON = 120
######MEMORY_MAX_TOPICS_PER_PERSON = 40
######
####### Idle micro-motion toggles and defaults (adjustable via set_enable_* functions)
######ENABLE_BREATHING = True
######BREATH_PERIOD = 10.0
######BREATH_AMPLITUDE_X = 6.0
######BREATH_AMPLITUDE_Y = 4.0
######
######ENABLE_JITTER = True
######JITTER_AMPLITUDE = 5
######JITTER_DURATION = 3.0
######JITTER_FREQ = 0.12
######
####### ---------------- Helpers: vision / LLM ----------------
######def llm_generate(prompt, timeout=20):
######    """Simple wrapper to call local Ollama-like endpoint. Returns string or ''."""
######    try:
######        payload = {"model": LANG_MODEL, "prompt": prompt, "stream": False}
######        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
######        d = r.json() if r is not None else {}
######        return d.get("response","") if isinstance(d, dict) else str(d)
######    except Exception:
######        return ""
######
######def scene_objects_from_frame(frame):
######    """Return list of short object names detected by LLM from an image frame. Non-fatal on errors."""
######    if frame is None:
######        return []
######    try:
######        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
######        b64 = base64.b64encode(buf).decode("utf-8")
######        prompt = "List distinct visible objects in this image as a comma-separated list (short words)."
######        payload = {"model": VISION_MODEL, "prompt": prompt, "images":[b64], "stream": False}
######        r = requests.post(OLLAMA_URL, json=payload, timeout=18)
######        j = r.json() if r is not None else {}
######        text = j.get("response", "") if isinstance(j, dict) else str(j)
######        items = re.split(r"[,\n]+", text)
######        items = [i.strip().lower() for i in items if i.strip()]
######        if not items:
######            matches = re.findall(r"\b(chair|table|door|window|sofa|phone|bag|book|cup|bottle|tv|lamp|plant|desk)\b", text.lower())
######            items = list(set(matches))
######        return items
######    except Exception:
######        return []
######
####### ---------------- PersonMemory (kept, with caps enforcement) ----------------
######class PersonMemory:
######    def __init__(self, filename=MEMORY_FILE):
######        self.filename = filename
######        self.people = []
######        self._load()
######
######    def _load(self):
######        try:
######            if os.path.exists(self.filename):
######                with open(self.filename, "r", encoding="utf-8") as f:
######                    self.people = json.load(f)
######        except Exception:
######            self.people = []
######
######    def _save(self):
######        try:
######            with open(self.filename, "w", encoding="utf-8") as f:
######                json.dump(self.people, f, indent=2)
######        except Exception:
######            pass
######
######    def _ensure_limits(self):
######        global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
######        if MEMORY_MAX_PEOPLE is not None:
######            while len(self.people) > MEMORY_MAX_PEOPLE:
######                self.people.pop(0)
######        # per-person caps
######        for p in self.people:
######            if "conversation_history" in p and MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
######                while len(p["conversation_history"]) > MEMORY_MAX_CONVERSATIONS_PER_PERSON:
######                    p["conversation_history"].pop(0)
######            if "mentioned_topics" in p and MEMORY_MAX_TOPICS_PER_PERSON is not None:
######                while len(p["mentioned_topics"]) > MEMORY_MAX_TOPICS_PER_PERSON:
######                    p["mentioned_topics"].pop(0)
######        self._save()
######
######    def add_person(self, name, enc=None):
######        rec = {"name": name, "enc_type": None, "enc": enc, "last_seen": time.time(), "last_interaction": time.time(), "mentioned_topics": [], "conversation_history": []}
######        self.people.append(rec)
######        self._ensure_limits()
######
######    def append_conversation(self, name, text):
######        for p in self.people:
######            if p.get("name") == name:
######                p.setdefault("conversation_history", []).append({"ts": time.time(), "text": text})
######                p["last_interaction"] = time.time()
######                p["last_seen"] = time.time()
######                self._ensure_limits()
######                return
######
######    def add_mentioned_topic(self, name, topic):
######        for p in self.people:
######            if p.get("name") == name:
######                lst = p.setdefault("mentioned_topics", [])
######                if topic not in lst:
######                    lst.append(topic)
######                    self._ensure_limits()
######                return
######
######    def update_last_seen(self, name):
######        for p in self.people:
######            if p.get("name") == name:
######                p["last_seen"] = time.time()
######                p["last_interaction"] = time.time()
######                self._save()
######                return
######
######    def get_person(self, name):
######        for p in self.people:
######            if p.get("name") == name:
######                return p
######        return None
######
####### ---------------- HeadController (sends arduino commands only) ----------------
######class HeadController:
######    def __init__(self):
######        self.last_x = int(FRAME_W // 2)
######        self.last_y = int(FRAME_H // 2)
######
######        # breathing/jitter threads
######        self._breath_thread = None
######        self._breath_stop = threading.Event()
######        self._jitter_thread = None
######        self._jitter_stop = threading.Event()
######
######    def _send_cmd(self, val, cmd):
######        msg = f"{cmd}{int(val)}Z"
######        # prefer user's arduino API if present
######        if external_arduino and hasattr(external_arduino, "send_arduino"):
######            try:
######                external_arduino.send_arduino(msg)
######                return
######            except Exception:
######                pass
######        # if external_arduino missing, do nothing (user asked to avoid in-module serial)
######        return
######
######    def force_move(self, tx, ty):
######        txi = int(tx); tyi = int(ty)
######        # clamp
######        txi = max(HEAD_X_MIN, min(HEAD_X_MAX, txi))
######        tyi = max(HEAD_Y_MIN, min(HEAD_Y_MAX, tyi))
######        self._send_cmd(txi, HEAD_CMD_X)
######        # small delay optional
######        time.sleep(0.02)
######        self._send_cmd(tyi, HEAD_CMD_Y)
######        self.last_x = txi; self.last_y = tyi
######
######    def start_breathing(self, amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD):
######        if self._breath_thread and self._breath_thread.is_alive():
######            return
######        self._breath_stop.clear()
######        def _loop():
######            t0 = time.time()
######            while not self._breath_stop.is_set():
######                t = time.time() - t0
######                ox = amplitude_x * math.sin(2.0 * math.pi * t / (period + 1e-6))
######                oy = amplitude_y * math.sin(2.0 * math.pi * t / (period + 1e-6) + 0.5)
######                try:
######                    self._send_cmd(int((FRAME_W//2) + ox), HEAD_CMD_X)
######                    time.sleep(0.02)
######                    self._send_cmd(int((FRAME_H//2) + oy), HEAD_CMD_Y)
######                except Exception:
######                    pass
######                time.sleep(max(0.08, period/80.0))
######        self._breath_thread = threading.Thread(target=_loop, daemon=True)
######        self._breath_thread.start()
######
######    def stop_breathing(self):
######        if self._breath_thread:
######            self._breath_stop.set()
######            self._breath_thread = None
######
######    def start_jitter(self, duration=JITTER_DURATION, amplitude=JITTER_AMPLITUDE):
######        if self._jitter_thread and self._jitter_thread.is_alive():
######            return
######        self._jitter_stop.clear()
######        def _loop():
######            end = time.time() + duration
######            while not self._jitter_stop.is_set() and time.time() < end:
######                rx = int(random.uniform(-amplitude, amplitude))
######                ry = int(random.uniform(-amplitude, amplitude))
######                try:
######                    self._send_cmd(self.last_x + rx, HEAD_CMD_X)
######                    time.sleep(0.01)
######                    self._send_cmd(self.last_y + ry, HEAD_CMD_Y)
######                except Exception:
######                    pass
######                time.sleep(max(0.02, JITTER_FREQ))
######            # restore
######            try:
######                self._send_cmd(self.last_x, HEAD_CMD_X)
######                time.sleep(0.01)
######                self._send_cmd(self.last_y, HEAD_CMD_Y)
######            except Exception:
######                pass
######        self._jitter_thread = threading.Thread(target=_loop, daemon=True)
######        self._jitter_thread.start()
######
######    def stop_jitter(self):
######        if self._jitter_thread:
######            self._jitter_stop.set()
######            self._jitter_thread = None
######
####### ---------------- CameraManager (minimal) ----------------
######class CameraManager:
######    def __init__(self, input_url, size=(FRAME_W, FRAME_H)):
######        self.input_url = input_url
######        self.w, self.h = size
######        self.lock = threading.Lock()
######        self.cap = None
######
######    def _open(self):
######        try:
######            cap = cv2.VideoCapture(self.input_url)
######            if not cap or not cap.isOpened():
######                if cap: cap.release()
######                return None
######            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
######            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
######            return cap
######        except Exception:
######            return None
######
######    def get_frame(self):
######        with self.lock:
######            if self.cap is None:
######                self.cap = self._open()
######                if self.cap is None:
######                    return None
######            try:
######                ret, frame = self.cap.read()
######                if not ret or frame is None:
######                    try: self.cap.release()
######                    except Exception: pass
######                    self.cap = None
######                    return None
######                if (frame.shape[1], frame.shape[0]) != (self.w, self.h):
######                    frame = cv2.resize(frame, (self.w, self.h))
######                return frame
######            except Exception:
######                try: self.cap.release()
######                except Exception: pass
######                self.cap = None
######                return None
######
######    def close(self):
######        try:
######            if self.cap:
######                self.cap.release()
######        except Exception:
######            pass
######
####### ---------------- Vision worker (optional helper thread) ----------------
######class VisionWorker(threading.Thread):
######    def __init__(self, cam_mgr, interval=VISION_INTERVAL, model=VISION_MODEL):
######        super().__init__(daemon=True)
######        self.cam = cam_mgr
######        self.interval = interval
######        self.model = model
######        self.running = False
######        self.lock = threading.Lock()
######        self.latest_text = ""
######        self.latest_persons = 0
######        self.latest_x = FRAME_W//2
######        self.latest_y = FRAME_H//2
######        self.latest_frame = None
######
######    def run(self):
######        self.running = True
######        while self.running:
######            frame = self.cam.get_frame()
######            if frame is None:
######                with self.lock:
######                    self.latest_text = "[no camera]"
######                    self.latest_persons = 0
######                    self.latest_x = FRAME_W//2
######                    self.latest_y = FRAME_H//2
######                    self.latest_frame = None
######                time.sleep(self.interval)
######                continue
######            # Optionally call an LLM to describe — kept minimal and resilient
######            try:
######                # we just use a cheap rule: call scene_objects_from_frame and set coarse x/y
######                items = scene_objects_from_frame(frame)
######                txt = ", ".join(items[:6]) if items else ""
######            except Exception:
######                txt = ""
######            persons = 1 if re.search(r"\b(person|people|human)\b", txt.lower()) else 0
######            tx = FRAME_W//2; ty = FRAME_H//2
######            t = (txt or "").lower()
######            if "left" in t: tx = int(FRAME_W * 0.25)
######            if "right" in t: tx = int(FRAME_W * 0.75)
######            if "up" in t: ty = int(FRAME_H * 0.25)
######            if "down" in t: ty = int(FRAME_H * 0.75)
######            with self.lock:
######                self.latest_text = txt
######                self.latest_persons = persons
######                self.latest_x = tx
######                self.latest_y = ty
######                self.latest_frame = frame.copy()
######            time.sleep(self.interval)
######
######    def get_latest(self):
######        with self.lock:
######            return (self.latest_text, self.latest_persons, self.latest_x, self.latest_y, (self.latest_frame.copy() if self.latest_frame is not None else None))
######
######    def stop(self):
######        self.running = False
######
####### ---------------- Conversation manager skeleton ----------------
######class ConversationManager:
######    """
######    Minimal conversation manager scaffolding that *uses* your modules:
######    - uses external_speech.AlfredSpeak(text) for speaking
######    - uses external_listen.listen() (or callable) for listening
######    - uses external_arduino.send_arduino(cmd) for scanning moves
######    This manager does NOT start automatically on import.
######    Call start() to start the background loop thread; stop() to stop it.
######    """
######    def __init__(self, cam_mgr, vision_worker, head_ctrl, memory):
######        self.cam = cam_mgr
######        self.vision = vision_worker
######        self.head = head_ctrl
######        self.memory = memory
######        self.state = "idle"
######        self._thread = None
######        self._stop = threading.Event()
######
######    def start(self):
######        if self._thread and self._thread.is_alive():
######            return
######        self._stop.clear()
######        self._thread = threading.Thread(target=self._loop, daemon=True)
######        self._thread.start()
######
######    def stop(self):
######        self._stop.set()
######        if self._thread:
######            self._thread.join(timeout=1.0)
######            self._thread = None
######
######    def _speak(self, text, wait=True):
######        """Use user's speech module if available, otherwise raise."""
######        if external_speech and hasattr(external_speech, "AlfredSpeak"):
######            try:
######                external_speech.AlfredSpeak(text)
######                return True
######            except Exception:
######                return False
######        else:
######            # no speech module available — nothing to do (user asked to call their modules)
######            return False
######
######    def _listen(self, timeout=None):
######        """Call user's listen module if available."""
######        if external_listen is None:
######            return None
######        try:
######            if hasattr(external_listen, "listen"):
######                return external_listen.listen()
######            elif callable(external_listen):
######                return external_listen()
######            else:
######                return None
######        except Exception:
######            return None
######
######    def _scan_for_object(self, obj):
######        """Use SCAN_SEQUENCE and external_arduino to move and inspect vision frames."""
######        for cmd, delay in SCAN_SEQUENCE:
######            try:
######                if external_arduino and hasattr(external_arduino, "send_arduino"):
######                    external_arduino.send_arduino(cmd)
######                time.sleep(delay)
######                # get frame and check quickly
######                vtext, persons, tx, ty, frame = self.vision.get_latest()
######                items = scene_objects_from_frame(frame) if frame is not None else []
######                if any(obj == it or obj in it for it in items) or obj in (vtext or "").lower():
######                    # focus
######                    if self.head:
######                        self.head.force_move(tx, ty)
######                    if external_speech:
######                        external_speech.AlfredSpeak(f"Found the {obj}.")
######                    return True
######            except Exception:
######                continue
######        if external_speech:
######            external_speech.AlfredSpeak(f"Sorry, I couldn't find a {obj}.")
######        return False
######
######    def _loop(self):
######        """Very small placeholder loop — keeps minimal behaviour and demonstrates calls to your modules.
######           The user requested to keep module minimal; extend inside main app as needed."""
######        while not self._stop.is_set():
######            # sample: if vision sees something interesting, announce (non-invasive)
######            vtext, persons, tx, ty, frame = self.vision.get_latest()
######            # simple example behaviour — you can replace/extend in your main app
######            time.sleep(1.0)
######
####### ---------------- Module-level start/stop API ----------------
######MODULE_STATE = {"running": False}
######MODULE_COMPONENTS = {}
######
######def start_service(background=True, use_camera_input=Alfred_config.CHEST_CAMERA_INPUT):
######    """
######    Create components and start vision and conversation manager.
######    This function does not run automatically; call it from your main program to start the system.
######    Returns True on success.
######    """
######    if MODULE_STATE["running"]:
######        return False
######
######    # choose camera input
######    cam_input = use_camera_input or CAMERA_INPUT
######    if not cam_input:
######        # no camera provided (user asked to use Alfred_config). Fail safely.
######        raise RuntimeError("No camera input provided. Set Alfred_config.CHEST_CAMERA_INPUT or pass use_camera_input.")
######
######    cam = CameraManager(cam_input, size=(FRAME_W, FRAME_H))
######    vision = VisionWorker(cam)
######    head = HeadController()
######    memory = PersonMemory()
######
######    # start vision worker
######    vision.start()
######
######    conv = ConversationManager(cam, vision, head, memory)
######    conv.start()
######
######    # start breathing if enabled
######    if ENABLE_BREATHING:
######        head.start_breathing(amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD)
######
######    MODULE_COMPONENTS.update({
######        "cam": cam,
######        "vision": vision,
######        "head": head,
######        "memory": memory,
######        "conv": conv
######    })
######    MODULE_STATE["running"] = True
######    return True
######
######def stop_service():
######    """Stop the running components started by start_service."""
######    if not MODULE_STATE["running"]:
######        return False
######    comp = MODULE_COMPONENTS
######    try:
######        if comp.get("conv"):
######            comp["conv"].stop()
######    except Exception:
######        pass
######    try:
######        if comp.get("vision"):
######            comp["vision"].stop()
######    except Exception:
######        pass
######    try:
######        if comp.get("cam"):
######            comp["cam"].close()
######    except Exception:
######        pass
######    try:
######        if comp.get("head"):
######            comp["head"].stop_breathing()
######            comp["head"].stop_jitter()
######    except Exception:
######        pass
######    MODULE_COMPONENTS.clear()
######    MODULE_STATE["running"] = False
######    return True
######
####### ---------------- Helpers to adjust runtime settings from main.py ----------------
######def set_memory_limits(max_people=None, max_conversations=None, max_topics=None):
######    global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
######    if max_people is not None:
######        MEMORY_MAX_PEOPLE = int(max_people)
######    if max_conversations is not None:
######        MEMORY_MAX_CONVERSATIONS_PER_PERSON = int(max_conversations)
######    if max_topics is not None:
######        MEMORY_MAX_TOPICS_PER_PERSON = int(max_topics)
######    # enforce now if running
######    mem = MODULE_COMPONENTS.get("memory")
######    if mem:
######        mem._ensure_limits()
######
######def set_enable_breathing(flag: bool):
######    global ENABLE_BREATHING
######    ENABLE_BREATHING = bool(flag)
######    if not ENABLE_BREATHING and MODULE_COMPONENTS.get("head"):
######        MODULE_COMPONENTS["head"].stop_breathing()
######
######def set_enable_jitter(flag: bool):
######    global ENABLE_JITTER
######    ENABLE_JITTER = bool(flag)
######    if not ENABLE_JITTER and MODULE_COMPONENTS.get("head"):
######        MODULE_COMPONENTS["head"].stop_jitter()
######
####### no code runs on import — caller must call start_service() to run


















### alfred_conversation_whisper_tiny_full_listen_fix.py
### (Updated: adds Arduino HEAD_POS parsing + small/exreme head move commands + small-step control
### Minimal modifications only; original structure preserved.)
###
### Changes in this fixed version:
### - Memory size controls and helper to set them at runtime.
### - Exposed start_service/stop_service API so main.py can start/stop this module.
### - Option to use external speech.listen/arduino modules (via flags).
### - Breathing + jitter micro-motion features with toggle functions and easy-to-set parameters.
### - Use Alfred_config.CHEST_CAMERA_INPUT for camera input if available (fallbacks remain).
### - Scanning uses the prescribed SCAN_SEQUENCE (arduino commands + delays).
### - TTS: option to delegate speak calls to external speech module (speech.AlfredSpeak).
### - Avoid SSML: plain text only by default; if you have a custom external module using edge-tts,
###   it will run that module; module does not craft SSML itself.
### - Did not change structure beyond the above; kept your original logic, names and flow.
##
##import threading
##import json
##import queue
##import traceback
##import re
##import datetime
##import cv2
##import base64
##import requests
##import numpy as np
##import sounddevice as sd
##import scipy.io.wavfile as wavfile
##import serial
##import os, sys
##import time
##from collections import Counter
##from math import sqrt
##import math
##import random
##
##try:
##    import Alfred_config
##    if PRIMARY_CAM is None:
##        PRIMARY_CAM = Alfred_config.CHEST_CAMERA_INPUT
##    if FALLBACK_CAM is None:
##        FALLBACK_CAM = Alfred_config.LEFT_EYE_CAMERA_INPUT
##except Exception:
##    pass
##
### ---------------- CONFIG ----------------
##SERIAL_PORT = "COM6"
##SERIAL_BAUD = 9600
##SERIAL_TIMEOUT = 0.1
##
##FALLBACK_LEFT_IP  = Alfred_config.LEFT_EYE_CAMERA_INPUT_NEW
##FALLBACK_RIGHT_IP = Alfred_config.RIGHT_EYE_CAMERA_INPUT_NEW
##def make_stream_url(ip): return f"http://{ip}:81/stream"
##
##FRAME_W, FRAME_H = 640, 480
##
####VISION_MODEL = "llava-phi3:latest"
##VISION_MODEL = "qwen3-vl:2b"
##LANG_MODEL = "dolphin-phi"
##OLLAMA_URL = "http://localhost:11434/api/generate"
##
##VISION_INTERVAL = 0.3
##RECORD_SR = 16000
##RECORD_LANG_SECONDS = 10
##WAKE_LISTEN_SECONDS = 2
##CONTEMPLATE_PAUSE = 6
##
##MEMORY_FILE = "people_memory.json"
##AUDIO_TEMP = "alfred_temp.wav"
##
### whisper tiny cache (your path)
##WHISPER_CACHE_DIR = r"C:\Users\ITF\.cache"
##WHISPER_MODEL_NAME = "tiny"
##
### ASR & face prefs
##TRY_WHISPER = True
##TRY_VOSK = True
##TRY_FACE_RECOG = True
##
### TTS local fallback
##USE_PYTTX3 = True
##IS_SPEAKING = False
##
### Head smoothing & thresholds
##SMOOTH = 0.18
##SEND_THRESHOLD_PIX = 3
##
### Wake words
##WAKE_WORDS = ["hallo","hello","alfred","hey","hi","can you","what is your","yes","yeah","yep","sure"]
##
### ---------------- HEAD MAPPING CONFIG (edit if needed) ----------------
##HEAD_CMD_X = "X"    # horizontal
##HEAD_CMD_Y = "Y"    # vertical
##SWAP_AXES = False
##INVERT_X = False
##INVERT_Y = True
##HEAD_X_MIN = 0
##HEAD_X_MAX = 640
##HEAD_Y_MIN = 0
##HEAD_Y_MAX = 480
##
### How long of inactivity (since last_interaction) before we scan for someone else
##INACTIVITY_SCAN_SECONDS = 30.0
##
### ---------------- Memory caps (customizable) ----------------
##MEMORY_MAX_PEOPLE = 40
##MEMORY_MAX_CONVERSATIONS_PER_PERSON = 120
##MEMORY_MAX_TOPICS_PER_PERSON = 40
##
### ---------- SCAN SEQUENCE (Arduino commands + delays) ----------
### Full sequence: (command, delay_seconds)
##SCAN_SEQUENCE = [
##    ("p", 2), ("f", 2), ("q", 3), ("e", 3),
##    ("M", 3), ("f", 3), ("q", 3), ("w", 3),
##    ("f", 3), ("e", 3)
##]
##
### ---------- Idle micro-motion parameters / toggles ----------
##ENABLE_BREATHING = True
##BREATH_PERIOD = 10.0        # seconds per breathing cycle
##BREATH_AMPLITUDE_X = 6.0    # pixels (or normalized units)
##BREATH_AMPLITUDE_Y = 4.0
##
##ENABLE_JITTER = True
##JITTER_AMPLITUDE = 5       # pixels
##JITTER_DURATION = 3.0      # seconds for jitter burst
##JITTER_FREQ = 0.12         # jitter update freq
##
### ---------- EXTERNAL INTEGRATION FLAGS ----------
##USE_EXTERNAL_SPEECH = False
##USE_EXTERNAL_LISTEN = False
##USE_EXTERNAL_ARDUINO = False
##
### try importing external helper modules if requested
##external_speech_module = None
##external_listen_module = None
##external_arduino_module = None
##if USE_EXTERNAL_SPEECH:
##    try:
##        import speech as external_speech_module
##    except Exception as e:
##        print('[ext speech] import failed', e); external_speech_module = None
##if USE_EXTERNAL_LISTEN:
##    try:
##        import listen as external_listen_module
##    except Exception as e:
##        print('[ext listen] import failed', e); external_listen_module = None
##if USE_EXTERNAL_ARDUINO:
##    try:
##        import arduino_com as external_arduino_module
##    except Exception as e:
##        print('[ext arduino] import failed', e); external_arduino_module = None
##
### ---------------- optional imports ----------------
##_face_recognition = None
##try:
##    if TRY_FACE_RECOG:
##        import face_recognition as _face_recognition
##except Exception:
##    _face_recognition = None
##
##_whisper = None
##try:
##    if TRY_WHISPER:
##        os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE_DIR
##        import whisper as _whisper
##except Exception:
##    _whisper = None
##
##_vosk_available = False
##try:
##    if TRY_VOSK:
##        from vosk import Model as VoskModel, KaldiRecognizer
##        _vosk_available = True
##except Exception:
##    _vosk_available = False
##
##_pyttsx3 = None
##try:
##    import pyttsx3 as _pyttsx3
##except Exception:
##    _pyttsx3 = None
##
### ---------------- utilities ----------------
##def now_ts():
##    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
##
##def now_seconds():
##    return time.time()
##
### ---------------- Serial IP reader (extended to parse HEAD_POS) ----------------
##class SerialIPReader:
##    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT):
##        self.port = port; self.baud = baud; self.timeout = timeout
##        self.ser = None; self.alive = False
##        self.ips = {"left": None, "right": None}
##        self._event = threading.Event()
##        self.recent_lines = []
##        self._max_lines = 12
##        self._thread = None
##
##        # new: track head pos reported by Arduino (angles 30..150 typically)
##        self.head_pos = {"x": None, "y": None}
##        self.head_event = threading.Event()  # set when a HEAD_POS line is parsed
##
##        self.open()
##
##    def open(self):
##        try:
##            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
##            time.sleep(1.0)
##            self.alive = True
##            print(f"[serial] opened {self.port}@{self.baud}")
##            self._thread = threading.Thread(target=self._reader, daemon=True)
##            self._thread.start()
##        except Exception as e:
##            print("[serial] open error:", e); self.ser = None; self.alive = False
##
##    def _push(self, ln):
##        self.recent_lines.insert(0, f"{now_ts()} {ln}")
##        if len(self.recent_lines) > self._max_lines:
##            self.recent_lines = self.recent_lines[:self._max_lines]
##
##    def _extract_ips(self, line):
##        found = {}
##        m = re.search(r"left[^0-9]*?(?:http://)?\s*'?(\d{1,3}(?:\.\d{1,3}){3})", line, re.IGNORECASE)
##        if m: found["left"] = m.group(1)
##        m = re.search(r"right[^0-9]*?(?:http://)?\s*'?(\d{1,3}(?:\.\d{1,3}){3})", line, re.IGNORECASE)
##        if m: found["right"] = m.group(1)
##        if not found:
##            ips = re.findall(r'(\d{1,3}(?:\.\d{1,3}){3})', line)
##            if ips:
##                if len(ips) == 1: found['left'] = ips[0]
##                else: found['left'] = ips[0]; found['right'] = ips[-1]
##        return found
##
##    def _parser_head_pos(self, line):
##        # expects "HEAD_POS X:<num> Y:<num>"
##        m = re.search(r"HEAD_POS\s+X\s*:\s*([0-9]+)\s*Y\s*:\s*([0-9]+)", line)
##        if m:
##            try:
##                x = int(m.group(1))
##                y = int(m.group(2))
##                prev = (self.head_pos["x"], self.head_pos["y"])
##                self.head_pos["x"] = x
##                self.head_pos["y"] = y
##                # notify waiters
##                self.head_event.set()
##                return True
##            except Exception:
##                return False
##        return False
##
##    def _reader(self):
##        while True:
##            if not self.ser or not self.ser.is_open:
##                break
##            try:
##                raw = self.ser.readline()
##                if not raw: continue
##                line = raw.decode("latin-1", errors="ignore").strip()
##                if not line: continue
##                self._push(line)
##                print("[serial rx]", line)
##
##                # parse HEAD_POS first
##                if self._parser_head_pos(line):
##                    # don't early-return; also try to extract IPs if present
##                    pass
##
##                ex = self._extract_ips(line)
##                if "left" in ex and not self.ips["left"]:
##                    self.ips["left"] = ex["left"]; print("[serial] left ip:", ex["left"])
##                if "right" in ex and not self.ips["right"]:
##                    self.ips["right"] = ex["right"]; print("[serial] right ip:", ex["right"])
##                if self.ips["left"] and self.ips["right"]:
##                    self._event.set()
##            except Exception:
##                traceback.print_exc(); time.sleep(0.1)
##
##    def wait_for_ips(self, timeout=None):
##        if self.ips["left"] and self.ips["right"]:
##            return dict(self.ips)
##        print("[serial] waiting for ips...")
##        self._event.wait(timeout=timeout)
##        return dict(self.ips)
##
##    def send(self, s):
##        if not s.endswith("\n"): s = s + "\n"
##        if self.ser and getattr(self.ser, "is_open", False):
##            try:
##                self.ser.write(s.encode("utf-8")); self.ser.flush(); print("[serial tx]", s.strip())
##            except Exception as e:
##                print("[serial tx err]", e)
##        else:
##            print("[serial tx sim]", s.strip())
##
##    def get_head_pos(self):
##        # return a copy
##        return dict(self.head_pos)
##
##    def wait_for_head_update(self, prev_pos=None, timeout=2.0):
##        """
##        Wait for a HEAD_POS report. Optionally check that it differs from prev_pos (tuple or dict).
##        Returns new pos dict or None on timeout.
##        """
##        # quick check first
##        if self.head_pos["x"] is not None and self.head_pos["y"] is not None:
##            if prev_pos is None:
##                return self.get_head_pos()
##            else:
##                prevx, prevy = prev_pos
##                if (prevx != self.head_pos["x"]) or (prevy != self.head_pos["y"]):
##                    return self.get_head_pos()
##        # wait for event
##        got = self.head_event.wait(timeout=timeout)
##        if not got:
##            return None
##        # clear event for next wait
##        self.head_event.clear()
##        return self.get_head_pos()
##
##    def close(self):
##        try:
##            if self.ser: self.ser.close()
##        except Exception:
##            pass
##
### ---------------- Camera manager ----------------
##class CameraManager:
##    def __init__(self, left_url, right_url, preferred="left", size=(FRAME_W, FRAME_H)):
##        self.left = left_url; self.right = right_url; self.preferred = preferred
##        self.w, self.h = size; self.lock = threading.Lock()
##        self.cap = None; self.label = None
##        self._connect()
##
##    def _open(self, url):
##        try:
##            cap = cv2.VideoCapture(url)
##            if not cap or not cap.isOpened():
##                if cap: cap.release(); return None
##            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
##            ret, frame = cap.read()
##            if not ret or frame is None:
##                cap.release(); return None
##            return cap
##        except Exception:
##            return None
##
##    def _connect(self):
##        pref = self.left if self.preferred == "left" else self.right
##        # FIXED: choose alt correctly based on preferred choice
##        alt = self.right if self.preferred == "left" else self.left
##        c = self._open(pref)
##        if c: self.cap = c; self.label = "LEFT" if self.preferred=="left" else "RIGHT"; return
##        c2 = self._open(alt)
##        if c2: self.cap = c2; self.label = "RIGHT" if self.preferred=="left" else "LEFT"; return
##        self.cap = None; self.label = None
##
##    def get_frame(self):
##        with self.lock:
##            if not self.cap:
##                self._connect()
##                if not self.cap: return None, None
##            try:
##                ret, frame = self.cap.read()
##                if not ret or frame is None:
##                    try: self.cap.release()
##                    except Exception: pass
##                    self.cap = None; self.label = None
##                    return None, None
##                if (frame.shape[1], frame.shape[0]) != (self.w, self.h):
##                    frame = cv2.resize(frame, (self.w, self.h))
##                return frame, self.label
##            except Exception:
##                try: self.cap.release()
##                except Exception: pass
##                self.cap = None; self.label = None
##                return None, None
##
### ---------------- Person memory ----------------
##class PersonMemory:
##    def __init__(self, filename=MEMORY_FILE):
##        self.filename = filename
##        self.people = []
##        self._load()
##
##    def _load(self):
##        if os.path.exists(self.filename):
##            try:
##                with open(self.filename, "r", encoding="utf-8") as f:
##                    self.people = json.load(f)
##                print(f"[memory] loaded {len(self.people)} people")
##            except Exception:
##                print("[memory] load err"); self.people = []
##        else:
##            self.people = []
##
##    def _save(self):
##        try:
##            with open(self.filename, "w", encoding="utf-8") as f:
##                json.dump(self.people, f, indent=2)
##        except Exception as e:
##            print("[memory] save err", e)
##
##    def _compute_encoding(self, face_img):
##        if _face_recognition:
##            try:
##                rgb = face_img[:, :, ::-1]
##                encs = _face_recognition.face_encodings(rgb)
##                if encs:
##                    return ("fr", encs[0].tolist())
##            except Exception as e:
##                print("[fr enc err]", e)
##        try:
##            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
##            hist = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
##            cv2.normalize(hist, hist)
##            return ("hist", hist.flatten().tolist())
##        except Exception as e:
##            print("[hist enc err]", e)
##            return None
##
##    def _compare(self, enc_a, enc_b):
##        if not enc_a or not enc_b:
##            return 0.0
##        ta, va = enc_a[0], np.array(enc_a[1]); tb, vb = enc_b[0], np.array(enc_b[1])
##        if ta == "fr" and tb == "fr":
##            d = np.linalg.norm(va - vb)
##            sim = max(0.0, 1.0 - (d / 0.8))
##            return float(sim)
##        try:
##            sim = cv2.compareHist(va.astype("float32"), vb.astype("float32"), cv2.HISTCMP_CORREL)
##            return float((sim + 1.0) / 2.0)
##        except Exception:
##            return 0.0
##
##    def find_match(self, face_img, min_similarity=0.6):
##        enc = self._compute_encoding(face_img)
##        if not enc:
##            return None, 0.0
##        best = None; best_score = 0.0
##        for p in self.people:
##            candidate = (p.get("enc_type"), p.get("enc"))
##            score = self._compare(enc, candidate)
##            if score > best_score:
##                best_score = score; best = p
##        if best and best_score >= min_similarity:
##            return best, best_score
##        return None, best_score
##
##    def _ensure_limits(self):
##        """Ensure memory caps: number of people and per-person conversation/topic history"""
##        global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
##        if MEMORY_MAX_PEOPLE is None:
##            return
##        # Drop oldest entries if needed (oldest at beginning)
##        if len(self.people) <= MEMORY_MAX_PEOPLE:
##            # still enforce per-person sub-limits
##            for p in self.people:
##                if "conversation_history" in p and MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
##                    while len(p["conversation_history"]) > MEMORY_MAX_CONVERSATIONS_PER_PERSON:
##                        p["conversation_history"].pop(0)
##                if "mentioned_topics" in p and MEMORY_MAX_TOPICS_PER_PERSON is not None:
##                    while len(p["mentioned_topics"]) > MEMORY_MAX_TOPICS_PER_PERSON:
##                        p["mentioned_topics"].pop(0)
##            return
##        # Trim the people list by removing oldest entries
##        while len(self.people) > MEMORY_MAX_PEOPLE:
##            self.people.pop(0)
##        # then ensure per-person caps
##        for p in self.people:
##            if "conversation_history" in p and MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
##                while len(p["conversation_history"]) > MEMORY_MAX_CONVERSATIONS_PER_PERSON:
##                    p["conversation_history"].pop(0)
##            if "mentioned_topics" in p and MEMORY_MAX_TOPICS_PER_PERSON is not None:
##                while len(p["mentioned_topics"]) > MEMORY_MAX_TOPICS_PER_PERSON:
##                    p["mentioned_topics"].pop(0)
##        # save after trimming
##        self._save()
##
##    def add_person(self, name, face_img):
##        enc = self._compute_encoding(face_img)
##        rec = {"name": name, "enc_type": None, "enc": None, "last_seen": now_ts(), "last_interaction": now_seconds(), "mentioned_topics": [], "conversation_history": []}
##        if enc:
##            rec["enc_type"] = enc[0]; rec["enc"] = enc[1]
##        self.people.append(rec)
##        # enforce limits
##        self._ensure_limits()
##        self._save()
##        print(f"[memory] added person '{name}'")
##
##    def append_conversation(self, name, text):
##        for p in self.people:
##            if p.get("name") == name:
##                p.setdefault("conversation_history", []).append({"ts": now_ts(), "text": text})
##                p["last_interaction"] = now_seconds()
##                p["last_seen"] = now_ts()
##                # enforce limits for the person
##                if "conversation_history" in p and MEMORY_MAX_CONVERSATIONS_PER_PERSON is not None:
##                    while len(p["conversation_history"]) > MEMORY_MAX_CONVERSATIONS_PER_PERSON:
##                        p["conversation_history"].pop(0)
##                self._save()
##                return
##
##    def add_mentioned_topic(self, name, topic):
##        for p in self.people:
##            if p.get("name") == name:
##                lst = p.setdefault("mentioned_topics", [])
##                if topic not in lst:
##                    lst.append(topic)
##                    # enforce topics cap
##                    if MEMORY_MAX_TOPICS_PER_PERSON is not None:
##                        while len(lst) > MEMORY_MAX_TOPICS_PER_PERSON:
##                            lst.pop(0)
##                    self._save()
##                return
##
##    def update_last_seen(self, name):
##        for p in self.people:
##            if p.get("name") == name:
##                p["last_seen"] = now_ts()
##                p["last_interaction"] = now_seconds()
##                self._save()
##                return
##
##    def get_person(self, name):
##        for p in self.people:
##            if p.get("name") == name:
##                return p
##        return None
##
### ---------------- TTS (fixed) ----------------
##class TTSEngine:
##    def __init__(self):
##        self.engine = None
##        self.q = queue.Queue()
##        self.thread = None
##        self.speaking = threading.Event()
##        self._stop_flag = False
##
##        if USE_PYTTX3 and _pyttsx3:
##            try:
##                self.engine = _pyttsx3.init()
##                rate = self.engine.getProperty("rate")
##                self.engine.setProperty("rate", max(120, rate-20))
##                self.thread = threading.Thread(target=self._worker, daemon=True)
##                self.thread.start()
##            except Exception as e:
##                print("[tts init err]", e)
##                self.engine = None
##                self.thread = threading.Thread(target=self._worker_print, daemon=True)
##                self.thread.start()
##        else:
##            print("[tts] pyttsx3 not available; fallback prints")
##            self.thread = threading.Thread(target=self._worker_print, daemon=True)
##            self.thread.start()
##
##    def _worker(self):
##        while True:
##            text = self.q.get()
##            if text is None:
##                break
##            try:
##                self.speaking.set()
##                # note: IS_SPEAKING not used globally (kept original semantics)
##                self.engine.say(text)
##                self.engine.runAndWait()
##            except Exception as e:
##                print("[tts worker err]", e)
##            finally:
##                self.speaking.clear()
##
##    def _worker_print(self):
##        while True:
##            text = self.q.get()
##            if text is None:
##                break
##            try:
##                self.speaking.set()
##                print("[TTS - fallback]", text)
##                approx = max(0.5, 0.2 * len(text.split()))
##                time.sleep(approx)
##            except Exception as e:
##                print("[tts fallback err]", e)
##            finally:
##                self.speaking.clear()
##
##    def speak(self, text, wait=False, max_wait=15.0, voice="en-US-GuyNeural", style=None):
##        """
##        Speak text.
##        If USE_EXTERNAL_SPEECH is True and external_speech_module was imported, we first try to
##        call external_speech_module.speech.AlfredSpeak(text) or external_speech_module.AlfredSpeak(text).
##        Otherwise we use the internal queue (pyttsx3 fallback).
##        Returns True on success (or best-effort).
##        """
##        if not text:
##            return True
##
##        # If configured, delegate speaking to external speech module (synchronous call).
##        if USE_EXTERNAL_SPEECH and external_speech_module:
##            try:
##                # user's external module may export 'speech' object with AlfredSpeak
##                if hasattr(external_speech_module, "speech") and hasattr(external_speech_module.speech, "AlfredSpeak"):
##                    external_speech_module.speech.AlfredSpeak(text)
##                    return True
##                if hasattr(external_speech_module, "AlfredSpeak"):
##                    external_speech_module.AlfredSpeak(text)
##                    return True
##                # fallback - try to call as a function
##                if hasattr(external_speech_module, "speak"):
##                    external_speech_module.speak(text)
##                    return True
##            except Exception as e:
##                print("[external speech err]", e)
##                # fall back to local engine
##
##        # probabilistic filler insertion before main text (small human-like filler)
##        if random.random() < 0.12:
##            filler = random.choice(["Hmm.", "Let me think...", "Ah, yes."])
##            # do not block if 'wait' is False
##            self.q.put(filler)
##
##        # push main utterance
##        self.q.put(text)
##        if not wait:
##            return True
##        start = time.time()
##        while not self.speaking.is_set():
##            if time.time() - start > max_wait:
##                return False
##            time.sleep(0.01)
##        while self.speaking.is_set():
##            if time.time() - start > max_wait:
##                return False
##            time.sleep(0.01)
##        return True
##
##    def wait_until_done(self, timeout=None):
##        start = time.time()
##        while True:
##            if self.q.empty() and not self.speaking.is_set():
##                return True
##            if timeout is not None and (time.time() - start) >= timeout:
##                return False
##            time.sleep(0.01)
##
##    def stop(self):
##        try:
##            self.q.put(None)
##        except Exception:
##            pass
##
### ---------------- Audio + ASR (safe record) ----------------
##def record_wav_safe(tts_engine, filename, seconds, sr=RECORD_SR, wait_timeout=8.0):
##
##    # ensure TTS finished before recording
##    if tts_engine is not None:
##        ok = tts_engine.wait_until_done(timeout=wait_timeout)
##        if not ok:
##            print("[record_wav_safe] warning: TTS did not finish within timeout; proceeding to record.")
##    try:
##        data = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype='int16')
##        sd.wait()
##        wavfile.write(filename, sr, data)
##        return filename
##    except Exception as e:
##        print("[audio rec err]", e); return None
##
##_whisper_model = None
##def transcribe_whisper(wav_path):
##    global _whisper_model
##    if not _whisper:
##        raise RuntimeError("whisper not installed")
##    try:
##        if _whisper_model is None:
##            _whisper_model = _whisper.load_model(WHISPER_MODEL_NAME)
##        res = _whisper_model.transcribe(wav_path, language="en")
##        txt = res.get("text","").strip()
##        print("[whisper] ->", txt)
##        return txt
##    except Exception as e:
##        print("[whisper err]", e); return ""
##
##def transcribe_vosk(wav_path, model_path=None):
##    if not _vosk_available:
##        raise RuntimeError("vosk not available")
##    if not model_path:
##        model_path = os.environ.get("VOSK_MODEL_PATH", None)
##    if not model_path or not os.path.exists(model_path):
##        raise RuntimeError("Vosk model not found")
##    import wave, json
##    wf = wave.open(wav_path, "rb")
##    model = VoskModel(model_path)
##    rec = KaldiRecognizer(model, wf.getframerate())
##    rec.SetWords(False)
##    parts = []
##    while True:
##        data = wf.readframes(4000)
##        if len(data) == 0:
##            break
##        if rec.AcceptWaveform(data):
##            parts.append(json.loads(rec.Result()).get("text",""))
##    parts.append(json.loads(rec.FinalResult()).get("text",""))
##    txt = " ".join(parts).strip()
##    print("[vosk] ->", txt)
##    return txt
##
##def detect_speech_energy(wav_path, rms_threshold=400.0):
##    """
##    Simple energy-based speech detection.
##    Returns True if audio RMS > threshold.
##    Threshold chosen for typical int16 recorded audio; adjust if needed.
##    """
##    try:
##        sr, data = wavfile.read(wav_path)
##        if data is None or data.size == 0:
##            return False
##        # normalize for mono or stereo
##        if data.ndim > 1:
##            data = data.mean(axis=1)
##        # convert to float
##        data = data.astype(np.float32)
##        rms = np.sqrt(np.mean(np.square(data)))
##        print(f"[VAD] RMS={rms:.1f} (threshold={rms_threshold})")
##        return rms >= rms_threshold
##    except Exception as e:
##        print("[VAD err]", e)
##        return False
##
##def transcribe_any(wav):
##    """
##    Attempt to transcribe using whisper -> vosk -> fallback None.
##    We do NOT block if both not installed; return empty string so callers can use energy fallback.
##    """
##    if _whisper:
##        try:
##            txt = transcribe_whisper(wav)
##            if txt:
##                return txt
##        except Exception as e:
##            print("[asr] whisper fail", e)
##    if _vosk_available:
##        try:
##            txt = transcribe_vosk(wav)
##            if txt:
##                return txt
##        except Exception as e:
##            print("[asr] vosk fail", e)
##    # nothing recognized by ASR
##    return ""
##
### ---------------- Vision worker (LLM-based descriptions) ----------------
##class VisionWorker(threading.Thread):
##    def __init__(self, cam_mgr, interval=VISION_INTERVAL, model=VISION_MODEL):
##        super().__init__(daemon=True)
##        self.cam = cam_mgr; self.interval = interval; self.model = model
##        self.latest_text = ""; self.latest_persons = 0; self.latest_x = FRAME_W//2; self.latest_y = FRAME_H//2
##        self.latest_frame = None
##        self.running = True; self.lock = threading.Lock()
##
##    def run(self):
##        while self.running:
##            frame, label = self.cam.get_frame()
##            if frame is None:
##                with self.lock:
##                    self.latest_text = "[no camera]"; self.latest_persons = 0
##                    self.latest_x = FRAME_W//2; self.latest_y = FRAME_H//2; self.latest_frame = None
##                time.sleep(self.interval); continue
##            try:
##                _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
##                b64 = base64.b64encode(buf).decode("utf-8")
##                payload = {"model": self.model, "prompt":"Describe people in the image (count and relative position). Reply concisely.", "images":[b64], "stream":False}
##                r = requests.post(OLLAMA_URL, json=payload, timeout=18)
##                data = r.json() if r is not None else {}
##                text = data.get("response","") if isinstance(data, dict) else str(data)
##            except Exception as e:
##                text = f"[vision err] {e}"
##            persons = 1 if re.search(r"\b(person|people|human)\b", text.lower()) else 0
##            tx = FRAME_W//2; ty = FRAME_H//2
##            t = text.lower()
##            if "left" in t: tx = int(FRAME_W*0.25)
##            if "right" in t: tx = int(FRAME_W*0.75)
##            if "up" in t: ty = int(FRAME_H*0.25)
##            if "down" in t: ty = int(FRAME_H*0.75)
##            with self.lock:
##                self.latest_text = text; self.latest_persons = persons; self.latest_x = tx; self.latest_y = ty; self.latest_frame = frame.copy()
##            time.sleep(self.interval)
##
##    def get_latest(self):
##        with self.lock:
##            return self.latest_text, self.latest_persons, self.latest_x, self.latest_y, (self.latest_frame.copy() if self.latest_frame is not None else None)
##
##    def stop(self):
##        self.running = False
##
### ---------------- head controller (configurable mapping & small/exreme commands) ----------------
##class HeadController:
##    def __init__(self, serial_reader):
##        self.sr = serial_reader
##        self.cx = float(FRAME_W//2); self.cy = float(FRAME_H//2)
##        self.last_x = int(round(self.cx)); self.last_y = int(round(self.cy))
##
##        # internal idle motion threads
##        self._breath_thread = None
##        self._breath_stop = threading.Event()
##        self._jitter_thread = None
##        self._jitter_stop = threading.Event()
##
##    def _clamp(self, v, lo, hi):
##        return max(lo, min(hi, v))
##
##    def _prepare_and_send(self, val, cmd):
##        cmd_str = f"{cmd}{int(val)}Z"
##        if self.sr:
##            try:
##                self.sr.send(cmd_str)
##            except Exception as e:
##                print("[HEAD TX ERR]", e, cmd_str)
##        else:
##            print("[HEAD TX SIM]", cmd_str)
##
##    def update_send(self, tx, ty):
##        try:
##            self.cx += (tx - self.cx) * SMOOTH
##            self.cy += (ty - self.cy) * SMOOTH
##        except Exception:
##            self.cx = float(tx if isinstance(tx, (int,float)) else self.cx)
##            self.cy = float(ty if isinstance(ty, (int,float)) else self.cy)
##
##        ix = int(round(self.cx)); iy = int(round(self.cy))
##
##        send_x = ix; send_y = iy
##        if SWAP_AXES:
##            send_x, send_y = iy, ix
##
##        if INVERT_X:
##            send_x = int(HEAD_X_MAX - (send_x - HEAD_X_MIN))
##        if INVERT_Y:
##            send_y = int(HEAD_Y_MAX - (send_y - HEAD_Y_MIN))
##
##        send_x = self._clamp(send_x, HEAD_X_MIN, HEAD_X_MAX)
##        send_y = self._clamp(send_y, HEAD_Y_MIN, HEAD_Y_MAX)
##
##        dx = abs(send_x - self.last_x)
##        dy = abs(send_y - self.last_y)
##
##        if dx > SEND_THRESHOLD_PIX:
##            self._prepare_and_send(send_x, HEAD_CMD_X)
##            print(f"[HEAD] Sent X => {send_x} (dx={dx})")
##            self.last_x = int(send_x)
##
##        if dy > SEND_THRESHOLD_PIX:
##            self._prepare_and_send(send_y, HEAD_CMD_Y)
##            print(f"[HEAD] Sent Y => {send_y} (dy={dy})")
##            self.last_y = int(send_y)
##
##    def force_move(self, tx, ty, send_both=True):
##        try:
##            txf = float(tx); tyf = float(ty)
##        except Exception:
##            return
##
##        ix = int(round(txf)); iy = int(round(tyf))
##        send_x = ix; send_y = iy
##        if SWAP_AXES:
##            send_x, send_y = send_y, send_x
##        if INVERT_X:
##            send_x = int(HEAD_X_MAX - (send_x - HEAD_X_MIN))
##        if INVERT_Y:
##            send_y = int(HEAD_Y_MAX - (send_y - HEAD_Y_MIN))
##        send_x = self._clamp(send_x, HEAD_X_MIN, HEAD_X_MAX)
##        send_y = self._clamp(send_y, HEAD_Y_MIN, HEAD_Y_MAX)
##
##        if send_both:
##            self._prepare_and_send(send_x, HEAD_CMD_X)
##            time.sleep(0.02)
##            self._prepare_and_send(send_y, HEAD_CMD_Y)
##            print(f"[HEAD FORCE] X->{send_x} Y->{send_y}")
##            self.last_x = int(send_x); self.last_y = int(send_y)
##            self.cx = float(ix); self.cy = float(iy)
##        else:
##            self._prepare_and_send(send_x, HEAD_CMD_X)
##            print(f"[HEAD FORCE] X->{send_x}")
##            self.last_x = int(send_x)
##            self.cx = float(ix)
##
##    # ---------- New convenience methods to use new Arduino commands ----------
##    def small_move(self, direction, wait_for_report=True, timeout=1.5):
##        """
##        direction: 'left','right','up','down' -> sends 'l','r','u','d'
##        waits for a HEAD_POS report if serial provides it.
##        """
##        if not self.sr:
##            return None
##        mapping = {"left":"l","right":"r","up":"u","down":"d", "forward":"f", "straight":"s"}
##        cmd = mapping.get(direction)
##        if not cmd:
##            return None
##        prev = (self.sr.head_pos.get("x"), self.sr.head_pos.get("y"))
##        self.sr.send(cmd)
##        if wait_for_report:
##            return self.sr.wait_for_head_update(prev_pos=prev, timeout=timeout)
##        return None
##
##    def extreme_move(self, direction, wait_for_report=True, timeout=1.5):
##        """
##        direction: 'left'->q, 'right'->e, 'up'->w, 'down'->s
##        """
##        if not self.sr:
##            return None
##        mapping = {"left":"q","right":"e","up":"w","down":"s", "forward":"f", "straight":"r"}
##        cmd = mapping.get(direction)
##        if not cmd:
##            return None
##        prev = (self.sr.head_pos.get("x"), self.sr.head_pos.get("y"))
##        self.sr.send(cmd)
##        if wait_for_report:
##            return self.sr.wait_for_head_update(prev_pos=prev, timeout=timeout)
##        return None
##
##    def set_small_step(self, degrees):
##        """
##        Set the small-step size on Arduino: send 'M<degrees>' (Arduino reports SMALL_STEP <n>)
##        """
##        if not self.sr:
##            return None
##        self.sr.send(f"M{int(degrees)}")
##        # Arduino prints "SMALL_STEP <n>" — we don't block on that, but it will be visible in serial log
##        return True
##
##    # ---------- breathing + jitter helpers ----------
##    def _breath_loop(self, amplitude_x, amplitude_y, period_s):
##        t0 = time.time()
##        while not self._breath_stop.is_set():
##            # slow sinusoidal motion around center
##            t = time.time() - t0
##            ox = amplitude_x * math.sin(2.0 * math.pi * t / (period_s + 0.001))
##            oy = amplitude_y * math.sin(2.0 * math.pi * t / (period_s + 0.001) + 0.5)
##            tx = int(round((FRAME_W // 2) + ox))
##            ty = int(round((FRAME_H // 2) + oy))
##            try:
##                self._prepare_and_send(tx, HEAD_CMD_X)
##                time.sleep(0.02)
##                self._prepare_and_send(ty, HEAD_CMD_Y)
##            except Exception:
##                pass
##            # gentle pace
##            time.sleep(max(0.08, period_s / 80.0))
##
##    def start_breathing(self, amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD):
##        if self._breath_thread and self._breath_thread.is_alive():
##            return
##        self._breath_stop.clear()
##        self._breath_thread = threading.Thread(target=self._breath_loop, args=(amplitude_x, amplitude_y, period), daemon=True)
##        self._breath_thread.start()
##        print("[HEAD] started breathing motion")
##
##    def stop_breathing(self):
##        if self._breath_thread:
##            self._breath_stop.set()
##            self._breath_thread = None
##            print("[HEAD] stopped breathing motion")
##
##    def _jitter_loop(self, duration, amplitude):
##        t_end = time.time() + duration
##        while not self._jitter_stop.is_set() and time.time() < t_end:
##            rx = int(random.uniform(-amplitude, amplitude))
##            ry = int(random.uniform(-amplitude, amplitude))
##            tx = self.last_x + rx
##            ty = self.last_y + ry
##            try:
##                self._prepare_and_send(self._clamp(tx, HEAD_X_MIN, HEAD_X_MAX), HEAD_CMD_X)
##                time.sleep(0.01)
##                self._prepare_and_send(self._clamp(ty, HEAD_Y_MIN, HEAD_Y_MAX), HEAD_CMD_Y)
##            except Exception:
##                pass
##            time.sleep(max(0.02, JITTER_FREQ))
##        # restore last known position
##        try:
##            self._prepare_and_send(self.last_x, HEAD_CMD_X)
##            time.sleep(0.01)
##            self._prepare_and_send(self.last_y, HEAD_CMD_Y)
##        except Exception:
##            pass
##
##    def start_micro_jitter(self, duration=JITTER_DURATION, amplitude=JITTER_AMPLITUDE):
##        if self._jitter_thread and self._jitter_thread.is_alive():
##            return
##        self._jitter_stop.clear()
##        self._jitter_thread = threading.Thread(target=self._jitter_loop, args=(duration, amplitude), daemon=True)
##        self._jitter_thread.start()
##        print("[HEAD] started micro-jitter")
##
##    def stop_micro_jitter(self):
##        if self._jitter_thread:
##            self._jitter_stop.set()
##            self._jitter_thread = None
##            print("[HEAD] stopped micro-jitter")
##
### ---------------- helpers: face detect & appearance analysis ----------------
##_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
##
##def detect_face_with_bbox(frame):
##    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
##    if len(faces) == 0:
##        return None, None
##    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
##    x, y, w, h = faces[0]
##    pad = int(0.15 * max(w, h))
##    x0 = max(0, x - pad); y0 = max(0, y - pad)
##    x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
##    crop = frame[y0:y1, x0:x1].copy()
##    return crop, (x0, y0, x1 - x0, y1 - y0)
##
##def dominant_color_name_from_region(img):
##    if img is None or img.size == 0:
##        return None
##    small = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
##    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(float)
##    bins = (rgb // 32).astype(int)
##    counts = Counter(map(tuple, bins.tolist()))
##    most = counts.most_common(1)[0][0]
##    avg = np.array(most) * 32 + 16
##    color_names = {
##        "black": (20,20,20),
##        "white": (240,240,240),
##        "red": (200,30,30),
##        "green": (30,160,30),
##        "blue": (40,80,200),
##        "yellow": (220,200,30),
##        "brown": (140,90,50),
##        "gray": (130,130,130),
##        "pink": (230,120,160),
##        "purple": (150,60,160),
##        "orange": (230,120,30)
##    }
##    def dist(a,b): return sqrt(((a-b)**2).sum())
##    best = None; bestd = 1e9
##    for name, val in color_names.items():
##        d = dist(avg, np.array(val))
##        if d < bestd:
##            bestd = d; best = name
##    return best
##
##def analyze_appearance(frame, bbox):
##    if frame is None or bbox is None:
##        return None, None
##    x, y, w, h = bbox
##    h_img, w_img = frame.shape[:2]
##    hair_y0 = max(0, y - int(0.6*h))
##    hair_y1 = max(0, y + int(0.1*h))
##    hair_x0 = x; hair_x1 = min(w_img, x + w)
##    hair_region = None
##    if hair_y1 > hair_y0 and hair_x1 > hair_x0:
##        hair_region = frame[hair_y0:hair_y1, hair_x0:hair_x1]
##    cloth_y0 = min(h_img, y + h + int(0.05*h))
##    cloth_y1 = min(h_img, y + h + int(0.9*h))
##    cloth_x0 = max(0, x - int(0.25*w))
##    cloth_x1 = min(w_img, x + w + int(0.25*w))
##    cloth_region = None
##    if cloth_y1 > cloth_y0 and cloth_x1 > cloth_x0:
##        cloth_region = frame[cloth_y0:cloth_y1, cloth_x0:cloth_x1]
##    hair_color = dominant_color_name_from_region(hair_region) if hair_region is not None else None
##    clothing_color = dominant_color_name_from_region(cloth_region) if cloth_region is not None else None
##    return hair_color, clothing_color
##
### ---------------- Scene extraction & LLM helpers ----------------
##def scene_objects_from_frame(frame):
##    try:
##        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
##        b64 = base64.b64encode(buf).decode("utf-8")
##        prompt = "List distinct visible objects in this image as a comma-separated list (short words)."
##        payload = {"model": VISION_MODEL, "prompt": prompt, "images":[b64], "stream": False}
##        r = requests.post(OLLAMA_URL, json=payload, timeout=18)
##        j = r.json() if r is not None else {}
##        text = j.get("response", "") if isinstance(j, dict) else str(j)
##        items = re.split(r"[,\n]+", text)
##        items = [i.strip().lower() for i in items if i.strip()]
##        if not items:
##            matches = re.findall(r"\b(chair|table|door|window|sofa|phone|bag|book|cup|bottle|tv|lamp|plant|desk|chair)\b", text.lower())
##            items = list(set(matches))
##        return items
##    except Exception as e:
##        print("[scene objects err]", e)
##        return []
##
##def llm_generate(prompt):
##    try:
##        payload = {"model": LANG_MODEL, "prompt": prompt, "stream": False}
##        r = requests.post(OLLAMA_URL, json=payload, timeout=20)
##        d = r.json() if r is not None else {}
##        return d.get("response","") if isinstance(d, dict) else str(d)
##    except Exception as e:
##        print("[llm err]", e); return ""
##
##def make_compliment(hair_color, clothing_color):
##    parts = []
##    if hair_color:
##        parts.append(f"the color of your hair ({hair_color})")
##    if clothing_color:
##        parts.append(f"your clothes ({clothing_color})")
##    if not parts:
##        return "I like your style."
##    prompt = f"Make a short friendly compliment about {', and '.join(parts)}. Keep it natural and not cheesy (one sentence)."
##    res = llm_generate(prompt)
##    if res:
##        return res.strip()
##    if hair_color and clothing_color:
##        return f"I really like your {hair_color} hair and the {clothing_color} clothes — nice look."
##    if hair_color:
##        return f"Nice {hair_color} hair — it suits you."
##    if clothing_color:
##        return f"I like the color of your {clothing_color} clothes."
##    return "You look great."
##
### ---------------- Conversation Manager ----------------
##class ConversationManager:
##    def __init__(self, serial_reader, cam_mgr, vision_worker, head_ctrl, tts_engine, memory):
##        self.sr = serial_reader; self.cam = cam_mgr; self.vision = vision_worker
##        self.head = head_ctrl; self.tts = tts_engine; self.memory = memory
##        self.state = "idle"   # idle, greeting, conversing, contemplating, scanning
##        self.prev_state = None
##        self.current_person = None   # name string
##        self.current_face_bbox = None
##        self.last_interaction_time = 0.0
##        self.thread = threading.Thread(target=self._main_loop, daemon=True)
##        self._stop = False
##        self.thread.start()
##
##    def stop(self):
##        self._stop = True
##
##    def _is_looking(self, tx, persons):
##        return persons > 0 and abs(tx - FRAME_W//2) < (FRAME_W * 0.25)
##
##    def _listen_and_transcribe(self, seconds):
##        """
##        Record and try ASR. If ASR returns empty, use energy-based VAD fallback to report
##        speech-detected placeholder so the conversation loop continues.
##        Also: if we get the <speech> placeholder, attempt to center on the closest face (if any).
##        Realism: start micro head jitter while listening if enabled.
##        """
##        # If configured, delegate to external listen module (synchronous)
##        if USE_EXTERNAL_LISTEN and external_listen_module:
##            try:
##                q = external_listen_module.listen()
##                print('[ext listen] ->', q)
##                return q if q else ""
##            except Exception as e:
##                print('[ext listen err]', e)
##        # ensure TTS finished and give a tiny natural "listening" pause
##        if self.tts:
##            self.tts.wait_until_done(timeout=8.0)
##        # start small micro-jitter to look attentive while listening (if enabled)
##        try:
##            if self.head and ENABLE_JITTER:
##                self.head.start_micro_jitter(duration=JITTER_DURATION, amplitude=JITTER_AMPLITUDE)
##        except Exception:
##            pass
##
##        wav = record_wav_safe(self.tts, AUDIO_TEMP, seconds, sr=RECORD_SR, wait_timeout=3.0)
##
##        # stop jitter as soon as we have the recording
##        try:
##            if self.head:
##                self.head.stop_micro_jitter()
##        except Exception:
##            pass
##
##        if not wav:
##            return ""
##        # try ASR
##        txt = transcribe_any(wav)
##        if txt and txt.strip():
##            print("[ASR] transcript:", txt.strip())
##            return txt.strip()
##        # no transcript -> energy-based fallback
##        vad = detect_speech_energy(wav, rms_threshold=400.0)
##        if vad:
##            print("[ASR fallback] speech detected by energy (no transcript)")
##            # attempt to focus on the closest face if visible
##            vtext, persons, tx, ty, frame = self.vision.get_latest()
##            if frame is not None:
##                face_crop, bbox = detect_face_with_bbox(frame)
##                if bbox:
##                    cx = bbox[0] + bbox[2]//2
##                    cy = bbox[1] + bbox[3]//2
##                    # force-center on the speaking face (closest face assumption)
##                    print("[ASR fallback] centering on detected face at", cx, cy)
##                    self.head.force_move(cx, cy)
##            # return placeholder so conversation continues
##            return "<speech>"
##        else:
##            print("[ASR] no speech detected")
##            return ""
##
##    def _choose_scene_topic(self, frame, person_name):
##        items = scene_objects_from_frame(frame) if frame is not None else []
##        for it in items:
##            if person_name not in (p.get("name") for p in self.memory.people):
##                return it
##        person_rec = self.memory.get_person(person_name)
##        if not person_rec:
##            return items[0] if items else None
##        used = set(person_rec.get("mentioned_topics", []))
##        for it in items:
##            if it not in used:
##                return it
##        return items[0] if items else None
##
##    def _handle_look_command(self, obj, wait_for_report=True, timeout=1.5):
##        """
##        Arduino-driven scanning sequence using serial commands.
##        Follows the pattern:
##        f → p → q → e → M → f → q → w → f → e
##        Then returns to center (f → p) if object not found.
##        Provides both spoken and console feedback.
##        """
##        obj = obj.lower().strip()
##        self.tts.speak(f"Let me look around for the {obj}.", wait=True, max_wait=5.0)
##        print(f"[SCAN] Starting Arduino-based search for '{obj}'")
##
##        # Full sequence: (command, delay_seconds)
##        scan_sequence = [
##            ("p", 2), ("f", 2), ("q", 3), ("e", 3),
##            ("M", 3), ("f", 3), ("q", 3), ("w", 3),
##            ("f", 3), ("e", 3)
##        ]
##
##        found = False
##
##        for cmd, delay_s in scan_sequence:
##            try:
##                # Announce and send each move
##                print(f"[SCAN] Sending '{cmd}' ...")
##                self.tts.speak(f"Looking around...", wait=False)
##                prev = (self.sr.head_pos.get("x"), self.sr.head_pos.get("y"))
##
##                # Send the Arduino command and wait for update
##                self.sr.send(cmd)
##                self.sr.wait_for_head_update(prev_pos=prev, timeout=delay_s + 1.0)
##                time.sleep(delay_s)
##
##                # Check vision frame after move
##                vtext, persons, tx, ty, frame = self.vision.get_latest();
##                items = scene_objects_from_frame(frame) if frame is not None else []
##                if any(obj == it or obj in it for it in items) or obj in (vtext or "").lower():
##                    print(f"[SCAN] Found the {obj}! Focusing now...")
##                    self.tts.speak(f"Found the {obj}! Focusing now.", wait=True, max_wait=6.0)
##                    self.head.force_move(tx, ty)
##                    found = True
##                    break
##            except Exception as e:
##                print("[SCAN ERR]", e)
##                continue
##
##        # If object not found after full sequence
##        if not found:
##            print(f"[SCAN] Object '{obj}' not found after full sequence.")
##            self.tts.speak(f"I couldn't find the {obj}. Going back to center.", wait=True, max_wait=3.0)
##            try:
##                self.sr.send("p")
##                time.sleep(2)
##                self.sr.send("f")
##                time.sleep(2)
##            except Exception as e:
##                print("[SCAN return center err]", e)
##            return False
##        else:
##            return True
##
##    def _interactive_conversation(self, name, face_crop, frame):
##        self.current_person = name
##        self.memory.update_last_seen(name)
##        self.last_interaction_time = now_seconds()
##        self.state = "conversing"
##        while True:
##            if frame is not None:
##                face_crop2, bbox2 = detect_face_with_bbox(frame)
##                if bbox2:
##                    cx = bbox2[0] + bbox2[2]//2
##                    cy = bbox2[1] + bbox2[3]//2
##                    # use update_send for smoother tracking while conversing
##                    self.head.update_send(cx, cy)
##            user_text = self._listen_and_transcribe(RECORD_LANG_SECONDS)
##            if user_text:
##                user_text_l = user_text.lower()
##                self.last_interaction_time = now_seconds()
##                self.memory.append_conversation(name, f"User: {user_text}")
##
##                # ---- New: handle small / extreme look commands ----
##                # small move (little bit left/right/up/down)
##                m_small = re.search(r"\b(?:look|turn)\b.*\b(?:a little bit|a little|little|slightly|small)\b.*\b(left|right|up|down|forward|straight)\b", user_text_l)
##                if not m_small:
##                    m_small = re.search(r"\b(?:look|turn)\b.*\b(left|right|up|down|forward|straight)\b.*\b(?:a little bit|a little|little|slightly|small)\b", user_text_l)
##                if m_small:
##                    dir = m_small.group(1)
##                    self.tts.speak(f"Okay, looking a little {dir}.", wait=True, max_wait=6.0)
##                    newpos = self.head.small_move(dir)
##                    if newpos:
##                        print("[head] new pos after small_move:", newpos)
##                    continue
##
##                # extreme / far / all the way
##                m_far = re.search(r"\b(?:look|turn)\b.*\b(?:far|all the way|max|fully|to the maximum)\b.*\b(left|right|up|down|forward|straight)\b", user_text_l)
##                if not m_far:
##                    m_far = re.search(r"\b(?:look|turn)\b.*\b(left|right|up|down|forward|straight)\b.*\b(?:far|all the way|max|fully)\b", user_text_l)
##                if m_far:
##                    dir = m_far.group(1)
##                    self.tts.speak(f"Okay, looking as far {dir} as I can.", wait=True, max_wait=6.0)
##                    newpos = self.head.extreme_move(dir)
##                    if newpos:
##                        print("[head] new pos after extreme_move:", newpos)
##                    continue
##
##                # simple directional commands without 'little' or 'far'
##                m_dir = re.search(r"\b(?:look|turn)\b.*\b(left|right|up|down|forward|straight)\b", user_text_l)
##                if m_dir and not (m_small or m_far):
##                    dir = m_dir.group(1)
##                    # treat as a moderate move: do small_move by default
##                    self.tts.speak(f"Okay, looking {dir}.", wait=True, max_wait=6.0)
##                    newpos = self.head.small_move(dir)
##                    if newpos:
##                        print("[head] new pos after direction:", newpos)
##                    continue
##
##                # handle "look at <item>" requests explicitly
##                m = re.search(r"\blook (?:at )?([a-z0-9 _\\-]+)", user_text_l)
##                if m:
##                    obj = m.group(1).strip()
##                    handled = self._handle_look_command(obj)
##                    if handled:
##                        # after looking, prompt a short follow up
##                        self.tts.speak("Would you like to know more about it?", wait=True, max_wait=6.0)
##                        self.memory.append_conversation(name, f"Alfred: looked at {obj}")
##                        continue
##                    # if not handled, continue the loop with default reply
##
##                # reply using LLM
##                scene_items = scene_objects_from_frame(frame) if frame is not None else []
##                prompt = f"You are Alfred, a friendly robot. Continue a very short engaging conversation with {name}. The user said: \"{user_text}\". Use available scene items: {scene_items}. Keep reply short and ask a follow-up question."
##                reply = llm_generate(prompt) or "Interesting — tell me more."
##                self.tts.speak(reply, wait=True, max_wait=20.0)
##                self.memory.append_conversation(name, f"Alfred: {reply}")
##                continue
##            else:
##                elapsed = now_seconds() - self.last_interaction_time
##                if elapsed >= INACTIVITY_SCAN_SECONDS:
##                    print("[conversation] inactivity exceeded — scanning for other faces")
##                    self.state = "scanning"
##                    return "scan"
##                self.tts.speak("Hmm, I haven't heard you. I'll think for a moment.", wait=True, max_wait=6.0)
##                while True:
##                    topic = self._choose_scene_topic(frame, name)
##                    if topic:
##                        prompt = f"You are Alfred. Produce a short internal monologue about this topic '{topic}' and keep it interesting; then suggest a question you could ask the person about it. Keep it one concise paragraph for internal thinking and one short question to eventually ask."
##                        thought = llm_generate(prompt)
##                        if thought:
##                            self.tts.speak("...I am thinking. " + thought, wait=True, max_wait=20.0)
##                            self.memory.add_mentioned_topic(name, topic)
##                            self.memory.append_conversation(name, f"[contemplate on {topic}]: {thought}")
##                        else:
##                            self.tts.speak("...I'm pondering something interesting.", wait=True, max_wait=6.0)
##                    else:
##                        thought = llm_generate("You are Alfred. Contemplate something interesting about the surroundings. One paragraph.")
##                        if thought:
##                            self.tts.speak("...I am thinking. " + thought, wait=True, max_wait=20.0)
##                            self.memory.append_conversation(name, f"[contemplate general]: {thought}")
##                    wake_txt = self._listen_for_wake(timeout=10)
##                    if wake_txt:
##                        self.tts.speak("Yes? I'm listening.", wait=True, max_wait=6.0)
##                        self.last_interaction_time = now_seconds()
##                        break
##                    elapsed = now_seconds() - self.last_interaction_time
##                    if elapsed >= INACTIVITY_SCAN_SECONDS:
##                        self.state = "scanning"
##                        return "scan"
##                    time.sleep(CONTEMPLATE_PAUSE)
##
##    def _listen_for_wake(self, timeout=10):
##        """
##        Try ASR for wake words; if ASR not available or empty, accept energy-based VAD as wake.
##        """
##        deadline = time.time() + timeout if timeout else None
##        while True:
##            if self.tts:
##                self.tts.wait_until_done(timeout=5.0)
##            wav = record_wav_safe(self.tts, AUDIO_TEMP, WAKE_LISTEN_SECONDS, sr=RECORD_SR, wait_timeout=1.0)
##            if not wav:
##                continue
##            txt = transcribe_any(wav).lower()
##            print("[wake listen] ASR ->", txt)
##            if txt and any(w in txt for w in WAKE_WORDS):
##                _, persons, tx, ty, _ = self.vision.get_latest()
##                if self._is_looking(tx, persons):
##                    print("[wake] detected by ASR wakeword")
##                    return txt
##            # ASR didn't find wakeword. Try energy-based detection as fallback:
##            vad = detect_speech_energy(wav, rms_threshold=400.0)
##            if vad:
##                # treat any detected speech as wake if we're in front of the person (coarse)
##                _, persons, tx, ty, _ = self.vision.get_latest()
##                if self._is_looking(tx, persons):
##                    print("[wake] detected by energy VAD fallback")
##                    return "<speech>"
##                else:
##                    print("[wake] energy detected but not looking at person -> ignoring")
##            if deadline and time.time() > deadline:
##                return None
##
##    def _attention_getter_sequence(self):
##        phrases = ["Hallo", "Hi", "Hey", "Excuse me", "Hey you! over there?"]
##        for ph in phrases:
##            self.tts.speak(ph, wait=True, max_wait=6.0)
##            found_looking = False
##            check_deadline = time.time() + 2.0
##            while time.time() < check_deadline:
##                vtext, persons, tx, ty, frame = self.vision.get_latest()
##                if persons > 0 and self._is_looking(tx, persons):
##                    found_looking = True
##                    break
##                time.sleep(0.2)
##            if found_looking:
##                return True
##        return False
##
##    def _ask_name_and_confirm(self):
##        tries = 0
##        name_candidate = None
##        while tries < 4:
##            self.tts.speak("What's your name?", wait=True, max_wait=8.0)
##            name_text = self._listen_and_transcribe(6)
##            if not name_text:
##                self.tts.speak("I didn't catch that. Could you repeat your name please?", wait=True, max_wait=6.0)
##                tries += 1
##                continue
##            name_candidate = name_text.splitlines()[0].strip()
##            if not name_candidate:
##                tries += 1
##                continue
##            self.tts.speak(f"Did you say {name_candidate}? Is that correct?", wait=True, max_wait=6.0)
##            answer = self._listen_and_transcribe(5).lower()
##            if any(w in answer for w in ["yes","yeah","yep","correct","that is correct","right","sure"]):
##                return name_candidate, True, "you"
##            else:
##                tries += 1
##        self.tts.speak("I couldn't confirm your name. Should I call you him, her, or you?", wait=True, max_wait=6.0)
##        pron = self._listen_and_transcribe(5).lower()
##        if "him" in pron:
##            return name_candidate or "friend", False, "him"
##        if "her" in pron:
##            return name_candidate or "friend", False, "her"
##        return name_candidate or "friend", False, "you"
##
##    def _scan_for_faces_and_attempt_engage(self, scan_seconds=20):
##        self.state = "scanning"
##        start = time.time()
##        pos_idx = 0
##        # If SCAN_SEQUENCE is defined and we have serial, drive Arduino scanning commands
##        if isinstance(SCAN_SEQUENCE, list) and self.sr and getattr(self.sr, 'send', None):
##            seq = list(SCAN_SEQUENCE)
##            for cmd, delay_s in seq:
##                # send command to Arduino and then poll camera for faces during delay
##                try:
##                    self.sr.send(cmd)
##                except Exception:
##                    pass
##                t_dead = time.time() + float(delay_s)
##                while time.time() < t_dead and (time.time() - start) < scan_seconds:
##                    vtext, persons, vx, vy, frame = self.vision.get_latest()
##                    if frame is not None:
##                        face_crop, bbox = detect_face_with_bbox(frame)
##                        if bbox is not None:
##                            cx = bbox[0] + bbox[2]//2
##                            cy = bbox[1] + bbox[3]//2
##                            self.head.update_send(cx, cy)
##                            engaged = self._attention_getter_sequence()
##                            if engaged:
##                                self.tts.speak("Yes you.", wait=True, max_wait=4.0)
##                                match, score = self.memory.find_match(face_crop, min_similarity=0.6)
##                                if match:
##                                    name = match.get("name")
##                                    self.tts.speak(f"Hi {name}. Good to see you.", wait=True, max_wait=6.0)
##                                    return name
##                                else:
##                                    name, confirmed, pron = self._ask_name_and_confirm()
##                                    self.memory.add_person(name, face_crop)
##                                    if confirmed:
##                                        self.tts.speak(f"Nice to meet you, {name}.", wait=True, max_wait=6.0)
##                                    else:
##                                        self.tts.speak(f"Alright. I'll call you {pron}.", wait=True, max_wait=4.0)
##                                    return name
##                    time.sleep(0.12)
##            # finished sequence
##            self.state = "idle"
##            return None
##
##        # Fallback: original sweep-based scanning
##        sweep_positions = [
##            (FRAME_W//8, FRAME_H//2),
##            (FRAME_W//2, FRAME_H//2),
##            (7*FRAME_W//8, FRAME_H//2),
##            (FRAME_W//2, FRAME_H//4),
##            (FRAME_W//2, 3*FRAME_H//4)
##        ]
##        pos_idx = 0
##        while time.time() - start < scan_seconds:
##            target = sweep_positions[pos_idx % len(sweep_positions)]
##            pos_idx += 1
##            tx, ty = target
##            self.head.force_move(tx, ty)
##            t_deadline = time.time() + 1.2
##            while time.time() < t_deadline:
##                vtext, persons, vx, vy, frame = self.vision.get_latest()
##                if frame is not None:
##                    face_crop, bbox = detect_face_with_bbox(frame)
##                    if bbox is not None:
##                        cx = bbox[0] + bbox[2]//2
##                        cy = bbox[1] + bbox[3]//2
##                        self.head.update_send(cx, cy)
##                        engaged = self._attention_getter_sequence()
##                        if engaged:
##                            self.tts.speak("Yes you.", wait=True, max_wait=4.0)
##                            match, score = self.memory.find_match(face_crop, min_similarity=0.6)
##                            if match:
##                                name = match.get("name")
##                                self.tts.speak(f"Hi {name}. Good to see you.", wait=True, max_wait=6.0)
##                                return name
##                            else:
##                                name, confirmed, pron = self._ask_name_and_confirm()
##                                self.memory.add_person(name, face_crop)
##                                if confirmed:
##                                    self.tts.speak(f"Nice to meet you, {name}.", wait=True, max_wait=6.0)
##                                else:
##                                    self.tts.speak(f"Alright. I'll call you {pron}.", wait=True, max_wait=4.0)
##                                return name
##                time.sleep(0.12)
##        self.state = "idle"
##        return None
##
##    def _interactive_conversation_resume(self, name, face_crop, frame):
##        self.tts.speak(f"Hi {name}, good to see you again.", wait=True, max_wait=6.0)
##        follow = llm_generate(f"You are Alfred. Say a short friendly follow-up to {name} after greeting.")
##        if follow:
##            self.tts.speak(follow, wait=True, max_wait=8.0)
##        return self._interactive_conversation(name, face_crop, frame)
##
##    def _main_loop(self):
##        while not self._stop:
##            try:
##                # detect state transitions and manage breathing
##                if self.prev_state != self.state:
##                    # leaving idle
##                    if self.prev_state == "idle" and self.head:
##                        try:
##                            self.head.stop_breathing()
##                        except Exception:
##                            pass
##                    # entering idle
##                    if self.state == "idle" and self.head:
##                        try:
##                            # gentle breathing while idle (only if enabled)
##                            if ENABLE_BREATHING:
##                                self.head.start_breathing(amplitude_x=BREATH_AMPLITUDE_X, amplitude_y=BREATH_AMPLITUDE_Y, period=BREATH_PERIOD)
##                        except Exception:
##                            pass
##                    self.prev_state = self.state
##
##                vtext, persons, vx, vy, frame = self.vision.get_latest()
##                face_crop, bbox = (None, None)
##                if frame is not None:
##                    face_crop, bbox = detect_face_with_bbox(frame)
##                if bbox:
##                    cx = bbox[0] + bbox[2]//2
##                    cy = bbox[1] + bbox[3]//2
##                    self.current_face_bbox = bbox
##                    self.head.update_send(cx, cy)
##                    if self.state == "idle":
##                        match, score = self.memory.find_match(face_crop, min_similarity=0.6)
##                        if match:
##                            name = match.get("name")
##                            res = self._interactive_conversation(name, face_crop, frame)
##                            if res == "scan":
##                                engaged_name = self._scan_for_faces_and_attempt_engage()
##                                if engaged_name:
##                                    self._interactive_conversation(engaged_name, face_crop, frame)
##                                self.state = "idle"
##                            else:
##                                self.state = "idle"
##                        else:
##                            self.state = "greeting"
##                            hair_color, clothing_color = analyze_appearance(frame, bbox)
##                            compliment = make_compliment(hair_color, clothing_color)
##                            greet = "Hi. Hello. Hallo. Good day."
##                            self.tts.speak(greet + " " + compliment, wait=True, max_wait=8.0)
##                            time.sleep(0.25)
##                            self.tts.speak("What's your name?", wait=True, max_wait=8.0)
##                            name_text = self._listen_and_transcribe(6)
##                            if not name_text:
##                                self.tts.speak("I didn't catch that. Please type your name.", wait=True, max_wait=6.0)
##                                try:
##                                    name_text = input("Name: ").strip()
##                                except Exception:
##                                    name_text = "friend"
##                            name_name = (name_text.splitlines()[0] or "friend").strip()
##                            self.memory.add_person(name_name, face_crop)
##                            if hair_color:
##                                self.memory.add_mentioned_topic(name_name, f"hair:{hair_color}")
##                            if clothing_color:
##                                self.memory.add_mentioned_topic(name_name, f"clothes:{clothing_color}")
##                            ret = self._interactive_conversation(name_name, face_crop, frame)
##                            if ret == "scan":
##                                engaged_name = self._scan_for_faces_and_attempt_engage()
##                                if engaged_name:
##                                    self._interactive_conversation(engaged_name, face_crop, frame)
##                            self.state = "idle"
##                    else:
##                        if self.current_person:
##                            self.memory.update_last_seen(self.current_person)
##                            self.last_interaction_time = now_seconds()
##                else:
##                    if persons > 0:
##                        self.head.update_send(vx, vy)
##                        if self.state == "idle" and frame is not None:
##                            h, w = frame.shape[:2]
##                            fcrop = frame[h//4:3*h//4, w//4:3*w//4]
##                            match, score = self.memory.find_match(fcrop, min_similarity=0.6)
##                            if match:
##                                name = match.get("name")
##                                res = self._interactive_conversation(name, fcrop, frame)
##                                if res == "scan":
##                                    engaged_name = self._scan_for_faces_and_attempt_engage()
##                                    if engaged_name:
##                                        self._interactive_conversation(engaged_name, fcrop, frame)
##                                self.state = "idle"
##                            else:
##                                engaged = self._attention_getter_sequence()
##                                if engaged:
##                                    name, confirmed, pron = self._ask_name_and_confirm()
##                                    self.memory.add_person(name, fcrop)
##                                    if confirmed:
##                                        self.tts.speak(f"Nice to meet you, {name}.", wait=True, max_wait=6.0)
##                                    else:
##                                        self.tts.speak(f"Alright. I'll call you {pron}.", wait=True, max_wait=4.0)
##                                    self._interactive_conversation(name, fcrop, frame)
##                    else:
##                        if self.current_person:
##                            last_rec = self.memory.get_person(self.current_person)
##                            last_inter = last_rec.get("last_interaction", 0) if last_rec else 0
##                            elapsed = now_seconds() - last_inter
##                            if elapsed >= INACTIVITY_SCAN_SECONDS:
##                                engaged_name = self._scan_for_faces_and_attempt_engage(scan_seconds=12)
##                                if engaged_name:
##                                    self._interactive_conversation(engaged_name, None, None)
##                                else:
##                                    self._contemplate_room()
##                            else:
##                                self._contemplate_room()
##                        else:
##                            engaged_name = self._scan_for_faces_and_attempt_engage(scan_seconds=10)
##                            if engaged_name:
##                                self._interactive_conversation(engaged_name, None, None)
##                            else:
##                                self._contemplate_room()
##                time.sleep(0.25)
##            except Exception:
##                traceback.print_exc()
##                time.sleep(0.5)
##
##    def _contemplate_room(self):
##        vtext, persons, vx, vy, frame = self.vision.get_latest()
##        items = scene_objects_from_frame(frame) if frame is not None else []
##        if items:
##            topic = items[0]
##            thought = llm_generate(f"You are Alfred. Contemplate an interesting short thought about '{topic}' in the room and suggest an action or question. One short paragraph.")
##            if thought:
##                self.tts.speak("...I am thinking. " + thought, wait=True, max_wait=12.0)
##        else:
##            thought = llm_generate("You are Alfred. Contemplate something interesting about the surroundings. One short paragraph.")
##            if thought:
##                self.tts.speak("...I am thinking. " + thought, wait=True, max_wait=12.0)
##        time.sleep(1.0)
##
### ---------------- Main program ----------------
##def head_test(sr):
##    print("Running head-test sequence (watch head). Ctrl-C to stop.")
##    hc = HeadController(sr)
##    try:
##        seq = [
##            (0, FRAME_H//2),
##            (FRAME_W//2, FRAME_H//2),
##            (FRAME_W, FRAME_H//2),
##            (FRAME_W//2, 0),
##            (FRAME_W//2, FRAME_H//2),
##            (FRAME_W//2, FRAME_H)
##        ]
##        for tx, ty in seq:
##            print("Sending (force):", tx, ty)
##            hc.force_move(tx, ty)
##            time.sleep(1.2)
##        print("Head test done.")
##    except KeyboardInterrupt:
##        print("Head test interrupted.")
##
##def main():
##    head_only = False
##    if len(sys.argv) > 1 and sys.argv[1] == "--head-test":
##        head_only = True
##
##    # Attempt to use Alfred_config camera input if available
##    left_ip = FALLBACK_LEFT_IP; right_ip = FALLBACK_RIGHT_IP
##    try:
##        import Alfred_config
##        if hasattr(Alfred_config, "CHEST_CAMERA_INPUT"):
##            left_url = Alfred_config.CHEST_CAMERA_INPUT
##            right_url = Alfred_config.CHEST_CAMERA_INPUT
##        else:
##            left_url = make_stream_url(left_ip); right_url = make_stream_url(right_ip)
##    except Exception:
##        left_url = make_stream_url(left_ip); right_url = make_stream_url(right_ip)
##
##    sr = SerialIPReader()
##    if not sr.alive:
##        print("[main] serial not available; using fallback IPs")
##    left_url = left_url; right_url = right_url
##    print("[main] left:", left_url, " right:", right_url)
##
##    if head_only:
##        head_test(sr)
##        sr.close()
##        return
##
##    cam = CameraManager(left_url, right_url, preferred="left", size=(FRAME_W, FRAME_H))
##    vision = VisionWorker(cam)
##    vision.start()
##    head = HeadController(sr)
##    memory = PersonMemory()
##    tts = TTSEngine()
##    conv = ConversationManager(sr, cam, vision, head, tts, memory)
##
##    cv2.namedWindow("Alfred View", cv2.WINDOW_NORMAL)
##    last_time = time.time(); frames = 0; fps = 0.0
##    try:
##        while True:
##            loop_start = time.time()
##            frame, label = cam.get_frame()
##            if frame is None:
##                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
##                vis_text, persons, tx, ty, _ = "[no camera]", 0, FRAME_W//2, FRAME_H//2, None
##            else:
##                vis_text, persons, tx, ty, _ = vision.get_latest()
##            # head.update_send is done in ConversationManager (face priority) but call here to ensure general tracking
##            head.update_send(tx, ty)
##            # overlay UI
##            cv2.rectangle(frame, (0,0), (FRAME_W,60), (0,0,0), -1)
##            cv2.putText(frame, f"CAM: {label or 'NONE'}", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),1)
##            cv2.putText(frame, f"Persons: {persons}", (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
##            cv2.putText(frame, f"Vision: {vis_text[:80]}", (140,40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220),1)
##            # show reported Arduino head pos if available
##            hp = sr.get_head_pos()
##            if hp["x"] is not None and hp["y"] is not None:
##                cv2.putText(frame, f"HeadArd X:{hp['x']} Y:{hp['y']}", (8,60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0),1)
##            cv2.putText(frame, f"State: {conv.state}", (8, FRAME_H-28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,160,200),1)
##            cv2.imshow("Alfred View", frame)
##            frames += 1
##            if time.time() - last_time >= 1.0:
##                fps = frames; frames = 0; last_time = time.time()
##            key = cv2.waitKey(1) & 0xFF
##            if key == ord('q'):
##                break
##            time.sleep(max(0.0, 1/20 - (time.time()-loop_start)))
##    except KeyboardInterrupt:
##        print("Interrupted")
##    finally:
##        print("Shutting down")
##        conv.stop(); vision.stop(); tts.stop(); sr.close()
##        cv2.destroyAllWindows()
##
### ---------------- Module start/stop API for external caller (main.py) ----------------
##MODULE_STATE = {'running': False}
##MODULE_COMPONENTS = {}
##
##def start_service(background=True, camera_input=Alfred_config.CHEST_CAMERA_INPUT, head_only=False):
##    """
##    Start the module (in-process). Returns True on success.
##    Example (from main.py):
##        import Newest_VLM_AI_edge_tts_MoreReal_Breathing_Adjust_V16_fixed as alfred
##        alfred.start_service(background=True)
##    Use camera_input to pass Alfred_config.CHEST_CAMERA_INPUT string.
##    """
##    global MODULE_STATE, MODULE_COMPONENTS
##    if MODULE_STATE.get('running'):
##        print("[module] already running")
##        return False
##
##    # prepare serial (or use external arduino module if requested)
##    if USE_EXTERNAL_ARDUINO and external_arduino_module:
##        sr = None
##        try:
##            sr = external_arduino_module.arduino
##        except Exception:
##            sr = None
##    else:
##        sr = SerialIPReader()
##
##    # camera urls
##    left_url = right_url = None
##    if camera_input:
##        left_url = camera_input; right_url = camera_input
##    else:
##        try:
##            import Alfred_config
##            if hasattr(Alfred_config, "CHEST_CAMERA_INPUT"):
##                left_url = Alfred_config.CHEST_CAMERA_INPUT
##                right_url = Alfred_config.CHEST_CAMERA_INPUT
##        except Exception:
##            pass
##    if not left_url:
##        left_url = make_stream_url(FALLBACK_LEFT_IP)
##        right_url = make_stream_url(FALLBACK_RIGHT_IP)
##
##    cam = CameraManager(left_url, right_url, preferred="left", size=(FRAME_W, FRAME_H))
##    vision = VisionWorker(cam)
##    vision.start()
##    head = HeadController(sr)
##    memory = PersonMemory()
##    tts = TTSEngine()
##    conv = ConversationManager(sr, cam, vision, head, tts, memory)
##
##    MODULE_COMPONENTS = {'sr': sr, 'cam': cam, 'vision': vision, 'head': head, 'memory': memory, 'tts': tts, 'conv': conv}
##    MODULE_STATE['running'] = True
##    print("[module] started")
##
##    if not background:
##        try:
##            while MODULE_STATE['running']:
##                time.sleep(0.25)
##        except KeyboardInterrupt:
##            pass
##    return True
##
##def stop_service():
##    global MODULE_STATE, MODULE_COMPONENTS
##    if not MODULE_STATE.get('running'):
##        print("[module] already stopped")
##        return
##    comp = MODULE_COMPONENTS
##    try:
##        if comp.get('conv'): comp['conv'].stop()
##    except Exception:
##        pass
##    try:
##        if comp.get('vision'): comp['vision'].stop()
##    except Exception:
##        pass
##    try:
##        if comp.get('tts'): comp['tts'].stop()
##    except Exception:
##        pass
##    try:
##        if comp.get('sr') and hasattr(comp['sr'], 'close'): comp['sr'].close()
##    except Exception:
##        pass
##    MODULE_STATE['running'] = False
##    MODULE_COMPONENTS = {}
##    print('[module] stopped')
##
##def set_enable_breathing(flag: bool):
##    global ENABLE_BREATHING
##    ENABLE_BREATHING = bool(flag)
##    if not ENABLE_BREATHING and MODULE_COMPONENTS.get('head'):
##        try: MODULE_COMPONENTS['head'].stop_breathing()
##        except Exception: pass
##
##def set_enable_jitter(flag: bool):
##    global ENABLE_JITTER
##    ENABLE_JITTER = bool(flag)
##    if not ENABLE_JITTER and MODULE_COMPONENTS.get('head'):
##        try: MODULE_COMPONENTS['head'].stop_micro_jitter()
##        except Exception: pass
##
##def set_memory_limits(max_people=None, max_conversations_per_person=None, max_topics_per_person=None):
##    global MEMORY_MAX_PEOPLE, MEMORY_MAX_CONVERSATIONS_PER_PERSON, MEMORY_MAX_TOPICS_PER_PERSON
##    if max_people is not None:
##        MEMORY_MAX_PEOPLE = int(max_people)
##    if max_conversations_per_person is not None:
##        MEMORY_MAX_CONVERSATIONS_PER_PERSON = int(max_conversations_per_person)
##    if max_topics_per_person is not None:
##        MEMORY_MAX_TOPICS_PER_PERSON = int(max_topics_per_person)
##    # Immediately enforce limits if running
##    if MODULE_COMPONENTS.get('memory'):
##        mem = MODULE_COMPONENTS['memory']
##        try:
##            mem._ensure_limits()
##        except Exception:
##            pass
##
##if __name__ == "__main__":
##    main()
##
##











