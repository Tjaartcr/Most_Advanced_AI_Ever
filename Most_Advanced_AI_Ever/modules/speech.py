
# speech.py (drop-in)
# Synchronous TTS controller with reliable edge-tts -> pyttsx3 fallback
# Playback uses python-vlc or external vlc binary when available for pause/resume/stop support.
import os
import sys
import time
import asyncio
import tempfile
import traceback
import subprocess
import shutil

import pyttsx3
from playsound import playsound
import edge_tts

# keep these imports to preserve API surface
import sounddevice as sd
import vosk

from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
from arduino_com import arduino

print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])

# ----------------- try python-vlc / libvlc -----------------
_HAS_PYTHON_VLC = False
_HAS_LIBVLC = False
vlc = None
try:
    import vlc as _vlc
    vlc = _vlc
    _HAS_PYTHON_VLC = True
    try:
        inst = vlc.Instance()
        _HAS_LIBVLC = True
        del inst
        print("python-vlc and libvlc available")
    except Exception as e:
        _HAS_LIBVLC = False
        print("python-vlc imported but libvlc.Instance() failed:", e)
except Exception as e:
    print("python-vlc import failed:", e)

# fallback: check external vlc binary
_HAS_VLC_BINARY = False
_VLC_BINARY = None
if not _HAS_LIBVLC:
    for candidate in ("cvlc", "vlc"):
        p = shutil.which(candidate)
        if p:
            _HAS_VLC_BINARY = True
            _VLC_BINARY = candidate
            print("Found external VLC binary:", candidate, "->", p)
            break

if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
    print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")

# ----------------- SpeechModule (no threads) -----------------
class SpeechModule:
    def __init__(self):
        # pyttsx3 engine for onboard fallback (used to save-to-file)
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 190)
            voices = self.engine.getProperty("voices")
            if voices:
                self.engine.setProperty("voice", voices[0].id)
        except Exception as e:
            print("pyttsx3 init failed:", e)
            self.engine = None

        # control flags (simple booleans, not threading events)
        self.wake_word_on_off = False

        # flags for GUI control
        self.tk_start_speech = False
        self.tk_stop_speech = False
        self.tk_pause_speech = False

        # runtime control flags
        self._stop_request = False
        self._pause_requested = False
        self._paused_file = None       # for external-cli fallback resume emulation
        self._player = None            # python-vlc player (if used)
        self._vlc_instance = None
        if _HAS_LIBVLC:
            try:
                self._vlc_instance = vlc.Instance()
            except Exception as e:
                print("Failed to create vlc.Instance():", e)
                self._vlc_instance = None

        # public flag used by main() to avoid auto-listen while TTS active/paused
        self._suppress_auto_listen = False

    # ----------------- small helpers exposed to main() -----------------

##    def is_paused(self):
##        """Return True if speech is currently paused (user pressed Pause)."""
##        return bool(self._pause_requested)


    # --- small helper so main() can check pause state ---
    def is_paused(self) -> bool:
        """Return True when speech module is in a pause/halt state."""
        # prefer explicit public flag if available; fall back to internal name
        try:
            return bool(getattr(self, "_pause_requested", False) or getattr(self, "_pause_in_progress", False))
        except Exception:
            return False


    def is_playing(self):
        """Return True if a player object is playing (best-effort)."""
        try:
            if _HAS_LIBVLC and self._player:
                return bool(self._player.is_playing())
        except Exception:
            pass
        # external process or playsound not easily checkable here
        return False

    # ----------------- Wake Word -----------------
    def set_wake_word_on(self, enabled=True):
        self.wake_word_on_off = True
        print("Wake word ON (synchronous mode)")

    def set_wake_word_off(self, enabled=False):
        self.wake_word_on_off = False
        print("Wake word OFF (synchronous mode)")

    # ---------------- Set Play, Stop, Pause --------------
    def set_tk_start_speech(self):
        """Resume or start playback; clears stop/pause flags."""
        print("speech.set_tk_start_speech() called (sync)")
        self.tk_start_speech = True
        self.tk_pause_speech = False
        self.tk_stop_speech = False
        self._stop_request = False
        self._pause_requested = False

        # keep suppress flag True while resuming playback; will be cleared when playback finishes
        self._suppress_auto_listen = True

        # If a python-vlc player exists and is paused, attempt resume
        try:
            if _HAS_LIBVLC and self._player:
                try:
                    print("Attempting to resume existing python-vlc player (sync)")
                    self._player.set_pause(False)
                    return
                except Exception as e:
                    print("Resume via set_pause failed:", e)
                    try:
                        self._player.play()
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # If we have a paused file from external VLC, play it now (blocking)
        if self._paused_file:
            fname = self._paused_file
            self._paused_file = None
            print("Resuming paused file (external fallback):", fname)
            self._play_file_controlled(fname)

    def set_tk_stop_speech(self):
        """Immediate stop: set stop flag and attempt to stop player/process."""
        print("speech.set_tk_stop_speech() called (sync)")
        self.tk_stop_speech = True
        self.tk_pause_speech = False
        self.tk_start_speech = False
        self._stop_request = True
        self._pause_requested = False

        # stop python-vlc player if present
        try:
            if _HAS_LIBVLC and self._player:
                print("Stopping python-vlc player (sync)")
                self._player.stop()
        except Exception:
            pass

        # stop external vlc process if present (best-effort)
        try:
            if getattr(self, "_vlc_process", None):
                try:
                    self._vlc_process.terminate()
                except Exception:
                    try:
                        self._vlc_process.kill()
                    except Exception:
                        pass
                self._vlc_process = None
        except Exception:
            pass

        # clear paused-file and suppress listen flag
        self._paused_file = None
        self._suppress_auto_listen = False

        # stop pyttsx3 if being used to write or speak
        try:
            if self.engine:
                self.engine.stop()
        except Exception:
            pass

    def set_tk_pause_speech(self):
        """Pause playback: if player exists, pause; otherwise emulate by stopping process and remembering file."""
        print("speech.set_tk_pause_speech() called (sync)")
        self.tk_pause_speech = True
        self.tk_start_speech = False
        self.tk_stop_speech = False
        self._pause_requested = True

        # ensure we continue to suppress listening while paused
        self._suppress_auto_listen = True

        # Try to pause python-vlc player in-place
        try:
            if _HAS_LIBVLC and self._player:
                try:
                    is_playing = False
                    try:
                        is_playing = bool(self._player.is_playing())
                    except Exception:
                        try:
                            st = self._player.get_state()
                            is_playing = (st == vlc.State.Playing)
                        except Exception:
                            is_playing = False
                    if is_playing:
                        print("Pausing python-vlc player (sync set_pause True)")
                        self._player.set_pause(True)
                        return
                    else:
                        print("python-vlc player not playing; cannot pause in-place (sync)")
                except Exception as e:
                    print("Error while pausing python-vlc player (sync):", e)
        except Exception:
            pass

        # Emulate pause for external-vlc: set stop flag (the play loop will preserve the filename)
        print("Emulating pause in sync mode (will stop playback and retain file for resume)")
        self._stop_request = True
        # pyttsx3: we do not call engine.pause() since it's unreliable — we preserve text->file for resume
        try:
            if self.engine:
                # No direct pause: stop any speaking; resume will play saved file again
                self.engine.stop()
        except Exception:
            pass

    def stop_current(self):
        """Convenience immediate stop (same as set_tk_stop_speech)."""
        print("speech.stop_current() called (sync)")
        self.set_tk_stop_speech()



    def AlfredSpeakPYTTSX3(self, text):
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 190)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('volume', 1)
        print('engine: ' + str(text), end = "\r")
##        print('\033c', end = '')
        engine.say(text)
        engine.runAndWait()

    # ----------------- Public API -----------------


##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking / synchronous from the caller's perspective, but we protect against
##        long hangs when edge-tts or pyttsx3 stall by using short timeouts.
##        - Try edge-tts first (safe pre-check + timeout)
##        - Fallback to pyttsx3 (run in a subprocess so COM/SAPI is isolated)
##        """
##        if not text:
##            return
##
##        self._suppress_auto_listen = True
##        self._stop_request = False
##        self._pause_requested = False
##
##        fname = None
##
##        # ---------------------------
##        # EDGE-TTS (SAFE) with network pre-check + timeouted thread
##        # ---------------------------
##        try:
##            import threading
##            import asyncio
##            import os
##            import traceback
##            import socket
##            import time
##
##            # If HTTP(S) proxy env var is set, skip direct host check (we assume proxy will be used)
##            proxy = (
##                os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
##                or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
##            )
##            if proxy:
##                print("[TTS] proxy detected via env, skipping direct DNS/connect pre-check:", proxy)
##                network_ok = True
##            else:
##                try:
##                    host = "speech.platform.bing.com"
##                    port = 443
##                    socket.getaddrinfo(host, port)
##                    s = socket.create_connection((host, port), timeout=3)
##                    s.close()
##                    network_ok = True
##                except Exception as net_e:
##                    print("[TTS] network pre-check failed (cannot reach edge-tts host):", net_e)
##                    network_ok = False
##
##            if not network_ok:
##                print("[TTS] skipping edge-tts due to failed pre-check; falling back to pyttsx3")
##                fname = None
##            else:
##                container = {}
##
##                def _runner():
##                    try:
##                        container['result'] = asyncio.run(
##                            self._speak_edge_tts_save_only(
##                                text,
##                                voice=voice,
##                                style=style
##                            )
##                        )
##                    except Exception as e_inner:
##                        container['error'] = e_inner
##
##                t = threading.Thread(target=_runner, daemon=True)
##                t.start()
##
##                join_timeout = 15.0  # seconds
##                t.join(timeout=join_timeout)
##
##                if t.is_alive():
##                    print(f"[TTS] edge-tts coroutine timed out after {join_timeout}s — falling back to pyttsx3")
##                    fname = None
##                else:
##                    if 'error' in container:
##                        raise container['error']
##                    fname = container.get('result')
##                    if not fname or not os.path.exists(fname) or os.path.getsize(fname) == 0:
##                        print("[TTS] edge-tts returned invalid/empty file:", fname)
##                        fname = None
##
##        except Exception as e:
##            print("[TTS] edge-tts failed → fallback to pyttsx3:", e)
##            traceback.print_exc()
##            fname = None
##
##        # ---------------------------
##        # PYTTSX3 FALLBACK (USING SUBPROCESS FOR RELIABILITY ON WINDOWS)
##        # ---------------------------
##        if not fname:
##            try:
##                import tempfile
##                import time
##                import subprocess
##                import sys
##
##                tmp = os.path.join(
##                    tempfile.gettempdir(),
##                    f"alfred_pyttsx3_{int(time.time()*1000)}.wav"
##                )
##
##                # Helper: write a small one-off Python script that uses pyttsx3 to save or speak.
##                def _run_pyttsx3_subprocess_save(text_to_speak, out_path, timeout_s=12):
##                    """
##                    Returns True if file created successfully, False on failure or timeout.
##                    """
##                    script = f"""
##    import pyttsx3, sys
##    engine = pyttsx3.init('sapi5')
##    engine.setProperty('rate', 190)
##    voices = engine.getProperty('voices')
##    if voices:
##        engine.setProperty('voice', voices[0].id)
##    engine.setProperty('volume', 1.0)
##    engine.save_to_file({repr(text_to_speak)}, {repr(out_path)})
##    engine.runAndWait()
##    """
##                    # write temp script
##                    import tempfile as _tf
##                    fh = _tf.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
##                    fh.write(script)
##                    fh.flush()
##                    fh.close()
##                    try:
##                        subprocess.run([sys.executable, fh.name], check=True, timeout=timeout_s)
##                        # check file
##                        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
##                            return True
##                        return False
##                    except subprocess.TimeoutExpired:
##                        print(f"[TTS] pyttsx3 subprocess save timed out after {timeout_s}s")
##                        return False
##                    except Exception as e_sub:
##                        print("[TTS] pyttsx3 subprocess save error:", e_sub)
##                        return False
##                    finally:
##                        try:
##                            os.unlink(fh.name)
##                        except Exception:
##                            pass
##
##                def _run_pyttsx3_subprocess_speak(text_to_speak, timeout_s=8):
##                    """
##                    Returns True if subprocess finished (attempted to speak) or False on error/timeout.
##                    """
##                    script = f"""
##    import pyttsx3
##    engine = pyttsx3.init('sapi5')
##    engine.setProperty('rate', 190)
##    voices = engine.getProperty('voices')
##    if voices:
##        engine.setProperty('voice', voices[0].id)
##    engine.setProperty('volume', 1.0)
##    engine.say({repr(text_to_speak)})
##    engine.runAndWait()
##    """
##                    import tempfile as _tf
##                    fh = _tf.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
##                    fh.write(script)
##                    fh.flush()
##                    fh.close()
##                    try:
##                        subprocess.run([sys.executable, fh.name], check=True, timeout=timeout_s)
##                        return True
##                    except subprocess.TimeoutExpired:
##                        print(f"[TTS] pyttsx3 subprocess speak timed out after {timeout_s}s")
##                        return False
##                    except Exception as e_sub:
##                        print("[TTS] pyttsx3 subprocess speak error:", e_sub)
##                        return False
##                    finally:
##                        try:
##                            os.unlink(fh.name)
##                        except Exception:
##                            pass
##
##                # 1) Try to save to file via subprocess (reliable and isolated)
##                save_timeout = 14.0
##                saved_ok = _run_pyttsx3_subprocess_save(text, tmp, timeout_s=save_timeout)
##                if saved_ok:
##                    fname = tmp
##                else:
##                    # 2) Try direct speak in subprocess (won't produce a file)
##                    say_timeout = 10.0
##                    said_ok = _run_pyttsx3_subprocess_speak(text, timeout_s=say_timeout)
##                    if said_ok:
##                        print("[TTS] pyttsx3 subprocess direct speak completed successfully")
##                        fname = None  # audio was produced live (subprocess did it)
##                    else:
##                        print("[TTS] pyttsx3 subprocess direct speak failed; skipping audio output")
##                        fname = None
##
##            except Exception as e:
##                print("[TTS] pyttsx3 fallback (subprocess) failed completely:", e)
##                traceback.print_exc()
##                fname = None
##
##        # ---------------------------
##        # PLAYBACK (CONTROLLED)
##        # ---------------------------
##        if fname:
##            try:
##                self._play_file_controlled(fname)
##            except Exception as e:
##                print("[TTS] playback failed:", e)
##                traceback.print_exc()
##
##        # ---------------------------
##        # CLEAN EXIT (IMPORTANT)
##        # ---------------------------
##        self._suppress_auto_listen = False
##        self._pause_requested = False
##        self._stop_request = False




    
######    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
######        """
######        Blocking / synchronous:
######        - Try edge-tts first (safe timeout)
######        - Fallback to pyttsx3 (guaranteed)
######        """
######        if not text:
######            return
######
######        self._suppress_auto_listen = True
######        self._stop_request = False
######        self._pause_requested = False
######
######        fname = None
######
######        # ---------------------------
######        # EDGE-TTS (SAFE) with network pre-check + timeouted thread
######        # ---------------------------
######        try:
######            import threading
######            import asyncio
######            import os
######            import traceback
######            import socket
######            import time
######
######            # If HTTP(S) proxy env var is set, skip direct host check (we assume proxy will be used)
######            proxy = (
######                os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
######                or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
######            )
######            if proxy:
######                print("[TTS] proxy detected via env, skipping direct DNS/connect pre-check:", proxy)
######                network_ok = True
######            else:
######                # Quick DNS + TCP connect check to the edge-tts host (short timeout)
######                try:
######                    host = "speech.platform.bing.com"
######                    port = 443
######                    socket.getaddrinfo(host, port)
######                    s = socket.create_connection((host, port), timeout=3)
######                    s.close()
######                    network_ok = True
######                except Exception as net_e:
######                    print("[TTS] network pre-check failed (cannot reach edge-tts host):", net_e)
######                    network_ok = False
######
######            # If network is not ok, skip trying edge-tts and fall back to pyttsx3
######            if not network_ok:
######                print("[TTS] skipping edge-tts due to failed pre-check; falling back to pyttsx3")
######                fname = None
######            else:
######                # Run coroutine inside a daemon thread and wait with timeout to avoid hangs
######                container = {}
######
######                def _runner():
######                    try:
######                        # call the coroutine and store the result
######                        container['result'] = asyncio.run(
######                            self._speak_edge_tts_save_only(
######                                text,
######                                voice=voice,
######                                style=style
######                            )
######                        )
######                    except Exception as e_inner:
######                        container['error'] = e_inner
######
######                t = threading.Thread(target=_runner, daemon=True)
######                t.start()
######
######                # wait up to N seconds for the coroutine to finish
######                join_timeout = 15.0  # seconds; tweak if you want longer
######                t.join(timeout=join_timeout)
######
######                if t.is_alive():
######                    # The worker is still running → don't hang the main thread; fall back
######                    print(f"[TTS] edge-tts coroutine timed out after {join_timeout}s — falling back to pyttsx3")
######                    fname = None
######                    # Note: the background thread may still finish later; we ignore it safely.
######                else:
######                    # Thread finished — check result or error
######                    if 'error' in container:
######                        raise container['error']
######                    fname = container.get('result')
######
######                    # Validate returned filename (exists & non-empty)
######                    if not fname or not os.path.exists(fname) or os.path.getsize(fname) == 0:
######                        print("[TTS] edge-tts returned invalid/empty file:", fname)
######                        fname = None
######
######        except Exception as e:
######            print("[TTS] edge-tts failed → fallback to pyttsx3:", e)
######            traceback.print_exc()
######            fname = None
######
######        # ---------------------------
######        # PYTTSX3 FALLBACK (FIXED)
######        # ---------------------------
######        if not fname:
######            try:
######                import tempfile
######                import time
######                import pyttsx3
######
######                if not getattr(self, "engine", None):
######                    self.engine = pyttsx3.init('sapi5')
######                    self.engine.setProperty('rate', 190)
######                    voices = self.engine.getProperty('voices')
######                    if voices:
######                        self.engine.setProperty('voice', voices[0].id)
######                    self.engine.setProperty('volume', 1.0)
######
######                tmp = os.path.join(
######                    tempfile.gettempdir(),
######                    f"alfred_pyttsx3_{int(time.time()*1000)}.wav"
######                )
######
######                try:
######                    self.engine.save_to_file(text, tmp)
######                    self.engine.runAndWait()
######                    fname = tmp
######                except Exception as e2:
######                    print("[TTS] pyttsx3 save_to_file failed:", e2)
######                    print("[TTS] speaking directly via pyttsx3")
######
######                    # LAST RESORT — DIRECT SPEAK (never freeze)
######                    self.engine.say(text)
######                    self.engine.runAndWait()
######                    fname = None
######
######            except Exception as e:
######                print("[TTS] pyttsx3 fallback failed completely:", e)
######                fname = None
######
######        # ---------------------------
######        # PLAYBACK (CONTROLLED)
######        # ---------------------------
######        if fname:
######            try:
######                self._play_file_controlled(fname)
######            except Exception as e:
######                print("[TTS] playback failed:", e)
######                traceback.print_exc()
######
######        # ---------------------------
######        # CLEAN EXIT (IMPORTANT)
######        # ---------------------------
######        self._suppress_auto_listen = False
######        self._pause_requested = False
######        self._stop_request = False


####    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
####        """
####        Blocking / synchronous:
####        - Try edge-tts first (safe timeout)
####        - Fallback to pyttsx3 (guaranteed)
####        """
####        if not text:
####            return
####
####        self._suppress_auto_listen = True
####        self._stop_request = False
####        self._pause_requested = False
####
####        fname = None
####
####        # ---------------------------
####        # EDGE-TTS (SAFE)
####        # ---------------------------
####        try:
####            import threading
####            import asyncio
####            import os
####            import traceback
####
####            # Run coroutine in a separate thread using asyncio.run inside that thread
####            # This avoids calling asyncio.run when an event loop is already running.
####            def _run_coro_in_thread(coro):
####                container = {}
####
####                def _runner():
####                    try:
####                        container['result'] = asyncio.run(coro)
####                    except Exception as e_inner:
####                        container['error'] = e_inner
####
####                t = threading.Thread(target=_runner)
####                t.start()
####                t.join()
####                if 'error' in container:
####                    raise container['error']
####                return container.get('result')
####
####            # Call your coroutine safely
####            fname = _run_coro_in_thread(
####                self._speak_edge_tts_save_only(
####                    text,
####                    voice=voice,
####                    style=style
####                )
####            )
####
####            # Validate returned filename (exists & non-empty)
####            if not fname or not os.path.exists(fname) or os.path.getsize(fname) == 0:
####                raise RuntimeError(f"edge-tts returned invalid/empty file: {fname!r}")
####
####        except Exception as e:
####            print("[TTS] edge-tts failed → fallback to pyttsx3:", e)
####            traceback.print_exc()
####            fname = None
####
####        # ---------------------------
####        # PYTTSX3 FALLBACK (FIXED)
####        # ---------------------------
####        if not fname:
####            try:
####                if not self.engine:
####                    self.engine = pyttsx3.init('sapi5')
####                    self.engine.setProperty('rate', 190)
####                    voices = self.engine.getProperty('voices')
####                    if voices:
####                        self.engine.setProperty('voice', voices[0].id)
####                    self.engine.setProperty('volume', 1.0)
####
####                tmp = os.path.join(
####                    tempfile.gettempdir(),
####                    f"alfred_pyttsx3_{int(time.time()*1000)}.wav"
####                )
####
####                try:
####                    self.engine.save_to_file(text, tmp)
####                    self.engine.runAndWait()
####                    fname = tmp
####                except Exception as e2:
####                    print("[TTS] pyttsx3 save_to_file failed:", e2)
####                    print("[TTS] speaking directly via pyttsx3")
####
####                    # LAST RESORT — DIRECT SPEAK (never freeze)
####                    self.engine.say(text)
####                    self.engine.runAndWait()
####                    fname = None
####
####            except Exception as e:
####                print("[TTS] pyttsx3 fallback failed completely:", e)
####                fname = None
####
####        # ---------------------------
####        # PLAYBACK (CONTROLLED)
####        # ---------------------------
####        if fname:
####            try:
####                self._play_file_controlled(fname)
####            except Exception as e:
####                print("[TTS] playback failed:", e)
####                traceback.print_exc()
####
####        # ---------------------------
####        # CLEAN EXIT (IMPORTANT)
####        # ---------------------------
####        self._suppress_auto_listen = False
####        self._pause_requested = False
####        self._stop_request = False
####
####





##
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking / synchronous:
##        - Try edge-tts first (safe timeout)
##        - Fallback to pyttsx3 (guaranteed)
##        """
##        if not text:
##            return
##
##        self._suppress_auto_listen = True
##        self._stop_request = False
##        self._pause_requested = False
##
##        fname = None
##
##        # ---------------------------
##        # EDGE-TTS (SAFE)
##        # ---------------------------
##        try:
##            try:
##                loop = asyncio.get_running_loop()
##                # If we already have a loop, we cannot use asyncio.run()
##                raise RuntimeError("Running loop detected")
##            except RuntimeError:
##                # No running loop → safe
##                fname = asyncio.run(
##                    self._speak_edge_tts_save_only(
##                        text,
##                        voice=voice,
##                        style=style
##                    )
##                )
##        except Exception as e:
##            print("[TTS] edge-tts failed → fallback to pyttsx3:", e)
##            fname = None
##
##        # ---------------------------
##        # PYTTSX3 FALLBACK (FIXED)
##        # ---------------------------
##        if not fname:
##            try:
##                if not self.engine:
##                    self.engine = pyttsx3.init('sapi5')
##                    self.engine.setProperty('rate', 190)
##                    voices = self.engine.getProperty('voices')
##                    if voices:
##                        self.engine.setProperty('voice', voices[0].id)
##                    self.engine.setProperty('volume', 1.0)
##
##                tmp = os.path.join(
##                    tempfile.gettempdir(),
##                    f"alfred_pyttsx3_{int(time.time()*1000)}.wav"
##                )
##
##                try:
##                    self.engine.save_to_file(text, tmp)
##                    self.engine.runAndWait()
##                    fname = tmp
##                except Exception as e2:
##                    print("[TTS] pyttsx3 save_to_file failed:", e2)
##                    print("[TTS] speaking directly via pyttsx3")
##
##                    # LAST RESORT — DIRECT SPEAK (never freeze)
##                    self.engine.say(text)
##                    self.engine.runAndWait()
##                    fname = None
##
##            except Exception as e:
##                print("[TTS] pyttsx3 fallback failed completely:", e)
##                fname = None
##
##        # ---------------------------
##        # PLAYBACK (CONTROLLED)
##        # ---------------------------
##        if fname:
##            try:
##                self._play_file_controlled(fname)
##            except Exception as e:
##                print("[TTS] playback failed:", e)
##                traceback.print_exc()
##
##        # ---------------------------
##        # CLEAN EXIT (IMPORTANT)
##        # ---------------------------
##        self._suppress_auto_listen = False
##        self._pause_requested = False
##        self._stop_request = False

    
    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
        """
        Blocking / synchronous: try edge-tts first. On failure, fall back to pyttsx3-generated file.
        While speaking (or paused), self._suppress_auto_listen is True so main() will not call listen.listen().
        """
        if not text:
            return

        # signal: suppress auto listen while we generate/play
        self._suppress_auto_listen = True
        self._stop_request = False
        self._pause_requested = False

        # Try edge-tts -> save file and play
        fname = None
        try:
            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
        except Exception as e:
            # log the edge error and fallback
            print("Error generating TTS via edge-tts (sync):", e)
            self.AlfredSpeakPYTTSX3(text)
            
        # If edge failed to create a file, fallback to pyttsx3 -> save_to_file
        if not fname:
            try:
                # save using pyttsx3 to a wav/mp3 file then play it (so pause/stop are consistent)
                if not self.engine:
                    # try to create a local engine to save file
##                    self.engine = pyttsx3.init()
##                    self.engine.setProperty("rate", 190)

                    self.engine = pyttsx3.init('sapi5')
                    self.engine.setProperty('rate', 190)
                    self.voices = engine.getProperty('voices')
                    self.engine.setProperty('voice', voices[0].id)
                    self.engine.setProperty('volume', 1)
                    print('engine: ' + str(text), end = "\r")
                    print('\033c', end = '')
                    
                tmp = os.path.join(tempfile.gettempdir(), f"alfred_pyttsx3_{int(time.time()*1000)}.mp3")
                # pyttsx3 supports save_to_file; the driver decides file format - use .mp3 or .wav
                try:
                    self.engine.save_to_file(text, tmp)
                    self.engine.runAndWait()  # blocks while writing file
                    fname = tmp
                except Exception as e2:
                    print("pyttsx3 save_to_file failed:", e2)
                    # Last resort: speak directly via engine (no pause/resume control) — still suppress listening while speaking
                    try:
                        print("speech: falling back to direct pyttsx3 speak (no file)")
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e3:
                        print("direct pyttsx3 speak also failed:", e3)
                        fname = None
            except Exception as e:
                print("pyttsx3 fallback failed entirely:", e)
                fname = None

        # If we have a file, play it under control so pause/resume/stop work
        if fname:
            try:
                # keep _suppress_auto_listen True while playing; it will be cleared at the end.
                self._play_file_controlled(fname)
            except Exception as e:
                print("Error during controlled playback (sync):", e)
                traceback.print_exc()

        # finished speaking (or stopped) -> allow listening again
        self._suppress_auto_listen = False
        self._pause_requested = False
        self._stop_request = False

    def AlfredSpeak_Onboard(self, text):
        """Blocking onboard TTS (pyttsx3) — kept for API compatibility, but main flow uses save->play for control."""
        if not self.engine:
            self.engine = pyttsx3.init()
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("AlfredSpeak_Onboard failed:", e)

    def AlfredSpeak_Bluetooth(self, text):
        try:
            if hasattr(arduino, "send_bluetooth"):
                arduino.send_bluetooth(text)
        except Exception as e:
            print("Error sending bluetooth:", e)
        # synchronous speak
        self.AlfredSpeak(text)

    # ---------------- helpers -----------------
    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
        ssml = f"{text}"
        communicate = edge_tts.Communicate(text=ssml, voice=voice)
        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
        await communicate.save(fname)
        return fname

    def _onboard_speak_blocking_with_stop(self, text):
        if not self.engine:
            raise RuntimeError("pyttsx3 engine not initialized")
        if self._stop_request:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("pyttsx3 speak failed (sync):", e)
            traceback.print_exc()

    # ---------------- playback core (blocking) ----------------
    def _play_file_controlled(self, fname):
        """
        Blocking playback that honors self._stop_request and self._pause_requested.
        Uses python-vlc (libvlc) if available, else external vlc process (if found), else playsound.
        """
        if not os.path.exists(fname):
            print("_play_file_controlled: file not found:", fname)
            return

        print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)

        try:
            # python-vlc primary path
            if _HAS_LIBVLC and self._vlc_instance:
                print(" _play_file_controlled (sync): using python-vlc (libvlc)")
                player = self._vlc_instance.media_player_new()
                media = self._vlc_instance.media_new(str(fname))
                player.set_media(media)
                self._player = player
                player.play()

                # wait up to a short timeout for actual play to begin
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    try:
                        if player.is_playing():
                            break
                    except Exception:
                        pass
                    if self._stop_request:
                        break
                    time.sleep(0.05)

                # playback loop (blocking) honoring pause/stop flags
                while True:
                    if self._stop_request:
                        print(" _play_file_controlled (sync): stop requested -> stopping python-vlc player")
                        try:
                            player.stop()
                        except Exception:
                            pass
                        break

                    if self._pause_requested:
                        # pause in-place using set_pause(True)
                        try:
                            print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
                            player.set_pause(True)
                        except Exception:
                            pass
                        # busy-wait until resume or stop
                        while self._pause_requested and (not self._stop_request):
                            time.sleep(0.05)
                        # when resumed
                        try:
                            player.set_pause(False)
                        except Exception:
                            try:
                                player.play()
                            except Exception:
                                pass

                    # check normal end
                    try:
                        st = player.get_state()
                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
                            break
                    except Exception:
                        pass
                    time.sleep(0.06)

                # cleanup
                try:
                    player.stop()
                except Exception:
                    pass
                self._player = None

            # external vlc binary fallback (runs subprocess synchronously but poll-loop)
            elif _HAS_VLC_BINARY:
                print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._paused_file = fname  # keep for resume if pause emulation used
                self._vlc_process = proc

                while True:
                    if self._stop_request:
                        try:
                            proc.terminate()
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        break
                    if self._pause_requested:
                        # emulate pause: terminate process and keep file path for resume
                        try:
                            proc.terminate()
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        break
                    # check if finished
                    if proc.poll() is not None:
                        break
                    time.sleep(0.06)

                try:
                    _ = proc.wait(timeout=0.1)
                except Exception:
                    pass
                self._vlc_process = None

            # last fallback: blocking playsound
            else:
                print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
                if self._stop_request:
                    print(" _play_file_controlled (sync): stop already requested; skipping playsound")
                    return
                try:
                    playsound(fname)
                except Exception as e:
                    print("playsound failed (sync):", e)

        finally:
            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
            try:
                if fname and (self._paused_file != fname):
                    os.remove(fname)
            except Exception:
                pass

        print("_play_file_controlled (sync): playback finished/aborted")

# create module-level instance
speech = SpeechModule()
print("speech (synchronous) initialized — done")




####### speech.py -- synchronous, thread-free TTS playback with pause/stop support
######import os
######import sys
######import time
######import asyncio
######import tempfile
######import traceback
######import subprocess
######import shutil
######
######import pyttsx3
######from playsound import playsound
######import edge_tts
####### keep these imports to preserve API surface
######import sounddevice as sd
######import vosk
######
######from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
######from arduino_com import arduino
######
######print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
######
####### ----------------- try python-vlc / libvlc -----------------
######_HAS_PYTHON_VLC = False
######_HAS_LIBVLC = False
######vlc = None
######try:
######    import vlc as _vlc
######    vlc = _vlc
######    _HAS_PYTHON_VLC = True
######    try:
######        inst = vlc.Instance()
######        _HAS_LIBVLC = True
######        del inst
######        print("python-vlc and libvlc available")
######    except Exception as e:
######        _HAS_LIBVLC = False
######        print("python-vlc imported but libvlc.Instance() failed:", e)
######except Exception as e:
######    print("python-vlc import failed:", e)
######
####### fallback: check external vlc binary
######_HAS_VLC_BINARY = False
######_VLC_BINARY = None
######if not _HAS_LIBVLC:
######    for candidate in ("cvlc", "vlc"):
######        p = shutil.which(candidate)
######        if p:
######            _HAS_VLC_BINARY = True
######            _VLC_BINARY = candidate
######            print("Found external VLC binary:", candidate, "->", p)
######            break
######
######if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
######    print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
######
####### ----------------- SpeechModule (no threads) -----------------
######class SpeechModule:
######    def __init__(self):
######        # pyttsx3 engine for onboard fallback
######        try:
######            self.engine = pyttsx3.init()
######            self.engine.setProperty("rate", 190)
######            voices = self.engine.getProperty("voices")
######            if voices:
######                self.engine.setProperty("voice", voices[0].id)
######        except Exception as e:
######            print("pyttsx3 init failed:", e)
######            self.engine = None
######
######        # control flags (simple booleans, not threading events)
######        self.wake_word_on_off = False
######
######        # flags for GUI control
######        self.tk_start_speech = False
######        self.tk_stop_speech = False
######        self.tk_pause_speech = False
######
######        # runtime control flags
######        self._stop_request = False
######        self._pause_requested = False
######        self._paused_file = None       # for external-cli fallback resume emulation
######        self._player = None            # python-vlc player (if used)
######        self._vlc_instance = None
######        if _HAS_LIBVLC:
######            try:
######                self._vlc_instance = vlc.Instance()
######            except Exception as e:
######                print("Failed to create vlc.Instance():", e)
######                self._vlc_instance = None
######
######    # ----------------- Wake Word -----------------
######    def set_wake_word_on(self, enabled=True):
######        self.wake_word_on_off = True
######        print("Wake word ON (synchronous mode)")
######
######    def set_wake_word_off(self, enabled=False):
######        self.wake_word_on_off = False
######        print("Wake word OFF (synchronous mode)")
######
######    # ---------------- Set Play, Stop, Pause --------------
######    def set_tk_start_speech(self):
######        """Resume or start playback; clears stop/pause flags."""
######        print("speech.set_tk_start_speech() called (sync)")
######        self.tk_start_speech = True
######        self.tk_pause_speech = False
######        self.tk_stop_speech = False
######        self._stop_request = False
######        self._pause_requested = False
######
######        # If a python-vlc player exists and is paused, attempt resume
######        if _HAS_LIBVLC and self._player:
######            try:
######                print("Attempting to resume existing python-vlc player (sync)")
######                self._player.set_pause(False)
######                return
######            except Exception as e:
######                print("Resume via set_pause failed:", e)
######                try:
######                    self._player.play()
######                    return
######                except Exception:
######                    pass
######
######        # If we have a paused file from external VLC, play it now (blocking)
######        if self._paused_file:
######            fname = self._paused_file
######            self._paused_file = None
######            print("Resuming paused file (external fallback):", fname)
######            self._play_file_controlled(fname)
######
######    def set_tk_stop_speech(self):
######        """Immediate stop: set stop flag and attempt to stop player/process."""
######        print("speech.set_tk_stop_speech() called (sync)")
######        self.tk_stop_speech = True
######        self.tk_pause_speech = False
######        self.tk_start_speech = False
######        self._stop_request = True
######        self._pause_requested = False
######
######        # stop python-vlc player if present
######        try:
######            if _HAS_LIBVLC and self._player:
######                print("Stopping python-vlc player (sync)")
######                self._player.stop()
######        except Exception:
######            pass
######
######        # cannot forcibly stop playsound easily; external process is handled in _play_file_controlled
######        # clear paused file
######        self._paused_file = None
######
######        # stop pyttsx3 if in use
######        try:
######            if self.engine:
######                self.engine.stop()
######        except Exception:
######            pass
######
######    def set_tk_pause_speech(self):
######        """Pause playback: if player exists, pause; otherwise emulate by stopping process and remembering file."""
######        print("speech.set_tk_pause_speech() called (sync)")
######        self.tk_pause_speech = True
######        self.tk_start_speech = False
######        self.tk_stop_speech = False
######        self._pause_requested = True
######
######        # Try to pause python-vlc player in-place
######        try:
######            if _HAS_LIBVLC and self._player:
######                try:
######                    is_playing = False
######                    try:
######                        is_playing = bool(self._player.is_playing())
######                    except Exception:
######                        try:
######                            st = self._player.get_state()
######                            is_playing = (st == vlc.State.Playing)
######                        except Exception:
######                            is_playing = False
######                    if is_playing:
######                        print("Pausing python-vlc player (sync set_pause True)")
######                        self._player.set_pause(True)
######                        return
######                    else:
######                        print("python-vlc player not playing; cannot pause in-place (sync)")
######                except Exception as e:
######                    print("Error while pausing python-vlc player (sync):", e)
######        except Exception:
######            pass
######
######        # Emulate pause for external-vlc: set stop flag (the play loop will preserve the filename)
######        print("Emulating pause in sync mode (will stop playback and retain file for resume)")
######        self._stop_request = True
######        # pyttsx3 stop as well
######        try:
######            if self.engine:
######                self.engine.pause()
######        except Exception:
######            pass
######
######    def stop_current(self):
######        """Convenience immediate stop (same as set_tk_stop_speech)."""
######        print("speech.stop_current() called (sync)")
######        self.set_tk_stop_speech()
######
######    # ----------------- Public API -----------------
######    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
######        """
######        Blocking / synchronous: generate audio (edge-tts) then play it under control.
######        NOTE: This function blocks the caller until playback finishes or is stopped.
######        """
######        if not text:
######            return
######
######        # If wake_word_on_off is supported, queueing is not available in sync mode; behave similarly to previous logic
######        if self.wake_word_on_off:
######            # In a no-thread setup, we simply perform the speak immediately
######            pass
######
######        # Generate audio file synchronously (edge-tts)
######        try:
######            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
######        except Exception as e:
######            print("Error generating TTS via edge-tts (sync):", e)
######            fname = None
######
######        if fname:
######            try:
######                # play blocking but controllable (pause/stop)
######                self._play_file_controlled(fname)
######            except Exception as e:
######                print("Error during controlled playback (sync):", e)
######                traceback.print_exc()
######        else:
######            # fallback to onboard blocking speak
######            try:
######                self._onboard_speak_blocking_with_stop(text)
######            except Exception as e:
######                print("Onboard fallback failed (sync):", e)
######
######    def AlfredSpeak_Onboard(self, text):
######        """Blocking onboard TTS (pyttsx3)."""
######        engine = pyttsx3.init("sapi5")
######        engine.setProperty("rate", 190)
######        voices = engine.getProperty("voices")
######        engine.setProperty("voice", voices[0].id)
######        engine.setProperty("volume", 1)
######
######        if self.tk_start_speech:
######            engine.say(text)
######            engine.runAndWait()
######        else:   
######            self.tk_pause_speech = True
######            self.engine.pause()
######
######    def AlfredSpeak_Bluetooth(self, text):
######        try:
######            if hasattr(arduino, "send_bluetooth"):
######                arduino.send_bluetooth(text)
######        except Exception as e:
######            print("Error sending bluetooth:", e)
######        # synchronous speak
######        self.AlfredSpeak(text)
######
######    # ---------------- helpers -----------------
######    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
######        ssml = f"{text}"
######        communicate = edge_tts.Communicate(text=ssml, voice=voice)
######        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
######        await communicate.save(fname)
######        return fname
######
######    def _onboard_speak_blocking_with_stop(self, text):
######        if not self.engine:
######            raise RuntimeError("pyttsx3 engine not initialized")
######        if self._stop_request:
######            return
######        try:
######            self.engine.say(text)
######            self.engine.runAndWait()
######        except Exception as e:
######            print("pyttsx3 speak failed (sync):", e)
######            traceback.print_exc()
######
######    # ---------------- playback core (blocking) ----------------
######    def _play_file_controlled(self, fname):
######        """
######        Blocking playback that honors self._stop_request and self._pause_requested.
######        Uses python-vlc (libvlc) if available, else external vlc process (if found), else playsound.
######        """
######        if not os.path.exists(fname):
######            print("_play_file_controlled: file not found:", fname)
######            return
######
######        print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)
######
######        try:
######            # python-vlc primary path
######            if _HAS_LIBVLC and self._vlc_instance:
######                print(" _play_file_controlled (sync): using python-vlc (libvlc)")
######                player = self._vlc_instance.media_player_new()
######                media = self._vlc_instance.media_new(str(fname))
######                player.set_media(media)
######                self._player = player
######                player.play()
######
######                # wait up to a short timeout for actual play to begin
######                t0 = time.time()
######                while time.time() - t0 < 3.0:
######                    try:
######                        if player.is_playing():
######                            break
######                    except Exception:
######                        pass
######                    if self._stop_request:
######                        break
######                    time.sleep(0.05)
######
######                # playback loop (blocking) honoring pause/stop flags
######                while True:
######                    if self._stop_request:
######                        print(" _play_file_controlled (sync): stop requested -> stopping python-vlc player")
######                        try:
######                            player.stop()
######                        except Exception:
######                            pass
######                        break
######
######                    if self._pause_requested:
######                        # pause in-place using set_pause(True)
######                        try:
######                            print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
######                            player.set_pause(True)
######                        except Exception:
######                            pass
######                        # busy-wait until resume or stop
######                        while self._pause_requested and (not self._stop_request):
######                            time.sleep(0.05)
######                        # when resumed
######                        try:
######                            player.set_pause(False)
######                        except Exception:
######                            try:
######                                player.play()
######                            except Exception:
######                                pass
######
######                    # check normal end
######                    try:
######                        st = player.get_state()
######                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
######                            break
######                    except Exception:
######                        pass
######                    time.sleep(0.06)
######
######                # cleanup
######                try:
######                    player.stop()
######                except Exception:
######                    pass
######                self._player = None
######
######            # external vlc binary fallback (runs subprocess synchronously but poll-loop)
######            elif _HAS_VLC_BINARY:
######                print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
######                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
######                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
######                self._paused_file = fname  # keep for resume if pause emulation used
######
######                while True:
######                    if self._stop_request:
######                        try:
######                            proc.terminate()
######                        except Exception:
######                            try:
######                                proc.kill()
######                            except Exception:
######                                pass
######                        break
######                    if self._pause_requested:
######                        # emulate pause: terminate process and keep file path for resume
######                        try:
######                            proc.terminate()
######                        except Exception:
######                            try:
######                                proc.kill()
######                            except Exception:
######                                pass
######                        break
######                    # check if finished
######                    if proc.poll() is not None:
######                        break
######                    time.sleep(0.06)
######
######                try:
######                    _ = proc.wait(timeout=0.1)
######                except Exception:
######                    pass
######                self._vlc_process = None
######
######            # last fallback: blocking playsound
######            else:
######                print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
######                if self._stop_request:
######                    print(" _play_file_controlled (sync): stop already requested; skipping playsound")
######                    return
######                try:
######                    playsound(fname)
######                except Exception as e:
######                    print("playsound failed (sync):", e)
######
######        finally:
######            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
######            try:
######                if fname and (self._paused_file != fname):
######                    os.remove(fname)
######            except Exception:
######                pass
######
######        print("_play_file_controlled (sync): playback finished/aborted")
######
####### create module-level instance
######speech = SpeechModule()
######print("speech (synchronous) initialized — done")







### speech.py (drop-in) — synchronous TTS controller with pause/halt + speaking-state flag
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import re
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
##
### Keep these imports available for compatibility
##import sounddevice as sd
##import vosk
##
### Adjust these imports to your environment layout if necessary
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (drop-in) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### ----------------- vlc detection -----------------
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##        print("speech.py: python-vlc/libvlc available")
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("speech.py: python-vlc imported but libvlc.Instance() failed:", e)
##except Exception as e:
##    print("speech.py: python-vlc import failed:", e)
##
### External VLC binary fallback
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("speech.py: found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("speech.py: WARNING: No VLC available, falling back to playsound (limited pause/stop)")
##
### ----------------- SpeechModule -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine (shared for onboard fallback)
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("speech.py: pyttsx3 init failed:", e)
##            self.engine = None
##
##        # basic state flags
##        self.wake_word_on_off = False
##
##        # GUI-visible flags
##        self.tk_start_speech = True   # Start available by default (user requested)
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##
##        # runtime control flags (synchronous)
##        self._stop_request = False
##        self._pause_requested = False
##
##        # playback/chunking state
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # pause/wake-word suppression helpers
##        self._prev_wake_word = None
##        self._suppress_auto_listen = False
##        self._pause_in_progress = False
##
##        # python-vlc / subprocess handles
##        self._player = None
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("speech.py: Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##        self._vlc_process = None
##        self._paused_file = None
##
##        # speaking state flag (NEW)
##        self._currently_speaking = False
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("speech.py: Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("speech.py: Wake word OFF (synchronous mode)")
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_pause_speech(self):
##        """
##        Pause-as-halt: interrupt playback, preserve remaining chunks, and suppress auto-listen.
##        Idempotent.
##        """
##        print("speech.set_tk_pause_speech() called (sync)")
##        if self._pause_in_progress:
##            print("speech: already paused — ignoring repeated pause")
##            return
##
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##
##        self._pause_requested = True
##        self._pause_in_progress = True
##
##        # preserve and suppress wake-word so other parts won't auto-start mic
##        self._prev_wake_word = self.wake_word_on_off
##        if self._prev_wake_word:
##            print("speech: suppressing wake_word_on_off while paused (was True)")
##        self.wake_word_on_off = False
##        self._suppress_auto_listen = True
##
##        # Try to pause python-vlc player in-place if playing
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        st = self._player.get_state()
##                        is_playing = (st == vlc.State.Playing)
##                    if is_playing:
##                        print("speech: pausing python-vlc player in-place")
##                        self._player.set_pause(True)
##                        return
##                    else:
##                        print("speech: python-vlc player not playing (pause no-op)")
##                except Exception as e:
##                    print("speech: error while pausing libvlc player:", e)
##        except Exception:
##            pass
##
##        # For pyttsx3 onboard: engine.stop() will interrupt runAndWait(); chunk loop will save remaining chunks
##        try:
##            if self.engine:
##                print("speech: calling engine.stop() to interrupt onboard playback and preserve remaining chunks")
##                self.engine.stop()
##        except Exception as e:
##            print("speech: engine.stop() raised:", e)
##
##        # For external vlc subprocess fallback: terminate process to emulate pause (play loop preserves _paused_file)
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                print("speech: terminated external VLC to emulate pause; paused file:", self._paused_file)
##        except Exception:
##            pass
##
##    def set_tk_start_speech(self):
##        """
##        Resume speech playback only. Does NOT automatically re-enable listening.
##        To re-enable listening explicitly call speech.restore_listening() or have GUI call listen.set_listen_whisper(True).
##        """
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##
##        # clear pause flags for playback; do NOT clear _suppress_auto_listen
##        self._pause_requested = False
##        self._pause_in_progress = False
##        self._stop_request = False
##
##        # Resume onboard remaining chunks if they exist (blocking)
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##            try:
##                self._play_onboard_chunks(chunks)
##            finally:
##                self._pause_requested = False
##                self._stop_request = False
##
##        # Resume python-vlc player if present and paused
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    self._player.set_pause(False)
##                except Exception:
##                    try:
##                        self._player.play()
##                    except Exception:
##                        pass
##        except Exception as e:
##            print("speech: libvlc resume attempt failed:", e)
##
##        # If an external paused file exists, play it now (this keeps _suppress_auto_listen True)
##        if self._paused_file:
##            fname = self._paused_file
##            self._paused_file = None
##            self._play_file_controlled(fname)
##
##    def set_tk_stop_speech(self):
##        """Stop playback and clear pending/onboard buffers immediately."""
##        print("speech.set_tk_stop_speech() called (sync)")
##
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##
##        # request stop and clear pause markers
##        self._stop_request = True
##        self._pause_requested = False
##        self._pause_in_progress = False
##
##        # stop pyttsx3 and clear remaining chunks
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # stop python-vlc player
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("speech: stopping python-vlc player")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # stop external vlc process if any
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##
##        # clear paused file pointer and do not restore auto-listen
##        self._paused_file = None
##
##        # clear speaking flag
##        self._currently_speaking = False
##
##    def restore_listening(self):
##        """
##        Explicitly restore listening behavior that was suppressed by pause.
##        Call this from GUI if the user wants the mic to be allowed to start again.
##        """
##        print("speech.restore_listening() called")
##        if self._prev_wake_word is not None:
##            self.wake_word_on_off = self._prev_wake_word
##            self._prev_wake_word = None
##            print("speech: wake_word_on_off restored to", self.wake_word_on_off)
##        # clear suppression so main() will allow listen.listen()
##        self._suppress_auto_listen = False
##
##    def stop_current(self):
##        """Convenience immediate stop (alias)."""
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    # ----------------- Public TTS API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking synchronous speak. Prefer file-based playback (edge-tts -> file -> controlled playback).
##        If that is not available, fallback to chunked pyttsx3 (honors pause/halt).
##        """
##        if not text:
##            return
##
##        # mark speaking
##        self._currently_speaking = True
##        try:
##            # Attempt to generate a file with edge-tts synchronously
##            fname = None
##            try:
##                fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##            except Exception as e:
##                print("speech: edge-tts generation failed:", e)
##                fname = None
##
##            if fname and (_HAS_LIBVLC or _HAS_VLC_BINARY):
##                # play file in blocking, controllable way
##                self._play_file_controlled(fname)
##            else:
##                # fallback: chunk and play with pyttsx3 so pause/halt works
##                chunks = self._chunk_text_for_onboard(text)
##                if self._onboard_remaining_chunks:
##                    chunks = self._onboard_remaining_chunks
##                    self._onboard_remaining_chunks = None
##                self._onboard_current_voice = voice
##                self._onboard_current_style = style
##                self._play_onboard_chunks(chunks)
##        finally:
##            # ensure speaking flag cleared when done or aborted
##            self._currently_speaking = False
##
##    def AlfredSpeak_Onboard(self, text):
##        """Direct onboard blocking speak using chunking (so pause works)."""
##        if not text:
##            return
##        chunks = self._chunk_text_for_onboard(text)
##        # mark speaking around the chunk playback
##        self._currently_speaking = True
##        try:
##            self._play_onboard_chunks(chunks)
##        finally:
##            self._currently_speaking = False
##
##    def AlfredSpeak_Bluetooth(self, text):
##        """Send to bluetooth device if configured, then speak normally."""
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("speech: error sending bluetooth:", e)
##        self.AlfredSpeak(text)
##
##    # ---------------- helpers for pyttsx3 chunking ----------------
##    def _chunk_text_for_onboard(self, text):
##        """
##        Split text into sentence-like chunks for pausable playback.
##        Conservative sentence split + sub-splitting for long segments.
##        """
##        if not text:
##            return []
##        # split on sentence-ending punctuation followed by a capital or digit (heuristic)
##        pattern = re.compile(r'(?<=\S[.!?])\s+(?=[A-Z0-9"\'“”])')
##        parts = pattern.split(text.strip())
##
##        maxlen = 250
##        chunks = []
##        for p in parts:
##            if len(p) <= maxlen:
##                chunks.append(p.strip())
##            else:
##                sub = re.split(r'([,;]\s+)', p)
##                acc = ""
##                for s in sub:
##                    acc += s
##                    if len(acc) >= maxlen:
##                        chunks.append(acc.strip())
##                        acc = ""
##                if acc.strip():
##                    chunks.append(acc.strip())
##        return [c for c in chunks if c]
##
##    def _play_onboard_chunks(self, chunks):
##        """
##        Blocking play of chunk list with pyttsx3.
##        Honors _pause_requested (HALT) and _stop_request.
##        Preserves remaining chunks in _onboard_remaining_chunks when paused/interrupted.
##        """
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        # If resume scenario: use saved remaining
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##
##        # set speaking flag (in case caller didn't)
##        self._currently_speaking = True
##        try:
##            for idx, chunk in enumerate(chunks):
##                # check immediate stop before chunk
##                if self._stop_request:
##                    print("speech: onboard playback — stop requested before chunk, aborting")
##                    self._onboard_remaining_chunks = None
##                    return
##
##                # if paused before chunk start -> save remaining and return
##                if self._pause_requested and self._pause_in_progress:
##                    remaining = chunks[idx:]
##                    print("speech: onboard playback — pause requested before chunk; preserving", len(remaining), "chunks")
##                    self._onboard_remaining_chunks = remaining
##                    return
##
##                try:
##                    # Speak chunk — blocking runAndWait
##                    self.engine.say(chunk)
##                    self.engine.runAndWait()
##                except Exception as e:
##                    # interruption likely due to engine.stop() called by pause/stop
##                    print("speech: pyttsx3 chunk interrupted/exception:", e)
##                    if self._pause_requested:
##                        remaining = chunks[idx:]
##                        print("speech: pyttsx3 interrupted during chunk; preserving", len(remaining), "chunks")
##                        self._onboard_remaining_chunks = remaining
##                        return
##                    if self._stop_request:
##                        self._onboard_remaining_chunks = None
##                        return
##                    # Otherwise continue
##
##                # after chunk checks
##                if self._stop_request:
##                    print("speech: onboard playback — stop requested after chunk, aborting")
##                    self._onboard_remaining_chunks = None
##                    return
##                if self._pause_requested and self._pause_in_progress:
##                    remaining = chunks[idx+1:]
##                    print("speech: onboard playback — pause requested after chunk; preserving", len(remaining), "chunks")
##                    self._onboard_remaining_chunks = remaining
##                    return
##
##            # played all chunks
##            print("speech: onboard playback finished all chunks")
##            self._onboard_remaining_chunks = None
##            # reset local pause/stop markers but do NOT automatically restore listening
##            self._pause_requested = False
##            self._stop_request = False
##            self._pause_in_progress = False
##        finally:
##            # clear speaking flag
##            self._currently_speaking = False
##
##    # ---------------- edge-tts file generation ----------------
##    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        # produce a temp file and return its path
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await communicate.save(fname)
##        return fname
##
##    # ---------------- playback core (blocking) for files ----------------
##    def _play_file_controlled(self, fname):
##        """
##        Blocking playback respecting _pause_requested and _stop_request.
##        Uses python-vlc/libvlc if available, else external vlc subprocess, else playsound.
##        """
##        if not os.path.exists(fname):
##            print("speech._play_file_controlled: file not found:", fname)
##            return
##
##        print("speech._play_file_controlled: starting playback for", fname, "stop_request=", self._stop_request)
##
##        # set speaking flag
##        self._currently_speaking = True
##        try:
##            # python-vlc path
##            if _HAS_LIBVLC and self._vlc_instance:
##                print("speech: using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait briefly for playing to begin
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request:
##                        break
##                    time.sleep(0.05)
##
##                # playback loop: honor pause/stop as HALT
##                while True:
##                    if self._stop_request:
##                        try:
##                            player.stop()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested and self._pause_in_progress:
##                        try:
##                            player.set_pause(True)
##                        except Exception:
##                            pass
##                        # wait until resumed or stopped
##                        while self._pause_requested and (not self._stop_request):
##                            time.sleep(0.05)
##                        # resume
##                        try:
##                            player.set_pause(False)
##                        except Exception:
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            # external VLC binary fallback (synchronous subprocess)
##            elif _HAS_VLC_BINARY:
##                print("speech: using external VLC binary:", _VLC_BINARY)
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                self._vlc_process = proc
##                self._paused_file = fname
##
##                while True:
##                    if self._stop_request:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if self._pause_requested and self._pause_in_progress:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        # paused file is preserved in self._paused_file
##                        break
##                    if proc.poll() is not None:
##                        break
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##                self._vlc_process = None
##
##            # final fallback: playsound (limited control)
##            else:
##                print("speech: no VLC available, falling back to playsound (limited)")
##                if self._stop_request:
##                    print("speech: stop already requested; skipping playsound")
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("speech: playsound failed:", e)
##
##        finally:
##            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##            # clear speaking flag
##            self._currently_speaking = False
##
##        print("speech._play_file_controlled: playback finished/aborted")
##
##    # ---------------- helpers / status ----------------
##    def is_paused(self):
##        """Public helper for other modules to check if speech is in halted pause state."""
##        return bool(getattr(self, "_pause_in_progress", False))
##
##    def is_speaking(self):
##        """Return True while TTS playback is in progress (file or pyttsx3)."""
##        return bool(getattr(self, "_currently_speaking", False))
##
### module-level instance
##speech = SpeechModule()
##print("speech module (drop-in) initialized — done")
##






### speech.py (drop-in) ALMOST THERE
###
### Synchronous TTS controller with pyttsx3 chunking for pause/halt behavior,
### edge-tts file generation + python-vlc (or external vlc / playsound) playback.
###
### Pause semantics:
###   - set_tk_pause_speech() acts like a HALT: it interrupts playback, preserves remaining
###     chunks, and sets _suppress_auto_listen=True so other parts won't start the mic.
###   - set_tk_start_speech() resumes speech playback only (does NOT re-enable auto-listen).
###   - restore_listening() must be called explicitly to re-enable auto-listen / wake-word.
###
### This file is synchronous: calling AlfredSpeak(...) or resuming onboard chunks will block the caller.
### If you call these from the Tk main thread, the GUI will block. Use a separate process/thread if you want non-blocking UI.
##
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import re
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
##
### Keep these imports in place so other modules can import them if needed
##import sounddevice as sd
##import vosk
##
### Adjust these imports to your environment layout if necessary
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (drop-in) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### ----------------- vlc detection -----------------
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##        print("speech.py: python-vlc/libvlc available")
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("speech.py: python-vlc imported but libvlc.Instance() failed:", e)
##except Exception as e:
##    print("speech.py: python-vlc import failed:", e)
##
### External VLC binary fallback
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("speech.py: found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("speech.py: WARNING: No VLC available, falling back to playsound (limited pause/stop)")
##
### ----------------- SpeechModule -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine (shared for onboard fallback)
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("speech.py: pyttsx3 init failed:", e)
##            self.engine = None
##
##        # basic state flags
##        self.wake_word_on_off = False
##
##        # GUI-visible flags
##        self.tk_start_speech = True   # Start available by default per your request
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##
##        # runtime control flags (synchronous)
##        self._stop_request = False
##        self._pause_requested = False
##
##        # chunking state for onboard playback (pyttsx3)
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # pause/wake-word suppression helpers
##        self._prev_wake_word = None
##        self._suppress_auto_listen = False
##        self._pause_in_progress = False
##
##        # VLC & subprocess handles
##        self._player = None
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("speech.py: Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##        self._vlc_process = None
##        self._paused_file = None
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("speech.py: Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("speech.py: Wake word OFF (synchronous mode)")
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_pause_speech(self):
##        """
##        Pause-as-halt: interrupt playback, preserve remaining chunks, and suppress auto-listen.
##        Idempotent: repeated calls are ignored while already paused.
##        """
##        print("speech.set_tk_pause_speech() called (sync)")
##        if self._pause_in_progress:
##            print("speech: already paused — ignoring repeated pause")
##            return
##
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##
##        self._pause_requested = True
##        self._pause_in_progress = True
##
##        # preserve and suppress wake-word so other parts won't auto-start mic
##        self._prev_wake_word = self.wake_word_on_off
##        if self._prev_wake_word:
##            print("speech: suppressing wake_word_on_off while paused (was True)")
##        self.wake_word_on_off = False
##        # This flag prevents main() or other modules from auto-starting listen.listen()
##        self._suppress_auto_listen = True
##
##        # Try to pause python-vlc player in-place if playing
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        st = self._player.get_state()
##                        is_playing = (st == vlc.State.Playing)
##                    if is_playing:
##                        print("speech: pausing python-vlc player in-place")
##                        self._player.set_pause(True)
##                        return
##                    else:
##                        print("speech: python-vlc player not playing (pause no-op)")
##                except Exception as e:
##                    print("speech: error while pausing libvlc player:", e)
##        except Exception:
##            pass
##
##        # For pyttsx3 onboard: engine.stop() will interrupt runAndWait(); chunk loop will save remaining chunks
##        try:
##            if self.engine:
##                print("speech: calling engine.stop() to interrupt onboard playback and preserve remaining chunks")
##                self.engine.stop()
##        except Exception as e:
##            print("speech: engine.stop() raised:", e)
##
##        # For external vlc subprocess fallback: terminate process to emulate pause (play loop preserves _paused_file)
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                print("speech: terminated external VLC to emulate pause; paused file:", self._paused_file)
##        except Exception:
##            pass
##
##    def set_tk_start_speech(self):
##        """
##        Resume speech playback only. Does NOT automatically re-enable listening.
##        To re-enable listening explicitly call speech.restore_listening() or have GUI call listen.set_listen_whisper(True).
##        """
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##
##        # clear pause flags for playback; do NOT clear _suppress_auto_listen
##        self._pause_requested = False
##        self._pause_in_progress = False
##        self._stop_request = False
##
##        # Resume onboard remaining chunks if they exist (blocking)
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##            try:
##                # blocking call that will honor pause/stop flags
##                self._play_onboard_chunks(chunks)
##            finally:
##                self._pause_requested = False
##                self._stop_request = False
##
##        # Resume python-vlc player if present and paused
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    self._player.set_pause(False)
##                except Exception:
##                    try:
##                        self._player.play()
##                    except Exception:
##                        pass
##        except Exception as e:
##            print("speech: libvlc resume attempt failed:", e)
##
##        # If an external paused file exists, play it now (this keeps _suppress_auto_listen True)
##        if self._paused_file:
##            fname = self._paused_file
##            self._paused_file = None
##            self._play_file_controlled(fname)
##
##    def set_tk_stop_speech(self):
##        """Stop playback and clear pending/onboard buffers immediately."""
##        print("speech.set_tk_stop_speech() called (sync)")
##
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##
##        # request stop and clear pause markers
##        self._stop_request = True
##        self._pause_requested = False
##        self._pause_in_progress = False
##
##        # stop pyttsx3 and clear remaining chunks
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # stop python-vlc player
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("speech: stopping python-vlc player")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # stop external vlc process if any
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##
##        # clear paused file pointer and do not restore auto-listen
##        self._paused_file = None
##
##    def restore_listening(self):
##        """
##        Explicitly restore listening behavior that was suppressed by pause.
##        Call this from GUI if the user wants the mic to be allowed to start again.
##        """
##        print("speech.restore_listening() called")
##        if self._prev_wake_word is not None:
##            self.wake_word_on_off = self._prev_wake_word
##            self._prev_wake_word = None
##            print("speech: wake_word_on_off restored to", self.wake_word_on_off)
##        # clear suppression so main() will allow listen.listen()
##        self._suppress_auto_listen = False
##
##    def stop_current(self):
##        """Convenience immediate stop (alias)."""
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    # ----------------- Public TTS API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking synchronous speak. Prefer file-based playback (edge-tts -> file -> controlled playback).
##        If that is not available or not appropriate, fallback to chunked pyttsx3 (honors pause/halt).
##        """
##        if not text:
##            return
##
##        # Attempt to generate a file with edge-tts synchronously
##        fname = None
##        try:
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##        except Exception as e:
##            print("speech: edge-tts generation failed:", e)
##            fname = None
##
##        if fname and (_HAS_LIBVLC or _HAS_VLC_BINARY):
##            # play file in blocking, controllable way
##            self._play_file_controlled(fname)
##        else:
##            # fallback: chunk and play with pyttsx3 so pause/halt works
##            chunks = self._chunk_text_for_onboard(text)
##            if self._onboard_remaining_chunks:
##                # if resume existed, prefer remaining chunks
##                chunks = self._onboard_remaining_chunks
##                self._onboard_remaining_chunks = None
##            self._onboard_current_voice = voice
##            self._onboard_current_style = style
##            self._play_onboard_chunks(chunks)
##
##    def AlfredSpeak_Onboard(self, text):
##        """Direct onboard blocking speak using chunking (so pause works)."""
##        if not text:
##            return
##        chunks = self._chunk_text_for_onboard(text)
##        self._play_onboard_chunks(chunks)
##
##    def AlfredSpeak_Bluetooth(self, text):
##        """Send to bluetooth device if configured, then speak normally."""
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("speech: error sending bluetooth:", e)
##        self.AlfredSpeak(text)
##
##    # ---------------- helpers for pyttsx3 chunking ----------------
##    def _chunk_text_for_onboard(self, text):
##        """
##        Split text into sentence-like chunks for pausable playback.
##        Conservative sentence split + sub-splitting for long segments.
##        """
##        if not text:
##            return []
##        # split on sentence-ending punctuation followed by a capital or digit (heuristic)
##        pattern = re.compile(r'(?<=\S[.!?])\s+(?=[A-Z0-9"\'“”])')
##        parts = pattern.split(text.strip())
##
##        maxlen = 250
##        chunks = []
##        for p in parts:
##            if len(p) <= maxlen:
##                chunks.append(p.strip())
##            else:
##                sub = re.split(r'([,;]\s+)', p)
##                acc = ""
##                for s in sub:
##                    acc += s
##                    if len(acc) >= maxlen:
##                        chunks.append(acc.strip())
##                        acc = ""
##                if acc.strip():
##                    chunks.append(acc.strip())
##        return [c for c in chunks if c]
##
##    def _play_onboard_chunks(self, chunks):
##        """
##        Blocking play of chunk list with pyttsx3.
##        Honors _pause_requested (HALT) and _stop_request.
##        Preserves remaining chunks in _onboard_remaining_chunks when paused/interrupted.
##        """
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        # If resume scenario: use saved remaining
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##
##        for idx, chunk in enumerate(chunks):
##            # check immediate stop before chunk
##            if self._stop_request:
##                print("speech: onboard playback — stop requested before chunk, aborting")
##                self._onboard_remaining_chunks = None
##                return
##
##            # if paused before chunk start -> save remaining and return
##            if self._pause_requested and self._pause_in_progress:
##                remaining = chunks[idx:]
##                print("speech: onboard playback — pause requested before chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                return
##
##            try:
##                # Speak chunk — blocking runAndWait
##                self.engine.say(chunk)
##                self.engine.runAndWait()
##            except Exception as e:
##                # interruption likely due to engine.stop() called by pause/stop
##                print("speech: pyttsx3 chunk interrupted/exception:", e)
##                if self._pause_requested:
##                    remaining = chunks[idx:]
##                    print("speech: pyttsx3 interrupted during chunk; preserving", len(remaining), "chunks")
##                    self._onboard_remaining_chunks = remaining
##                    return
##                if self._stop_request:
##                    self._onboard_remaining_chunks = None
##                    return
##                # Otherwise continue
##
##            # after chunk checks
##            if self._stop_request:
##                print("speech: onboard playback — stop requested after chunk, aborting")
##                self._onboard_remaining_chunks = None
##                return
##            if self._pause_requested and self._pause_in_progress:
##                remaining = chunks[idx+1:]
##                print("speech: onboard playback — pause requested after chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                return
##
##        # played all chunks
##        print("speech: onboard playback finished all chunks")
##        self._onboard_remaining_chunks = None
##        # reset local pause/stop markers but do NOT automatically restore listening
##        self._pause_requested = False
##        self._stop_request = False
##        self._pause_in_progress = False
##
##    # ---------------- edge-tts file generation ----------------
##    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        # produce a temp file and return its path
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await communicate.save(fname)
##        return fname
##
##    # ---------------- playback core (blocking) for files ----------------
##    def _play_file_controlled(self, fname):
##        """
##        Blocking playback respecting _pause_requested and _stop_request.
##        Uses python-vlc/libvlc if available, else external vlc subprocess, else playsound.
##        """
##        if not os.path.exists(fname):
##            print("speech._play_file_controlled: file not found:", fname)
##            return
##
##        print("speech._play_file_controlled: starting playback for", fname, "stop_request=", self._stop_request)
##
##        try:
##            # python-vlc path
##            if _HAS_LIBVLC and self._vlc_instance:
##                print("speech: using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait briefly for playing to begin
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request:
##                        break
##                    time.sleep(0.05)
##
##                # playback loop: honor pause/stop as HALT
##                while True:
##                    if self._stop_request:
##                        try:
##                            player.stop()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested and self._pause_in_progress:
##                        try:
##                            player.set_pause(True)
##                        except Exception:
##                            pass
##                        # wait until resumed or stopped
##                        while self._pause_requested and (not self._stop_request):
##                            time.sleep(0.05)
##                        # resume
##                        try:
##                            player.set_pause(False)
##                        except Exception:
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            # external VLC binary fallback (synchronous subprocess)
##            elif _HAS_VLC_BINARY:
##                print("speech: using external VLC binary:", _VLC_BINARY)
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                self._vlc_process = proc
##                self._paused_file = fname
##
##                while True:
##                    if self._stop_request:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if self._pause_requested and self._pause_in_progress:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        # paused file is preserved in self._paused_file
##                        break
##                    if proc.poll() is not None:
##                        break
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##                self._vlc_process = None
##
##            # final fallback: playsound (limited control)
##            else:
##                print("speech: no VLC available, falling back to playsound (limited)")
##                if self._stop_request:
##                    print("speech: stop already requested; skipping playsound")
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("speech: playsound failed:", e)
##
##        finally:
##            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##        print("speech._play_file_controlled: playback finished/aborted")
##
##    # ---------------- helpers / status ----------------
##    def is_paused(self):
##        """Public helper for other modules to check if speech is in halted pause state."""
##        return bool(getattr(self, "_pause_in_progress", False))
##
### module-level instance
##speech = SpeechModule()
##print("speech module (drop-in) initialized — done")





#####   RESTART LISTEN WHEN PUSH PLAY NOT GOOD
##
### speech.py (drop-in) — synchronous, pyttsx3 chunked playback with robust pause=hault behavior
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import re
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (sync + robust pause/halt) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### Try python-vlc / libvlc
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("python-vlc imported but libvlc.Instance() failed:", e)
##except Exception as e:
##    print("python-vlc import failed:", e)
##
### fallback: check external vlc binary
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
##
### ----------------- SpeechModule (no threads) -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine for onboard fallback
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("pyttsx3 init failed:", e)
##            self.engine = None
##
##        # wake-word / listen control (external code can rely on this)
##        self.wake_word_on_off = False
##
##        # GUI-visible flags
##        self.tk_start_speech = True   # you wanted Start active at startup
##        self.tk_stop_speech  = False
##        self.tk_pause_speech = False
##
##        # runtime control flags (synchronous booleans)
##        self._stop_request = False
##        self._pause_requested = False
##
##        # pyttsx3 chunk bookkeeping (for pause/resume)
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # wake-word preservation during pause
##        self._prev_wake_word = None    # None means "no saved state"
##        self._pause_in_progress = False
##
##        # vlc vars
##        self._player = None
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##        self._vlc_process = None
##        self._paused_file = None
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF (synchronous mode)")
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_start_speech(self):
##        """
##        Resume/start playback. If pausing previously suppressed listening,
##        this will restore the previous wake_word state.
##        """
##        print("speech.set_tk_start_speech() called (sync)")
##        # If nothing to resume/enable, still clear pause flags
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##        self._stop_request = False
##        # Clear pause request BEFORE resuming so playback loop won't immediately re-pause
##        was_paused = self._pause_in_progress
##        self._pause_requested = False
##        self._pause_in_progress = False
##
##        # restore previous wake-word state if we suppressed it on pause
##        if self._prev_wake_word is not None:
##            self.wake_word_on_off = self._prev_wake_word
##            self._prev_wake_word = None
##            print("speech: restored wake_word_on_off to", self.wake_word_on_off)
##
##        # Resume onboard chunks if they exist (blocking)
##        if self._onboard_remaining_chunks:
##            print("speech: resuming onboard remaining chunks:", len(self._onboard_remaining_chunks))
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##            # Play blocking; this will respect self._pause_requested/_stop_request
##            try:
##                self._play_onboard_chunks(chunks)
##            finally:
##                # ensure flags are reset on exit
##                self._pause_requested = False
##                self._stop_request = False
##
##        # If python-vlc player exists and is paused, try to resume
##        if _HAS_LIBVLC and self._player:
##            try:
##                print("speech: attempting to resume python-vlc player")
##                try:
##                    self._player.set_pause(False)
##                except Exception:
##                    self._player.play()
##            except Exception as e:
##                print("speech: resume via libvlc failed:", e)
##
##        # If an external paused file exists, play it now
##        if self._paused_file:
##            fname = self._paused_file
##            self._paused_file = None
##            print("speech: resuming paused external file:", fname)
##            self._play_file_controlled(fname)
##
##    def set_tk_stop_speech(self):
##        """Stop and clear everything immediately."""
##        print("speech.set_tk_stop_speech() called (sync)")
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##
##        # request stop and clear pause flags
##        self._stop_request = True
##        self._pause_requested = False
##        self._pause_in_progress = False
##
##        # clear onboard buffers
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # stop libvlc player
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("speech: stopping python-vlc player")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # terminate external process if any
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##
##        # clear paused-file pointer
##        self._paused_file = None
##
##    def set_tk_pause_speech(self):
##        """
##        Pause as a HALT: do not re-enable listening, do not clear remaining chunks.
##        Implemented idempotently (multiple pause presses safe).
##        """
##        print("speech.set_tk_pause_speech() called (sync)")
##        # if already paused — ignore (idempotent)
##        if self._pause_in_progress:
##            print("speech: already in pause/halt state — ignoring repeated pause call")
##            return
##
##        # record pause state
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self._pause_requested = True
##        self._pause_in_progress = True
##
##        # preserve wake-word and disable listening while paused so no automatic re-listen happens
##        self._prev_wake_word = self.wake_word_on_off
##        if self._prev_wake_word:
##            print("speech: suppressing wake_word_on_off while paused (was True)")
##        self.wake_word_on_off = False
##
##        # If libvlc player exists and is playing -> pause in-place
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    pl_state = None
##                    try:
##                        pl_state = self._player.get_state()
##                    except Exception:
##                        pass
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        is_playing = (pl_state == vlc.State.Playing)
##                    if is_playing:
##                        print("speech: pausing python-vlc player in-place")
##                        self._player.set_pause(True)
##                        return
##                    else:
##                        print("speech: python-vlc not playing; will apply pause via engine.stop()/process termination")
##                except Exception as e:
##                    print("speech: error while pausing libvlc player:", e)
##        except Exception:
##            pass
##
##        # If playing onboard (pyttsx3) -> calling engine.stop() will interrupt runAndWait but we keep remaining chunks
##        try:
##            if self.engine:
##                print("speech: calling engine.stop() to interrupt onboard playback and preserve remaining chunks")
##                # engine.stop() usually raises/interupts runAndWait() — our chunk loop will capture that and store remaining
##                self.engine.stop()
##        except Exception as e:
##            print("speech: engine.stop() raised:", e)
##
##        # If using external process, terminate it to emulate pause (play loop preserved _paused_file)
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                print("speech: terminated external vlc process to emulate pause; paused file:", self._paused_file)
##        except Exception:
##            pass
##
##    def stop_current(self):
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    # ----------------- Public API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking synchronous speak: prefer generating an audio file (edge-tts) and playing it controllably.
##        If not possible (or if no VLC), falls back to chunked pyttsx3 so pause/resume works.
##        """
##        if not text:
##            return
##
##        # if wake_word_on_off is True we still do immediate speak (synchronous design)
##        try:
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##        except Exception as e:
##            print("speech: edge-tts generation failed:", e)
##            fname = None
##
##        if fname and (_HAS_LIBVLC or _HAS_VLC_BINARY):
##            # play file under control
##            self._play_file_controlled(fname)
##        else:
##            # fallback to chunked onboard pyttsx3 playback (allows pause/halt)
##            chunks = self._chunk_text_for_onboard(text)
##            # If there are previously saved remaining chunks (resume scenario), prefer them
##            if self._onboard_remaining_chunks:
##                chunks = self._onboard_remaining_chunks
##                self._onboard_remaining_chunks = None
##            self._onboard_current_voice = voice
##            self._onboard_current_style = style
##            self._play_onboard_chunks(chunks)
##
##    def AlfredSpeak_Onboard(self, text):
##        if not text:
##            return
##        chunks = self._chunk_text_for_onboard(text)
##        self._play_onboard_chunks(chunks)
##
##    def AlfredSpeak_Bluetooth(self, text):
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("speech: error sending bluetooth:", e)
##        self.AlfredSpeak(text)
##
##    # ---------------- helpers for pyttsx3 chunking ----------------
##    def _chunk_text_for_onboard(self, text):
##        pattern = re.compile(r'(?<=\S[.!?])\s+(?=[A-Z0-9"\'“”])')
##        parts = pattern.split(text.strip())
##        maxlen = 250
##        chunks = []
##        for p in parts:
##            if len(p) <= maxlen:
##                chunks.append(p.strip())
##            else:
##                sub = re.split(r'([,;]\s+)', p)
##                acc = ""
##                for s in sub:
##                    acc += s
##                    if len(acc) >= maxlen:
##                        chunks.append(acc.strip())
##                        acc = ""
##                if acc.strip():
##                    chunks.append(acc.strip())
##        return [c for c in chunks if c]
##
##    def _play_onboard_chunks(self, chunks):
##        """
##        Blocking chunked playback honoring pause as HALT (we store remaining chunks).
##        If pause requested mid-chunk, engine.stop() will interrupt runAndWait and we store leftover chunks.
##        """
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        # If resume had preserved remaining chunks, use them
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##            self._onboard_remaining_chunks = None
##
##        # ensure pause flag is only honored if set BEFORE chunk start or triggered by engine.stop interruption
##        for idx, chunk in enumerate(chunks):
##            if self._stop_request:
##                print("speech: onboard playback — stop requested before chunk, aborting")
##                self._onboard_remaining_chunks = None
##                return
##
##            if self._pause_requested and self._pause_in_progress:
##                # preserve remaining and return immediately (do not re-enable listening)
##                remaining = chunks[idx:]
##                print("speech: onboard playback — pause requested before chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                return
##
##            try:
##                # Speak this chunk (blocking)
##                self.engine.say(chunk)
##                self.engine.runAndWait()
##            except Exception as e:
##                # engine.stop() or external interruption will typically raise/interrupt here
##                print("speech: pyttsx3 chunk interrupted/exception:", e)
##                # If pause was requested, preserve current chunk + rest
##                if self._pause_requested:
##                    remaining = chunks[idx:]
##                    print("speech: pyttsx3 interrupted during chunk; preserving", len(remaining), "chunks")
##                    self._onboard_remaining_chunks = remaining
##                    return
##                # If stop was requested, clear remaining and return
##                if self._stop_request:
##                    self._onboard_remaining_chunks = None
##                    return
##                # Otherwise continue to next chunk
##
##            # After a chunk finishes check pause/stop again
##            if self._stop_request:
##                print("speech: onboard playback — stop requested after chunk, aborting")
##                self._onboard_remaining_chunks = None
##                return
##            if self._pause_requested and self._pause_in_progress:
##                remaining = chunks[idx+1:]
##                print("speech: onboard playback — pause requested after chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                return
##
##        # Completed all chunks
##        print("speech: onboard playback finished all chunks")
##        self._onboard_remaining_chunks = None
##        # reset pause/stop flags
##        self._pause_requested = False
##        self._stop_request = False
##        self._pause_in_progress = False
##        # do NOT automatically re-enable wake_word_on_off here; start handler is responsible for restoring it
##
##    # ---------------- playback core (blocking) for files ----------------
##    def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        return asyncio.get_event_loop().run_until_complete(communicate.save(fname)) or fname
##
##    def _play_file_controlled(self, fname):
##        if not os.path.exists(fname):
##            print("_play_file_controlled: file not found:", fname)
##            return
##
##        print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)
##
##        try:
##            # python-vlc primary path
##            if _HAS_LIBVLC and self._vlc_instance:
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait briefly for playing to begin
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request:
##                        break
##                    time.sleep(0.05)
##
##                # playback loop honoring pause as HALT and stop
##                while True:
##                    if self._stop_request:
##                        try:
##                            player.stop()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested:
##                        try:
##                            player.set_pause(True)
##                        except Exception:
##                            pass
##                        # wait until resumed or stopped (busy-wait)
##                        while self._pause_requested and (not self._stop_request):
##                            time.sleep(0.05)
##                        try:
##                            player.set_pause(False)
##                        except Exception:
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            # external vlc binary fallback (synchronous)
##            elif _HAS_VLC_BINARY:
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                self._vlc_process = proc
##                self._paused_file = fname
##
##                while True:
##                    if self._stop_request:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if self._pause_requested:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if proc.poll() is not None:
##                        break
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##                self._vlc_process = None
##
##            else:
##                # playsound fallback (blocking, limited control)
##                if self._stop_request:
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("playsound failed:", e)
##
##        finally:
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##        print("_play_file_controlled: playback finished/aborted")
##
### module instance
##speech = SpeechModule()
##print("speech (sync + robust pause/halt) initialized — done")




#####    Not too bad PAUSE Restart LISTEN NOT GOOD maybe
##
### speech.py -- synchronous (no threads) with pyttsx3 pause/resume/stop emulation
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import re
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (synchronous w/ pyttsx3 chunking) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### ----------------- try python-vlc / libvlc -----------------
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##        print("python-vlc and libvlc available")
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("python-vlc imported but libvlc.Instance() failed:", e)
##except Exception as e:
##    print("python-vlc import failed:", e)
##
### fallback: check external vlc binary
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("Found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
##
### ----------------- SpeechModule (no threads) -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine for onboard fallback
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("pyttsx3 init failed:", e)
##            self.engine = None
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##
##        # flags for GUI control
##        self.tk_start_speech = True   # you said you want start active on startup
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # runtime control flags (synchronous)
##        self._stop_request = False
##        self._pause_requested = False
##
##        # For pyttsx3 chunked playback:
##        self._onboard_remaining_chunks = None  # list[str] of chunks yet to play
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # vlc vars
##        self._player = None            # python-vlc player (if used)
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##        # for external vlc subprocess
##        self._vlc_process = None
##        self._paused_file = None
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF (synchronous mode)")
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_start_speech(self):
##        """Resume or start playback; clears stop/pause flags.
##           If onboard (pyttsx3) had remaining chunks, resume them (blocking)."""
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##        self._stop_request = False
##        self._pause_requested = False
##
##        # If pyttsx3 had remaining chunks, resume them
##        if self._onboard_remaining_chunks:
##            print("Resuming onboard (pyttsx3) remaining chunks...")
##            try:
##                # play blocking; note: this will block the caller/UI thread
##                self._play_onboard_chunks(self._onboard_remaining_chunks)
##            finally:
##                # when finished or stopped we clear remaining chunks
##                self._onboard_remaining_chunks = None
##                self._onboard_current_voice = None
##                self._onboard_current_style = None
##            return
##
##        # If python-vlc player exists and is paused: resume it
##        if _HAS_LIBVLC and self._player:
##            try:
##                print("Attempting to resume existing python-vlc player (sync)")
##                try:
##                    self._player.set_pause(False)
##                except Exception:
##                    self._player.play()
##                return
##            except Exception as e:
##                print("Resume via set_pause/play failed:", e)
##
##        # If we have a paused external file from earlier, play it now (blocking)
##        if self._paused_file:
##            fname = self._paused_file
##            self._paused_file = None
##            print("Resuming paused file (external fallback):", fname)
##            self._play_file_controlled(fname)
##
##    def set_tk_stop_speech(self):
##        """Immediate stop: set stop flag and attempt to stop player/process and clear onboard buffers."""
##        print("speech.set_tk_stop_speech() called (sync)")
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##        self._stop_request = True
##        self._pause_requested = False
##
##        # stop pyttsx3 engine and clear remaining onboard chunks
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##        self._onboard_remaining_chunks = None
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##
##        # stop python-vlc player if present
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("Stopping python-vlc player (sync)")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # stop external vlc process if present
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##
##        # clear paused file
##        self._paused_file = None
##
##    def set_tk_pause_speech(self):
##        """Pause playback: for pyttsx3 -> stop engine and keep remaining chunks; for libvlc -> set_pause True; else emulate."""
##        print("speech.set_tk_pause_speech() called (sync)")
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self._pause_requested = True
##
##        # If playing via python-vlc, try to pause in-place
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        try:
##                            st = self._player.get_state()
##                            is_playing = (st == vlc.State.Playing)
##                        except Exception:
##                            is_playing = False
##                    if is_playing:
##                        print("Pausing python-vlc player (sync set_pause True)")
##                        self._player.set_pause(True)
##                        return
##                except Exception as e:
##                    print("Error while pausing python-vlc player (sync):", e)
##        except Exception:
##            pass
##
##        # If playing onboard (pyttsx3) — stop engine to interrupt and preserve remaining chunks
##        try:
##            if self.engine:
##                # engine.stop() interrupts runAndWait(); we'll preserve leftover chunks in _onboard_remaining_chunks
##                print("Requesting engine.stop() to interrupt pyttsx3 and preserve remaining chunks")
##                self.engine.stop()
##        except Exception as e:
##            print("pyttsx3 engine.stop() raised:", e)
##
##        # For external-vlc fallback we emulate pause by terminating process (playback loop preserved file)
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                # _paused_file should already be stored by the play loop
##                print("Terminated external VLC process to emulate pause; paused file:", self._paused_file)
##        except Exception:
##            pass
##
##    def stop_current(self):
##        """Convenience immediate stop (same as set_tk_stop_speech)."""
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    # ----------------- Public API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking / synchronous: choose backend.
##        If a pyttsx3-onboard path is used (fallback), it will play chunked and honor pause/stop.
##        """
##        if not text:
##            return
##
##        # Generate audio file synchronously (edge-tts) and play controllably if libvlc or external vlc available
##        try:
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##        except Exception as e:
##            print("Error generating TTS via edge-tts (sync):", e)
##            fname = None
##
##        if fname and (_HAS_LIBVLC or _HAS_VLC_BINARY):
##            try:
##                self._play_file_controlled(fname)
##            except Exception as e:
##                print("Error during controlled playback (sync):", e)
##                traceback.print_exc()
##        else:
##            # fallback to onboard blocking speak (pyttsx3) but we will play it chunked so pause/resume works
##            try:
##                # prepare chunks and play them respecting pause/stop
##                chunks = self._chunk_text_for_onboard(text)
##                self._onboard_current_voice = voice
##                self._onboard_current_style = style
##                # play blocks until complete or stopped/paused
##                self._play_onboard_chunks(chunks)
##            except Exception as e:
##                print("Onboard fallback failed (sync):", e)
##                traceback.print_exc()
##
##    def AlfredSpeak_Onboard(self, text):
##        """Blocking onboard TTS (pyttsx3) using chunking so pause/resume works."""
##        if not text:
##            return
##        chunks = self._chunk_text_for_onboard(text)
##        # play blocking chunked
##        self._onboard_current_voice = None
##        self._onboard_current_style = None
##        self._play_onboard_chunks(chunks)
##
##    def AlfredSpeak_Bluetooth(self, text):
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("Error sending bluetooth:", e)
##        # synchronous speak
##        self.AlfredSpeak(text)
##
##    # ---------------- helpers for pyttsx3 chunking ----------------
##    def _chunk_text_for_onboard(self, text):
##        """Split text into reasonable-sized chunks (sentences)."""
##        # split into sentences conservatively; keep punctuation
##        # This regex splits on sentence-ending punctuation followed by space+capital or end-of-string.
##        pattern = re.compile(r'(?<=\S[.!?])\s+(?=[A-Z0-9"\'“”])')
##        parts = pattern.split(text.strip())
##        # further break parts that are longer than e.g. 250 chars into smaller segments
##        maxlen = 250
##        chunks = []
##        for p in parts:
##            if len(p) <= maxlen:
##                chunks.append(p.strip())
##            else:
##                # break on commas or spaces
##                sub = re.split(r'([,;]\s+)', p)
##                acc = ""
##                for s in sub:
##                    acc += s
##                    if len(acc) >= maxlen:
##                        chunks.append(acc.strip())
##                        acc = ""
##                if acc.strip():
##                    chunks.append(acc.strip())
##        return [c for c in chunks if c]
##
##    def _play_onboard_chunks(self, chunks):
##        """
##        Play list of chunks sequentially with pyttsx3. Honors self._pause_requested and self._stop_request.
##        If paused mid-play, leftover chunks are stored into self._onboard_remaining_chunks and playback returns.
##        This function is blocking.
##        """
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        # If there's already remaining chunks saved and caller calls resume, prefer that saved list
##        if self._onboard_remaining_chunks:
##            chunks = self._onboard_remaining_chunks
##
##        self._onboard_remaining_chunks = None  # we'll refill if paused mid-play
##
##        for idx, chunk in enumerate(chunks):
##            # check stop before every chunk
##            if self._stop_request:
##                print("Onboard playback: stop requested before chunk, aborting.")
##                self._onboard_remaining_chunks = None
##                return
##
##            # if pause requested BEFORE starting this chunk, preserve remaining and return
##            if self._pause_requested:
##                remaining = chunks[idx:]
##                print("Onboard playback: pause requested before chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                # clear pause flag (we honor it until resume call sets it off)
##                # do NOT clear _pause_requested here — the resume handler will clear it.
##                return
##
##            try:
##                # speak this chunk — blocking runAndWait
##                # set voice/params here if you want to change voice per-chunk; currently using engine defaults
##                self.engine.say(chunk)
##                self.engine.runAndWait()
##            except Exception as e:
##                # runAndWait can be interrupted by engine.stop(); when pause/stop requested engine.stop()
##                print("pyttsx3 chunk play exception (likely interrupted):", e)
##
##            # After chunk finished, check pause/stop
##            if self._stop_request:
##                print("Onboard playback: stop requested after chunk, aborting.")
##                self._onboard_remaining_chunks = None
##                return
##            if self._pause_requested:
##                # store remaining chunks and return
##                remaining = chunks[idx+1:]
##                print("Onboard playback: pause requested after chunk; preserving", len(remaining), "chunks")
##                self._onboard_remaining_chunks = remaining
##                return
##
##        # finished all chunks
##        print("Onboard playback: finished all chunks")
##        self._onboard_remaining_chunks = None
##        # reset control flags
##        self._pause_requested = False
##        self._stop_request = False
##
##    # ---------------- playback core (blocking) for files ----------------
##    def _play_file_controlled(self, fname):
##        """
##        Blocking playback that honors self._stop_request and self._pause_requested.
##        Uses python-vlc (libvlc) if available, else external vlc process (if found), else playsound.
##        """
##        if not os.path.exists(fname):
##            print("_play_file_controlled: file not found:", fname)
##            return
##
##        print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)
##
##        try:
##            # python-vlc primary path
##            if _HAS_LIBVLC and self._vlc_instance:
##                print(" _play_file_controlled (sync): using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait up to a short timeout for actual play to begin
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request:
##                        break
##                    time.sleep(0.05)
##
##                # playback loop (blocking) honoring pause/stop flags
##                while True:
##                    if self._stop_request:
##                        print(" _play_file_controlled (sync): stop requested -> stopping python-vlc player")
##                        try:
##                            player.stop()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested:
##                        # pause in-place using set_pause(True)
##                        try:
##                            print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
##                            player.set_pause(True)
##                        except Exception:
##                            pass
##                        # busy-wait until resume or stop
##                        while self._pause_requested and (not self._stop_request):
##                            time.sleep(0.05)
##                        # when resumed
##                        try:
##                            player.set_pause(False)
##                        except Exception:
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    # check normal end
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##                # cleanup
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            # external vlc binary fallback (runs subprocess synchronously but poll-loop)
##            elif _HAS_VLC_BINARY:
##                print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                self._vlc_process = proc
##                self._paused_file = fname  # keep for resume if pause emulation used
##
##                while True:
##                    if self._stop_request:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if self._pause_requested:
##                        # emulate pause: terminate process and keep file path for resume
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    # check if finished
##                    if proc.poll() is not None:
##                        break
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##                self._vlc_process = None
##
##            # last fallback: blocking playsound
##            else:
##                print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
##                if self._stop_request:
##                    print(" _play_file_controlled (sync): stop already requested; skipping playsound")
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("playsound failed (sync):", e)
##
##        finally:
##            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##        print("_play_file_controlled (sync): playback finished/aborted")
##
##    # ---------------- fallback pyttsx3 single-call (kept for reference) ----------------
##    def _onboard_speak_blocking_with_stop(self, text):
##        """Legacy single-run fallback; replaced by chunked _play_onboard_chunks for pause support."""
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##        if self._stop_request:
##            return
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed (sync):", e)
##            traceback.print_exc()
##
### create module-level instance
##speech = SpeechModule()
##print("speech (synchronous w/ pyttsx3 chunking) initialized — done")





#######   NOT so Great No Threading NOT FOR PYTTSX3
####
####
### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### ----------------- try python-vlc / libvlc -----------------
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##        print("python-vlc and libvlc available")
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("python-vlc imported but libvlc.Instance() failed:", e)
##except Exception as e:
##    print("python-vlc import failed:", e)
##
### fallback: check external vlc binary
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("Found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
##
### ----------------- SpeechModule (no threads) -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine for onboard fallback
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("pyttsx3 init failed:", e)
##            self.engine = None
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##
##        # flags for GUI control
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # runtime control flags
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None       # for external-cli fallback resume emulation
##        self._player = None            # python-vlc player (if used)
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF (synchronous mode)")
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_start_speech(self):
##        """Resume or start playback; clears stop/pause flags."""
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##        self._stop_request = False
##        self._pause_requested = False
##
##        # If a python-vlc player exists and is paused, attempt resume
##        if _HAS_LIBVLC and self._player:
##            try:
##                print("Attempting to resume existing python-vlc player (sync)")
##                self._player.set_pause(False)
##                return
##            except Exception as e:
##                print("Resume via set_pause failed:", e)
##                try:
##                    self._player.play()
##                    return
##                except Exception:
##                    pass
##
##        # If we have a paused file from external VLC, play it now (blocking)
##        if self._paused_file:
##            fname = self._paused_file
##            self._paused_file = None
##            print("Resuming paused file (external fallback):", fname)
##            self._play_file_controlled(fname)
##
##    def set_tk_stop_speech(self):
##        """Immediate stop: set stop flag and attempt to stop player/process."""
##        print("speech.set_tk_stop_speech() called (sync)")
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##        self._stop_request = True
##        self._pause_requested = False
##
##        # stop python-vlc player if present
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("Stopping python-vlc player (sync)")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # cannot forcibly stop playsound easily; external process is handled in _play_file_controlled
##        # clear paused file
##        self._paused_file = None
##
##        # stop pyttsx3 if in use
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##
##    def set_tk_pause_speech(self):
##        """Pause playback: if player exists, pause; otherwise emulate by stopping process and remembering file."""
##        print("speech.set_tk_pause_speech() called (sync)")
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self._pause_requested = True
##
##        # Try to pause python-vlc player in-place
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        try:
##                            st = self._player.get_state()
##                            is_playing = (st == vlc.State.Playing)
##                        except Exception:
##                            is_playing = False
##                    if is_playing:
##                        print("Pausing python-vlc player (sync set_pause True)")
##                        self._player.set_pause(True)
##                        return
##                    else:
##                        print("python-vlc player not playing; cannot pause in-place (sync)")
##                except Exception as e:
##                    print("Error while pausing python-vlc player (sync):", e)
##        except Exception:
##            pass
##
##        # Emulate pause for external-vlc: set stop flag (the play loop will preserve the filename)
##        print("Emulating pause in sync mode (will stop playback and retain file for resume)")
##        self._stop_request = True
##        # pyttsx3 stop as well
##        try:
##            if self.engine:
##                self.engine.pause()
##        except Exception:
##            pass
##
##    def stop_current(self):
##        """Convenience immediate stop (same as set_tk_stop_speech)."""
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    # ----------------- Public API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        Blocking / synchronous: generate audio (edge-tts) then play it under control.
##        NOTE: This function blocks the caller until playback finishes or is stopped.
##        """
##        if not text:
##            return
##
##        # If wake_word_on_off is supported, queueing is not available in sync mode; behave similarly to previous logic
##        if self.wake_word_on_off:
##            # In a no-thread setup, we simply perform the speak immediately
##            pass
##
##        # Generate audio file synchronously (edge-tts)
##        try:
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##        except Exception as e:
##            print("Error generating TTS via edge-tts (sync):", e)
##            fname = None
##
##        if fname:
##            try:
##                # play blocking but controllable (pause/stop)
##                self._play_file_controlled(fname)
##            except Exception as e:
##                print("Error during controlled playback (sync):", e)
##                traceback.print_exc()
##        else:
##            # fallback to onboard blocking speak
##            try:
##                self._onboard_speak_blocking_with_stop(text)
##            except Exception as e:
##                print("Onboard fallback failed (sync):", e)
##
##    def AlfredSpeak_Onboard(self, text):
##        """Blocking onboard TTS (pyttsx3)."""
##        engine = pyttsx3.init("sapi5")
##        engine.setProperty("rate", 190)
##        voices = engine.getProperty("voices")
##        engine.setProperty("voice", voices[0].id)
##        engine.setProperty("volume", 1)
##
##        if self.tk_start_speech:
##            engine.say(text)
##            engine.runAndWait()
##        else:   
##            self.tk_pause_speech = True
##            self.engine.pause()
##
##    def AlfredSpeak_Bluetooth(self, text):
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("Error sending bluetooth:", e)
##        # synchronous speak
##        self.AlfredSpeak(text)
##
##    # ---------------- helpers -----------------
##    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await communicate.save(fname)
##        return fname
##
##    def _onboard_speak_blocking_with_stop(self, text):
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##        if self._stop_request:
##            return
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed (sync):", e)
##            traceback.print_exc()
##
##    # ---------------- playback core (blocking) ----------------
##    def _play_file_controlled(self, fname):
##        """
##        Blocking playback that honors self._stop_request and self._pause_requested.
##        Uses python-vlc (libvlc) if available, else external vlc process (if found), else playsound.
##        """
##        if not os.path.exists(fname):
##            print("_play_file_controlled: file not found:", fname)
##            return
##
##        print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)
##
##        try:
##            # python-vlc primary path
##            if _HAS_LIBVLC and self._vlc_instance:
##                print(" _play_file_controlled (sync): using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait up to a short timeout for actual play to begin
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request:
##                        break
##                    time.sleep(0.05)
##
##                # playback loop (blocking) honoring pause/stop flags
##                while True:
##                    if self._stop_request:
##                        print(" _play_file_controlled (sync): stop requested -> stopping python-vlc player")
##                        try:
##                            player.stop()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested:
##                        # pause in-place using set_pause(True)
##                        try:
##                            print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
##                            player.set_pause(True)
##                        except Exception:
##                            pass
##                        # busy-wait until resume or stop
##                        while self._pause_requested and (not self._stop_request):
##                            time.sleep(0.05)
##                        # when resumed
##                        try:
##                            player.set_pause(False)
##                        except Exception:
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    # check normal end
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##                # cleanup
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            # external vlc binary fallback (runs subprocess synchronously but poll-loop)
##            elif _HAS_VLC_BINARY:
##                print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                self._paused_file = fname  # keep for resume if pause emulation used
##
##                while True:
##                    if self._stop_request:
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    if self._pause_requested:
##                        # emulate pause: terminate process and keep file path for resume
##                        try:
##                            proc.terminate()
##                        except Exception:
##                            try:
##                                proc.kill()
##                            except Exception:
##                                pass
##                        break
##                    # check if finished
##                    if proc.poll() is not None:
##                        break
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##                self._vlc_process = None
##
##            # last fallback: blocking playsound
##            else:
##                print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
##                if self._stop_request:
##                    print(" _play_file_controlled (sync): stop already requested; skipping playsound")
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("playsound failed (sync):", e)
##
##        finally:
##            # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##        print("_play_file_controlled (sync): playback finished/aborted")
##
### create module-level instance
##speech = SpeechModule()
##print("speech (synchronous) initialized — done")
##
##
##
##


#####   NOT so Great Threading NOT FOR PYTTSX3
##
##
### speech.py -- improved pause/resume behavior for python-vlc + robust fallbacks
##import os
##import sys
##import time
##import queue
##import asyncio
##import tempfile
##import threading
##import traceback
##import subprocess
##import shutil
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
##import sounddevice as sd
##import vosk
##
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### Try python-vlc import and lib detection
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
##    vlc = _vlc
##    _HAS_PYTHON_VLC = True
##    # test lib availability
##    try:
##        inst = vlc.Instance()
##        _HAS_LIBVLC = True
##        del inst
##        print("python-vlc and libvlc available")
##    except Exception as e:
##        _HAS_LIBVLC = False
##        print("python-vlc imported but libvlc Instance() failed:", e)
##except Exception as e:
##    print("python-vlc import failed:", e)
##
### Find external vlc binary if libvlc not available
##_HAS_VLC_BINARY = False
##_VLC_BINARY = None
##if not _HAS_LIBVLC:
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("Found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print("WARNING: No usable VLC backend. Falling back to playsound (pause/stop limited).")
##
##class SpeechModule:
##    def __init__(self):
##        # initialize pyttsx3 engine
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("pyttsx3 init failed:", e)
##            self.engine = None
##
##        self.wake_word_on_off = False
##        self.speak_queue = queue.Queue()
##        self.audio_queue = queue.Queue()
##
##        self.Sending_On = False
##        self.Start_Speech = False
##        self.Stop_Speech = False
##
##        self.tk_start_speech = True
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # control flags & handles
##        self._currently_speaking = False
##        self._stop_request = threading.Event()
##        self._pause_event = threading.Event(); self._pause_event.set()
##        self._pause_requested = False
##        self._current_text = None
##
##        self._player = None            # python-vlc MediaPlayer
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##        self._vlc_process = None       # external vlc subprocess
##        self._paused_file = None       # file preserved when using external subprocess pause-emulation
##
##        # worker thread
##        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
##        self._worker_thread.start()
##
##    # --- wake-word ---
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF")
##
##    # --- GUI controls ---
##    def set_tk_start_speech(self):
##        """Resume or Start playback."""
##        print("speech.set_tk_start_speech() called")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##
##        # clear stop and unpause
##        self._stop_request.clear()
##        self._pause_requested = False
##        self._pause_event.set()
##        self.wake_word_on_off = True
##
##        # If python-vlc player exists and is paused: resume it
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    # use set_pause(False) to resume reliably
##                    print("Attempting to resume python-vlc player")
##                    self._player.set_pause(False)   # 0 => resume
##                    return
##                except Exception as e:
##                    print("Error resuming python-vlc player via set_pause:", e)
##                    try:
##                        self._player.play()
##                        return
##                    except Exception:
##                        pass
##        except Exception:
##            pass
##
##        # If we have a paused external file (subprocess fallback), restart playback in background
##        if self._paused_file:
##            fname = self._paused_file
##            print("Resuming paused external file:", fname)
##            self._paused_file = None
##            threading.Thread(target=self._play_file_controlled, args=(fname,), daemon=True).start()
##
##    def set_tk_stop_speech(self):
##        """Stop playback and clear queue."""
##        print("speech.set_tk_stop_speech() called")
##        self.tk_stop_speech = True
##        self.tk_pause_speech = False
##        self.tk_start_speech = False
##
##        # request stop and unpause so worker can react
##        self._stop_request.set()
##        self._pause_requested = False
##        self._pause_event.set()
##
##        # stop python-vlc player if present
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("Stopping python-vlc player")
##                self._player.stop()
##        except Exception:
##            pass
##
##        # stop external vlc process if present
##        try:
##            if getattr(self, "_vlc_process", None):
##                print("Terminating external VLC process")
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##
##        # stop pyttsx3 engine
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##
##        # drain the queue
##        try:
##            while True:
##                self.speak_queue.get_nowait()
##                self.speak_queue.task_done()
##        except queue.Empty:
##            pass
##
##        self._current_text = None
##        self._paused_file = None
##
##    def set_tk_pause_speech(self):
##        """Pause playback. For python-vlc: pause in-place. For external CLI: emulate pause by terminating process and remembering file."""
##        print("speech.set_tk_pause_speech() called")
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##
##        self._pause_requested = True
##        self._pause_event.clear()
##
##        # Try python-vlc pause in-place (do NOT set stop_request)
##        try:
##            if _HAS_LIBVLC and self._player:
##                try:
##                    # Only attempt pause if player reports playing
##                    is_playing = False
##                    try:
##                        is_playing = bool(self._player.is_playing())
##                    except Exception:
##                        # fallback to state check
##                        try:
##                            st = self._player.get_state()
##                            is_playing = (st == vlc.State.Playing)
##                        except Exception:
##                            is_playing = False
##                    if is_playing:
##                        print("Pausing python-vlc player (set_pause True)")
##                        # set_pause(1) reliably pauses; player.pause() toggles
##                        self._player.set_pause(True)
##                        # small wait loop for state change
##                        t0 = time.time()
##                        while time.time() - t0 < 1.5:
##                            try:
##                                st = self._player.get_state()
##                                if st == vlc.State.Paused:
##                                    break
##                            except Exception:
##                                pass
##                            time.sleep(0.03)
##                        print("python-vlc pause attempted; player state:", getattr(self._player, "get_state", lambda: None)())
##                        return
##                    else:
##                        print("python-vlc player not currently playing; cannot pause in-place")
##                except Exception as e:
##                    print("Error trying to pause python-vlc player:", e)
##        except Exception:
##            pass
##
##        # Not python-vlc or pause didn't succeed -> emulate pause by stopping and remembering file for resume
##        print("Emulating pause: setting stop_request and terminating external process if any")
##        self._stop_request.set()
##        try:
##            if getattr(self, "_vlc_process", None):
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                # keep _paused_file which was set by _play_file_controlled
##                print("External VLC process terminated; paused file preserved:", self._paused_file)
##        except Exception:
##            pass
##
##        # also stop pyttsx3 to avoid overlaps
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##
##    # ----------------- Public API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        if not text:
##            return
##        if self.wake_word_on_off:
##            self.speak_queue.put((text, voice, style))
##            return
##
##        # blocking path: generate file and play under control
##        try:
##            self._current_text = (text, voice, style)
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##            if fname:
##                self._play_file_controlled(fname)
##                self.Stop_Speech = True
##                self.Start_Speech = False
##        except Exception as e:
##            print("Error during online speech (blocking):", e)
##            try:
##                self._onboard_speak_blocking_with_stop(text)
##            except Exception as e2:
##                print("Error during onboard fallback:", e2)
##        finally:
##            self._current_text = None
##
##    def AlfredSpeak_Onboard(self, text):
##        engine = pyttsx3.init('sapi5')
##        engine.setProperty('rate', 190)
##        voices = engine.getProperty('voices')
##        engine.setProperty('voice', voices[0].id)
##        engine.setProperty('volume', 1)
##        engine.say(text)
##        engine.runAndWait()
##
##    def AlfredSpeak_Bluetooth(self, text):
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("Error sending bluetooth:", e)
##        self.AlfredSpeak(text)
##
##    def stop_current(self):
##        print("speech.stop_current() called")
##        self._stop_request.set()
##        try:
##            if _HAS_LIBVLC and self._player:
##                print("Stopping python-vlc player (stop_current)")
##                self._player.stop()
##        except Exception:
##            pass
##        try:
##            if getattr(self, "_vlc_process", None):
##                print("Terminating external VLC process (stop_current)")
##                try:
##                    self._vlc_process.terminate()
##                except Exception:
##                    try:
##                        self._vlc_process.kill()
##                    except Exception:
##                        pass
##                self._vlc_process = None
##        except Exception:
##            pass
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##
##    def callback(self, indata, frames, time_info, status):
##        if status:
##            print("sounddevice callback status:", status)
##        try:
##            self.audio_queue.put(bytes(indata))
##        except Exception as e:
##            print("Error putting audio callback data:", e)
##
##    # ----------------- Worker & helpers -----------------
##    def _speech_worker(self):
##        while True:
##            while not self.wake_word_on_off:
##                time.sleep(0.05)
##            self._pause_event.wait()
##            try:
##                text, voice, style = self.speak_queue.get()
##            except Exception:
##                continue
##
##            self._current_text = (text, voice, style)
##            self._stop_request.clear()
##            self.Start_Speech = True
##            self.Stop_Speech = False
##            self._currently_speaking = True
##
##            try:
##                if self._pause_requested:
##                    try:
##                        self.speak_queue.put(self._current_text)
##                    except Exception:
##                        pass
##                    self._pause_event.wait()
##
##                try:
##                    fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##                    if fname:
##                        self._play_file_controlled(fname)
##                except Exception as online_exc:
##                    print("Error during online Edge TTS:", online_exc)
##                    traceback.print_exc()
##                    try:
##                        self._onboard_speak_blocking_with_stop(text)
##                    except Exception as onboard_exc:
##                        print("Error onboard fallback:", onboard_exc)
##                        traceback.print_exc()
##
##                if self._stop_request.is_set():
##                    self._current_text = None
##
##            finally:
##                self._currently_speaking = False
##                self.Start_Speech = False
##                self.Stop_Speech = True
##                try:
##                    self.speak_queue.task_done()
##                except Exception:
##                    pass
##
##    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await communicate.save(fname)
##        return fname
##
##    def _play_file_controlled(self, fname):
##        if not os.path.exists(fname):
##            print("_play_file_controlled: file not found:", fname)
##            return
##
##        print("_play_file_controlled: starting playback for", fname, "stop_event=", self._stop_request.is_set())
##
##        try:
##            # python-vlc primary path
##            if _HAS_LIBVLC and self._vlc_instance:
##                print(" _play_file_controlled: using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait briefly for playing state
##                t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##                    if self._stop_request.is_set():
##                        break
##                    time.sleep(0.05)
##
##                # playback loop: honor pause/stop
##                while True:
##                    if self._stop_request.is_set():
##                        print(" _play_file_controlled: stop requested -> stopping python-vlc player")
##                        try: player.stop()
##                        except Exception: pass
##                        break
##
##                    # if pause requested -> set_pause(True) and wait until unpaused
##                    if not self._pause_event.is_set():
##                        try:
##                            print(" _play_file_controlled: pause requested -> set_pause(True)")
##                            player.set_pause(True)
##                        except Exception as e:
##                            print(" _play_file_controlled: set_pause(True) failed:", e)
##                        # block until resume or stop
##                        self._pause_event.wait()
##                        print(" _play_file_controlled: resume (pause_event set) -> set_pause(False)")
##                        try:
##                            player.set_pause(False)
##                        except Exception as e:
##                            print(" _play_file_controlled: set_pause(False) failed:", e)
##                            try:
##                                player.play()
##                            except Exception:
##                                pass
##
##                    # check normal end
##                    try:
##                        st = player.get_state()
##                        if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                            break
##                    except Exception:
##                        pass
##                    time.sleep(0.06)
##
##            # external VLC binary fallback
##            elif _HAS_VLC_BINARY:
##                print(f" _play_file_controlled: using external VLC binary ({_VLC_BINARY})")
##                try:
##                    args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##                    self._vlc_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##                    # keep paused-file for resume if we emulate pause by killing process
##                    self._paused_file = fname
##
##                    while True:
##                        if self._stop_request.is_set():
##                            try:
##                                self._vlc_process.terminate()
##                            except Exception:
##                                try:
##                                    self._vlc_process.kill()
##                                except Exception:
##                                    pass
##                            break
##                        if not self._pause_event.is_set():
##                            # emulate pause by terminating process and keeping file
##                            try:
##                                self._vlc_process.terminate()
##                            except Exception:
##                                try:
##                                    self._vlc_process.kill()
##                                except Exception:
##                                    pass
##                            break
##                        ret = self._vlc_process.poll()
##                        if ret is not None:
##                            break
##                        time.sleep(0.06)
##                finally:
##                    try:
##                        if getattr(self, "_vlc_process", None):
##                            self._vlc_process.wait(timeout=0.1)
##                    except Exception:
##                        pass
##                    self._vlc_process = None
##
##            # final fallback: playsound (limited control)
##            else:
##                print(" _play_file_controlled: no VLC available, falling back to playsound (limited)")
##                if self._stop_request.is_set():
##                    print(" _play_file_controlled: stop already requested, skipping playsound")
##                    return
##                try:
##                    playsound(fname)
##                except Exception as e:
##                    print("playsound failed:", e)
##
##        finally:
##            try:
##                # if this file is not stored as paused-file, remove it; otherwise resume will use it
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##        print("_play_file_controlled: playback finished/aborted")
##
##    def _onboard_speak_blocking_with_stop(self, text):
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##        if self._stop_request.is_set():
##            return
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed:", e)
##            traceback.print_exc()
##
##    # helpers
##    def is_speaking(self):
##        return self._currently_speaking
##
##    def queue_length(self):
##        return self.speak_queue.qsize()
##
### create module-level instance
##speech = SpeechModule()
##print("speech module initialized — done")



#####     SPEECH TREADING  LASTEST WORKING
### speech.py
##import os
##import time
##import queue
##import asyncio
##import tempfile
##import threading
##import traceback
##
##import pyttsx3
##from playsound import playsound
##import edge_tts
### sounddevice/vosk/serial kept if you use the callback or recognition elsewhere
##import sounddevice as sd
##import vosk
##
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech start")
##
##
##
##class SpeechModule:
##    def __init__(self):
##        """Initialize TTS, queues and worker thread."""
##        # onboard engine (init once)
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            # If available, use the first (or whichever) voice
##            if voices and len(voices) > 0:
##                self.engine.setProperty("voice", voices[0].id)
##        except Exception as e:
##            print("pyttsx3 init failed:", e)
##            self.engine = None
##
##        # wake-word control (start off)
##        self.wake_word_on_off = False
##
##        # queue for speech requests: items are tuples (text, voice, style)
##        self.speak_queue = queue.Queue()
##
##        # audio callback queue (for vosk/sounddevice) — keeps naming consistent
##        self.audio_queue = queue.Queue()
##
##        # control flags for compatibility with your existing code
##        self.Sending_On = False
##        self.Start_Speech = False
##        self.Stop_Speech = False
##
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # internal state
##        self._currently_speaking = False
##        self._stop_request = threading.Event()  # used to request stopping onboard engine
##
##        # start worker thread (it will sleep until wake_word_on_off is True)
##        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
##        self._worker_thread.start()
##
##    # ----------------- Wake Word -----------------
##    def set_wake_word_on(self, enabled: bool = True):
##        self.wake_word_on_off = True
##        print("🔔 Speech Wake Word On  🔔 Threading Speech Resumed")
##
##    def set_wake_word_off(self, enabled: bool = False):
##        self.wake_word_on_off = False
##        print("🔔 Speech Wake Word Off  🔔 Normal Speech Resumed")
##
##
##    # ---------------- Set Play, Stop, Pause --------------
##    def set_tk_start_speech(self)
##        self.tk_start_speech = False
##
##    def set_tk_stop_speech(self)
##        self.tk_stop_speech = False
##
##
##    def set_tk_pause_speech(self)
##        self.tk_pause_speech = False
##
##
##
##    # ----------------- Public API -----------------
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """
##        If wake-word is ON -> queue for threaded processing (non-blocking).
##        If wake-word is OFF -> run the blocking path (asyncio.run on speak_edge_tts)
##        (this preserves your original behavior when wake-word is off).
##        """
##        if not text:
##            return
##
##        if self.wake_word_on_off:
##            # Threaded, queue-based path (non-blocking)
##            self.speak_queue.put((text, voice, style))
##        else:
##            # Blocking path — same as your original: run async tts or fallback to onboard
##            try:
##                asyncio.run(self.speak_edge_tts(text, voice=voice, style=style))
##                self.Stop_Speech = True
##                self.Start_Speech = False
##            except Exception as e:
##                print("Error during online speech:", e)
##                try:
##                    self.AlfredSpeak_Onboard(text)
##                except Exception as e2:
##                    print("Error during onboard speech:", e2)
##
##
##    def AlfredSpeak_Onboard(self, text):
##        """Fallback TTS using onboard voice (blocking)."""
##        engine = pyttsx3.init('sapi5')
##        engine.setProperty('rate', 190)
##        voices = engine.getProperty('voices')
##        engine.setProperty('voice', voices[0].id)
##        engine.setProperty('volume', 1)
##        print('engine: ' + str(text), end="\r")
##        print('\033c', end='')
##        engine.say(text)
##        engine.runAndWait()
##
##    def AlfredSpeak_Bluetooth(self, text):
##        # send to arduino (use your arduino module)
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##            else:
##                print("arduino has no send_bluetooth attribute")
##        except Exception as e:
##            print("Error sending bluetooth:", e)
##
##        # queue / speak (follow wake-word behavior)
##        self.AlfredSpeak(text)
##
##    def stop_current(self):
##        """Request immediate stop of current speech (best-effort)."""
##        self._stop_request.set()
##        try:
##            if self.engine:
##                self.engine.stop()
##        except Exception:
##            pass
##
##    def callback(self, indata, frames, time_info, status):
##        """sounddevice callback — push raw audio to audio_queue for later processing."""
##        if status:
##            print(f"❌ Error: {status}")
##        try:
##            self.audio_queue.put(bytes(indata))
##        except Exception as e:
##            print("Error putting audio callback data:", e)
##
##    # ----------------- Worker & helpers -----------------
##    def _speech_worker(self):
##        """
##        Worker loop: sleeps while wake-word is OFF; when wake-word is ON, processes queued items.
##        Uses asyncio.run(self.speak_edge_tts(...)) for online TTS and falls back to onboard if needed.
##        """
##        while True:
##            # Wait until wake-word processing is enabled
##            while not self.wake_word_on_off:
##                time.sleep(0.05)
##
##            # Now wake-word is ON, process next queued item (blocks until available)
##            try:
##                text, voice, style = self.speak_queue.get()
##            except Exception:
##                continue
##
##            # reset stop request
##            self._stop_request.clear()
##            self.Start_Speech = True
##            self.Stop_Speech = False
##            self._currently_speaking = True
##
##            try:
##                try:
##                    # run your async TTS (blocking only this worker thread)
##                    asyncio.run(self.speak_edge_tts(text, voice=voice, style=style))
##                except Exception as online_exc:
##                    print("Error during online Edge TTS:", online_exc)
##                    traceback.print_exc()
##                    # fallback to onboard
##                    try:
##                        self._onboard_speak_blocking(text)
##                    except Exception as onboard_exc:
##                        print("Error during onboard TTS fallback:", onboard_exc)
##                        traceback.print_exc()
##            finally:
##                self._currently_speaking = False
##                self.Start_Speech = False
##                self.Stop_Speech = True
##                try:
##                    self.speak_queue.task_done()
##                except Exception:
##                    pass
##
##    async def speak_edge_tts(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        """Your async Edge TTS (kept simple like you had it)."""
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        await communicate.save("output.mp3")
##        playsound("output.mp3")
##
##    def _onboard_speak_blocking(self, text):
##        """Use the pre-initialized pyttsx3 engine in worker thread (blocking only worker)."""
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        if self._stop_request.is_set():
##            return
##
##        try:
##            print('engine: ' + str(text), end="\r")
##            print('\033c', end='')
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed:", e)
##            traceback.print_exc()
##
##    # ----------------- helpers / status -----------------
##    def is_speaking(self):
##        return self._currently_speaking
##
##    def queue_length(self):
##        return self.speak_queue.qsize()
##
##
### ✅ Initialize Speech Module
##speech = SpeechModule()
##
##print("speech end")
##
##
##





