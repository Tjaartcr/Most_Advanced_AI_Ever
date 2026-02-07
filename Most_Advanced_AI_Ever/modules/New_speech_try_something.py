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
        # pyttsx3 engine for onboard fallback
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
        self.wake_word_on_off = False  # flags for GUI control
        self.tk_start_speech = False
        self.tk_stop_speech = False
        self.tk_pause_speech = False

        # runtime control flags
        self._stop_request = False
        self._pause_requested = False
        self._paused_file = None

        # for external-cli fallback resume emulation
        self._player = None  # python-vlc player (if used)
        self._vlc_instance = None if _HAS_LIBVLC else None

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

        if self._paused_file:
            fname = self._paused_file = None
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

        if _HAS_LIBVLC and self._player:
            try:
                print("Stopping python-vlc player (sync)")
                self._player.stop()
            except Exception:
                pass

        # cannot forcibly stop playsound easily; external process is handled in _play_file_controlled
        # clear paused file
        self._paused_file = None

        # stop pyttsx3 if in use
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

        # Try to pause python-vlc player in-place
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
            except Exception as e:
                print(" _play_file_controlled (sync): pause requested -> set_pause(True)")

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
            self._paused_file = fname

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





##
###   LOOK GOOD
##
### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##import edge_tts
##
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH from arduino_com import arduino
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
##    if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##        print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
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
##        self.engine = None
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##        # flags for GUI control
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # runtime control flags
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None
##
##        # for external-cli fallback resume emulation
##        self._player = None
##        # python-vlc player (if used)
##        self._vlc_instance = None
##        if _HAS_LIBVLC:
##            try:
##                self._vlc_instance = vlc.Instance()
##            except Exception as e:
##                print("Failed to create vlc.Instance():", e)
##                self._vlc_instance = None
##
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF (synchronous mode)")
##
##    def set_tk_start_speech(self):
##        """Resume or start playback; clears stop/pause flags."""
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##        self._stop_request = False
##        self._pause_requested = False
##
##        # If a python-vlc player exists and is paused, attempt resume if _HAS_LIBVLC and self._player:
##        try:
##            print("Attempting to resume existing python-vlc player (sync)")
##            self._player.set_pause(False)
##            return
##        except Exception as e:
##            print("Resume via set_pause failed:", e)
##        try:
##            self._player.play()
##            return
##        except Exception:
##            pass
##
##        # If we have a paused file from external VLC, play it now (blocking)
##        if self._paused_file:
##            fname = self._paused_file = None
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
##                    is_playing = self._player.is_playing()
##                except Exception as e:
##                    print("Failed to check if player is playing:", e)
##                    return
##
##                if is_playing:
##                    print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
##                    self._player.set_pause(True)
##            else:
##                # emulate pause: terminate process and keep file path for resume
##                try:
##                    pid = os.getpid()
##                    args = [sys.executable, __file__]
##                    kwargs = {'args': sys.argv[1:], 'wdir': os.getcwd()}
##                    proc = subprocess.Popen(args, **kwargs)
##
##                    self._paused_file = fname
##
##                    while True:
##                        if self._stop_request:
##                            try:
##                                proc.terminate()
##                            except Exception:
##                                try:
##                                    proc.kill()
##                                except Exception:
##                                    pass
##                            break
##
##                        if self._pause_requested:
##                            # pause in-place using set_pause(True)
##                            try:
##                                print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
##                                self._player.set_pause(True)
##                            except Exception:
##                                pass
##                            break
##
##                        time.sleep(0.05)
##
##                    # check if finished
##                    if proc.poll() is not None:
##                        break
##
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##
##                self._vlc_process = None
##        except Exception as e:
##            print("Failed to pause/stop python-vlc player:", e)
##            return
##
##    def set_tk_resume_speech(self):
##        """Resume playback: if paused, start from paused file."""
##        print("speech.set_tk_resume_speech() called (sync)")
##        self.tk_pause_speech = False
##        self._player = None
##
##        try:
##            fname = self._paused_file
##            if not os.path.exists(fname):
##                print("_play_file_controlled: file not found:", fname)
##                return
##
##            print("_play_file_controlled (sync): starting playback for", fname, "stop_request=", self._stop_request)
##
##            # python-vlc primary path if _HAS_LIBVLC and self._vlc_instance:
##            try:
##                print(" _play_file_controlled (sync): using python-vlc (libvlc)")
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                # wait up to a short timeout for actual play to begin t0 = time.time()
##                while time.time() - t0 < 3.0:
##                    try:
##                        if player.is_playing():
##                            break
##                    except Exception:
##                        pass
##
##                if self._stop_request:
##                    break
##
##                time.sleep(0.05)
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
##                            self._player.set_pause(True)
##                        except Exception:
##                            pass
##                        break
##
##                    time.sleep(0.05)
##
##                    st = player.get_state()
##                    if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                        break
##
##                    time.sleep(0.06)
##
##                try:
##                    player.stop()
##                except Exception:
##                    pass
##                self._player = None
##
##            except Exception as e:
##                print("Failed to start python-vlc player:", e)
##                return
##
##        # external vlc binary fallback (runs subprocess synchronously but poll-loop)
##        elif _HAS_VLC_BINARY:
##            print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
##            args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##
##            self._paused_file = fname
##
##            while True:
##                if self._stop_request:
##                    try:
##                        proc.terminate()
##                    except Exception:
##                        try:
##                            proc.kill()
##                        except Exception:
##                            pass
##                        break
##
##                    if self._pause_requested:
##                        # emulate pause: terminate process and keep file path for resume
##                        try:
##                            pid = os.getpid()
##                            args = [sys.executable, __file__]
##                            kwargs = {'args': sys.argv[1:], 'wdir': os.getcwd()}
##                            proc = subprocess.Popen(args, **kwargs)
##
##                            self._paused_file = fname
##
##                            while True:
##                                if self._stop_request:
##                                    try:
##                                        proc.terminate()
##                                    except Exception:
##                                        try:
##                                            proc.kill()
##                                        except Exception:
##                                            pass
##                                        break
##
##                        except Exception as e:
##                            print("Failed to pause/stop python-vlc player:", e)
##                            return
##
##                    # check if finished
##                    if proc.poll() is not None:
##                        break
##
##                    time.sleep(0.06)
##
##                try:
##                    _ = proc.wait(timeout=0.1)
##                except Exception:
##                    pass
##
##                self._vlc_process = None
##
##        # last fallback: blocking playsound
##        else:
##            print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
##            if self._stop_request:
##                print(" _play_file_controlled (sync): stop already requested; skipping playsound")
##                return
##
##            try:
##                playsound(fname)
##            except Exception as e:
##                print("playsound failed (sync):", e)
##
##        finally:
##            # cleanup: remove file if not used for paused resume
##            try:
##                if fname and (self._paused_file != fname):
##                    os.remove(fname)
##            except Exception:
##                pass
##
##    def init(self):
##        print("speech (synchronous) initialized — done")




##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##import edge_tts
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH from arduino_com import arduino
##
##print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### ----------------- try python-vlc / libvlc -----------------
##_HAS_PYTHON_VLC = False
##_HAS_LIBVLC = False
##vlc = None
##try:
##    import vlc as _vlc
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
##    if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##        print("WARNING: No usable VLC backend detected. Falling back to playsound (limited pause/stop).")
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
##        self.engine = None
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##        # flags for GUI control
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##
##        # runtime control flags
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None
##
##        # for external-cli fallback resume emulation
##        self._player = None
##        # python-vlc player (if used)
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
##        # If a python-vlc player exists and is paused, attempt resume if _HAS_LIBVLC and self._player:
##            try:
##                print("Attempting to resume existing python-vlc player (sync)")
##                self._player.set_pause(False)
##                return
##            except Exception as e:
##                print("Resume via set_pause failed:", e)
##            try:
##                self._player.play()
##                return
##            except Exception:
##                pass
##
##        # If we have a paused file from external VLC, play it now (blocking)
##        if self._paused_file:
##            fname = self._paused_file = None
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
##                        is_playing = player.is_playing()
##                    except Exception:
##                        pass
##                    if not is_playing:
##                        break
##                except Exception:
##                    pass
##                if self._stop_request:
##                    break
##                time.sleep(0.05)
##
##                # when resumed
##                player.set_pause(True)
##            else:
##                # emulate pause: terminate process and keep file path for resume
##                try:
##                    proc.terminate()
##                except Exception:
##                    try:
##                        proc.kill()
##                    except Exception:
##                        pass
##                break
##
##                if self._stop_request:
##                    break
##                time.sleep(0.05)
##
##                if proc.poll() is not None:
##                    break
##                time.sleep(0.06)
##
##            player.stop()
##            self._player = None
##
##        # external vlc binary fallback (runs subprocess synchronously but poll-loop)
##        elif _HAS_VLC_BINARY:
##            print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
##            args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##            self._paused_file = fname
##
##        # last fallback: blocking playsound
##        else:
##            print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
##            if self._stop_request:
##                print(" _play_file_controlled (sync): stop already requested; skipping playsound")
##                return
##
##            try:
##                playsound(fname)
##            except Exception as e:
##                print("playsound failed (sync):", e)
##
##            finally:
##                # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file)
##                try:
##                    if fname and (self._paused_file != fname):
##                        os.remove(fname)
##                except Exception:
##                    pass
##                print("_play_file_controlled (sync): playback finished/aborted")
##
##        speech = SpeechModule()
##        print("speech (synchronous) initialized — done")






##
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
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
##except Exception as e:
##    _HAS_LIBVLC = False
##    print("python-vlc imported but libvlc.Instance() failed:", e)
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
### ----------------- SpeechModule (no threads) -----------------
##class SpeechModule:
##    def __init__(self):
##        # pyttsx3 engine for onboard fallback try: self.engine = pyttsx3.init() self.engine.setProperty("rate", 190) voices = self.engine.getProperty("voices") if voices: self.engine.setProperty("voice", voices[0].id) except Exception as e: print("pyttsx3 init failed:", e) self.engine = None # control flags (simple booleans, not threading events) self.wake_word_on_off = False # flags for GUI control self.tk_start_speech = False self.tk_stop_speech = False self.tk_pause_speech = False # runtime control flags self._stop_request = False self._pause_requested = False self._paused_file = None # for external-cli fallback resume emulation self._player = None # python-vlc player (if used) self._vlc_instance = None if _HAS_LIBVLC: try: self._vlc_instance = vlc.Instance() except Exception as e: print("Failed to create vlc.Instance():", e) self._vlc_instance = None
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
##        # If a python-vlc player exists and is paused, attempt resume if _HAS_LIBVLC and self._player:
##        try:
##            print("Attempting to resume existing python-vlc player (sync)")
##            self._player.set_pause(False)
##            return
##        except Exception as e:
##            print("Resume via set_pause failed:", e)
##
##        try:
##            self._player.play()
##            return
##        except Exception:
##            pass
##
##        # If we have a paused file from external VLC, play it now (blocking) if self._paused_file:
##            fname = self._paused_file = None
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
##        # stop python-vlc player if present try:
##        if _HAS_LIBVLC and self._player:
##            print("Stopping python-vlc player (sync)")
##            self._player.stop()
##
##        # cannot forcibly stop playsound easily; external process is handled in _play_file_controlled
##        # clear paused file self._paused_file = None
##
##        # stop pyttsx3 if in use try:
##        if self.engine:
##            self.engine.stop()
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
##        # Try to pause python-vlc player in-place try:
##        if _HAS_LIBVLC and self._player:
##            try:
##                is_playing = False
##                try:
##                    is_playing = bool(self._player.is_playing())
##                except Exception:
##                    try:
##                        st = self._player.get_state()
##                        is_playing = (st == vlc.State.Playing)
##                    except Exception:
##                        is_playing = False
##                if is_playing:
##                    print("Pausing python-vlc player (sync set_pause True)")
##                    self._player.set_pause(True)
##                    return
##                else:
##                    print("python-vlc player not playing; cannot pause in-place (sync)")
##            except Exception as e:
##                print("Error while pausing python-vlc player (sync):", e)
##
##        except Exception:
##            pass
##
##        # Emulate pause for external-vlc: set stop request
##        elif _HAS_VLC_BINARY:
##            args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", self._paused_file]
##            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##            self._paused_file = None
##
##            # keep for resume if pause emulation used
##            while True:
##                if self._stop_request:
##                    try:
##                        proc.terminate()
##                    except Exception:
##                        try:
##                            proc.kill()
##                        except Exception:
##                            pass
##                    break
##
##                if self._pause_requested:
##                    # emulate pause: terminate process and keep file path for resume
##                    try:
##                        proc.terminate()
##                    except Exception:
##                        try:
##                            proc.kill()
##                        except Exception:
##                            pass
##                    break
##
##                # check if finished if proc.poll() is not None:
##                break
##            time.sleep(0.06)
##
##        try:
##            _ = proc.wait(timeout=0.1)
##        except Exception:
##            pass
##
##        self._vlc_process = None
##
##        # last fallback: blocking playsound else:
##        print("speech (synchronous) initialized — done")






##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##import edge_tts
### keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##
### keep these imports to preserve API surface
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
##            break
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
##        self.engine = None
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##        # flags for GUI control
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##        # runtime control flags
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None
##        # for external-cli fallback resume emulation
##        self._player = None  # python-vlc player (if used)
##        self._vlc_instance = None if _HAS_LIBVLC else None
##
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
##        # If a python-vlc player exists and is paused, attempt resume if _HAS_LIBVLC and self._player:
##        try:
##            print("Attempting to resume existing python-vlc player (sync)")
##            self._player.set_pause(False)
##        except Exception as e:
##            print("Resume via set_pause failed:", e)
##        try:
##            self._player.play()
##        except Exception:
##            pass
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
##                except Exception as e:
##                    print("Error while pausing python-vlc player:", e)
##        except Exception as e:
##            try:
##                self._player.stop()
##            except Exception:
##                pass
##        break
##
##        # busy-wait until resume or stop
##        while self._pause_requested and (not self._stop_request):
##            time.sleep(0.05)
##
##        # when resumed
##        try:
##            self._player.set_pause(False)
##        except Exception:
##            try:
##                self._player.play()
##            except Exception:
##                pass
##
##        # check normal end
##        while True:
##            if self._stop_request:
##                print(" _play_file_controlled (sync): stop requested -> stopping python-vlc player")
##                try:
##                    self._player.stop()
##                except Exception:
##                    pass
##                break
##            if self._pause_requested:
##                # pause in-place using set_pause(True)
##                try:
##                    print(" _play_file_controlled (sync): pause requested -> set_pause(True)")
##                    self._player.set_pause(True)
##                except Exception as e:
##                    print("Error while pausing python-vlc player:", e)
##            # busy-wait until resume or stop
##            while self._pause_requested and (not self._stop_request):
##                time.sleep(0.05)
##            # when resumed
##            try:
##                self._player.set_pause(False)
##            except Exception:
##                try:
##                    self._player.play()
##                except Exception:
##                    pass
##            # check normal end
##            try:
##                st = self._player.get_state()
##                if st in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error):
##                    break
##            except Exception:
##                pass
##            time.sleep(0.06)
##        try:
##            player.stop()
##        except Exception:
##            pass
##        self._player = None
##
##        # external vlc binary fallback (runs subprocess synchronously but poll-loop)
##        elif _HAS_VLC_BINARY:
##            print(f" _play_file_controlled (sync): using external VLC binary ({_VLC_BINARY})")
##            args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", fname]
##            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##            self._paused_file = fname
##
##        # keep for resume if pause emulation used
##        while True:
##            if self._stop_request:
##                try:
##                    proc.terminate()
##                except Exception:
##                    try:
##                        proc.kill()
##                    except Exception:
##                        pass
##                break
##            if self._pause_requested:
##                # emulate pause: terminate process and keep file path for resume
##                try:
##                    proc.terminate()
##                except Exception:
##                    try:
##                        proc.kill()
##                    except Exception:
##                        pass
##                break
##
##        # check if finished
##        if proc.poll() is not None:
##            break
##        time.sleep(0.06)
##
##        try:
##            _ = proc.wait(timeout=0.1)
##        except Exception:
##            pass
##        self._vlc_process = None
##
##        # last fallback: blocking playsound else: playsound fails (limited pause/stop)
##        else:
##            print(" _play_file_controlled (sync): no VLC available, falling back to playsound (limited)")
##            if self._stop_request:
##                print(" _play_file_controlled (sync): stop already requested; skipping playsound")
##                return try: playsound(fname) except Exception as e: print("playsound failed (sync):", e) finally: # cleanup: remove file if not used for paused resume (external fallback keeps _paused_file) try: if fname and (self._paused_file != fname): os.remove(fname) except Exception: pass print("_play_file_controlled (sync): playback finished/aborted")
##
##    def initialize(self):
##        print("speech (synchronous) initialized — done")










### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3 from playsound import playsound
##from vosk import Vosk  # Assuming vosk is installed for speech recognition if needed.
##
### Keep imports to preserve API surface (no additional changes)
##import edge_tts
##from vlc import Instance, media_player_new
##
##def setup_vlc_instance():
##    """Create and return a new vlc instance."""
##    try:
##        vlc_instance = Instance()
##        _HAS_LIBVLC = True
##        print("python-vlc initialized with instance.")
##        return vlc_instance
##    except Exception as e:
##        print("Failed to create vlc.Instance():", e)
##        raise RuntimeError("VLC initialization failed") from e
##
##class SpeechModule:
##
##    def __init__(self, wake_word_on_off=True):
##        self.wake_word_on_off = wake_word_on_off
##
##        # Initialize engine and voices if needed.
##        try:
##            self.engine = pyttsx3.init()
##            self.engine.setProperty("rate", 190)
##            voices = self.engine.getProperty("voices")
##            if voices:
##                self.engine.setProperty("voice", voices[0].id)
##                print("Using voice:", voices[0].id)
##
##            # Other initializations...
##
##        except Exception as e:
##            print(f"pyttsx3 initialization failed: {e}")
##            raise RuntimeError("PyTTSX3 engine not initialized") from e
##
##    def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        """Blocking TTS generation and saving audio to disk."""
##        ssml = f"{text}"
##
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##
##        # Save generated audio file.
##        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
##            temp_file.write(communicate.output_data)
##            fname = temp_file.name
##
##        return fname
##
##    async def play_audio(self, filepath):
##      """Block until audio is played."""
##      if not filepath or not os.path.exists(filepath):  # Skip if file doesn't exist.
##          print("Audio file not found.")
##          return
##
##      try:
##          playsound(filepath)
##      except Exception as e:
##          print(f"Error playing sound: {e}")
##
##    def AlfredSpeak(self, text):
##      """Blocking call for synchronous speech generation."""
##      if self.wake_word_on_off is True:
##          raise RuntimeError("Wake word cannot be ON in sync mode.")
##
##      # Generate and play the audio.
##      try:
##          fname = await asyncio.get_event_loop().run_in_executor(
##              None,
##              self._speak_edge_tts_save_only,
##              text,
##              style="cheerful"
##          )
##
##          if not os.path.exists(fname):
##              raise FileNotFoundError(f"Audio file '{fname}' does not exist.")
##
##          print("Starting to play audio...")
##          await asyncio.get_event_loop().run_in_executor(None, self.play_audio, fname)
##
##      except Exception as e:
##          traceback.print_exc()
##
### Example usage:
##speech_module = SpeechModule(wake_word_on_off=False)
##print(speech_module.AlfredSpeak("Hello! How can I assist you today?"))



##python
### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##import edge_tts  # keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
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
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    for candidate in ("cvlc", "vlc"):
##        p = shutil.which(candidate)
##        if p:
##            _HAS_VLC_BINARY = True
##            _VLC_BINARY = candidate
##            print("Found external VLC binary:", candidate, "->", p)
##            break
##
##if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##    print(
##        "WARNING: No VLC media player found. Please install python-vlc or have the external VLC media player installed."
##    )
##
##class SpeechModule:
##    def __init__(self):
##        self._pause_requested = False
##        self._playback_thread = None
##
##    # ... (previous methods)
##
##    def set_tk_play_speech(self):
##        """Start playing the speech with a new thread."""
##        self._playback_thread = Thread(target=self._play_file, args=(self._current_file,))
##        self._playback_thread.start()
##
##    def _play_file(self, file):
##        """Play the given file using VLC or Playsound."""
##        if _HAS_LIBVLC and self._vlc_instance:
##            player = self._vlc_instance.media_player_new()
##            media = self._vlc_instance.media_new(str(file))
##            player.set_media(media)
##            player.play()
##            while player.get_state() != vlc.State.Ended and player.get_state() != vlc.State.Stopped:
##                time.sleep(0.1)
##            self._vlc_instance.media_player_set_MediaPlayer(player, None)
##        elif _HAS_VLC_BINARY or _HAS_PYTHON_VLC:
##            try:
##                subprocess.Popen([_VLC_BINARY, "--intf", "dummy", "--play-and-exit", file])
##            except Exception as e:
##                print("Error playing the file using VLC:", e)
##                playsound(file)
##        else:
##            playsound(file)
##
##    def _onboard_speak_blocking_with_stop(self, text):
##        """Blocking onboard TTS (pyttsx3)."""
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##        if self._stop_request or self._pause_requested:
##            return
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed:", e)
##            self._onboard_speak_blocking_with_stop.pause()
##
### ... (previous methods)
##
##speech = SpeechModule()
##print("speech (synchronous) initialized — done")



##
### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##import tkinter as tk
##from playsound import playsound
##import edge_tts  # keep these imports to preserve API surface
##import sounddevice as sd
##import vosk
##from Alfred_config import VOSK_MODEL_PATH, SERIAL_PORT_BLUETOOTH, BAUDRATE_BLUETOOTH
##from arduino_com import arduino
##
##print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
##try:
##    # python-vlc / libvlc -----------------
##    _HAS_PYTHON_VLC = False
##    _HAS_LIBVLC = False
##    vlc = None
##
##    try:
##        import vlc as _vlc
##        vlc = _vlc
##        _HAS_PYTHON_VLC = True
##        try:
##            inst = vlc.Instance()
##            _HAS_LIBVLC = True
##            del inst
##        except Exception as e:
##            _HAS_LIBVLC = False
##            print("python-vlc imported but libvlc.Instance() failed:", e)
##    except Exception as e:
##        print("python-vlc import failed:", e)
##
##    # fallback: check external vlc binary --------------
##    _HAS_VLC_BINARY = False
##    _VLC_BINARY = None
##
##    if not (_HAS_LIBVLC or _HAS_VLC_BINARY):
##        for candidate in ("cvlc", "vlc"):
##            p = shutil.which(candidate)
##            if p:
##                _HAS_VLC_BINARY = True
##                _VLC_BINARY = candidate
##                print("found external vlc binary:", _VLC_BINARY)
##                break
##
##except Exception as e:
##    print(f"Initialization failed with error: {e}")
##
##class SpeechButton(tk.Frame):
##    def __init__(self, master, text, command):
##        super().__init__(master)
##        self.config(relief='flat', width=3, height=1)
##        self.label = tk.Label(self, text=text, relief="flat", borderwidth=0)
##        self.label.pack_propagate(False)
##        self.label.pack()
##        self.command = command
##        self.bind("<Button-1>", self.handle_click)
##
##    def handle_click(self, event):
##        self.command()
##
##class SpeechModule:
##    def __init__(self):
##        self.master = tk.Tk()
##        self.master.title("Speech Module")
##        self.frame = tk.Frame(self.master)
##        self.frame.pack(side="left", fill="both", expand=True)
##
##        self.play_button = SpeechButton(self.frame, "Play", self.play)
##        self.play_button.pack(side="top")
##        self.pause_button = SpeechButton(self.frame, "Pause", self.pause)
##        self.pause_button.pack(side="top")
##        self.stop_button = SpeechButton(self.frame, "Stop", self.stop)
##        self.stop_button.pack(side="top")
##
##        self.text = ""
##
##        self.engine = pyttsx3.init()
##
##        self.master.mainloop()
##
##    def play(self):
##        self.pause()
##        self.engine.say(self.text)
##        self.engine.runAndWait()
##
##    def pause(self):
##        if self.engine.isSpeaking():
##            self.engine.pause()
##
##    def stop(self):
##        self.engine.stop()
##        self.text = ""
##
##speech = SpeechModule()
##print("speech (synchronous) initialized — done")





##
##
##python
##import os
##import sys from playsound
##import playsound
##Ensure this module or library exists and is correctly imported
##class SpeechModule:
##    def __init__(self):
##        self.engine = pyttsx3.init()
##        self.engine.setProperty('rate', 190)
##        voices = self.engine.getProperty('voices')
##        if voices: self.engine.setProperty('voice', voices[0].id)
##        Keep these imports to preserve API surface
##        import sounddevice as sd
##        from vosk import Model, KaldiRecognizer
##        model_path = "path_to_your_model"
##        Ensure this path is correct or define it properly
##        if not os.path.exists(model_path):
##            raise FileNotFoundError(f"Vosk model not found at {model_path}")
##        def set_wake_word_on(self, enabled=True):
##            print("Wake word ON (synchronous mode)")
##            def set_wake_word_off(self, enabled=False):
##                print("Wake word OFF (synchronous mode)")
##                def _play_file_controlled(self, fname):
##                    if not os.path.exists(fname):
##                        raise FileNotFoundError(f"File not found: {fname}")
##                    t0 = time.time()
##                    while True:
##                        if self._stop_request or time.time() - t0 > 3.0:
##                            break
##                        try:
##                            if self.engine.is_playing():
##                            break
##                        except Exception as e:
##                            pass time.sleep(0.1)
##                        print(f" _play_file_controlled (sync): playback finished/aborted")
##                        def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##                            if not text:
##                                return self.set_wake_word_on()
##                            try:
##                                fname = self._speak_edge_tts_save_only(text, voice=voice, style=style)
##                                if not os.path.exists(fname):
##                                    raise FileNotFoundError(f"File not saved: {fname}")
##                                self.engine.stop()
##                                Stop the previous playback playsound(fname)  Use playsound for playing and controlling audio
##                                t0 = time.time()
##                                while True:
##                                    if self._stop_request or time.time() - t0 > 3.0:
##                                        break
##                                    try:
##                                        if not self.engine.is_playing():
##                                            break
##                                        except Exception as e:
##                                            pass time.sleep(0.1)
##                                        finally:  Clean up by stopping the engine and removing file (if no _stop_request)
##                                        self.set_wake_word_off()
##                                        self.engine.stop()
##                                        def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##                                            ssml = f"{text}"
##                                            with open("speech_text.txt", "w") as f:
##                                                f.write(ssml)
##                                                os.system(f"edge-tts -negative voice {voice} -negative style {style} speech_text.txt negative o speech_output.mp3")
##                                                return "speech_output.mp3"
##                                            def AlfredSpeak_Onboard(self, text):
##                                                self.engine = pyttsx3.init("sapi5")
##                                                self.engine.setProperty('rate', 190)
##                                                voices = self.engine.getProperty('voices')
##                                                if voices: self.engine.setProperty('voice', voices[0].id)
##                                                Use playsound for playback playsound(text)
##                                                def AlfredSpeak_Bluetooth(self, text):
##                                                    arduino.send_bluetooth(text)
##                                                    Ensure this function exists
##                                                    speech_module = SpeechModule()
##                                                    print("speech (synchronous) initialized — done")
##
##

##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##from edge_tts import Communicate, Message
##
##print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
### -------------- try python-vlc / libvlc -----------------
##_has_python_vlc = False
##_has_libvlc = False
##_vlc_instance = None
##_player = None
##
##try:
##    import vlc as _vlc
##    _has_python_vlc = True
##except ImportError:
##    print("python-vlc import failed")
##
##if _has_python_vlc and _vlc.Instance().playback_enabled():
##    _libvcl = _vlc.Instance()
##    _player = _libvcl.media_player_new()
##
##if _has_python_vlc:
##    _has_libvlc = True
##else:
##    _has_libvlc = False
##
##print("python-vlc and libvlc available: " + str(_has_python_vlc), str(_has_libvlc))
##
### ----------------- SpeechModule (no threads) -----------------
##class SpeechModule:
##    def __init__(self):
##        self.engine = pyttsx3.init()
##        self.engine.setProperty('rate', 190)
##        voices = self.engine.getProperty('voices')
##        if voices:
##            self.engine.setProperty('voice', voices[0].id)
##
##        # control flags (simple booleans, not threading events)
##        self.wake_word_on_off = False
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None
##
##    def set_wake_word_on(self, enabled=True):
##        print("Wake word ON (synchronous mode)")
##        self.wake_word_on_off = True
##
##    def set_wake_word_off(self, enabled=False):
##        print("Wake word OFF (synchronous mode)")
##        self.wake_word_on_off = False
##
##    def set_tk_start_speech(self):
##        if _has_libvlc and _player:
##            try:
##                print("Attempting to resume existing python-vlc player (sync)")
##                _player.set_pause(False)
##                _player.play()
##            except Exception as e:
##                print("Resume via set_pause failed:", e)
##
##    def set_tk_stop_speech(self):
##        self.tk_stop_speech = True
##        self._stop_request = True
##
##    def stop_current(self):
##        if _has_libvlc and _player:
##            try:
##                print("Stopping python-vlc player (sync)")
##                _player.stop()
##            except Exception as e:
##                print("Error stopping python-vlc player (sync):", e)
##
##    def set_tk_pause_speech(self):
##        self.tk_pause_speech = True
##        self._pause_requested = True
##
##    def stop_current(self):
##        if _has_libvlc and _player:
##            try:
##                print("Stopping python-vlc player (sync)")
##                _player.stop()
##            except Exception as e:
##                print("Error stopping python-vlc player (sync):", e)
##
##    async def _speak_edge_tts_save_only(self, text):
##        ssml = f"<speak><prosody rate='190'>{text}</prosody></speak>"
##        msg = Message(text=ssml)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await Communicate(message=msg).save(fname=fname)
##
##    def AlfredSpeak(self, text):
##        if not text:
##            return
##        try:
##            if self._speak_edge_tts_save_only(text):
##                self.set_tk_start_speech()
##        except Exception as e:
##            print("Error generating TTS via edge-tts (sync):", e)
##            traceback.print_exc()
##
##    def AlfredSpeak_Onboard(self, text):
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed (sync):", e)
##            traceback.print_exc()
##
##    def AlfredSpeak_Bluetooth(self, text):
##        if hasattr(arduino, "send_bluetooth"):
##            arduino.send_bluetooth(text)
##
##print("speech.py (synchronous) initialized — done")



### speech.py -- synchronous, thread-free TTS playback with pause/stop support
##
##import os
##import sys
##import time
##import asyncio
##import tempfile
##import traceback
##import subprocess
##import shutil
##import pyttsx3
##from playsound import playsound
##import edge_tts  # make sure this is imported correctly if using external TTS engine
##
### keep these imports to preserve API surface
##print("speech.py (synchronous) loading — PID:", os.getpid(), "Python:", sys.version.splitlines()[0])
##
##class SpeechModule:
##    def __init__(self):
##        self.engine = None  # pyttsx3 engine for onboard fallback
##        self.wake_word_on_off = False
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self.tk_pause_speech = False
##        self._stop_request = False
##        self._pause_requested = False
##        self._paused_file = None
##
##    def set_wake_word_on(self, enabled=True):
##        self.wake_word_on_off = True
##        print("Wake word ON (synchronous mode)")
##
##    def set_wake_word_off(self, enabled=False):
##        self.wake_word_on_off = False
##        print("Wake word OFF (synchronous mode)")
##
##    def set_tk_start_speech(self):
##        """Resume or start playback; clears stop/pause flags."""
##        print("speech.set_tk_start_speech() called (sync)")
##        self.tk_start_speech = True
##        self.tk_pause_speech = False
##        self.tk_stop_speech = False
##        self._stop_request = False
##        self._pause_requested = False
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
##    def set_tk_pause_speech(self):
##        """Pause playback: if player exists, pause; otherwise emulate by stopping process and remembering file."""
##        print("speech.set_tk_pause_speech() called (sync)")
##        self.tk_pause_speech = True
##        self.tk_start_speech = False
##        self.tk_stop_speech = False
##        self._pause_requested = True
##
##    def stop_current(self):
##        """Convenience immediate stop (same as set_tk_stop_speech)."""
##        print("speech.stop_current() called (sync)")
##        self.set_tk_stop_speech()
##
##    def AlfredSpeak(self, text, voice="en-US-GuyNeural", style="cheerful"):
##        """Blocking / synchronous: generate audio (edge-tts) then play it under control."""
##        if not text:
##            return
##        try:
##            fname = asyncio.run(self._speak_edge_tts_save_only(text, voice=voice, style=style))
##        except Exception as e:
##            print("Error generating TTS via edge-tts (sync):", e)
##            return
##
##        self._play_file_controlled(fname)
##
##    def AlfredSpeak_Onboard(self, text):
##        """Blocking onboard TTS (pyttsx3)."""
##        if not self.engine:
##            raise RuntimeError("pyttsx3 engine not initialized")
##
##        try:
##            self.engine.say(text)
##            self.engine.runAndWait()
##        except Exception as e:
##            print("pyttsx3 speak failed (sync):", e)
##            traceback.print_exc()
##
##    def AlfredSpeak_Bluetooth(self, text):
##        try:
##            if hasattr(arduino, "send_bluetooth"):
##                arduino.send_bluetooth(text)
##        except Exception as e:
##            print("Error sending bluetooth:", e)
##
##    async def _speak_edge_tts_save_only(self, text, voice="af-ZA-WillemNeural", style="cheerful"):
##        ssml = f"{text}"
##        communicate = edge_tts.Communicate(text=ssml, voice=voice)
##        fname = os.path.join(tempfile.gettempdir(), f"alfred_edge_tts_output_{int(time.time()*1000)}.mp3")
##        await communicate.save(fname)
##
##    def _play_file_controlled(self, fname):
##        """Blocking playback that honors self._stop_request and self._pause_requested."""
##        if not os.path.exists(fname):
##            print(f"_play_file_controlled: file not found: {fname}")
##            return
##
##        try:
##            player = None
##            media = None
##            if _HAS_LIBVLC and self.engine:
##                player = self._vlc_instance.media_player_new()
##                media = self._vlc_instance.media_new(str(fname))
##                player.set_media(media)
##                self._player = player
##                player.play()
##
##                t0 = time.time()
##                while True:
##                    if player.is_playing():
##                        break
##                    if self._stop_request or (time.time() - t0 > 3.0):
##                        break
##                    time.sleep(0.1)
##
##            elif _HAS_VLC_BINARY and fname:
##                print(f"_play_file_controlled: using external VLC binary ({_VLC_BINARY})")
##                args = [_VLC_BINARY, "--intf", "dummy", "--play-and-exit", str(fname)]
##                proc = subprocess.Popen(args)
##                self._paused_file = fname
##                try:
##                    while True:
##                        if self._stop_request or (proc.poll() is not None):
##                            break
##                        time.sleep(0.1)
##                    # Cleanup: remove file if used for paused resume
##                    if self._paused_file == fname and os.path.exists(fname):
##                        os.remove(fname)
##                except Exception as e:
##                    traceback.print_exc()
##            else:
##                print("_play_file_controlled: no VLC available, falling back to playsound (limited)")
##                try:
##                    playsound(str(fname))
##                except Exception as e:
##                    print("playsound failed (sync):", e)
##
##        finally:
##            # Cleanup: remove file if used for paused resume
##            if self._paused_file == fname and os.path.exists(fname):
##                os.remove(fname)
##
##
### Create module-level instance
##speech = SpeechModule()
##print("speech (synchronous) initialized — done")
