



import cv2
from ultralytics import YOLO
import Alfred_config
import datetime
import threading
import ollama
import time
import queue
import re

from memory import memory
from speech import speech
from listen import listen
from Repeat_Last import repeat
from GUI import gui


import re
from typing import Tuple, Optional
    
class AI_VisionModule:
    def __init__(self, gui):

        if not gui:
            raise ValueError("GUI instance must be provided to AI_VisionModule!")

        self.gui = gui
        self.response_queue = queue.Queue()

##        try:
##            self.camera = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##            self.camera.set(3, 640)
##            self.camera.set(4, 480)
##        except:
##            self.camera = cv2.VideoCapture(Alfred_config.LEFT_CAMERA_INPUT)
##            self.camera.set(3, 640)
##            self.camera.set(4, 480)            


        if isinstance(self.gui.vision_model, str):
            raise TypeError("Expected StringVar for vision_model, got str!")

        self.AI_vision_model = self.gui.vision_model.get()
        print(f"✅ AI_Vision_Model: {self.AI_vision_model}")



    import re
    from typing import Optional, Tuple

    def AI_vision_extract_text_from_query(
        self, AlfredQueryOffline
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], str, Optional[str]]:
        """
        Return: (message, score, gender, gender_conf, username, timestamp)

        - message: inner text of the FIRST triple-single-quote fence (''' ... ''') if present,
                   otherwise a best-effort message string.
        - score, gender, gender_conf: parsed label values (string) or None
        - username: parsed username or fallback self.current_user / 'ITF'
        - timestamp: "YYYY-MM-DD HH:MM:SS" or None

        Works with both dict and string inputs. Handles patterns like:
          "'message':'''... : 'score':None : 'gender':None : 'gender_conf':None''' : 'username':ITF : 'timestamp':2025-10-10 13:24:19"
        """
        fallback_user = getattr(self, "current_user", "ITF") or "ITF"

        def _strip_triple_single_if_present(s: str) -> str:
            if not isinstance(s, str):
                return s
            s2 = s.strip()
            m = re.match(r"^('{3})(?P<body>.*)(\1)$", s2, flags=re.DOTALL)
            return m.group("body") if m else s

        # helper to find label (general) in a text; returns first match or None
        def _find_label_general(text: str, key: str) -> Optional[str]:
            if not text:
                return None
            # try a few tolerant variants; stop at next " : " or end
            patterns = [
                rf"""['"]?{re.escape(key)}['"]?\s*[:=]\s*['"]?(?P<v>[^'"\n:][^'"\n]*?)['"]?(?=\s*[:]|$)""",
                rf"""['"]?{re.escape(key)}['"]?\s*[:=]\s*['"]?(?P<v>[^'"\n]+?)['"]?""",
                rf"""{re.escape(key)}\s*[:=]\s*(?P<v>[^: \n]+)"""
            ]
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m and m.group("v") is not None:
                    val = m.group("v").strip()
                    return val if val != "" else None
            return None

        # timestamp needs special pattern (contains spaces and colons)
        def _find_timestamp(text: str) -> Optional[str]:
            if not text:
                return None
            # common form: 2025-10-10 13:24:19 or 'timestamp':2025-10-10 13:24:19
            m = re.search(r"""['"]?timestamp['"]?\s*[:=]\s*['"]?(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})['"]?""", text, flags=re.IGNORECASE)
            if m:
                return m.group("ts").strip()
            # fallback: try to find a leading timestamp in the form yyyy-mm-dd : hh:mm:ss
            m2 = re.search(r"(?P<date>\d{4}-\d{2}-\d{2})\s*[:]\s*(?P<time>\d{2}:\d{2}:\d{2})", text)
            if m2:
                return f"{m2.group('date')} {m2.group('time')}"
            return None

        # ---- start ----
        if not AlfredQueryOffline:
            return "", None, None, None, fallback_user, None

        # --- Case: dict input ---
        if isinstance(AlfredQueryOffline, dict):
            # message may be triple-wrapped
            message_raw = str(
                AlfredQueryOffline.get("text")
                or AlfredQueryOffline.get("message")
                or AlfredQueryOffline.get("query")
                or AlfredQueryOffline.get("q")
                or ""
            )
            message_inner = _strip_triple_single_if_present(message_raw).strip()

            # Attempt to extract labels from dict directly (highest priority)
            score = AlfredQueryOffline.get("score", None)
            gender = AlfredQueryOffline.get("gender", None)
            gender_conf = AlfredQueryOffline.get("gender_conf", None)

            # timestamp fields in dict
            timestamp = None
            if AlfredQueryOffline.get("timestamp"):
                timestamp = str(AlfredQueryOffline.get("timestamp")).strip()
            else:
                ts_date = AlfredQueryOffline.get("date") or AlfredQueryOffline.get("timestamp_date")
                ts_time = AlfredQueryOffline.get("time") or AlfredQueryOffline.get("timestamp_time")
                if ts_date and ts_time:
                    timestamp = f"{str(ts_date).strip()} {str(ts_time).strip()}"

            # username
            username = (
                str(AlfredQueryOffline.get("username"))
                if AlfredQueryOffline.get("username") is not None
                else str(AlfredQueryOffline.get("user") or fallback_user)
            )

            # If some labels missing, try to parse them from the message_inner (in-case they were embedded)
            if score is None:
                score = _find_label_general(message_inner, "score")
            if gender is None:
                gender = _find_label_general(message_inner, "gender")
            if gender_conf is None:
                gender_conf = _find_label_general(message_inner, "gender_conf")
            if timestamp is None:
                timestamp = _find_timestamp(message_inner)
            # Clean "None" string to actual None
            score = None if score in (None, "None", "") else str(score)
            gender = None if gender in (None, "None", "") else str(gender)
            gender_conf = None if gender_conf in (None, "None", "") else str(gender_conf)

            # strip any inline labels from message_inner (remove score/gender/gender_conf fragments)
            # remove patterns like " : 'score':None" etc.
            message_clean = re.sub(r"\s*:\s*'score'\s*:\s*[^:]*", "", message_inner, flags=re.IGNORECASE)
            message_clean = re.sub(r"\s*:\s*'gender'\s*:\s*[^:]*", "", message_clean, flags=re.IGNORECASE)
            message_clean = re.sub(r"\s*:\s*'gender_conf'\s*:\s*[^:]*", "", message_clean, flags=re.IGNORECASE)
            message_clean = message_clean.strip(" :\n\r\t")
            return message_clean, score, gender, gender_conf, username, timestamp

        # --- Case: string input ---
        if isinstance(AlfredQueryOffline, str):
            s_full = AlfredQueryOffline.strip()

            # If there's an explicit "'message':" label with a triple-fence after it, handle that too
            m_label_msg = re.search(r"""['"]?message['"]?\s*[:=]\s*(?P<val>('{3}.*?'{3}|.+))""", s_full, flags=re.IGNORECASE | re.DOTALL)
            if m_label_msg:
                val = m_label_msg.group("val").strip()
                # if val is a fence, extract body, otherwise treat as normal
                if val.startswith("'''") and val.endswith("'''"):
                    body = val[3:-3]
                    meta = (s_full[: m_label_msg.start()] + s_full[m_label_msg.end():]).strip()
                else:
                    # not fenced; try to find fence elsewhere
                    m_fence_else = re.search(r"('{3})(?P<body>.*?)(\1)", s_full, flags=re.DOTALL)
                    if m_fence_else:
                        body = m_fence_else.group("body")
                        meta = (s_full[: m_fence_else.start()] + s_full[m_fence_else.end():]).strip()
                    else:
                        body = val
                        meta = (s_full[: m_label_msg.start()] + s_full[m_label_msg.end():]).strip()
            else:
                # 1) find first '''...''' block (non-greedy) and treat that as canonical message
                m_fence = re.search(r"('{3})(?P<body>.*?)(\1)", s_full, flags=re.DOTALL)
                if m_fence:
                    body = m_fence.group("body")
                    meta = (s_full[: m_fence.start()] + s_full[m_fence.end():]).strip()
                else:
                    # No fence at all: treat whole string as body and meta empty
                    body = s_full
                    meta = ""

            # Now parse labels from both body and meta (score/gender often inside body; username/timestamp often in meta)
            # Priority: explicit label in meta first, then body, then fallbacks.
            score = _find_label_general(meta, "score") or _find_label_general(body, "score")
            gender = _find_label_general(meta, "gender") or _find_label_general(body, "gender")
            gender_conf = _find_label_general(meta, "gender_conf") or _find_label_general(body, "gender_conf")
            username = _find_label_general(meta, "username") or _find_label_general(s_full, "username") or fallback_user
            timestamp = _find_timestamp(meta) or _find_timestamp(s_full) or None

            # normalize "None" -> None, ensure strings otherwise
            score = None if score in (None, "None", "") else str(score)
            gender = None if gender in (None, "None", "") else str(gender)
            gender_conf = None if gender_conf in (None, "None", "") else str(gender_conf)
            username = username or fallback_user

            # Build clean message: if body contains labelled suffixes (like " : 'score':None ..."), strip them.
            # Find first labelled key inside body and cut message before it (score/gender/gender_conf)
            label_positions = []
            for k in ("'score'", "score", "'gender_conf'", "gender_conf", "'gender'", "gender"):
                idx = body.lower().find(k.lower())
                if idx != -1:
                    label_positions.append(idx)
            if label_positions:
                cut_at = min(label_positions)
                message_clean = body[:cut_at].rstrip(" :\n\r\t").strip()
            else:
                message_clean = body.strip(" :\n\r\t ")

            return message_clean, score, gender, gender_conf, username, timestamp

        # --- fallback for other types ---
        return str(AlfredQueryOffline).strip(), None, None, None, fallback_user, None



###########################


    def get_frame(self):
        """
        Returns: (frame, cap)
        - frame: np.ndarray or None
        - cap: cv2.VideoCapture or None (ONLY if we opened it)
        """
        import cv2

        # 1) Preferred source: face_tracking
        try:
            import face_tracking
            try:
                frame = face_tracking.get_latest_frame(timeout=1.0)
            except TypeError:
                frame = face_tracking.get_latest_frame()

            if frame is not None and frame.size > 0:
                return frame, None
        except Exception:
            pass

        # 2) Fallback: open camera ONCE (cached)
        try:
            cap = getattr(self, "_cached_cap", None)
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # store cached cap to avoid reopening next time
                self._cached_cap = cap
        except Exception:
            cap = None

        if cap is None or not cap.isOpened():
            # ensure we don't keep a broken cached handle
            try:
                if getattr(self, "_cached_cap", None) is not None:
                    try:
                        self._cached_cap.release()
                    except Exception:
                        pass
                    self._cached_cap = None
            except Exception:
                pass
            return None, None

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            return None, None

        return frame, cap


    def record_video(self, cap, duration=5):
        """
        Non-blocking background recording.
        Uses an EXISTING VideoCapture only.
        """
        import threading
        import time
        import datetime
        import os
        import cv2

        if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()) or duration <= 0:
            return

        def _record():
            try:
                img_dir = os.path.join(
                    Alfred_config.DRIVE_LETTER,
                    "Python_Env", "New_Virtual_Env", "Personal", "Moondream_Image_File"
                )
                os.makedirs(img_dir, exist_ok=True)

                # Try to get width/height/fps from the capture; fall back to reasonable defaults.
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = float(cap_fps) if cap_fps and cap_fps > 0 else 20.0

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(img_dir, f"MoonDream_Front_{ts}.mp4")

                out = cv2.VideoWriter(path, fourcc, fps, (width, height))
                start = time.time()

                # compute a sleep target that roughly matches camera fps to avoid busy-looping
                frame_interval = max(0.0, 1.0 / fps - 0.001)

                while time.time() - start < duration:
                    ret, frame = cap.read()
                    if ret and frame is not None and getattr(frame, "size", 0) > 0:
                        out.write(frame)
                    # sleep a small amount to avoid burning CPU and let camera catch up
                    if frame_interval > 0:
                        time.sleep(frame_interval)

                out.release()
                print(f"Saved video: {path}")

            except Exception as e:
                print("Recording error:", e)

        threading.Thread(target=_record, daemon=True).start()


    def run_vision(self, frame, prompt, model):
        import cv2
        import ollama

        if frame is None or frame.size == 0:
            raise RuntimeError("Invalid frame for vision inference")

        # Use JPEG encoding for speed and smaller payloads vs PNG
        is_ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not is_ok:
            raise RuntimeError("Failed to encode frame")

        system_message = {
            "role": "system",
            "content": (
                "You are a Vision AI assistant. "
                "You respond as if seeing with your own eyes. "
                "You are concise, friendly, and encouraging."
            )
        }

        user_message = {
            "role": "user",
            "content": prompt,
            "images": [buffer.tobytes()]
        }

        response = ollama.chat(
            model=model,
            messages=[system_message, user_message],
            options={"temperature": 0.3},
            stream=False
        )

        return response["message"]["content"].strip()


    def Vision_Model_Front_Look(self, prompt):
        import time

        start = time.time()

        model = getattr(self, "AI_vision_model", "moondream")
        try:
            model = self.gui.vision_model.get()
        except Exception:
            pass

        frame, cap = self.get_frame()
        if frame is None:
            print("No valid camera frame")
            return

        # 🔴 Start recording AFTER frame capture (non-blocking)
        record_seconds = getattr(self, "vision_record_seconds", 5)
        self.record_video(cap, record_seconds)

        # 🧠 Instant inference
        response = self.run_vision(frame, prompt, model)

        print(f"{model}: {response}")
        print(f"Vision latency: {time.time() - start:.3f}s")

        try:
            speech.AlfredSpeak(response)
            listen.send_bluetooth(response)
        except Exception:
            pass

        # Only release the capture if it is NOT our cached persistent cap.
        try:
            if cap is not None:
                if getattr(self, "_cached_cap", None) is not cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                # if cap is the cached one, leave it open for faster future captures
        except Exception:
            pass








##    def get_frame(self):
##        """
##        Returns: (frame, cap)
##        - frame: np.ndarray or None
##        - cap: cv2.VideoCapture or None (ONLY if we opened it)
##        """
##        import cv2
##
##        # 1) Preferred source: face_tracking
##        try:
##            import face_tracking
##            try:
##                frame = face_tracking.get_latest_frame(timeout=1.0)
##            except TypeError:
##                frame = face_tracking.get_latest_frame()
##
##            if frame is not None and frame.size > 0:
##                return frame, None
##        except Exception:
##            pass
##
##        # 2) Fallback: open camera ONCE
##        cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##
##        if not cap.isOpened():
##            cap.release()
##            return None, None
##
##        ret, frame = cap.read()
##        if not ret or frame is None or frame.size == 0:
##            cap.release()
##            return None, None
##
##        return frame, cap
##
##
##    def record_video(self, cap, duration=5):
##        """
##        Non-blocking background recording.
##        Uses an EXISTING VideoCapture only.
##        """
##        import threading
##        import time
##        import datetime
##        import os
##        import cv2
##
##        if cap is None or not cap.isOpened() or duration <= 0:
##            return
##
##        def _record():
##            try:
##                img_dir = os.path.join(
##                    Alfred_config.DRIVE_LETTER,
##                    "Python_Env", "New_Virtual_Env", "Personal", "Moondream_Image_File"
##                )
##                os.makedirs(img_dir, exist_ok=True)
##
##                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
##                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
##                fps = 20.0
##                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
##
##                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
##                path = os.path.join(img_dir, f"MoonDream_Front_{ts}.mp4")
##
##                out = cv2.VideoWriter(path, fourcc, fps, (width, height))
##                start = time.time()
##
##                while time.time() - start < duration:
##                    ret, frame = cap.read()
##                    if ret and frame is not None:
##                        out.write(frame)
##                    time.sleep(0.01)
##
##                out.release()
##                print(f"Saved video: {path}")
##
##            except Exception as e:
##                print("Recording error:", e)
##
##        threading.Thread(target=_record, daemon=True).start()
##
##
##    def run_vision(self, frame, prompt, model):
##        import cv2
##        import ollama
##
##        if frame is None or frame.size == 0:
##            raise RuntimeError("Invalid frame for vision inference")
##
##        is_ok, buffer = cv2.imencode(".png", frame)
##        if not is_ok:
##            raise RuntimeError("Failed to encode frame")
##
##        system_message = {
##            "role": "system",
##            "content": (
##                "You are a Vision AI assistant. "
##                "You respond as if seeing with your own eyes. "
##                "You are concise, friendly, and encouraging."
##            )
##        }
##
##        user_message = {
##            "role": "user",
##            "content": prompt,
##            "images": [buffer.tobytes()]
##        }
##
##        response = ollama.chat(
##            model=model,
##            messages=[system_message, user_message],
##            options={"temperature": 0.3},
##            stream=False
##        )
##
##        return response["message"]["content"].strip()
##
##
##    def Vision_Model_Front_Look(self, prompt):
##        import time
##
##        start = time.time()
##
##        model = getattr(self, "AI_vision_model", "moondream")
##        try:
##            model = self.gui.vision_model.get()
##        except Exception:
##            pass
##
##        frame, cap = self.get_frame()
##        if frame is None:
##            print("No valid camera frame")
##            return
##
##        # 🔴 Start recording AFTER frame capture (non-blocking)
##        record_seconds = getattr(self, "vision_record_seconds", 5)
##        self.record_video(cap, record_seconds)
##
##        # 🧠 Instant inference
##        response = self.run_vision(frame, prompt, model)
##
##        print(f"{model}: {response}")
##        print(f"Vision latency: {time.time() - start:.3f}s")
##
##        try:
##            speech.AlfredSpeak(response)
##            listen.send_bluetooth(response)
##        except Exception:
##            pass
##
##        if cap is not None:
##            cap.release()






###########################
    
##
##    def Vision_Model_Front_Look(self, prompt):
##        """
##        Capture a frame (prefer from face_tracking.get_latest_frame), call the chosen
##        vision model via ollama.chat (non-streaming), log and speak the reply.
##
##        Added: short video recording (mp4) using OpenCV VideoWriter.
##        Minimal changes only — recording is optional and defaults to 5 seconds.
##        """
##        import time
##        import datetime
##        import os
##        import cv2
##
##        model = None
##        try:
##            model = self.gui.vision_model.get()
##        except Exception:
##            model = getattr(self, "AI_vision_model", "moondream")
##        print(f"AI_Vision_Running : {model}")
##
##        msg, score, gender, gender_conf, user, timestamp = self.AI_vision_extract_text_from_query(prompt)
##
##        print("message:", msg)
##        print("score  :", score)
##        print("gender:", gender)
##        print("gender_conf:", gender_conf)
##        print("speaker:", user)
##        print("timestamp:", timestamp)
##        
##        start_vision = time.time()
##
##        frame = None
##        local_cap = None
##        window_title = f"Vision Language Model : {model} System"
##
##        # 1) Preferred: try to get latest frame published by face_tracking
##        try:
##            import face_tracking
##            try:
##                frame = face_tracking.get_latest_frame(timeout=1.0)
##            except Exception:
##                # function may not accept timeout or behave differently; try no-arg call
##                try:
##                    frame = face_tracking.get_latest_frame()
##                except Exception:
##                    frame = None
##        except Exception:
##            frame = None
##
##        # 2) Fallback: open a local VideoCapture only if no frame available
##        try:
##            if frame is None:
##                try:
##                    local_cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##                    local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##                    local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##                    ret, frame = local_cap.read()
##                    if not ret or frame is None or frame.size == 0:
##                        print("Camera read failed; aborting vision call.")
##                        # cleanup local cap and exit
##                        if local_cap is not None:
##                            try:
##                                local_cap.release()
##                            except Exception:
##                                pass
##                        return
##                except Exception as e:
##                    print("Failed to open fallback camera:", e)
##                    if local_cap is not None:
##                        try:
##                            local_cap.release()
##                        except Exception:
##                            pass
##                    return
##
##            # ---------------------- NEW: short video recording ----------------------
##            # Minimal, safe recording code. Adjust record_seconds as needed.
##            try:
##                # Decide how long to record (seconds). Change value if you want different length.
##                record_seconds = getattr(self, "vision_record_seconds", 5)  # default 5s
##                # If someone explicitly set 0 or False, skip recording
##                if record_seconds and float(record_seconds) > 0:
##                    # Ensure image directory exists (reuse your img_dir later)
##                    try:
##                        img_dir = os.path.join(
##                            Alfred_config.DRIVE_LETTER,
##                            "Python_Env", "New_Virtual_Env", "Personal", "Moondream_Image_File"
##                        )
##                        os.makedirs(img_dir, exist_ok=True)
##                    except Exception:
##                        img_dir = None
##
##                    # Ensure we have a capture object to read from for recording.
##                    if local_cap is None:
##                        try:
##                            local_cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##                            local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##                            local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##                        except Exception as e:
##                            print("Recording: failed to open camera:", e)
##                            local_cap = None
##
##                    if local_cap is not None and local_cap.isOpened():
##                        # Recording parameters
##                        fps = 20.0
##                        width = int(local_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
##                        height = int(local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
##                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
##                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
##                        if img_dir is not None:
##                            video_path = os.path.join(img_dir, f"MoonDream_Front_{ts}.mp4")
##                        else:
##                            # fallback to cwd
##                            video_path = f"MoonDream_Front_{ts}.mp4"
##
##                        out = None
##                        try:
##                            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
##                            record_start = time.time()
##                            last_frame = frame  # preserve existing frame if available
##                            while (time.time() - record_start) < float(record_seconds):
##                                ret_v, vframe = local_cap.read()
##                                if not ret_v or vframe is None:
##                                    # brief pause then continue trying until time expires
##                                    time.sleep(0.05)
##                                    continue
##                                out.write(vframe)
##                                last_frame = vframe
##                            # use last_frame as the frame to send to vision model (if available)
##                            if last_frame is not None:
##                                frame = last_frame
##                            print(f"Saved short video to: {video_path}")
##                        except Exception as e:
##                            print("Error during video recording:", e)
##                        finally:
##                            try:
##                                if out is not None:
##                                    out.release()
##                            except Exception:
##                                pass
##                    else:
##                        print("Recording skipped: camera not available for recording.")
##            except Exception as e:
##                print("Recording block error:", e)
##            # -------------------- end NEW recording block ----------------------------
##
##            # 4) Ensure directory exists and save the frame
##            try:
##                # if img_dir not created above (recording block), create it here
##                try:
##                    img_dir
##                except NameError:
##                    img_dir = os.path.join(
##                        Alfred_config.DRIVE_LETTER,
##                        "Python_Env", "New_Virtual_Env", "Personal", "Moondream_Image_File"
##                    )
##                os.makedirs(img_dir, exist_ok=True)
##                img_path = os.path.join(img_dir, "MoonDream_Front.png")
##                cv2.imwrite(img_path, frame)
##            except Exception as e:
##                print("Failed to save image for vision model:", e)
##                img_path = None
##
##            # 5) Prepare messages for ollama
##            system_message = {
##                'role': 'system',
##                'content': (
##                    "You are a Vision AI assistant and always happy and uplifting. "
##                    "You will always be encouraging and friendly. You will always give concise answers. "
##                    "You will not answer as if you were an image processor; instead answer as if you are looking at it with your own eyes."
##                )
##            }
##            user_message = {
##                'role': 'user',
##                'content': msg,
##            }
##
##            # Prefer to pass the image bytes (safer)
##            images_payload = []
##            if img_path is not None:
##                try:
##                    with open(img_path, "rb") as f:
##                        images_payload = [f.read()]
##                except Exception as e:
##                    print("Could not read image as bytes; falling back to path. Error:", e)
##                    images_payload = [img_path]
##            else:
##                # If no saved image, try to send raw bytes from frame
##                try:
##                    is_success, buffer = cv2.imencode(".png", frame)
##                    if is_success:
##                        images_payload = [buffer.tobytes()]
##                except Exception:
##                    images_payload = []
##
##            if images_payload:
##                user_message['images'] = images_payload
##
##            # 6) Call ollama.chat - non-streaming
##            try:
##                response = ollama.chat(
##                    model=model,
##                    messages=[system_message, user_message],
##                    options={'temperature': 0.3},
##                    stream=False
##                )
##            except TypeError as e:
##                print("Error calling ollama.chat (TypeError):", e)
##                raise
##            except Exception as e:
##                print("Error calling ollama.chat:", e)
##                raise
##
##
##            # 7) Robust extraction of assistant content from different response shapes
##            def extract_content(resp):
##                """
##                Return only the assistant/user-visible text from a response structure.
##                Handles: dicts with 'message'/'content'/'text', 'choices' lists,
##                streaming chunks with 'delta', lists/tuples of chunks, and objects
##                with .content/.text/.message attributes. Skips obvious metadata keys.
##                """
##                if not resp:
##                    return ""
##
##                # simple string
##                if isinstance(resp, str):
##                    return resp.strip()
##
##                # lists/tuples/streams of chunks: collect useful parts
##                if isinstance(resp, (list, tuple)):
##                    parts = []
##                    for item in resp:
##                        p = extract_content(item)
##                        if p:
##                            parts.append(p)
##                    return " ".join(parts).strip()
##
##                # dict-like handling
##                if isinstance(resp, dict):
##                    # common direct shapes
##                    # 1) message -> { 'role':..., 'content': '...' } or similar
##                    if 'message' in resp and resp['message']:
##                        return extract_content(resp['message'])
##
##                    # 2) direct 'content' or 'text'
##                    if 'content' in resp and isinstance(resp['content'], str) and resp['content'].strip():
##                        return resp['content'].strip()
##                    if 'text' in resp and isinstance(resp['text'], str) and resp['text'].strip():
##                        return resp['text'].strip()
##
##                    # 3) choices: usually a list of choice dicts
##                    if 'choices' in resp and isinstance(resp['choices'], (list, tuple)) and resp['choices']:
##                        # prefer first non-empty content found in choices
##                        for choice in resp['choices']:
##                            c = extract_content(choice)
##                            if c:
##                                return c
##                        # if nothing found, fall through
##
##                    # 4) streaming chunk style: 'delta' often contains partial 'content' or 'text'
##                    if 'delta' in resp and resp['delta']:
##                        return extract_content(resp['delta'])
##
##                    # 5) fallback: scan values but skip metadata-like keys
##                    skip_keys = {
##                        'model', 'created_at', 'done', 'done_reason', 'total_duration', 'load_duration',
##                        'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration'
##                    }
##                    for k, v in resp.items():
##                        if k in skip_keys:
##                            continue
##                        candidate = extract_content(v)
##                        if candidate:
##                            return candidate
##
##                    # nothing meaningful found
##                    return ""
##
##                # objects with attributes (Message-like objects)
##                for attr in ('content', 'text', 'message', 'data'):
##                    if hasattr(resp, attr):
##                        try:
##                            return extract_content(getattr(resp, attr))
##                        except Exception:
##                            pass
##
##                # last-resort: try converting to str but filter obvious metadata dumps
##                try:
##                    s = str(resp).strip()
##                    low = s.lower()
##                    if ('model' in low and 'created_at' in low) or ('total_duration' in low and 'load_duration' in low):
##                        return ""
##                    return s
##                except Exception:
##                    return ""
##
##            # Use the extractor on your `response` (or whatever variable you hold)
##            content = extract_content(response) or ""
##
##            # small, explicit wording-cleanup replacements you had
##            cleaned = content.replace("The image depicts", "I see")
##            cleaned = cleaned.replace("The image shows", "I see")
##            cleaned = cleaned.replace("In the image", "I see")
##            cleaned = cleaned.replace(" in the image ", " I see ")
##
##            # 'cleaned' now contains only the assistant-visible text (no metadata)
##            if "{'model': 'moondream', 'created_at':" in cleaned:
##                print(f"\n Excuse me. The prompt is unclear. Please check your prompt : \n {user_message} \n")
##                try:
##                    speech.AlfredSpeak(f"Excuse me {user}. The Query is unclear. Please check your Query, {user_message}")
##                    listen.send_bluetooth(f"Excuse me {user}. The Query is unclear. Please check your Query, {user_message}")
##                except Exception:
##                    pass
##                cleaned = ""
##
##            stop_vision = time.time()
##            print(f"Vision Response Time : {stop_vision - start_vision:.3f}s")
##            print(f"{model}: {cleaned}")
##
##            Alfred_Repeat_Previous_Response = cleaned
##
##
##            # 8) Build GUI-friendly logs
##            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
##            current_time = datetime.datetime.now().strftime('%H:%M:%S')
##
##            chat_entry = {
##                "date": current_date,
##                "time": current_time,
##                "query": prompt,
##                "response": cleaned,
##                "model": model
##            }
##
##            memory.add_to_memory(chat_entry)
##            repeat.add_to_repeat(chat_entry)
##
##            msg_log = f"At {current_date} :  {current_time} : You Asked: {prompt} "
##            query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {user}"
##
##            try:
##                self.gui.log_message(msg_log)
##                self.gui.log_response(query_resp)
##            except Exception as e:
##                print("GUI instance not available for logging message:", e)
##
##            # 9) Speak & bluetooth notify
##            try:
##                speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
##                listen.send_bluetooth(Alfred_Repeat_Previous_Response)
##                speech.AlfredSpeak("Listening...")
##                listen.send_bluetooth("Listening...")
##                
##            except Exception as e:
##                print("Error while speaking/sending bluetooth:", e)
##
##        finally:
##            # cleanup: release local cap only if we opened it here
##            try:
##                if local_cap is not None:
##                    local_cap.release()
##            except Exception:
##                pass
##            # destroy the preview window we created (don't destroy all windows)
##            try:
##                cv2.destroyWindow(window_title)
##            except Exception:
##                pass
##
##
##
##




##
##    def Vision_Model_Front_Look(self, prompt):
##        """
##        Capture a frame (prefer from face_tracking.get_latest_frame), call the chosen
##        vision model via ollama.chat (non-streaming), log and speak the reply.
##        """
##        import time
##        import datetime
##        import os
##        import cv2
##
##        model = None
##        try:
##            model = self.gui.vision_model.get()
##        except Exception:
##            model = getattr(self, "AI_vision_model", "moondream")
##        print(f"AI_Vision_Running : {model}")
##
##        msg, score, gender, gender_conf, user, timestamp = self.AI_vision_extract_text_from_query(prompt)
##
##        print("message:", msg)
##        print("score  :", score)
##        print("gender:", gender)
##        print("gender_conf:", gender_conf)
##        print("speaker:", user)
##        print("timestamp:", timestamp)
##        
##        start_vision = time.time()
##
##        frame = None
##        local_cap = None
##        window_title = f"Vision Language Model : {model} System"
##
##        # 1) Preferred: try to get latest frame published by face_tracking
##        try:
##            import face_tracking
##            try:
##                frame = face_tracking.get_latest_frame(timeout=1.0)
##            except Exception:
##                # function may not accept timeout or behave differently; try no-arg call
##                try:
##                    frame = face_tracking.get_latest_frame()
##                except Exception:
##                    frame = None
##        except Exception:
##            frame = None
##
##        # 2) Fallback: open a local VideoCapture only if no frame available
##        try:
##            if frame is None:
##                try:
##                    local_cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##                    local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##                    local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##                    ret, frame = local_cap.read()
##                    if not ret or frame is None or frame.size == 0:
##                        print("Camera read failed; aborting vision call.")
##                        # cleanup local cap and exit
##                        if local_cap is not None:
##                            try:
##                                local_cap.release()
##                            except Exception:
##                                pass
##                        return
##                except Exception as e:
##                    print("Failed to open fallback camera:", e)
##                    if local_cap is not None:
##                        try:
##                            local_cap.release()
##                        except Exception:
##                            pass
##                    return
##
##            # 4) Ensure directory exists and save the frame
##            try:
##                img_dir = os.path.join(
##                    Alfred_config.DRIVE_LETTER,
##                    "Python_Env", "New_Virtual_Env", "Personal", "Moondream_Image_File"
##                )
##                os.makedirs(img_dir, exist_ok=True)
##                img_path = os.path.join(img_dir, "MoonDream_Front.png")
##                cv2.imwrite(img_path, frame)
##            except Exception as e:
##                print("Failed to save image for vision model:", e)
##                img_path = None
##
##            # 5) Prepare messages for ollama
##            system_message = {
##                'role': 'system',
##                'content': (
##                    "You are a Vision AI assistant and always happy and uplifting. "
##                    "You will always be encouraging and friendly. You will always give concise answers. "
##                    "You will not answer as if you were an image processor; instead answer as if you are looking at it with your own eyes."
##                )
##            }
##            user_message = {
##                'role': 'user',
##                'content': msg,
##            }
##
##            # Prefer to pass the image bytes (safer)
##            images_payload = []
##            if img_path is not None:
##                try:
##                    with open(img_path, "rb") as f:
##                        images_payload = [f.read()]
##                except Exception as e:
##                    print("Could not read image as bytes; falling back to path. Error:", e)
##                    images_payload = [img_path]
##            else:
##                # If no saved image, try to send raw bytes from frame
##                try:
##                    is_success, buffer = cv2.imencode(".png", frame)
##                    if is_success:
##                        images_payload = [buffer.tobytes()]
##                except Exception:
##                    images_payload = []
##
##            if images_payload:
##                user_message['images'] = images_payload
##
##            # 6) Call ollama.chat - non-streaming
##            try:
##                response = ollama.chat(
##                    model=model,
##                    messages=[system_message, user_message],
##                    options={'temperature': 0.3},
##                    stream=False
##                )
##            except TypeError as e:
##                print("Error calling ollama.chat (TypeError):", e)
##                raise
##            except Exception as e:
##                print("Error calling ollama.chat:", e)
##                raise
##
##
##            # 7) Robust extraction of assistant content from different response shapes
##            def extract_content(resp):
##                """
##                Return only the assistant/user-visible text from a response structure.
##                Handles: dicts with 'message'/'content'/'text', 'choices' lists,
##                streaming chunks with 'delta', lists/tuples of chunks, and objects
##                with .content/.text/.message attributes. Skips obvious metadata keys.
##                """
##                if not resp:
##                    return ""
##
##                # simple string
##                if isinstance(resp, str):
##                    return resp.strip()
##
##                # lists/tuples/streams of chunks: collect useful parts
##                if isinstance(resp, (list, tuple)):
##                    parts = []
##                    for item in resp:
##                        p = extract_content(item)
##                        if p:
##                            parts.append(p)
##                    return " ".join(parts).strip()
##
##                # dict-like handling
##                if isinstance(resp, dict):
##                    # common direct shapes
##                    # 1) message -> { 'role':..., 'content': '...' } or similar
##                    if 'message' in resp and resp['message']:
##                        return extract_content(resp['message'])
##
##                    # 2) direct 'content' or 'text'
##                    if 'content' in resp and isinstance(resp['content'], str) and resp['content'].strip():
##                        return resp['content'].strip()
##                    if 'text' in resp and isinstance(resp['text'], str) and resp['text'].strip():
##                        return resp['text'].strip()
##
##                    # 3) choices: usually a list of choice dicts
##                    if 'choices' in resp and isinstance(resp['choices'], (list, tuple)) and resp['choices']:
##                        # prefer first non-empty content found in choices
##                        for choice in resp['choices']:
##                            c = extract_content(choice)
##                            if c:
##                                return c
##                        # if nothing found, fall through
##
##                    # 4) streaming chunk style: 'delta' often contains partial 'content' or 'text'
##                    if 'delta' in resp and resp['delta']:
##                        return extract_content(resp['delta'])
##
##                    # 5) fallback: scan values but skip metadata-like keys
##                    skip_keys = {
##                        'model', 'created_at', 'done', 'done_reason', 'total_duration', 'load_duration',
##                        'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration'
##                    }
##                    for k, v in resp.items():
##                        if k in skip_keys:
##                            continue
##                        candidate = extract_content(v)
##                        if candidate:
##                            return candidate
##
##                    # nothing meaningful found
##                    return ""
##
##                # objects with attributes (Message-like objects)
##                for attr in ('content', 'text', 'message', 'data'):
##                    if hasattr(resp, attr):
##                        try:
##                            return extract_content(getattr(resp, attr))
##                        except Exception:
##                            pass
##
##                # last-resort: try converting to str but filter obvious metadata dumps
##                try:
##                    s = str(resp).strip()
##                    low = s.lower()
##                    if ('model' in low and 'created_at' in low) or ('total_duration' in low and 'load_duration' in low):
##                        return ""
##                    return s
##                except Exception:
##                    return ""
##
##            # Use the extractor on your `response` (or whatever variable you hold)
##            content = extract_content(response) or ""
##
##            # small, explicit wording-cleanup replacements you had
##            cleaned = content.replace("The image depicts", "I see")
##            cleaned = cleaned.replace("The image shows", "I see")
##            cleaned = cleaned.replace("In the image", "I see")
##            cleaned = cleaned.replace(" in the image ", " I see ")
##
##            # 'cleaned' now contains only the assistant-visible text (no metadata)
##            if "{'model': 'moondream', 'created_at':" in cleaned:
##                print(f"\n Excuse me. The prompt is unclear. Please check your prompt : \n {user_message} \n")
##                try:
##                    speech.AlfredSpeak(f"Excuse me {user}. The Query is unclear. Please check your Query, {user_message}")
##                    listen.send_bluetooth(f"Excuse me {user}. The Query is unclear. Please check your Query, {user_message}")
##                except Exception:
##                    pass
##                cleaned = ""
##
##            stop_vision = time.time()
##            print(f"Vision Response Time : {stop_vision - start_vision:.3f}s")
##            print(f"{model}: {cleaned}")
##
##            Alfred_Repeat_Previous_Response = cleaned
##
##
##            # 8) Build GUI-friendly logs
##            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
##            current_time = datetime.datetime.now().strftime('%H:%M:%S')
##
##            chat_entry = {
##                "date": current_date,
##                "time": current_time,
##                "query": prompt,
##                "response": cleaned,
##                "model": model
##            }
##
##            memory.add_to_memory(chat_entry)
##            repeat.add_to_repeat(chat_entry)
##
##            msg_log = f"At {current_date} :  {current_time} : You Asked: {prompt} "
##            query_resp = f"At {current_date} :  {current_time} : {model} : {Alfred_Repeat_Previous_Response} : {user}"
##
##            try:
##                self.gui.log_message(msg_log)
##                self.gui.log_response(query_resp)
##            except Exception as e:
##                print("GUI instance not available for logging message:", e)
##
##            # 9) Speak & bluetooth notify
##            try:
##                speech.AlfredSpeak(Alfred_Repeat_Previous_Response)
##                listen.send_bluetooth(Alfred_Repeat_Previous_Response)
##                speech.AlfredSpeak("Listening...")
##                listen.send_bluetooth("Listening...")
##                
##            except Exception as e:
##                print("Error while speaking/sending bluetooth:", e)
##
##        finally:
##            # cleanup: release local cap only if we opened it here
##            try:
##                if local_cap is not None:
##                    local_cap.release()
##            except Exception:
##                pass
##            # destroy the preview window we created (don't destroy all windows)
##            try:
##                cv2.destroyWindow(window_title)
##            except Exception:
##                pass
##

    def Vision_Model_Where_Look(self, prompt):

        AlfredQueryOffline = prompt
        model = self.gui.vision_model.get()
        print(f"AI_Vision_Running : {model}")

        start_vision = time.time()

        cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
        cap.set(3, 640)
        cap.set(4, 480)

        img_path = Alfred_config.DRIVE_LETTER + "Python_Env//New_Virtual_Env//Personal//Moondream_Image_File//MoonDream_Front.jpg"
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Face Detection System", frame)
            cv2.imwrite(img_path, frame)

        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a Vision AI assistant and always happy and uplifting. You will always be encouraging and friendly. You will always give concise answers. You will never answer that it is an image you will answer as if you are looking at it from your own eyes.', 'temperature': '0.3', 'role': 'user', 'content': prompt, 'images': [img_path], 'stream': True},
            ]
        )

        content = response['message']['content']
        cleaned = content.replace("The image depicts", "I see")

        stop_vision = time.time()
        print(f"Vision Response Time : {stop_vision - start_vision}")
        print("MOONDREAM:", cleaned)

        locations = ["lounge", "living", "bedroom", "bathroom", "toilet", "kitchen"]
        spoken = "I am not sure, Sir"

        for place in locations:
            if place in cleaned:
                spoken = f"I am in the {place} room, Sir" if place != "toilet" else "I am in the toilet, Sir"
                break


        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        Alfred_Repeat_Previous_Response = spoken

        # GUI log if available
        msg = (
            f"At {current_date} :  {current_time} : You Asked: {AlfredQueryOffline} "
        )

        query_resp = f"At {current_date} :  {current_time} : I replied : \n \n {Alfred_Repeat_Previous_Response} \n \n "

        
        try:
            assistant.gui.log_message(msg)
            assistant.gui.log_response(query_resp)
        except NameError:
            print("GUI instance not available for logging message.")


        speech.AlfredSpeak(spoken)
        listen.send_bluetooth(spoken)

        speech.AlfredSpeak("Listening...")
        listen.send_bluetooth("Listening...")

