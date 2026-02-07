# shared_camera.py
import threading
import time
import cv2

class SharedCamera:
    def __init__(self, src=0, width=640, height=480, name="shared"):
        self.src = src
        self.width = width
        self.height = height
        self.name = name

        self._cap = None
        self._thread = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._latest = None

    def start(self, timeout=2.0):
        """Open capture and start background reader thread. Return True on success."""
        try:
            # Open capture
            self._cap = cv2.VideoCapture(self.src)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                time.sleep(0.03)
            if not (self._cap and self._cap.isOpened()):
                try:
                    self._cap.release()
                except Exception:
                    pass
                return False
            # set resolution if possible
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            except Exception:
                pass
        except Exception:
            return False

        self._stopped.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True, name=f"{self.name}-reader")
        self._thread.start()
        return True

    def _reader(self):
        while not self._stopped.is_set():
            if self._cap is None:
                time.sleep(0.02)
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            with self._lock:
                # store latest frame (consumers get a copy)
                self._latest = frame
        # thread exits

    def get_frame(self):
        """Return a copy of the latest frame or None if not available."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def read_blocking(self, timeout=1.0):
        """Block up to timeout seconds until a frame is available."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.get_frame()
            if f is not None:
                return f
            time.sleep(0.01)
        return None

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        self._thread = None
        with self._lock:
            self._latest = None


# optional: single global instance helper
_shared_instance = None
def get_shared_camera(src=0, width=640, height=480):
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = SharedCamera(src, width, height)
    return _shared_instance
