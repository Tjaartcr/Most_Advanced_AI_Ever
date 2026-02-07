

#!/usr/bin/env python3
"""
ollama_cam.py
Capture a frame from a local webcam (or an IP stream) and send it to Ollama.
Features:
 - Auto-detect available local camera indices
 - Accept camera source as integer index (0,1,...) or URL (http://...)
 - Optional preview & confirm before sending
 - Streams Ollama output to the terminal in real-time
 - Cleans up temp files and handles Ctrl+C gracefully
"""

import cv2
import subprocess
import tempfile
import os
import sys
import time
from typing import Union, List

DEFAULT_MODEL = "llava"
MAX_CAMERA_CHECK = 6  # how many indices to probe for local cameras


def list_local_cameras(max_index: int = MAX_CAMERA_CHECK) -> List[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            # try default backend if CAP_DSHOW failed
            try:
                cap.release()
            except Exception:
                pass
            continue
        found.append(i)
        cap.release()
    return found


def open_capture(source: Union[int, str]):
    # If source looks like an integer string, convert
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    # Use DirectShow on Windows name to improve camera detection
    if os.name == "nt" and isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def capture_frame(camera_source: Union[int, str], preview: bool = False):
    cap = open_capture(camera_source)
    if not cap or not cap.isOpened():
        print(f"❌ Could not open camera source: {camera_source}")
        return None

    # read a few frames (useful for IP streams / webcams to warm up)
    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    cap.release()

    if frame is None:
        print("❌ Failed to capture frame")
        return None

    if preview:
        # show preview window and wait for y/n key
        cv2.imshow("Preview - press 'y' to send, 'r' to retake, 'q' to cancel", frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('y'), ord('Y')):
                cv2.destroyAllWindows()
                break
            elif key in (ord('r'), ord('R')):
                cv2.destroyAllWindows()
                return capture_frame(camera_source, preview=True)  # retake
            elif key in (ord('q'), 27):  # q or ESC
                cv2.destroyAllWindows()
                return None

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, frame)
    return tmp.name


def run_ollama_with_image(prompt: str, img_path: str, model: str = DEFAULT_MODEL, timeout: int = None):
    cmd = ["ollama", "run", model, "--image", img_path]
    print(f"📷 Sending image {os.path.basename(img_path)} -> {model}")
    print("⏳ Ollama output (streaming):\n")

    # Use Popen so we can stream output in real time and also handle interrupts
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    try:
        # send prompt (add newline so some CLIs treat it as a finished submission)
        if proc.stdin:
            proc.stdin.write(prompt + "\n")
            proc.stdin.flush()
            proc.stdin.close()

        # Read and print output lines as they come
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="")

        # Wait for process to finish (with optional timeout)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"\n⚠️ Ollama process exceeded timeout ({timeout}s). Killing process.")
        proc.kill()
    except KeyboardInterrupt:
        print("\n✂️ Interrupted by user. Terminating Ollama process...")
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        # Ensure resources closed
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def usage_and_exit():
    print("Usage examples:")
    print("  python ollama_cam.py                -> detect local cameras, pick one interactively")
    print("  python ollama_cam.py 0              -> use local camera index 0")
    print("  python ollama_cam.py 1 llava        -> use camera 1 and model 'llava'")
    print("  python ollama_cam.py http://192.168.1.10:81/stream bakllava  -> use IP stream URL")
    print("\nOptional env var PREVIEW=1 to preview the capture before sending (requires a display).")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    preview_flag = os.environ.get("PREVIEW", "") == "1"
    model = DEFAULT_MODEL
    camera_source = None

    if not args:
        # try to auto-detect cameras
        cams = list_local_cameras()
        if not cams:
            print("No local cameras detected.")
            print("Provide a camera index or IP stream URL, e.g. python ollama_cam.py 0")
            usage_and_exit()
        print("Detected local cameras:", cams)
        print("Using first detected camera by default:", cams[0])
        camera_source = cams[0]
    elif len(args) == 1:
        camera_source = args[0]
    elif len(args) >= 2:
        camera_source = args[0]
        model = args[1]

    # If camera_source is numeric string, convert to int
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"Camera source = {camera_source}")
    print(f"Model         = {model}")
    print(f"Preview       = {preview_flag}\n")

    try:
        while True:
            prompt = input("You: ")
            if prompt.strip().lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            img = capture_frame(camera_source, preview=preview_flag)
            if not img:
                print("No image captured. Try a different camera source or disable PREVIEW.")
                continue

            try:
                run_ollama_with_image(prompt, img, model=model, timeout=None)
            finally:
                # cleanup
                try:
                    os.remove(img)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        return


if __name__ == "__main__":
    main()



##import cv2
##import subprocess
##import tempfile
##import os
##import sys
##
### Default model + camera source
##MODEL = "llava"
##CAMERA_SOURCE = 0  # 0 = default webcam, or replace with IP stream URL
##
### Allow overriding via command-line args
##if len(sys.argv) > 1:
##    CAMERA_SOURCE = sys.argv[1]  # e.g. python ollama_cam.py http://192.168.114.64:81/stream
##if len(sys.argv) > 2:
##    MODEL = sys.argv[2]          # e.g. python ollama_cam.py 0 bakllava
##
##
##def capture_frame():
##    cap = cv2.VideoCapture(CAMERA_SOURCE)
##    if not cap.isOpened():
##        print(f"❌ Could not open camera source: {CAMERA_SOURCE}")
##        return None
##
##    ret, frame = cap.read()
##    cap.release()
##
##    if not ret:
##        print("❌ Failed to capture frame")
##        return None
##
##    # Save snapshot
##    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
##    cv2.imwrite(tmp.name, frame)
##    return tmp.name
##
##
##def ollama_with_image(prompt, model=MODEL):
##    img_path = capture_frame()
##    if not img_path:
##        return
##
##    print(f"📷 Captured image: {img_path}")
##    print(f"🤖 Sending to Ollama model: {model}")
##
##    try:
##        subprocess.run(
##            ["ollama", "run", model, "--image", img_path],
##            input=prompt.encode(),
##        )
##    finally:
##        os.remove(img_path)
##
##
##if __name__ == "__main__":
##    print("🎥 Ollama Webcam CLI")
##    print(f"Camera source = {CAMERA_SOURCE}")
##    print(f"Model = {MODEL}")
##    print("Type your prompt and press Enter. (Type 'exit' to quit)\n")
##
##    while True:
##        user_input = input("You: ")
##        if user_input.strip().lower() in ["exit", "quit", "q"]:
##            break
##
##        ollama_with_image(user_input, MODEL)
