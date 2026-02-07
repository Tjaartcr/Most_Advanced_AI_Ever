# vision.py

from arduino_com import arduino
import Alfred_config
from speech import speech
##from communication import comm
from listen import listen

import cv2
from ultralytics import YOLO
from Face_Recognition_Image_Trainer_Best_Great import SimpleFacerec
import time
import datetime

import threading
import multiprocessing
from multiprocessing import Process, Value, Array, Lock
import queue
import serial
import os
import numpy as np

from shared_queues import query_queue, log_queue

My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images(My_Images_Path)

print("vision face trackink....started")

############################################
###         CAMERA CHANNEL SELECT

Camera_Input_Channel = 1

############################################
#       FACE RECOGNITION SOFTWARE

What_I_See_Front = []
What_Is_In_Front_Speak = ""

Who_Is_In_Front = []
Who_Is_In_Front_Speak = []

POI_Who_Is_In_Front = []
POI_String_New = 0

Names_and_POI_Together_List = []
Names_and_POI_Together = 0

Name_Only_For_Where = 0
Name_Only_For_Look_AT = 0

log_queue = queue.Queue()

print("Tracking start")

############################################
#       OBJECT DETECTION SOFTWARE

from ultralytics import YOLO

Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
print("Loading the Model YoloV8n.....")

print('\n')
print('____________________________________________________________________')
print('\n')

print("visio start 1")

# YOLO Model
Obsticle_Detection_Vision_Model = YOLO(Model_File)
with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
    classNames = [line.rstrip('\n') for line in f]

from collections import Counter

print("vision start 2")

# === Shared state for current speaker ===
_current_speaker = None
_speaker_lock = threading.Lock()


def _set_current_speaker(name: str):
    """Internal use only: set the current speaker."""
    global _current_speaker
    with _speaker_lock:
        _current_speaker = name


def _get_current_speaker() -> str:
    """Internal use only: return the current speaker."""
    with _speaker_lock:
        return _current_speaker

# frame-share helpers
import threading, time as _time
_latest_frame = None
_latest_frame_lock = threading.Lock()

def _set_latest_frame(frame):
    """Internal — called by the tracking loop to update the latest frame (copy stored)."""
    global _latest_frame
    if frame is None:
        with _latest_frame_lock:
            _latest_frame = None
        return
    with _latest_frame_lock:
        # store a copy so caller keeps original safe
        _latest_frame = frame.copy()

def get_latest_frame(timeout=1.0):
    """
    Return a copy of the latest frame available within `timeout` seconds, else None.
    Non-blocking if latest is already present.
    """
    t0 = _time.time()
    while _time.time() - t0 < float(timeout):
        with _latest_frame_lock:
            if _latest_frame is not None:
                return _latest_frame.copy()
        # short sleep to avoid busy-loop
        time.sleep(0.01)
    return None

from shutdown_helpers import app_shutdown_event





def Vision_Who_InFront_Look_At_New(get_speaker_func,
                                   sfr=None,
                                   arduino=None,
                                   PRIMARY_CAM=None,
                                   FALLBACK_CAM=None):
    import cv2
    import time
    import numpy as np
    from math import fabs
    import re
    try:
        import Alfred_config
        if PRIMARY_CAM is None:
            PRIMARY_CAM = Alfred_config.CHEST_CAMERA_INPUT
        if FALLBACK_CAM is None:
            FALLBACK_CAM = Alfred_config.LEFT_EYE_CAMERA_INPUT
    except Exception:
        pass
    if sfr is None:
        sfr = globals().get('sfr', None)
    if arduino is None:
        arduino = globals().get('arduino', None)
    def open_camera(src, timeout=2.0):
        if src is None:
            return None
        try:
            cap = cv2.VideoCapture(src)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        return cap
                time.sleep(0.05)
            if cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass
        except Exception:
            pass
        return None
    def create_csrt():
        tr = None
        try:
            tr = cv2.legacy.TrackerCSRT_create()
        except Exception:
            try:
                tr = cv2.TrackerCSRT_create()
            except Exception:
                tr = None
        return tr
    def normalize_name(s):
        if not s:
            return ""
        s = str(s).lower()
        s = re.sub(r'[^a-z0-9]', '', s)
        return s
    def iou(boxA, boxB):
        (xA, yA, wA, hA) = boxA
        (xB, yB, wB, hB) = boxB
        x1 = max(xA, xB)
        y1 = max(yA, yB)
        x2 = min(xA + wA, xB + wB)
        y2 = min(yA + hA, yB + hB)
        interW = max(0, x2 - x1)
        interH = max(0, y2 - y1)
        interArea = interW * interH
        boxAArea = wA * hA
        boxBArea = wB * hB
        union = boxAArea + boxBArea - interArea
        return interArea / (union + 1e-8)
    def face_loc_to_bbox(face_loc):
        try:
            # Common face_loc format: (top, right, bottom, left)
            top, right, bottom, left = face_loc[:4]
            x = int(left); y = int(top)
            w = int(max(1, right - left)); h = int(max(1, bottom - top))
            return (x, y, w, h)
        except Exception:
            try:
                x, y, w, h = face_loc[:4]
                return (int(x), int(y), int(w), int(h))
            except Exception:
                return None

    cap = open_camera(PRIMARY_CAM, timeout=2.0)
    using_primary = True
    if cap is None:
        cap = open_camera(FALLBACK_CAM, timeout=2.0)
        using_primary = False
        if cap is None:
            print("Vision_Who_InFront_Look_At_New: No camera available (primary & fallback failed). Aborting.")
            return
        else:
            print("Vision_Who_InFront_Look_At_New: Using fallback camera.")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    # --- tunables ---
    DETECTION_INTERVAL = 12
    MIN_FACE_AREA = 20 * 20
    LOST_FRAMES_THRESHOLD = 6
    SMOOTH_ALPHA = 0.25
    SCALE_CHANGE_THRESHOLD = 0.6
    IOU_THRESHOLD = 0.35
    PRIMARY_CHECK_INTERVAL = 5.0
    SEND_INTERVAL = 0.08
    # New: minimum confidence (0.0 - 1.0) to treat detection as reliable
    MIN_CONFIDENCE = 0.60

    tracker = None
    tracking_active = False
    locked_face = None
    name = None
    last_speaker = None
    frame_idx = 0
    lost_frames = 0
    smoothed_center = None
    last_primary_check = time.time()
    last_send_time = 0.0

    print("Vision_Who_InFront_Look_At_New: Tracker ready. Press 'q' to stop.")
    try:
        while True:
            # Respect app shutdown signal
            if app_shutdown_event.is_set():
                print("Vision_Who_InFront_Look_At_New: shutdown event received, exiting.")
                break
            ret, frame = cap.read()
            # publish latest frame for other modules to consume
            try:
                _set_latest_frame(frame)
            except Exception:
                pass
            now = time.time()
            if not ret or frame is None or frame.size == 0:
                # camera failure → try reopen logic (same as before)
                try:
                    cap.release()
                except Exception:
                    pass
                if using_primary:
                    cap = open_camera(FALLBACK_CAM, timeout=2.0)
                    if cap is not None:
                        using_primary = False
                        print("Switched to fallback camera.")
                    else:
                        cap = open_camera(PRIMARY_CAM, timeout=2.0)
                        if cap is not None:
                            using_primary = True
                            print("Primary recovered.")
                        else:
                            time.sleep(0.2)
                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
                            if cap is None:
                                time.sleep(0.2)
                                continue
                            else:
                                using_primary = False
                                print("Using fallback camera.")
                else:
                    if now - last_primary_check > PRIMARY_CHECK_INTERVAL:
                        last_primary_check = now
                        pcap = open_camera(PRIMARY_CAM, timeout=1.5)
                        if pcap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = pcap
                            using_primary = True
                            print("Primary camera recovered — switched back.")
                        else:
                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
                            if cap is None:
                                time.sleep(0.2)
                                continue
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            h_img, w_img = frame.shape[:2]
            try:
                cv2.line(frame, (0, int(h_img/2)), (w_img, int(h_img/2)), (0, 255, 0), 1)
                cv2.line(frame, (int(w_img/2), 0), (int(w_img/2), h_img), (0, 255, 0), 1)
                cv2.circle(frame, (int(w_img/2), int(h_img/2)), 2, (0, 0, 255), -1)
            except Exception:
                pass
            try:
                current_speaker = get_speaker_func() or ""
            except Exception:
                current_speaker = ""

            # if speaker changed externally, reset tracking target
            if (current_speaker or "") != (last_speaker or ""):
                last_speaker = current_speaker
                tracker = None
                tracking_active = False
                locked_face = None
                name = None
                smoothed_center = None
                lost_frames = 0
                if current_speaker:
                    print(f"Vision: switching target to speaker: {current_speaker}")

            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
            detected_locations = []
            detected_names = []
            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
                try:
                    try:
                        detected_locations, detected_names = sfr.detect_known_faces(frame)
                    except Exception:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        detected_locations, detected_names = sfr.detect_known_faces(rgb)
                    # Normalize results: many libs return numpy arrays; convert to lists
                    if detected_locations is None:
                        detected_locations = []
                    elif isinstance(detected_locations, np.ndarray):
                        try:
                            detected_locations = detected_locations.tolist()
                        except Exception:
                            detected_locations = list(detected_locations)
                    else:
                        try:
                            detected_locations = list(detected_locations)
                        except Exception:
                            detected_locations = []

                    if detected_names is None:
                        detected_names = []
                    elif isinstance(detected_names, np.ndarray):
                        try:
                            detected_names = [str(n) for n in detected_names.tolist()]
                        except Exception:
                            detected_names = [str(n) for n in detected_names]
                    else:
                        try:
                            # keep original objects because they may contain confidences (tuples/lists)
                            detected_names = list(detected_names)
                        except Exception:
                            detected_names = []
                except Exception:
                    detected_locations, detected_names = [], []

            # Parse confidences and normalize detected entries to (bbox, name, conf)
            parsed_bboxes = []
            parsed_names = []
            parsed_confs = []
            for i, loc in enumerate(detected_locations):
                bb = face_loc_to_bbox(loc)
                # Extract confidence from loc if present (e.g., loc length >=5)
                conf = None
                try:
                    if hasattr(loc, "__len__") and len(loc) >= 5:
                        last = loc[4]
                        if isinstance(last, (float, int)):
                            conf = float(last)
                except Exception:
                    conf = None
                # Name/conf might be embedded in detected_names
                detected_name_obj = None
                try:
                    detected_name_obj = detected_names[i] if i < len(detected_names) else None
                except Exception:
                    detected_name_obj = None
                nm = None
                try:
                    if isinstance(detected_name_obj, (list, tuple)) and len(detected_name_obj) >= 2:
                        nm = str(detected_name_obj[0])
                        if conf is None:
                            try:
                                candidate_conf = float(detected_name_obj[1])
                                conf = candidate_conf
                            except Exception:
                                pass
                    else:
                        nm = str(detected_name_obj) if detected_name_obj is not None else None
                except Exception:
                    nm = None
                # fallback default confidence
                if conf is None:
                    conf = 1.0
                parsed_bboxes.append(bb)
                parsed_names.append(nm)
                parsed_confs.append(float(conf))

            # Filter out None bboxes but keep corresponding names/confs aligned
            filtered_bboxes = []
            filtered_names = []
            filtered_confs = []
            for bb, nm, cf in zip(parsed_bboxes, parsed_names, parsed_confs):
                if bb is None:
                    continue
                filtered_bboxes.append(bb)
                filtered_names.append(nm)
                filtered_confs.append(cf)

            target_bbox = None
            target_name = None

            # If there's no current speaker, try to pick one from face recognition results
            if not current_speaker:
                # prefer known name (not 'Unknown') with largest area AND with sufficient confidence
                best_idx = None
                best_score = -1
                for i, bb in enumerate(filtered_bboxes):
                    area = bb[2] * bb[3]
                    nm = filtered_names[i] if i < len(filtered_names) else None
                    cf = filtered_confs[i] if i < len(filtered_confs) else 0.0
                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
                    # require minimum confidence to treat as known
                    known_bonus = 100000 if (is_known and cf >= MIN_CONFIDENCE) else 0
                    score = known_bonus + (area if cf >= 0.0 else 0)
                    if score > best_score and area >= MIN_FACE_AREA:
                        best_score = score
                        best_idx = i
                if best_idx is not None:
                    # chosen is already a bbox (x,y,w,h) — do NOT re-run face_loc_to_bbox on it
                    chosen = filtered_bboxes[best_idx]
                    if chosen is not None:
                        target_bbox = chosen
                    try:
                        target_name = filtered_names[best_idx] if best_idx < len(filtered_names) else None
                    except Exception:
                        target_name = None
                    # If we have a clear real name (not 'Unknown') and sufficient confidence, update speaker
                    try:
                        if target_name and str(target_name).strip().lower() not in ("unknown", "") and filtered_confs[best_idx] >= MIN_CONFIDENCE:
                            _set_current_speaker(target_name)
                            last_speaker = target_name
                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name} (conf={filtered_confs[best_idx]:.2f})")
                        else:
                            if target_name is None:
                                target_name = "Unknown"
                    except Exception:
                        last_speaker = target_name
            # SAFE check: if previous speaker exists or detection later picks someone, fallback selection logic below
            if target_bbox is None and len(filtered_bboxes) > 0:
                # filtered_bboxes already contains bbox tuples (x,y,w,h)
                bboxes = filtered_bboxes.copy()
                # selection logic (speaker match, IOU, largest face) — now using confidences
                norm_speaker = normalize_name(current_speaker)
                best_idx = None
                if norm_speaker:
                    for i, nm in enumerate(filtered_names):
                        if nm and normalize_name(nm) == norm_speaker and filtered_confs[i] >= MIN_CONFIDENCE:
                            best_idx = i
                            break
                    if best_idx is None:
                        for i, nm in enumerate(filtered_names):
                            if nm and norm_speaker in normalize_name(nm) and filtered_confs[i] >= MIN_CONFIDENCE:
                                best_idx = i
                                break
                if best_idx is None and tracking_active and locked_face is not None:
                    best_iou = -1.0
                    for i, b in enumerate(bboxes):
                        if b is None:
                            continue
                        cur_iou = iou(locked_face, b)
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            best_idx = i
                    if best_iou < 0.05:
                        best_idx = None
                if best_idx is None:
                    best_area = 0
                    for i, b in enumerate(bboxes):
                        if b is None:
                            continue
                        area = b[2] * b[3]
                        # require minimal face area AND minimum confidence for selection
                        if area > best_area and area >= MIN_FACE_AREA and filtered_confs[i] >= MIN_CONFIDENCE:
                            best_area = area
                            best_idx = i
                if best_idx is not None and bboxes[best_idx] is not None:
                    target_bbox = bboxes[best_idx]
                    try:
                        target_name = filtered_names[best_idx] if best_idx < len(filtered_names) else None
                    except Exception:
                        target_name = None

            # --- tracker init / update / draw / send logic (mostly unchanged) ---
            if target_bbox is not None:
                if not tracking_active:
                    tr = create_csrt()
                    if tr is not None:
                        try:
                            ok = tr.init(frame, tuple(map(int, (target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))))
                            if ok:
                                tracker = tr
                                tracking_active = True
                                locked_face = tuple(map(int, target_bbox))
                                name = target_name or "Unknown"
                                lost_frames = 0
                                smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
                        except Exception:
                            tracker = None
                else:
                    if locked_face is not None:
                        fx, fy, fw, fh = target_bbox
                        detected_area = fw * fh
                        locked_area = max(1, locked_face[2] * locked_face[3])
                        scale_change = fabs(detected_area - locked_area) / float(locked_area)
                        current_iou = iou(locked_face, target_bbox)
                        # Only re-init tracker if detection is confident enough OR IoU dramatically changed
                        # Find corresponding confidence if possible
                        rebuild_due_to_confidence = True
                        try:
                            conf_for_target = None
                            for i, b in enumerate(filtered_bboxes):
                                if b is None:
                                    continue
                                if b[0] == fx and b[1] == fy and b[2] == fw and b[3] == fh:
                                    conf_for_target = filtered_confs[i]
                                    break
                            if conf_for_target is not None and conf_for_target < MIN_CONFIDENCE:
                                rebuild_due_to_confidence = False
                        except Exception:
                            rebuild_due_to_confidence = True

                        if (current_iou < IOU_THRESHOLD or scale_change > SCALE_CHANGE_THRESHOLD) and rebuild_due_to_confidence:
                            tr = create_csrt()
                            if tr is not None:
                                try:
                                    ok = tr.init(frame, (int(fx), int(fy), int(fw), int(fh)))
                                    if ok:
                                        tracker = tr
                                        locked_face = (int(fx), int(fy), int(fw), int(fh))
                                        name = target_name or name or "Unknown"
                                        lost_frames = 0
                                        smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
                                except Exception:
                                    pass

            if tracking_active and tracker is not None:
                try:
                    success, bbox = tracker.update(frame)
                except Exception:
                    success = False
                    bbox = None
                if success and bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
                    w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
                    locked_face = (x, y, w, h)
                    lost_frames = 0
                    cx = x + w // 2; cy = y + h // 2
                    if smoothed_center is None:
                        smoothed_center = (cx, cy)
                    else:
                        # fixed smoothing math
                        sx = int(round(smoothed_center[0] * (1.0 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA))
                        sy = int(round(smoothed_center[1] * (1.0 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA))
                        smoothed_center = (sx, sy)
                else:
                    lost_frames += 1
                    if lost_frames > LOST_FRAMES_THRESHOLD:
                        tracking_active = False
                        tracker = None
                        locked_face = None
                        smoothed_center = None
                        name = None
                        lost_frames = 0

            if locked_face:
                x, y, w, h = locked_face
                try:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    if name:
                        cv2.putText(frame, str(name), (x, max(0, y - 8)),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                except Exception:
                    pass
                if smoothed_center:
                    send_x, send_y = smoothed_center
                else:
                    send_x = x + w // 2
                    send_y = y + h // 2
                if arduino is not None:
                    if time.time() - last_send_time > SEND_INTERVAL:
                        last_send_time = time.time()
                        data = f"X{int(send_x)}Y{int(send_y)}Z"
                        try:
                            arduino.send_arduino(data)
                        except Exception:
                            pass

            title = f'Face+Name Tracking ({ "CHEST" if using_primary else "LEFT" })'
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Vision_Who_InFront_Look_At_New: 'q' pressed, stopping.")
                break
            if (not using_primary) and (time.time() - last_primary_check > PRIMARY_CHECK_INTERVAL):
                last_primary_check = time.time()
                pcap = open_camera(PRIMARY_CAM, timeout=1.0)
                if pcap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = pcap
                    using_primary = True
                    print("Primary camera recovered — switched back to CHEST.")
            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Vision_Who_InFront_Look_At_New: Face Tracking Stopped.")













##def Vision_Who_InFront_Look_At_New(get_speaker_func,
##                                   sfr=None,
##                                   arduino=None,
##                                   PRIMARY_CAM=None,
##                                   FALLBACK_CAM=None):
##    import cv2
##    import time
##    import numpy as np
##    from math import fabs
##    import re
##    try:
##        import Alfred_config
##        if PRIMARY_CAM is None:
##            PRIMARY_CAM = Alfred_config.CHEST_CAMERA_INPUT
##        if FALLBACK_CAM is None:
##            FALLBACK_CAM = Alfred_config.LEFT_EYE_CAMERA_INPUT
##    except Exception:
##        pass
##    if sfr is None:
##        sfr = globals().get('sfr', None)
##    if arduino is None:
##        arduino = globals().get('arduino', None)
##    def open_camera(src, timeout=2.0):
##        if src is None:
##            return None
##        try:
##            cap = cv2.VideoCapture(src)
##            t0 = time.time()
##            while time.time() - t0 < timeout:
##                if cap.isOpened():
##                    ret, frame = cap.read()
##                    if ret and frame is not None and frame.size > 0:
##                        return cap
##                time.sleep(0.05)
##            if cap.isOpened():
##                return cap
##            try:
##                cap.release()
##            except Exception:
##                pass
##        except Exception:
##            pass
##        return None
##    def create_csrt():
##        tr = None
##        try:
##            tr = cv2.legacy.TrackerCSRT_create()
##        except Exception:
##            try:
##                tr = cv2.TrackerCSRT_create()
##            except Exception:
##                tr = None
##        return tr
##    def normalize_name(s):
##        if not s:
##            return ""
##        s = str(s).lower()
##        s = re.sub(r'[^a-z0-9]', '', s)
##        return s
##    def iou(boxA, boxB):
##        (xA, yA, wA, hA) = boxA
##        (xB, yB, wB, hB) = boxB
##        x1 = max(xA, xB)
##        y1 = max(yA, yB)
##        x2 = min(xA + wA, xB + wB)
##        y2 = min(yA + hA, yB + hB)
##        interW = max(0, x2 - x1)
##        interH = max(0, y2 - y1)
##        interArea = interW * interH
##        boxAArea = wA * hA
##        boxBArea = wB * hB
##        union = boxAArea + boxBArea - interArea
##        return interArea / (union + 1e-8)
##    def face_loc_to_bbox(face_loc):
##        try:
##            # Common face_loc format: (top, right, bottom, left)
##            top, right, bottom, left = face_loc[:4]
##            x = int(left); y = int(top)
##            w = int(max(1, right - left)); h = int(max(1, bottom - top))
##            return (x, y, w, h)
##        except Exception:
##            try:
##                x, y, w, h = face_loc[:4]
##                return (int(x), int(y), int(w), int(h))
##            except Exception:
##                return None
##
##    cap = open_camera(PRIMARY_CAM, timeout=2.0)
##    using_primary = True
##    if cap is None:
##        cap = open_camera(FALLBACK_CAM, timeout=2.0)
##        using_primary = False
##        if cap is None:
##            print("Vision_Who_InFront_Look_At_New: No camera available (primary & fallback failed). Aborting.")
##            return
##        else:
##            print("Vision_Who_InFront_Look_At_New: Using fallback camera.")
##    try:
##        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##    except Exception:
##        pass
##
##    # --- tunables ---
##    DETECTION_INTERVAL = 12
##    MIN_FACE_AREA = 20 * 20
##    LOST_FRAMES_THRESHOLD = 6
##    SMOOTH_ALPHA = 0.25
##    SCALE_CHANGE_THRESHOLD = 0.6
##    IOU_THRESHOLD = 0.35
##    PRIMARY_CHECK_INTERVAL = 5.0
##    SEND_INTERVAL = 0.08
##    # New: minimum confidence (0.0 - 1.0) to treat detection as reliable
##    MIN_CONFIDENCE = 0.60
##
##    tracker = None
##    tracking_active = False
##    locked_face = None
##    name = None
##    last_speaker = None
##    frame_idx = 0
##    lost_frames = 0
##    smoothed_center = None
##    last_primary_check = time.time()
##    last_send_time = 0.0
##
##    print("Vision_Who_InFront_Look_At_New: Tracker ready. Press 'q' to stop.")
##    try:
##        while True:
##            # Respect app shutdown signal
##            if app_shutdown_event.is_set():
##                print("Vision_Who_InFront_Look_At_New: shutdown event received, exiting.")
##                break
##            ret, frame = cap.read()
##            # publish latest frame for other modules to consume
##            try:
##                _set_latest_frame(frame)
##            except Exception:
##                pass
##            now = time.time()
##            if not ret or frame is None or frame.size == 0:
##                # camera failure → try reopen logic (same as before)
##                try:
##                    cap.release()
##                except Exception:
##                    pass
##                if using_primary:
##                    cap = open_camera(FALLBACK_CAM, timeout=2.0)
##                    if cap is not None:
##                        using_primary = False
##                        print("Switched to fallback camera.")
##                    else:
##                        cap = open_camera(PRIMARY_CAM, timeout=2.0)
##                        if cap is not None:
##                            using_primary = True
##                            print("Primary recovered.")
##                        else:
##                            time.sleep(0.2)
##                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
##                            if cap is None:
##                                time.sleep(0.2)
##                                continue
##                            else:
##                                using_primary = False
##                                print("Using fallback camera.")
##                else:
##                    if now - last_primary_check > PRIMARY_CHECK_INTERVAL:
##                        last_primary_check = now
##                        pcap = open_camera(PRIMARY_CAM, timeout=1.5)
##                        if pcap is not None:
##                            try:
##                                cap.release()
##                            except Exception:
##                                pass
##                            cap = pcap
##                            using_primary = True
##                            print("Primary camera recovered — switched back.")
##                        else:
##                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
##                            if cap is None:
##                                time.sleep(0.2)
##                                continue
##                ret, frame = cap.read()
##                if not ret:
##                    time.sleep(0.05)
##                    continue
##
##            h_img, w_img = frame.shape[:2]
##            try:
##                cv2.line(frame, (0, int(h_img/2)), (w_img, int(h_img/2)), (0, 255, 0), 1)
##                cv2.line(frame, (int(w_img/2), 0), (int(w_img/2), h_img), (0, 255, 0), 1)
##                cv2.circle(frame, (int(w_img/2), int(h_img/2)), 2, (0, 0, 255), -1)
##            except Exception:
##                pass
##            try:
##                current_speaker = get_speaker_func() or ""
##            except Exception:
##                current_speaker = ""
##
##            # if speaker changed externally, reset tracking target
##            if (current_speaker or "") != (last_speaker or ""):
##                last_speaker = current_speaker
##                tracker = None
##                tracking_active = False
##                locked_face = None
##                name = None
##                smoothed_center = None
##                lost_frames = 0
##                if current_speaker:
##                    print(f"Vision: switching target to speaker: {current_speaker}")
##
##            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
##            detected_locations = []
##            detected_names = []
##            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
##                try:
##                    try:
##                        detected_locations, detected_names = sfr.detect_known_faces(frame)
##                    except Exception:
##                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                        detected_locations, detected_names = sfr.detect_known_faces(rgb)
##                    # Normalize results: many libs return numpy arrays; convert to lists
##                    if detected_locations is None:
##                        detected_locations = []
##                    elif isinstance(detected_locations, np.ndarray):
##                        try:
##                            detected_locations = detected_locations.tolist()
##                        except Exception:
##                            detected_locations = list(detected_locations)
##                    else:
##                        try:
##                            detected_locations = list(detected_locations)
##                        except Exception:
##                            detected_locations = []
##
##                    if detected_names is None:
##                        detected_names = []
##                    elif isinstance(detected_names, np.ndarray):
##                        try:
##                            detected_names = [str(n) for n in detected_names.tolist()]
##                        except Exception:
##                            detected_names = [str(n) for n in detected_names]
##                    else:
##                        try:
##                            # keep original objects because they may contain confidences (tuples/lists)
##                            detected_names = list(detected_names)
##                        except Exception:
##                            detected_names = []
##                except Exception:
##                    detected_locations, detected_names = [], []
##
##            # Parse confidences and normalize detected entries to (bbox, name, conf)
##            parsed_bboxes = []
##            parsed_names = []
##            parsed_confs = []
##            for i, loc in enumerate(detected_locations):
##                bb = face_loc_to_bbox(loc)
##                # Extract confidence from loc if present (e.g., loc length >=5)
##                conf = None
##                try:
##                    if hasattr(loc, "__len__") and len(loc) >= 5:
##                        # last element might be confidence
##                        last = loc[4]
##                        if isinstance(last, (float, int)):
##                            conf = float(last)
##                except Exception:
##                    conf = None
##                # Name/conf might be embedded in detected_names
##                detected_name_obj = None
##                try:
##                    detected_name_obj = detected_names[i] if i < len(detected_names) else None
##                except Exception:
##                    detected_name_obj = None
##                nm = None
##                try:
##                    if isinstance(detected_name_obj, (list, tuple)) and len(detected_name_obj) >= 2:
##                        nm = str(detected_name_obj[0])
##                        if conf is None:
##                            try:
##                                candidate_conf = float(detected_name_obj[1])
##                                conf = candidate_conf
##                            except Exception:
##                                pass
##                    else:
##                        nm = str(detected_name_obj) if detected_name_obj is not None else None
##                except Exception:
##                    nm = None
##                # fallback default confidence
##                if conf is None:
##                    conf = 1.0
##                parsed_bboxes.append(bb)
##                parsed_names.append(nm)
##                parsed_confs.append(float(conf))
##
##            # Filter out None bboxes but keep corresponding names/confs aligned
##            filtered_bboxes = []
##            filtered_names = []
##            filtered_confs = []
##            for bb, nm, cf in zip(parsed_bboxes, parsed_names, parsed_confs):
##                if bb is None:
##                    continue
##                filtered_bboxes.append(bb)
##                filtered_names.append(nm)
##                filtered_confs.append(cf)
##
##            target_bbox = None
##            target_name = None
##
##            # If there's no current speaker, try to pick one from face recognition results
##            if not current_speaker:
##                # prefer known name (not 'Unknown') with largest area AND with sufficient confidence
##                best_idx = None
##                best_score = -1
##                for i, bb in enumerate(filtered_bboxes):
##                    area = bb[2] * bb[3]
##                    nm = filtered_names[i] if i < len(filtered_names) else None
##                    cf = filtered_confs[i] if i < len(filtered_confs) else 0.0
##                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
##                    # require minimum confidence to treat as known
##                    known_bonus = 100000 if (is_known and cf >= MIN_CONFIDENCE) else 0
##                    score = known_bonus + (area if cf >= 0.0 else 0)
##                    if score > best_score and area >= MIN_FACE_AREA:
##                        best_score = score
##                        best_idx = i
##                if best_idx is not None:
##                    chosen = filtered_bboxes[best_idx]
##                    bb = face_loc_to_bbox(chosen) if isinstance(chosen, (list, tuple)) else chosen
##                    if bb is not None:
##                        target_bbox = bb
##                    try:
##                        target_name = filtered_names[best_idx] if best_idx < len(filtered_names) else None
##                    except Exception:
##                        target_name = None
##                    # If we have a clear real name (not 'Unknown') and sufficient confidence, update speaker
##                    try:
##                        if target_name and str(target_name).strip().lower() not in ("unknown", "") and filtered_confs[best_idx] >= MIN_CONFIDENCE:
##                            _set_current_speaker(target_name)
##                            last_speaker = target_name
##                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name} (conf={filtered_confs[best_idx]:.2f})")
##                        else:
##                            if target_name is None:
##                                target_name = "Unknown"
##                    except Exception:
##                        last_speaker = target_name
##            # SAFE check: if previous speaker exists or detection later picks someone, fallback selection logic below
##            if target_bbox is None and len(filtered_bboxes) > 0:
##                bboxes = []
##                for loc in filtered_bboxes:
##                    b = face_loc_to_bbox(loc) if not isinstance(loc, tuple) or len(loc) != 4 else loc
##                    bboxes.append(b)
##                # selection logic (speaker match, IOU, largest face) — now using confidences
##                norm_speaker = normalize_name(current_speaker)
##                best_idx = None
##                if norm_speaker:
##                    for i, nm in enumerate(filtered_names):
##                        if nm and normalize_name(nm) == norm_speaker and filtered_confs[i] >= MIN_CONFIDENCE:
##                            best_idx = i
##                            break
##                    if best_idx is None:
##                        for i, nm in enumerate(filtered_names):
##                            if nm and norm_speaker in normalize_name(nm) and filtered_confs[i] >= MIN_CONFIDENCE:
##                                best_idx = i
##                                break
##                if best_idx is None and tracking_active and locked_face is not None:
##                    best_iou = -1.0
##                    for i, b in enumerate(bboxes):
##                        if b is None:
##                            continue
##                        cur_iou = iou(locked_face, b)
##                        if cur_iou > best_iou:
##                            best_iou = cur_iou
##                            best_idx = i
##                    if best_iou < 0.05:
##                        best_idx = None
##                if best_idx is None:
##                    best_area = 0
##                    for i, b in enumerate(bboxes):
##                        if b is None:
##                            continue
##                        area = b[2] * b[3]
##                        # require minimal face area AND minimum confidence for selection
##                        if area > best_area and area >= MIN_FACE_AREA and filtered_confs[i] >= MIN_CONFIDENCE:
##                            best_area = area
##                            best_idx = i
##                if best_idx is not None and bboxes[best_idx] is not None:
##                    target_bbox = bboxes[best_idx]
##                    try:
##                        target_name = filtered_names[best_idx] if best_idx < len(filtered_names) else None
##                    except Exception:
##                        target_name = None
##
##            # --- tracker init / update / draw / send logic (mostly unchanged) ---
##            if target_bbox is not None:
##                if not tracking_active:
##                    tr = create_csrt()
##                    if tr is not None:
##                        try:
##                            ok = tr.init(frame, tuple(map(int, (target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))))
##                            if ok:
##                                tracker = tr
##                                tracking_active = True
##                                locked_face = tuple(map(int, target_bbox))
##                                name = target_name or "Unknown"
##                                lost_frames = 0
##                                smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
##                        except Exception:
##                            tracker = None
##                else:
##                    if locked_face is not None:
##                        fx, fy, fw, fh = target_bbox
##                        detected_area = fw * fh
##                        locked_area = max(1, locked_face[2] * locked_face[3])
##                        scale_change = fabs(detected_area - locked_area) / float(locked_area)
##                        current_iou = iou(locked_face, target_bbox)
##                        # Only re-init tracker if detection is confident enough OR IoU dramatically changed
##                        # Find corresponding confidence if possible
##                        rebuild_due_to_confidence = True
##                        try:
##                            # try to locate index of target_bbox in filtered_bboxes to get conf
##                            conf_for_target = None
##                            for i, b in enumerate(filtered_bboxes):
##                                if b is None:
##                                    continue
##                                if b[0] == fx and b[1] == fy and b[2] == fw and b[3] == fh:
##                                    conf_for_target = filtered_confs[i]
##                                    break
##                            if conf_for_target is not None and conf_for_target < MIN_CONFIDENCE:
##                                rebuild_due_to_confidence = False
##                        except Exception:
##                            rebuild_due_to_confidence = True
##
##                        if (current_iou < IOU_THRESHOLD or scale_change > SCALE_CHANGE_THRESHOLD) and rebuild_due_to_confidence:
##                            tr = create_csrt()
##                            if tr is not None:
##                                try:
##                                    ok = tr.init(frame, (int(fx), int(fy), int(fw), int(fh)))
##                                    if ok:
##                                        tracker = tr
##                                        locked_face = (int(fx), int(fy), int(fw), int(fh))
##                                        name = target_name or name or "Unknown"
##                                        lost_frames = 0
##                                        smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
##                                except Exception:
##                                    pass
##
##            if tracking_active and tracker is not None:
##                try:
##                    success, bbox = tracker.update(frame)
##                except Exception:
##                    success = False
##                    bbox = None
##                if success and bbox is not None:
##                    x, y, w, h = [int(v) for v in bbox]
##                    x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
##                    w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
##                    locked_face = (x, y, w, h)
##                    lost_frames = 0
##                    cx = x + w // 2; cy = y + h // 2
##                    if smoothed_center is None:
##                        smoothed_center = (cx, cy)
##                    else:
##                        # fixed smoothing math (was corrupted previously)
##                        sx = int(round(smoothed_center[0] * (1.0 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA))
##                        sy = int(round(smoothed_center[1] * (1.0 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA))
##                        smoothed_center = (sx, sy)
##                else:
##                    lost_frames += 1
##                    if lost_frames > LOST_FRAMES_THRESHOLD:
##                        tracking_active = False
##                        tracker = None
##                        locked_face = None
##                        smoothed_center = None
##                        name = None
##                        lost_frames = 0
##
##            if locked_face:
##                x, y, w, h = locked_face
##                try:
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
##                    if name:
##                        cv2.putText(frame, str(name), (x, max(0, y - 8)),
##                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
##                except Exception:
##                    pass
##                if smoothed_center:
##                    send_x, send_y = smoothed_center
##                else:
##                    send_x = x + w // 2
##                    send_y = y + h // 2
##                if arduino is not None:
##                    if time.time() - last_send_time > SEND_INTERVAL:
##                        last_send_time = time.time()
##                        data = f"X{int(send_x)}Y{int(send_y)}Z"
##                        try:
##                            arduino.send_arduino(data)
##                        except Exception:
##                            pass
##
##            title = f'Face+Name Tracking ({ "CHEST" if using_primary else "LEFT" })'
##            cv2.imshow(title, frame)
##            key = cv2.waitKey(1) & 0xFF
##            if key == ord('q'):
##                print("Vision_Who_InFront_Look_At_New: 'q' pressed, stopping.")
##                break
##            if (not using_primary) and (time.time() - last_primary_check > PRIMARY_CHECK_INTERVAL):
##                last_primary_check = time.time()
##                pcap = open_camera(PRIMARY_CAM, timeout=1.0)
##                if pcap is not None:
##                    try:
##                        cap.release()
##                    except Exception:
##                        pass
##                    cap = pcap
##                    using_primary = True
##                    print("Primary camera recovered — switched back to CHEST.")
##            frame_idx += 1
##    finally:
##        try:
##            cap.release()
##        except Exception:
##            pass
##        cv2.destroyAllWindows()
##        print("Vision_Who_InFront_Look_At_New: Face Tracking Stopped.")










##def Vision_Who_InFront_Look_At_New(get_speaker_func,
##                                   sfr=None,
##                                   arduino=None,
##                                   PRIMARY_CAM=None,
##                                   FALLBACK_CAM=None):
##    import cv2
##    import time
##    import numpy as np
##    from math import fabs
##    import re
##
##    try:
##        import Alfred_config
##        if PRIMARY_CAM is None:
##            PRIMARY_CAM = Alfred_config.CHEST_CAMERA_INPUT
##        if FALLBACK_CAM is None:
##            FALLBACK_CAM = Alfred_config.LEFT_EYE_CAMERA_INPUT
##    except Exception:
##        pass
##
##    if sfr is None:
##        sfr = globals().get('sfr', None)
##    if arduino is None:
##        arduino = globals().get('arduino', None)
##
##    def open_camera(src, timeout=2.0):
##        if src is None:
##            return None
##        try:
##            cap = cv2.VideoCapture(src)
##            t0 = time.time()
##            while time.time() - t0 < timeout:
##                if cap.isOpened():
##                    ret, frame = cap.read()
##                    if ret and frame is not None and frame.size > 0:
##                        return cap
##                time.sleep(0.05)
##            if cap.isOpened():
##                return cap
##            try:
##                cap.release()
##            except Exception:
##                pass
##        except Exception:
##            pass
##        return None
##
##    def create_csrt():
##        tr = None
##        try:
##            tr = cv2.legacy.TrackerCSRT_create()
##        except Exception:
##            try:
##                tr = cv2.TrackerCSRT_create()
##            except Exception:
##                tr = None
##        return tr
##
##    def normalize_name(s):
##        if not s:
##            return ""
##        s = str(s).lower()
##        s = re.sub(r'[^a-z0-9]', '', s)
##        return s
##
##    def iou(boxA, boxB):
##        (xA, yA, wA, hA) = boxA
##        (xB, yB, wB, hB) = boxB
##        x1 = max(xA, xB)
##        y1 = max(yA, yB)
##        x2 = min(xA + wA, xB + wB)
##        y2 = min(yA + hA, yB + hB)
##        interW = max(0, x2 - x1)
##        interH = max(0, y2 - y1)
##        interArea = interW * interH
##        boxAArea = wA * hA
##        boxBArea = wB * hB
##        union = boxAArea + boxBArea - interArea
##        return interArea / (union + 1e-8)
##
##    def face_loc_to_bbox(face_loc):
##        try:
##            top, right, bottom, left = face_loc
##            x = int(left); y = int(top)
##            w = int(max(1, right - left)); h = int(max(1, bottom - top))
##            return (x, y, w, h)
##        except Exception:
##            try:
##                x, y, w, h = face_loc
##                return (int(x), int(y), int(w), int(h))
##            except Exception:
##                return None
##
##    cap = open_camera(PRIMARY_CAM, timeout=2.0)
##    using_primary = True
##    if cap is None:
##        cap = open_camera(FALLBACK_CAM, timeout=2.0)
##        using_primary = False
##        if cap is None:
##            print("Vision_Who_InFront_Look_At_New: No camera available (primary & fallback failed). Aborting.")
##            return
##        else:
##            print("Vision_Who_InFront_Look_At_New: Using fallback camera.")
##
##    try:
##        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
##        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
##    except Exception:
##        pass
##
##    DETECTION_INTERVAL = 12
##    MIN_FACE_AREA = 20 * 20
##    LOST_FRAMES_THRESHOLD = 6
##    SMOOTH_ALPHA = 0.25
##    SCALE_CHANGE_THRESHOLD = 0.6
##    IOU_THRESHOLD = 0.35
##    PRIMARY_CHECK_INTERVAL = 5.0
##    SEND_INTERVAL = 0.08
##
##    tracker = None
##    tracking_active = False
##    locked_face = None
##    name = None
##    last_speaker = None
##    frame_idx = 0
##    lost_frames = 0
##    smoothed_center = None
##    last_primary_check = time.time()
##    last_send_time = 0.0
##
##    print("Vision_Who_InFront_Look_At_New: Tracker ready. Press 'q' to stop.")
##
##    try:
##        while True:
##
##            # Respect app shutdown signal
##            if app_shutdown_event.is_set():
##                print("Vision_Who_InFront_Look_At_New: shutdown event received, exiting.")
##                break
##            
##            ret, frame = cap.read()
##
##            # publish latest frame for other modules to consume
##            try:
##                _set_latest_frame(frame)
##            except Exception:
##                pass
##
##            
##            now = time.time()
##            if not ret or frame is None or frame.size == 0:
##                # camera failure → try reopen logic (same as before)
##                try:
##                    cap.release()
##                except Exception:
##                    pass
##                if using_primary:
##                    cap = open_camera(FALLBACK_CAM, timeout=2.0)
##                    if cap is not None:
##                        using_primary = False
##                        print("Switched to fallback camera.")
##                    else:
##                        cap = open_camera(PRIMARY_CAM, timeout=2.0)
##                        if cap is not None:
##                            using_primary = True
##                            print("Primary recovered.")
##                        else:
##                            time.sleep(0.2)
##                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
##                            if cap is None:
##                                time.sleep(0.2)
##                                continue
##                            else:
##                                using_primary = False
##                                print("Using fallback camera.")
##                else:
##                    if now - last_primary_check > PRIMARY_CHECK_INTERVAL:
##                        last_primary_check = now
##                        pcap = open_camera(PRIMARY_CAM, timeout=1.5)
##                        if pcap is not None:
##                            try:
##                                cap.release()
##                            except Exception:
##                                pass
##                            cap = pcap
##                            using_primary = True
##                            print("Primary camera recovered — switched back.")
##                        else:
##                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
##                            if cap is None:
##                                time.sleep(0.2)
##                                continue
##
##                ret, frame = cap.read()
##                if not ret:
##                    time.sleep(0.05)
##                    continue
##
##            h_img, w_img = frame.shape[:2]
##
##            try:
##                cv2.line(frame, (0, int(h_img/2)), (w_img, int(h_img/2)), (0, 255, 0), 1)
##                cv2.line(frame, (int(w_img/2), 0), (int(w_img/2), h_img), (0, 255, 0), 1)
##                cv2.circle(frame, (int(w_img/2), int(h_img/2)), 2, (0, 0, 255), -1)
##            except Exception:
##                pass
##
##            try:
##                current_speaker = get_speaker_func() or ""
##            except Exception:
##                current_speaker = ""
##
##            # if speaker changed externally, reset tracking target
##            if (current_speaker or "") != (last_speaker or ""):
##                last_speaker = current_speaker
##                tracker = None
##                tracking_active = False
##                locked_face = None
##                name = None
##                smoothed_center = None
##                lost_frames = 0
##                if current_speaker:
##                    print(f"Vision: switching target to speaker: {current_speaker}")
##
##            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
##            detected_locations = []
##            detected_names = []
##            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
##                try:
##                    try:
##                        detected_locations, detected_names = sfr.detect_known_faces(frame)
##                    except Exception:
##                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                        detected_locations, detected_names = sfr.detect_known_faces(rgb)
##                    # Normalize results: many libs return numpy arrays; convert to lists
##                    if detected_locations is None:
##                        detected_locations = []
##                    elif isinstance(detected_locations, np.ndarray):
##                        try:
##                            detected_locations = detected_locations.tolist()
##                        except Exception:
##                            detected_locations = list(detected_locations)
##                    else:
##                        try:
##                            detected_locations = list(detected_locations)
##                        except Exception:
##                            detected_locations = []
##
##                    if detected_names is None:
##                        detected_names = []
##                    elif isinstance(detected_names, np.ndarray):
##                        try:
##                            detected_names = [str(n) for n in detected_names.tolist()]
##                        except Exception:
##                            detected_names = [str(n) for n in detected_names]
##                    else:
##                        try:
##                            detected_names = [str(n) for n in detected_names]
##                        except Exception:
##                            detected_names = []
##                except Exception:
##                    detected_locations, detected_names = [], []
##
##            target_bbox = None
##            target_name = None
##
##            # If there's no current speaker, try to pick one from face recognition results
##            if not current_speaker:
##                # prefer known name (not 'Unknown') with largest area
##                best_idx = None
##                best_area = 0
##                for i, loc in enumerate(detected_locations):
##                    bb = face_loc_to_bbox(loc)
##                    if bb is None:
##                        continue
##                    area = bb[2] * bb[3]
##                    nm = None
##                    try:
##                        nm = detected_names[i] if i < len(detected_names) else None
##                    except Exception:
##                        nm = None
##                    # prefer known names over Unknown, and prefer larger area
##                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
##                    score = (100000 if is_known else 0) + area
##                    if score > best_area:
##                        best_area = score
##                        best_idx = i
##
##                if best_idx is not None:
##                    # choose this as target (and if it's a known person, update speaker)
##                    chosen = detected_locations[best_idx]
##                    bb = face_loc_to_bbox(chosen)
##                    if bb is not None:
##                        target_bbox = bb
##                    try:
##                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
##                    except Exception:
##                        target_name = None
##
##                    # If we have a clear real name (not 'Unknown'), update module speaker
##                    if target_name and str(target_name).strip().lower() not in ("unknown", ""):
##                        try:
##                            # module-level setter exists in this file: _set_current_speaker
##                            _set_current_speaker(target_name)
##                            last_speaker = target_name
##                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name}")
##                        except Exception:
##                            # if setter not available, just set last_speaker locally
##                            last_speaker = target_name
##                    else:
##                        # keep name as Unknown (but we'll still track)
##                        if target_name is None:
##                            target_name = "Unknown"
##
##            # SAFE check: if previous speaker exists or detection later picks someone, fallback selection logic below
##            if target_bbox is None and len(detected_locations) > 0:
##                bboxes = []
##                for loc in detected_locations:
##                    b = face_loc_to_bbox(loc)
##                    bboxes.append(b)
##
##                # rest of selection logic (speaker match, IOU, largest face)...
##                norm_speaker = normalize_name(current_speaker)
##                best_idx = None
##                if norm_speaker:
##                    for i, nm in enumerate(detected_names):
##                        if nm and normalize_name(nm) == norm_speaker:
##                            best_idx = i
##                            break
##                    if best_idx is None:
##                        for i, nm in enumerate(detected_names):
##                            if nm and norm_speaker in normalize_name(nm):
##                                best_idx = i
##                                break
##
##                if best_idx is None and tracking_active and locked_face is not None:
##                    best_iou = -1.0
##                    for i, b in enumerate(bboxes):
##                        if b is None:
##                            continue
##                        cur_iou = iou(locked_face, b)
##                        if cur_iou > best_iou:
##                            best_iou = cur_iou
##                            best_idx = i
##                    if best_iou < 0.05:
##                        best_idx = None
##
##                if best_idx is None:
##                    best_area = 0
##                    for i, b in enumerate(bboxes):
##                        if b is None:
##                            continue
##                        area = b[2] * b[3]
##                        if area > best_area and area >= MIN_FACE_AREA:
##                            best_area = area
##                            best_idx = i
##
##                if best_idx is not None and bboxes[best_idx] is not None:
##                    target_bbox = bboxes[best_idx]
##                    try:
##                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
##                    except Exception:
##                        target_name = None
##
##            # --- then the existing tracker init / update / draw / send logic ---
##            if target_bbox is not None:
##                if not tracking_active:
##                    tr = create_csrt()
##                    if tr is not None:
##                        try:
##                            ok = tr.init(frame, tuple(map(int, (target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]))))
##                            if ok:
##                                tracker = tr
##                                tracking_active = True
##                                locked_face = tuple(map(int, target_bbox))
##                                name = target_name or "Unknown"
##                                lost_frames = 0
##                                smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
##                        except Exception:
##                            tracker = None
##                else:
##                    if locked_face is not None:
##                        fx, fy, fw, fh = target_bbox
##                        detected_area = fw * fh
##                        locked_area = max(1, locked_face[2] * locked_face[3])
##                        scale_change = fabs(detected_area - locked_area) / float(locked_area)
##                        current_iou = iou(locked_face, target_bbox)
##                        if current_iou < IOU_THRESHOLD or scale_change > SCALE_CHANGE_THRESHOLD:
##                            tr = create_csrt()
##                            if tr is not None:
##                                try:
##                                    ok = tr.init(frame, (int(fx), int(fy), int(fw), int(fh)))
##                                    if ok:
##                                        tracker = tr
##                                        locked_face = (int(fx), int(fy), int(fw), int(fh))
##                                        name = target_name or name or "Unknown"
##                                        lost_frames = 0
##                                        smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
##                                except Exception:
##                                    pass
##
##            if tracking_active and tracker is not None:
##                try:
##                    success, bbox = tracker.update(frame)
##                except Exception:
##                    success = False
##                    bbox = None
##
##                if success and bbox is not None:
##                    x, y, w, h = [int(v) for v in bbox]
##                    x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
##                    w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
##                    locked_face = (x, y, w, h)
##                    lost_frames = 0
##                    cx = x + w // 2; cy = y + h // 2
##                    if smoothed_center is None:
##                        smoothed_center = (cx, cy)
##                    else:
##                        sx = int(round(smoothed_center[0] * (1.0 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA))
##                        sy = int(round(smoothed_center[1] * (1.0 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA))
##                        smoothed_center = (sx, sy)
##                else:
##                    lost_frames += 1
##                    if lost_frames > LOST_FRAMES_THRESHOLD:
##                        tracking_active = False
##                        tracker = None
##                        locked_face = None
##                        smoothed_center = None
##                        name = None
##                        lost_frames = 0
##
##            if locked_face:
##                x, y, w, h = locked_face
##                try:
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
##                    if name:
##                        cv2.putText(frame, str(name), (x, max(0, y - 8)),
##                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
##                except Exception:
##                    pass
##
##                if smoothed_center:
##                    send_x, send_y = smoothed_center
##                else:
##                    send_x = x + w // 2
##                    send_y = y + h // 2
##
##                if arduino is not None:
##                    if time.time() - last_send_time > SEND_INTERVAL:
##                        last_send_time = time.time()
##                        data = f"X{int(send_x)}Y{int(send_y)}Z"
##                        try:
##                            arduino.send_arduino(data)
##                        except Exception:
##                            pass
##
##            title = f'Face+Name Tracking ({ "CHEST" if using_primary else "LEFT" })'
##            cv2.imshow(title, frame)
##            key = cv2.waitKey(1) & 0xFF
##            if key == ord('q'):
##                print("Vision_Who_InFront_Look_At_New: 'q' pressed, stopping.")
##                break
##
##            if (not using_primary) and (time.time() - last_primary_check > PRIMARY_CHECK_INTERVAL):
##                last_primary_check = time.time()
##                pcap = open_camera(PRIMARY_CAM, timeout=1.0)
##                if pcap is not None:
##                    try:
##                        cap.release()
##                    except Exception:
##                        pass
##                    cap = pcap
##                    using_primary = True
##                    print("Primary camera recovered — switched back to CHEST.")
##
##            frame_idx += 1
##
##    finally:
##        try:
##            cap.release()
##        except Exception:
##            pass
##        cv2.destroyAllWindows()
##        print("Vision_Who_InFront_Look_At_New: Face Tracking Stopped.")


class VisionFaceRecognitionTrackingModule:
    
    def __init__(self):

        self.camera = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
        self.camera.set(3, 640)  # Width
        self.camera.set(4, 480)  # Height

        self.Coord_Top_Left_X = 0
        self.Coord_Top_Left_Y = 480

        self.Coord_Middel_Left_X = 0  #   200
        self.Coord_Middel_Left_Y = 240  #   110

        self.Coord_Bottom_Left_X = 0
        self.Coord_Bottom_Left_Y = 0

        self.Coord_Top_Middel_X = 320
        self.Coord_Top_Middel_Y = 480

        self.Coord_Middel_Middel_X = 320
        self.Coord_Middel_Middel_Y = 240

        self.Coord_Bottom_Middel_X = 320
        self.Coord_Bottom_Middel_Y = 0

        self.Coord_Top_Right_X = 640
        self.Coord_Top_Right_Y = 480

        self.Coord_Middel_Right_X = 640
        self.Coord_Middel_Right_Y = 240

        self.Coord_Bottom_Right_X = 640
        self.Coord_Bottom_Right_Y = 0


    def detect_faces(self):
        """Detects faces and returns recognized names."""
        ret, frame = self.camera.read()
        if not ret:
            return []

        names = self.face_recognition.detect_known_faces(frame)
        return names

    def detect_objects(self):
        """Detects objects and returns list of recognized items."""
        ret, frame = self.camera.read()
        if not ret:
            return []
        
        results = self.object_detector(frame)
        detected_objects = [result['name'] for result in results]
        return detected_objects

###############################################################################################
###########################################################################################

    def Vision_Who_Persons_Left(self):

        from Face_Recognition_Image_Trainer_Best_Great import SimpleFacerec


        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")

        # Encode faces from a folder
        sfr = SimpleFacerec()
        sfr.load_encoding_images(My_Images_Path)


        # Load Camera
        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)

            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Frame not captured")
                break

            # --- ZOOM (crop + resize) ---
            zoom_factor = 1.3  # >1 zooms in
            h, w = frame.shape[:2]
            new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            x2, y2 = x1 + new_w, y1 + new_h
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (w, h))  # resize back to original

            # --- ENHANCEMENT ---
            # 1. Denoise (good for low-light cameras)
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

            # 2. Sharpen
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)

            # 3. Auto contrast (optional, improves visibility)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # --- FACE RECOGNITION ---
            face_locations, face_names = sfr.detect_known_faces(frame)

            for (y1, x2, y2, x1), name in zip(face_locations, face_names):
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # --- SHOW ---
            cv2.imshow("Enhanced Zoomed Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                break


    def Vision_Who_InFront(self, AlfredQueryOffline):
        """
        Detect who is in front. Prefer frames published by face_tracking.get_latest_frame().
        Fall back to opening CHEST_CAMERA_INPUT if no shared frame is available.
        """
        import cv2
        import os

        # Clear previous results
        Who_Is_In_Front.clear()
        POI_Who_Is_In_Front.clear()
        Names_and_POI_Together_List.clear()

        frame = None
        local_cap = None

        # 1) Preferred: get frame from face_tracking shared source
        try:
            import face_tracking
            try:
                frame = face_tracking.get_latest_frame(timeout=1.0)
            except Exception:
                try:
                    frame = face_tracking.get_latest_frame()
                except Exception:
                    frame = None
        except Exception:
            frame = None

        # 2) Fallback: open local VideoCapture if no frame
        try:
            if frame is None:
                local_cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
                if not local_cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")

                local_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                local_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                ret, frame = local_cap.read()
                if not ret or frame is None:
                    print("Failed to capture frame. Exiting Vision_Who_InFront.")
                    return

            # 3) Detect Faces (expects [y1, x2, y2, x1])
            face_locations, face_names = sfr.detect_known_faces(frame)
            print("Face Recognition Running....")

            for i, (face_loc, name) in enumerate(zip(face_locations, face_names)):
                y1, x2, y2, x1 = face_loc
                h, w = frame.shape[:2]

                # Bound safety
                x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
                y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))

                # Draw label + box
                cv2.putText(frame, name, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Center
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, center, 1, (0, 0, 255), -1)

                POI_Who_Is_In_Front.append(center)
                Who_Is_In_Front.append(name)
                Names_and_POI_Together_List.extend([name, center])

                # === ENHANCE & ZOOM ===
                pad_pct = 0.30
                box_w, box_h = x2 - x1, y2 - y1
                pad_x, pad_y = int(box_w * pad_pct), int(box_h * pad_pct)
                sx, sy = max(0, x1 - pad_x), max(0, y1 - pad_y)
                ex, ey = min(w, x2 + pad_x), min(h, y2 + pad_y)

                face_roi = frame[sy:ey, sx:ex].copy()
                if face_roi.size == 0:
                    continue

                target_size = (
                    min(1024, face_roi.shape[1] * 2),
                    min(1024, face_roi.shape[0] * 2)
                )
                face_up = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
                denoised = cv2.fastNlMeansDenoisingColored(face_up, None, 10, 10, 7, 21)
                lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
                sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
                cleaned = cv2.bilateralFilter(sharpened, 5, 75, 75)

                try:
                    paste_back = cv2.resize(cleaned, (ex - sx, ey - sy), interpolation=cv2.INTER_AREA)
                    frame[sy:ey, sx:ex] = paste_back
                except Exception:
                    pass

            # Prepare names string
            Who_Is_InFront_String_New = ' '.join(Who_Is_In_Front)
            Who_Is_InFront_String_New = (
                Who_Is_InFront_String_New
                .replace("tjaart", "Chart")
                .replace("celinda", "Selinda")
                .replace("dalinya", "Daalinya")
            )
            print("Who_Is_InFront_String_New:", Who_Is_InFront_String_New)

            if Who_Is_InFront_String_New.strip():
                speech.AlfredSpeak(f"I see, {Who_Is_InFront_String_New}")

        finally:
            if local_cap is not None:
                try:
                    local_cap.release()
                except Exception:
                    pass


    def Vision_Who_Persons_Right(self):

        # Load Camera
        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")



        while True:
            ret, frame = cap.read()

            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
           
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


    def Vision_What_Around(self):
        
        print("Object detection system is running.....")

        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")


        cap.set(3, 1023)
        cap.set(4, 768)

        Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"

        print("Loading the Model YoloV8n.....")

        # YOLO Model
        Obsticle_Detection_Vision_Model = YOLO(Model_File)

        with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
            classNames = [line.rstrip('//n') for line in f]

        print (classNames)

        print("Object detection Model is Loaded.....")

        while True:
            success, img = cap.read()
            results = Obsticle_Detection_Vision_Model(img, stream=False)

            print("Object Detection Software is running.....")

            
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.line(img, (x1, y1), (x1 + 15, y1), (0, 255, 0), 2)                      # Top line
                    cv2.line(img, (x1, y1), (x1, y1 + 15), (0, 255, 0), 2)                      # Left line
                    cv2.line(img, (x2, y1), (x2 - 15, y1), (0, 255, 0), 2)                      # Right line
                    cv2.line(img, (x1, y2), (x1 + 15, y2), (0, 255, 0), 2)                      # Bottom line
                    cv2.line(img, (x1, y2), (x1, y2 - 15), (0, 255, 0), 2)                      # Bottom left line
                    cv2.line(img, (x2, y2), (x2 - 15, y2), (0, 255, 0), 2)                      # Bottom right line
                    cv2.line(img, (x2, y1), (x2, y1 + 15), (0, 255, 0), 2)                      # Right top line
                    cv2.line(img, (x2, y2), (x2, y2 - 15), (0, 255, 0), 2)                      # Right bottom line

                    # Draw confidence and class name
                    confidence = round(float(box.conf[0]) * 100, 2)
                    class_index = int(box.cls[0])
                    class_name = classNames[class_index]
                    text = f"{class_name}: {confidence}%"
                    org = (x1, y1 - 10)  # Place text slightly above the bounding box
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.75
                    color = (0, 0, 255)
                    thickness = 1

                    cv2.putText(img, text, org, font, font_scale, color, thickness)

            cv2.imshow("Object Detection", img)
            if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()


    def Vision_Who_Persons_Around(self):

        from Face_Recognition_Image_Trainer_Best_Great import SimpleFacerec


        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")

        # Encode faces from a folder
        sfr = SimpleFacerec()
        sfr.load_encoding_images(My_Images_Path)


        # Load Camera
        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not cap.isOpened():
                    raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")

        while True:
            ret, frame = cap.read()

            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
           
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

##################################################################################################

    def Vision_Who_InFront_Where(self, Name_Only_For_Where):

        print(f"Name_Only_For_Where = {Name_Only_For_Where}")

        New_Name_Only_For_Where = Name_Only_For_Where

        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("can ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("you ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("tell ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("me ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("where ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("is ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace("are ","")
        New_Name_Only_For_Where = New_Name_Only_For_Where.replace(" ","")

        print(f"New_Name_Only_For_Where = {New_Name_Only_For_Where}")

        names = Who_Is_In_Front
        positions = POI_Who_Is_In_Front
    ##    positions = POI_String_New

        ##***********************************************************
        ##*****************  NAME AND LOCATION  *********************      

        print(f"names = {names}")
        print(f"positions = {positions}")

        New_Position_For_Name = positions

        # Combine the two lists into a dictionary
        name_position_map = dict(zip(names, positions))
        print(f"name_position_map = {name_position_map}")

        # Function to get position by name
        def get_position_by_name(name):
            return name_position_map.get(name, "not found")

        # Example usage
        name_to_search = New_Name_Only_For_Where
        position = get_position_by_name(name_to_search)
        print(f"The position of {name_to_search} is {position}")


        Who_Is_In_Front.clear()
        POI_Who_Is_In_Front.clear()


    ##########################################################################
    ###         FOR SINGLE FACE TRACKING NO LOCKING



###     Original Working
##
##    def Single_Face_Tracking_System(self, AlfredQueryOffline):
##        print("Starting Single Face Tracking System...")
##        print(f"Single Face Tracking AlfredQueryOffline : {AlfredQueryOffline}")
##
##        face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
##
##        try:
##            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
##            if not cap.isOpened():
##                    raise ConnectionError("ESP32 stream unreachable")
##        except Exception as e:
##            print(f"⚠️ Camera stream error: {e}")
##
##
##        locked_face = None  # Store locked face coordinates
##        tracker = None  # Tracker instance
##        tracking_active = False  # Track if a face is being followed
##
##        while True:
##            ret, img = cap.read()
##            if not ret:
##                print("Error: Failed to capture image from webcam.")
##                break
##
##            # Draw guiding lines
##            cv2.line(img, (640, 240), (0, 240), (0, 255, 0), 1)
##            cv2.line(img, (320, 0), (320, 640), (0, 255, 0), 1)
##            cv2.circle(img, (320, 240), 2, (0, 0, 255), -1)
##
##            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##
##            if not tracking_active:
##                # Search for faces if not already tracking
##                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
##                if len(faces) > 0:
##                    x, y, w, h = faces[0]
##                    locked_face = (x, y, w, h)
####                    tracker = cv2.TrackerCSRT_create()
##                    tracker = cv2.legacy.TrackerCSRT_create()
##                    tracker.init(img, locked_face)
##                    tracking_active = True
##            else:
##                # Update tracker if a face is locked
##                success, bbox = tracker.update(img)
##                if success:
##                    x, y, w, h = [int(v) for v in bbox]
##                    locked_face = (x, y, w, h)
##                else:
##                    tracking_active = False
##
##            # Draw the bounding box if a face is locked
##            if locked_face:
##                x, y, w, h = locked_face
##                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
##                center_x = x + w // 2
##                center_y = y + h // 2
##                data = f"X{center_x}Y{center_y}Z"
##                print(f"data: {data}")
##
##                # Send the coordinate data to Arduino
##                print(f"Sending to Arduino: {data}")
##                
##                Data_Revert = f"X{center_x}Y{center_y}Z"
##                
##                arduino.send_arduino(data)
##                time.sleep(0.001)
##
##            cv2.imshow('Face Detection', img)
##
##            key = cv2.waitKey(30) & 0xFF
##            # Exit on 'Esc' key or if stop command is detected in AlfredQueryOffline
##            if key == 27 or 'stop looking' in AlfredQueryOffline.lower():
##                print("Stopping Face Tracking...")
####                data_home = f"C{self.Coord_Middel_Left_Y}D{self.Coord_Middel_Left_X}Z"
##                data_home = f"D{240}E{640}Z"
##                print(f"data_home : {data_home}")
##
##                for i in range(10):
##                    i = i + 1
##                    arduino.send_arduino(data_home)
####                    time.sleep(0.1)
##                    print(f" i : {i}")
##
##                break
##
##        cap.release()
##        cv2.destroyAllWindows()
##        print("Face Tracking Stopped.")


##
##
    ##########################################################################

###     DEBATABLE 
##
##
    def Single_Face_Tracking_System(self, AlfredQueryOffline):
        import cv2
        import time
        import numpy as np
        from math import fabs

        print("Starting Single Face Tracking System...")
        print(f"Single Face Tracking AlfredQueryOffline : {AlfredQueryOffline}")

        # cascade
        face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')

        # camera sources
        PRIMARY_CAM = Alfred_config.CHEST_CAMERA_INPUT
        FALLBACK_CAM = Alfred_config.LEFT_EYE_CAMERA_INPUT

        def open_camera(src, timeout=2.0):
            """Try to open camera `src`. Return cap or None."""
            try:
                cap = cv2.VideoCapture(src)
                t0 = time.time()
                # wait briefly for it to open / provide frames
                while time.time() - t0 < timeout:
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            return cap
                    # small delay
                    time.sleep(0.05)
                # try one last time to see if isOpened() is True
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                pass
            return None

        # Try open primary first, then fallback
        cap = open_camera(PRIMARY_CAM, timeout=2.0)
        using_primary = True
        if cap is None:
            print("Primary camera failed to open, trying fallback (LEFT).")
            cap = open_camera(FALLBACK_CAM, timeout=2.0)
            using_primary = False
            if cap is None:
                print("Both primary and fallback cameras failed to open. Aborting.")
                return
            else:
                print("Using fallback LEFT camera.")

        # Optional: set expected resolution
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

        # Tracker state
        locked_face = None
        tracker = None
        tracking_active = False

        # Parameters
        DETECTION_INTERVAL = 15
        MIN_FACE_SIZE = 40
        LOST_FRAMES_THRESHOLD = 6
        SMOOTH_ALPHA = 0.25
        SCALE_CHANGE_THRESHOLD = 0.5

        # Fallback / primary retry policy
        last_primary_check = time.time()
        PRIMARY_CHECK_INTERVAL = 5.0   # seconds between attempts to reopen primary when using fallback
        last_frame_time = time.time()

        frame_idx = 0
        lost_frames = 0
        smoothed_center = None

        def iou(boxA, boxB):
            (xA, yA, wA, hA) = boxA
            (xB, yB, wB, hB) = boxB
            x1 = max(xA, xB)
            y1 = max(yA, yB)
            x2 = min(xA + wA, xB + wB)
            y2 = min(yA + hA, yB + hB)
            interW = max(0, x2 - x1)
            interH = max(0, y2 - y1)
            interArea = interW * interH
            boxAArea = wA * hA
            boxBArea = wB * hB
            union = boxAArea + boxBArea - interArea
            return interArea / (union + 1e-8)

        def choose_largest_face(faces):
            if len(faces) == 0:
                return None
            areas = [w*h for (x,y,w,h) in faces]
            idx = int(np.argmax(areas))
            return tuple(faces[idx])

        def init_tracker_with_bbox(frame, bbox):
            try:
                tr = cv2.legacy.TrackerCSRT_create()
            except Exception:
                try:
                    tr = cv2.TrackerCSRT_create()
                except Exception:
                    tr = None
            if tr is None:
                return None
            ok = tr.init(frame, bbox)
            return tr if ok else None

        print("Tracker ready. Press ESC to stop.")

        # main loop
        while True:
            ret, img = cap.read()
            now = time.time()
            if not ret or img is None or img.size == 0:
                # camera read failure: try immediate reopen of current cap once, otherwise switch
                print(f"Warning: failed to read frame from {'PRIMARY' if using_primary else 'FALLBACK'} camera.")
                try:
                    cap.release()
                except Exception:
                    pass

                # If we were using primary, try fallback; if using fallback, keep trying primary periodically
                if using_primary:
                    print("Attempting to open fallback camera (LEFT).")
                    cap = open_camera(FALLBACK_CAM, timeout=2.0)
                    if cap is not None:
                        using_primary = False
                        print("Switched to fallback LEFT camera.")
                    else:
                        # try reopening primary once more before aborting
                        print("Fallback failed. Retrying primary briefly.")
                        cap = open_camera(PRIMARY_CAM, timeout=2.0)
                        if cap is not None:
                            using_primary = True
                            print("Primary camera recovered.")
                        else:
                            print("Both cameras unavailable right now. Sleeping briefly then retrying.")
                            time.sleep(0.5)
                            # try to reopen fallback again in next loop iteration
                            cap = open_camera(FALLBACK_CAM, timeout=1.0)
                            if cap is None:
                                # nothing available, continue to next loop to attempt again
                                time.sleep(0.2)
                                continue
                            else:
                                using_primary = False
                                print("Using fallback LEFT camera.")
                else:
                    # currently using fallback
                    # try to open primary if enough time passed
                    if now - last_primary_check > PRIMARY_CHECK_INTERVAL:
                        print("Attempting to reopen primary camera...")
                        pcap = open_camera(PRIMARY_CAM, timeout=1.5)
                        last_primary_check = now
                        if pcap is not None:
                            # switch back to primary
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = pcap
                            using_primary = True
                            print("Primary camera recovered — switched back.")
                        else:
                            # try reopening fallback
                            print("Primary still down; retrying fallback.")
                            cap = open_camera(FALLBACK_CAM, timeout=1.5)
                            if cap is not None:
                                using_primary = False
                                print("Using fallback LEFT camera.")
                            else:
                                print("Fallback also unavailable; will retry shortly.")
                                time.sleep(0.2)
                                continue
                    else:
                        # keep trying fallback capture briefly
                        cap = open_camera(FALLBACK_CAM, timeout=1.0)
                        if cap is None:
                            print("Fallback unavailable; will retry.")
                            time.sleep(0.2)
                            continue

                # after switching/reopen attempt, try read again
                ret, img = cap.read()
                if not ret or img is None or img.size == 0:
                    # still failed — loop and retry
                    time.sleep(0.05)
                    continue

            h_img, w_img = img.shape[:2]

            # draw helpful overlay showing which camera in use
            cam_label = "PRIMARY (CHEST)" if using_primary else "FALLBACK (LEFT)"
            try:
                cv2.putText(img, cam_label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception:
                pass

            # optional guiding lines
            try:
                cv2.line(img, (0, int(h_img/2)), (w_img, int(h_img/2)), (0, 255, 0), 1)
                cv2.line(img, (int(w_img/2), 0), (int(w_img/2), h_img), (0, 255, 0), 1)
                cv2.circle(img, (int(w_img/2), int(h_img/2)), 2, (0, 0, 255), -1)
            except Exception:
                pass

            # detection prep
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            # detection / tracking logic (same as before)
            if not tracking_active:
                faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                if len(faces) > 0:
                    chosen = choose_largest_face(faces)
                    if chosen is not None:
                        x, y, w, h = chosen
                        locked_face = (int(x), int(y), int(w), int(h))
                        tracker = init_tracker_with_bbox(img, locked_face)
                        if tracker is not None:
                            tracking_active = True
                            lost_frames = 0
                            smoothed_center = (locked_face[0] + locked_face[2] // 2, locked_face[1] + locked_face[3] // 2)
            else:
                success, bbox = tracker.update(img) if tracker is not None else (False, None)
                if success and bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
                    w = max(1, min(w, w_img - x)); h = max(1, min(h, h_img - y))
                    new_bbox = (x, y, w, h)
                    locked_face = new_bbox
                    lost_frames = 0

                    cx = x + w//2; cy = y + h//2
                    if smoothed_center is None:
                        smoothed_center = (cx, cy)
                    else:
                        sx = int(round(smoothed_center[0] * (1.0 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA))
                        sy = int(round(smoothed_center[1] * (1.0 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA))
                        smoothed_center = (sx, sy)
                else:
                    lost_frames += 1
                    if lost_frames > LOST_FRAMES_THRESHOLD:
                        tracking_active = False
                        tracker = None
                        locked_face = None
                        smoothed_center = None
                        lost_frames = 0

            # periodic detection correction
            if frame_idx % DETECTION_INTERVAL == 0:
                faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
                if len(faces) > 0:
                    chosen = choose_largest_face(faces)
                    if chosen is not None:
                        fx, fy, fw, fh = [int(v) for v in chosen]
                        detected_bbox = (fx, fy, fw, fh)
                        if tracking_active and locked_face is not None:
                            current_iou = iou(locked_face, detected_bbox)
                            scale_change = fabs((fw*fh) - (locked_face[2]*locked_face[3])) / float(max(1, locked_face[2]*locked_face[3]))
                            if current_iou < 0.35 or scale_change > SCALE_CHANGE_THRESHOLD:
                                new_tracker = init_tracker_with_bbox(img, detected_bbox)
                                if new_tracker is not None:
                                    tracker = new_tracker
                                    locked_face = detected_bbox
                                    lost_frames = 0
                        elif not tracking_active:
                            new_tracker = init_tracker_with_bbox(img, detected_bbox)
                            if new_tracker is not None:
                                tracker = new_tracker
                                locked_face = detected_bbox
                                tracking_active = True
                                lost_frames = 0
                                smoothed_center = (fx + fw//2, fy + fh//2)

            # draw and send to arduino if locked
            if locked_face:
                x, y, w, h = locked_face
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
                cx = x + w//2
                cy = y + h//2
                if smoothed_center:
                    send_x, send_y = smoothed_center
                else:
                    send_x, send_y = cx, cy
                data = f"X{int(send_x)}Y{int(send_y)}Z"
                arduino.send_arduino(data)

            # show window with camera label
            title = f'Face Tracking System... ({ "CHEST" if using_primary else "LEFT" })'
            cv2.imshow(title, img)
            key = cv2.waitKey(1) & 0xFF

            # exit conditions
            if key == 27 or 'stop looking' in AlfredQueryOffline.lower():
                print("Stopping Face Tracking...")
                data_home = f"D{240}E{640}Z"
                for i in range(10):
                    arduino.send_arduino(data_home)
                break

            # periodically, if using fallback try reopening primary
            if (not using_primary) and (time.time() - last_primary_check > PRIMARY_CHECK_INTERVAL):
                last_primary_check = time.time()
                pcap = open_camera(PRIMARY_CAM, timeout=1.0)
                if pcap is not None:
                    # switch back
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = pcap
                    using_primary = True
                    print("Primary camera recovered — switched back to CHEST.")

            frame_idx += 1

        # cleanup
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Face Tracking Stopped.")

    ##########################################################################
    #       MOTION DETECTION AND MOTION TRACKING


    def Motion_Tracking_System(self, AlfredQueryOffline):
        print("🔍 Starting Motion Tracking System...")
        print(f"AlfredQueryOffline: {AlfredQueryOffline}")

        try:
            cap = cv2.VideoCapture(Alfred_config.CHEST_CAMERA_INPUT)
            if not cap.isOpened():
                raise ConnectionError("ESP32 stream unreachable")
        except Exception as e:
            print(f"⚠️ Camera stream error: {e}")
            return

        _, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Difference between current frame and previous
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            largest_area = 0
            target_center = (0, 0)

            for contour in contours:
                if cv2.contourArea(contour) < 800:  # Filter out small motions
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h

                if area > largest_area:
                    largest_area = area
                    target_center = (center_x, center_y)
                    motion_detected = True

            if motion_detected:
                cx, cy = target_center
                cv2.rectangle(frame, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                data = f"X{cx}Y{cy}Z"
                print(f"📡 Motion detected. Sending to Arduino: {data}")
                arduino.send_arduino(data)
                time.sleep(0.01)

            # Show grid and center dot
            cv2.line(frame, (640, 240), (0, 240), (255, 255, 255), 1)
            cv2.line(frame, (320, 0), (320, 640), (255, 255, 255), 1)
            cv2.circle(frame, (320, 240), 2, (0, 0, 255), -1)

            cv2.imshow("🎯 Motion Tracking", frame)
            prev_gray = gray.copy()

            key = cv2.waitKey(30) & 0xFF
            if key == 27 or 'stop looking' in AlfredQueryOffline.lower():
                print("🛑 Stopping Motion Tracking...")
                data_home = f"D{240}E{640}Z"
                for i in range(10):
                    arduino.send_arduino(data_home)
                    print(f"  ↪ Home Reset {i+1}")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Motion Tracking Ended.")

print("Tracking....ended")

face_tracking = VisionFaceRecognitionTrackingModule()


