



### BEST SO FAR 2026_01_14__23h30
##
### vision.py
##from arduino_com import arduino
##import Alfred_config
##from speech import speech
####from communication import comm
##from listen import listen
##
##import cv2
##from ultralytics import YOLO
##from Face_Recognition_Image_Trainer_Best_Great import SimpleFacerec
##import time
##import datetime
##
##import threading
##import multiprocessing
##from multiprocessing import Process, Value, Array, Lock
##import queue
##import serial
##import os
##import numpy as np
##
##from shared_queues import query_queue, log_queue
##
##My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")
##
### Encode faces from a folder (legacy fallback)
##sfr = SimpleFacerec()
##try:
##    sfr.load_encoding_images(My_Images_Path)
##except Exception:
##    # if SimpleFacerec fails, continue — facenet path may be used instead
##    pass
##
##print("vision face trackink....started")
##
##############################################
#####         CAMERA CHANNEL SELECT
##
##Camera_Input_Channel = 1
##
##############################################
###       FACE RECOGNITION SOFTWARE
##
##What_I_See_Front = []
##What_Is_In_Front_Speak = ""
##
##Who_Is_In_Front = []
##Who_Is_In_Front_Speak = []
##
##POI_Who_Is_In_Front = []
##POI_String_New = 0
##
##Names_and_POI_Together_List = []
##Names_and_POI_Together = 0
##
##Name_Only_For_Where = 0
##Name_Only_For_Look_AT = 0
##
##log_queue = queue.Queue()
##
##current_detection_time = 0
##
##print("Tracking start")
##
##############################################
###       OBJECT DETECTION SOFTWARE
##
##from ultralytics import YOLO
##
##Model_File = Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"
##print("Loading the Model YoloV8n.....")
##
##print('\n')
##print('____________________________________________________________________')
##print('\n')
##
##print("visio start 1")
##
### YOLO Model
##Obsticle_Detection_Vision_Model = YOLO(Model_File)
##with open(Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
##    classNames = [line.rstrip('\n') for line in f]
##
##from collections import Counter
##
##print("vision start 2")
##
### === Shared state for current speaker ===
##_current_speaker = None
##_speaker_lock = threading.Lock()
##
##
##def _set_current_speaker(name: str):
##    """Internal use only: set the current speaker."""
##    global _current_speaker
##    with _speaker_lock:
##        _current_speaker = name
##
##
##def _get_current_speaker() -> str:
##    """Internal use only: return the current speaker."""
##    with _speaker_lock:
##        return _current_speaker
##
### frame-share helpers
##import threading, time as _time
##_latest_frame = None
##_latest_frame_lock = threading.Lock()
##
##def _set_latest_frame(frame):
##    """Internal — called by the tracking loop to update the latest frame (copy stored)."""
##    global _latest_frame
##    if frame is None:
##        with _latest_frame_lock:
##            _latest_frame = None
##        return
##    with _latest_frame_lock:
##        # store a copy so caller keeps original safe
##        _latest_frame = frame.copy()
##
##def get_latest_frame(timeout=1.0):
##    """
##    Return a copy of the latest frame available within `timeout` seconds, else None.
##    Non-blocking if latest is already present.
##    """
##    t0 = _time.time()
##    while _time.time() - t0 < float(timeout):
##        with _latest_frame_lock:
##            if _latest_frame is not None:
##                return _latest_frame.copy()
##        # short sleep to avoid busy-loop
##        time.sleep(0.01)
##    return None
##
##from shutdown_helpers import app_shutdown_event
##
### Try import face_recognition for older confidence logic (optional)
##try:
##    import face_recognition
##    FACE_RECOG_AVAILABLE = True
##except Exception:
##    FACE_RECOG_AVAILABLE = False
##    print("vision.py: face_recognition not available — legacy confidence filtering disabled.")
##
### Try import facenet-pytorch (preferred new path)
##FACENET_AVAILABLE = False
##try:
##    import torch
##    from facenet_pytorch import MTCNN, InceptionResnetV1
##    from PIL import Image
##    FACENET_AVAILABLE = True
##    print("vision.py: facenet-pytorch available — will prefer Facenet detection if initialized.")
##except Exception:
##    FACENET_AVAILABLE = False
##    print("vision.py: facenet-pytorch not available — will fall back to SimpleFacerec/face_recognition if present.")
##
##
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
##    global current_detection_time, FACENET_AVAILABLE
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
##    # --- FACENET initialization & known embeddings (only if available) ---
##    mtcnn = None
##    facenet_model = None
##    known_names = []
##    known_embeddings = None
##    device = None
##
##    if FACENET_AVAILABLE:
##        try:
##            device = ('cuda' if torch.cuda.is_available() else 'cpu')
##            mtcnn = MTCNN(keep_all=True, device=device)
##            facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
##
##            print("[FACENET INIT] Building known embeddings from images in:", My_Images_Path)
##            imgs_paths = []
##            names = []
##            for root, _, files in os.walk(My_Images_Path):
##                for fn in sorted(files):
##                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
##                        path = os.path.join(root, fn)
##                        base = os.path.splitext(fn)[0]
##                        imgs_paths.append(path)
##                        names.append(base)
##
##            emb_list = []
##            keep_names = []
##            for idx, path in enumerate(imgs_paths):
##                try:
##                    pil = Image.open(path).convert('RGB')
##                    # get face tensors aligned from mtcnn
##                    face_tensors = mtcnn(pil)  # could be None, Tensor (3,160,160) or (n,3,160,160)
##                    if face_tensors is None:
##                        print(f"[FACENET INIT] no face tensor for {path}, skipping")
##                        continue
##                    # normalize shape to (n,3,160,160)
##                    if face_tensors.dim() == 3:
##                        face_tensors = face_tensors.unsqueeze(0)
##                    # take the first face found in each image
##                    ft = face_tensors[0].unsqueeze(0).to(device)
##                    with torch.no_grad():
##                        emb = facenet_model(ft)  # (1,512)
##                    emb_np = emb[0].cpu().numpy()
##                    emb_list.append(emb_np)
##                    keep_names.append(names[idx])
##                    print(f"[FACENET INIT] embedding added for {names[idx]}")
##                except Exception as e:
##                    print("[FACENET INIT] error processing", path, e)
##                    continue
##
##            if emb_list:
##                known_embeddings = np.vstack(emb_list)
##                known_names = keep_names
##                print(f"[FACENET INIT] built {len(known_names)} known embeddings.")
##            else:
##                known_embeddings = None
##                known_names = []
##                print("[FACENET INIT] no known embeddings built.")
##        except Exception as e:
##            # If facenet init fails, disable it and fall back.
##            print("vision.py: facenet init error — falling back. Error:", e)
##            FACENET_AVAILABLE = False
##            mtcnn = None
##            facenet_model = None
##            known_embeddings = None
##            known_names = []
##            device = None
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
##    MIN_FACE_AREA = 30 * 30
##    LOST_FRAMES_THRESHOLD = 6
##    SMOOTH_ALPHA = 0.25
##    SCALE_CHANGE_THRESHOLD = 0.6
##    IOU_THRESHOLD = 0.35
##    PRIMARY_CHECK_INTERVAL = 3.0
##    SEND_INTERVAL = 0.08
##
##    # Face-confidence / facenet specific settings
##    FACE_CONFIDENCE_ENABLED = True and (FACENET_AVAILABLE or FACE_RECOG_AVAILABLE)
##    FACE_MATCH_DISTANCE_THRESHOLD = 0.60  # for face_recognition (euclidean)
##    FACE_CONFIDENCE_MIN = 0.60           # minimum mapped confidence (0..1) to accept a match
##    FACENET_COSINE_THRESHOLD = 0.60      # cosine similarity threshold (higher is stricter)
##    # If encoding comparison fails but face bounding box area is large enough, accept as "Unknown".
##    MIN_FACE_AREA_KEEP_FOR_UNKNOWN = MIN_FACE_AREA
##
##    # NO-DETECTION behavior: configurable timeout (seconds). Default 3 minutes (180s).
##    try:
##        NO_DETECTION_TIMEOUT = float(getattr(Alfred_config, "NO_DETECTION_TIMEOUT", 180.0))
##    except Exception:
##        NO_DETECTION_TIMEOUT = 180.0
##
##    tracker = None
##    tracking_active = False
##    locked_face = None
##    name = None
##    locked_confidence = None
##    last_speaker = None
##    frame_idx = 0
##    lost_frames = 0
##    smoothed_center = None
##    last_primary_check = time.time()
##    last_send_time = 0.0
##
##    # Track last time we had any detection (face detected OR tracking active)
##    last_detection_time = time.time()
##
##    # guard to prevent races when movement sequence is being executed
##    movement_in_progress = False
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
##                locked_confidence = None
##                if current_speaker:
##                    print(f"Vision: switching target to speaker: {current_speaker}")
##
##            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
##            detected_locations = []
##            detected_names = []
##            detected_confidences = []
##
##            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
##                # Preferred: use facenet-pytorch MTCNN + InceptionResnet for detection+embedding if available
##                if FACENET_AVAILABLE and mtcnn is not None and facenet_model is not None:
##                    try:
##                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                        pil_im = Image.fromarray(rgb)
##                        # detect boxes + probs and get aligned face tensors
##                        boxes, probs = mtcnn.detect(pil_im)
##                        face_tensors = mtcnn(pil_im)  # returns None, or tensor (3,160,160) or (n,3,160,160)
##                        print(f"[FACENET DETECT] boxes: {None if boxes is None else len(boxes)}, probs: {None if probs is None else probs}")
##                        if face_tensors is None:
##                            # no face tensors found; ensure empty lists
##                            detected_locations = []
##                            detected_names = []
##                            detected_confidences = []
##                        else:
##                            # normalize to (n,3,160,160)
##                            if face_tensors.dim() == 3:
##                                face_tensors = face_tensors.unsqueeze(0)
##                            for i_ft, ft in enumerate(face_tensors):
##                                try:
##                                    # boxes may be None or shorter than tensors — protect indices
##                                    if boxes is not None and i_ft < len(boxes):
##                                        b = boxes[i_ft]
##                                        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
##                                    else:
##                                        # fallback: estimate bounding box from tensor (center)
##                                        # here we skip if we can't map a box
##                                        continue
##
##                                    w = x2 - x1; h = y2 - y1
##                                    if w * h < MIN_FACE_AREA:
##                                        continue
##
##                                    with torch.no_grad():
##                                        emb = facenet_model(ft.unsqueeze(0).to(device))
##                                        emb_np = emb[0].cpu().numpy()
##
##                                    matched_name = "Unknown"
##                                    conf_val = 0.0
##                                    if known_embeddings is not None and known_embeddings.shape[0] > 0:
##                                        e_norm = emb_np / (np.linalg.norm(emb_np) + 1e-8)
##                                        known_norms = known_embeddings / (np.linalg.norm(known_embeddings, axis=1, keepdims=True) + 1e-8)
##                                        sims = np.dot(known_norms, e_norm)  # cosine sims
##                                        best_idx = int(np.argmax(sims))
##                                        best_sim = float(sims[best_idx])
##                                        # compute a 0..1 confidence when > threshold
##                                        if best_sim >= FACENET_COSINE_THRESHOLD:
##                                            conf_val = float((best_sim - FACENET_COSINE_THRESHOLD) / (1.0 - FACENET_COSINE_THRESHOLD))
##                                            matched_name = known_names[best_idx]
##                                        else:
##                                            conf_val = 0.0
##                                            matched_name = "Unknown"
##                                        print(f"[FACENET MATCH] face#{i_ft} best_sim={best_sim:.3f} -> name={matched_name} conf={conf_val:.3f}")
##                                    else:
##                                        conf_val = 0.0
##                                        matched_name = "Unknown"
##
##                                    # record in same format used by your face pipeline (top,right,bottom,left)
##                                    detected_locations.append((y1, x2, y2, x1))
##                                    detected_names.append(matched_name)
##                                    detected_confidences.append(conf_val)
##                                except Exception as e:
##                                    # per-face failure: continue
##                                    print("[FACENET DETECT] per-face error:", e)
##                                    continue
##                    except Exception as e:
##                        print("[FACENET DETECT] error, falling back to SimpleFacerec:", e)
##                        # fall back to SimpleFacerec if facenet detection fails mid-loop
##                        try:
##                            detected_locations, detected_names = sfr.detect_known_faces(frame)
##                        except Exception:
##                            try:
##                                rgb_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                                detected_locations, detected_names = sfr.detect_known_faces(rgb_tmp)
##                            except Exception:
##                                detected_locations, detected_names = [], []
##                        # confidences will be produced below
##                else:
##                    # fallback: use existing SimpleFacerec detection
##                    try:
##                        try:
##                            detected_locations, detected_names = sfr.detect_known_faces(frame)
##                        except Exception:
##                            rgb_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                            detected_locations, detected_names = sfr.detect_known_faces(rgb_tmp)
##                        if detected_locations is None:
##                            detected_locations = []
##                        elif isinstance(detected_locations, np.ndarray):
##                            try:
##                                detected_locations = detected_locations.tolist()
##                            except Exception:
##                                detected_locations = list(detected_locations)
##                        else:
##                            try:
##                                detected_locations = list(detected_locations)
##                            except Exception:
##                                detected_locations = []
##
##                        if detected_names is None:
##                            detected_names = []
##                        elif isinstance(detected_names, np.ndarray):
##                            try:
##                                detected_names = [str(n) for n in detected_names.tolist()]
##                            except Exception:
##                                detected_names = [str(n) for n in detected_names]
##                        else:
##                            try:
##                                detected_names = [str(n) for n in detected_names]
##                            except Exception:
##                                detected_names = []
##                    except Exception:
##                        detected_locations, detected_names = [], []
##
##                    # If face_recognition is available, compute confidences as before
##                    if FACE_RECOG_AVAILABLE and getattr(sfr, 'known_face_encodings', None):
##                        try:
##                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                            known_encodings = getattr(sfr, 'known_face_encodings', None)
##                            for i, loc in enumerate(detected_locations):
##                                nm = None
##                                try:
##                                    nm = detected_names[i] if i < len(detected_names) else None
##                                except Exception:
##                                    nm = None
##                                conf_val = 1.0 if (nm and str(nm).strip().lower() not in ("unknown", "")) else 0.0
##                                bb = face_loc_to_bbox(loc)
##                                if bb is None:
##                                    detected_confidences.append(conf_val)
##                                    continue
##                                x, y, w, h = bb
##                                try:
##                                    crop = rgb[y:y+h, x:x+w]
##                                    encs = face_recognition.face_encodings(crop)
##                                    if not encs:
##                                        top = y; left = x; bottom = y + h; right = x + w
##                                        full_encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
##                                        if full_encs:
##                                            enc = full_encs[0]
##                                        else:
##                                            enc = None
##                                    else:
##                                        enc = encs[0]
##                                    if enc is None:
##                                        detected_confidences.append(conf_val)
##                                        continue
##                                    distances = face_recognition.face_distance(known_encodings, enc)
##                                    if distances is None or len(distances) == 0:
##                                        detected_confidences.append(conf_val)
##                                        continue
##                                    min_idx = int(np.argmin(distances))
##                                    min_dist = float(distances[min_idx])
##                                    mapped_conf = max(0.0, min(1.0, (FACE_MATCH_DISTANCE_THRESHOLD - min_dist) / FACE_MATCH_DISTANCE_THRESHOLD))
##                                    conf_val = mapped_conf
##                                except Exception:
##                                    conf_val = conf_val
##                                detected_confidences.append(conf_val)
##                        except Exception:
##                            # default confidences
##                            detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
##                    else:
##                        detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
##
##                # --- Filter out very low-confidence detections (to reduce false positives) ---
##                filtered_locations = []
##                filtered_names = []
##                filtered_confidences = []
##                for i, loc in enumerate(detected_locations):
##                    bb = face_loc_to_bbox(loc)
##                    area = 0
##                    if bb is not None:
##                        area = bb[2] * bb[3]
##                    conf_val = detected_confidences[i] if i < len(detected_confidences) else 0.0
##                    nm = detected_names[i] if i < len(detected_names) else None
##
##                    accept = False
##                    if (FACENET_AVAILABLE and known_embeddings is not None):
##                        # use facenet confidence rules
##                        if conf_val >= FACE_CONFIDENCE_MIN:
##                            accept = True
##                        else:
##                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                                accept = True
##                    elif FACE_RECOG_AVAILABLE:
##                        if conf_val >= FACE_CONFIDENCE_MIN:
##                            accept = True
##                        else:
##                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                                accept = True
##                    else:
##                        if nm and str(nm).strip().lower() not in ("unknown", ""):
##                            accept = True
##                        elif area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                            accept = True
##                        else:
##                            accept = False
##
##                    if accept:
##                        filtered_locations.append(loc)
##                        filtered_names.append(nm if nm is not None else "Unknown")
##                        filtered_confidences.append(conf_val)
##                    else:
##                        pass
##
##                detected_locations = filtered_locations
##                detected_names = filtered_names
##                detected_confidences = filtered_confidences
##
##                if detected_confidences is None:
##                    detected_confidences = [1.0] * len(detected_locations)
##                elif len(detected_confidences) != len(detected_locations):
##                    detected_confidences = (detected_confidences + [0.0] * len(detected_locations))[:len(detected_locations)]
##
##            # Update last_detection_time whenever we have detection results to avoid the no-detection trigger
##            try:
##                if (detected_locations and len(detected_locations) > 0) or tracking_active or locked_face:
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            target_bbox = None
##            target_name = None
##            target_confidence = None
##
##            # If there's no current speaker, try to pick one from face recognition results
##            if not current_speaker:
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
##                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
##                    score = (100000 if is_known else 0) + area
##                    if score > best_area:
##                        best_area = score
##                        best_idx = i
##
##                if best_idx is not None:
##                    chosen = detected_locations[best_idx]
##                    bb = face_loc_to_bbox(chosen)
##                    if bb is not None:
##                        target_bbox = bb
##                    try:
##                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
##                    except Exception:
##                        target_name = None
##
##                    try:
##                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
##                    except Exception:
##                        target_confidence = None
##
##                    if target_name and str(target_name).strip().lower() not in ("unknown", ""):
##                        try:
##                            _set_current_speaker(target_name)
##                            last_speaker = target_name
##                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name}")
##                        except Exception:
##                            last_speaker = target_name
##                    else:
##                        if target_name is None:
##                            target_name = "Unknown"
##
##            # SAFE selection fallback
##            if target_bbox is None and len(detected_locations) > 0:
##                bboxes = []
##                for loc in detected_locations:
##                    b = face_loc_to_bbox(loc)
##                    bboxes.append(b)
##
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
##                    try:
##                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
##                    except Exception:
##                        target_confidence = None
##
##            # --- tracker init / update / draw / send logic ---
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
##                                locked_confidence = target_confidence if target_confidence is not None else (1.0 if name and str(name).strip().lower() not in ("unknown", "") else 0.0)
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
##                                        locked_confidence = target_confidence if target_confidence is not None else locked_confidence
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
##                        name = None
##                        smoothed_center = None
##                        locked_confidence = None
##                        lost_frames = 0
##
##            # If we have a locked_face (tracking) consider that a detection too
##            try:
##                if locked_face:
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            # If no detections for longer than NO_DETECTION_TIMEOUT → perform the movement sequence
##            try:
##                current_detection_time = (time.time() - last_detection_time)
##                no_detections_now = (not detected_locations) and (not tracking_active) and (locked_face is None)
##                if current_detection_time > NO_DETECTION_TIMEOUT and no_detections_now and (not movement_in_progress):
##                    if arduino is not None:
##                        movement_in_progress = True
##                        try:
##                            # Movement commands (adjust as needed)
##                            data_front_UD = f"p"   # changed in your snippet
##                            data_front_LR = f"f"
##
##                            print(f"[NO-DETECT] data_front_UD : {data_front_UD}")
##                            print(f"[NO-DETECT] data_front_LR : {data_front_LR}")
##                            for _ in range(1):
##                                for i in range(10):
##                                    try:
##                                        if hasattr(arduino, "send_arduino"):
##                                            try:
##                                                arduino.send_arduino(data_front_UD)
##                                            except Exception:
##                                                if hasattr(arduino, "write"):
##                                                    try:
##                                                        arduino.write((data_front_UD + "\n").encode())
##                                                    except Exception:
##                                                        pass
##                                        else:
##                                            if hasattr(arduino, "write"):
##                                                try:
##                                                    arduino.write((data_front_UD + "\n").encode())
##                                                except Exception:
##                                                    pass
##                                    except Exception as e:
##                                        print("[NO-DETECT] exception while sending UD:", e)
##                                    print(f"[NO-DETECT] sent UD i : {i+1}")
##                                    try:
##                                        time.sleep(0.02)
##                                    except Exception:
##                                        pass
##
##                                try:
##                                    time.sleep(2.5)  # your original pause between sequences
##                                except Exception:
##                                    pass
##
##                                for i in range(10):
##                                    try:
##                                        if hasattr(arduino, "send_arduino"):
##                                            try:
##                                                arduino.send_arduino(data_front_LR)
##                                            except Exception:
##                                                if hasattr(arduino, "write"):
##                                                    try:
##                                                        arduino.write((data_front_LR + "\n").encode())
##                                                    except Exception:
##                                                        pass
##                                        else:
##                                            if hasattr(arduino, "write"):
##                                                try:
##                                                    arduino.write((data_front_LR + "\n").encode())
##                                                except Exception:
##                                                    pass
##                                    except Exception as e:
##                                        print("[NO-DETECT] exception while sending LR:", e)
##                                    print(f"[NO-DETECT] sent LR i : {i+1}")
##                                    try:
##                                        time.sleep(0.02)
##                                    except Exception:
##                                        pass
##
##                            print("[NO-DETECT] movement sequence complete.")
##                        finally:
##                            movement_in_progress = False
##
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            if locked_face:
##                x, y, w, h = locked_face
##                try:
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
##                    if name:
##                        label = str(name)
##                        try:
##                            if locked_confidence is not None:
##                                pct = int(round(max(0.0, min(1.0, locked_confidence)) * 100.0))
##                                label = f"{label} {pct}%"
##                        except Exception:
##                            pass
##                        cv2.putText(frame, label, (x, max(0, y - 8)),
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
##                            try:
##                                if hasattr(arduino, "write"):
##                                    arduino.write((data + "\n").encode())
##                            except Exception:
##                                pass
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
##











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

# Encode faces from a folder (legacy fallback)
sfr = SimpleFacerec()
try:
    sfr.load_encoding_images(My_Images_Path)
except Exception:
    # if SimpleFacerec fails, continue — facenet path may be used instead
    pass

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

current_detection_time = 0

# debug overlay toggle (can set Alfred_config.VISION_DEBUG = False to disable)
DEBUG_OVERLAY = getattr(Alfred_config, "VISION_DEBUG", False)

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

# Try import face_recognition for older confidence logic (optional)
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    FACE_RECOG_AVAILABLE = False
    print("vision.py: face_recognition not available — legacy confidence filtering disabled.")

# Try import facenet-pytorch (preferred new path)
FACENET_AVAILABLE = False
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    FACENET_AVAILABLE = True
    print("vision.py: facenet-pytorch available — will prefer Facenet detection if initialized.")
except Exception:
    FACENET_AVAILABLE = False
    print("vision.py: facenet-pytorch not available — will fall back to SimpleFacerec/face_recognition if present.")



# ---------------------------
# Vision_Who_InFront_Look_At_New (unchanged except for integration with improved SimpleFacerec)
# ---------------------------
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
    global current_detection_time, FACENET_AVAILABLE

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

    # --- FACENET initialization & known embeddings (only if available) ---
    mtcnn = None
    facenet_model = None
    known_names = []
    known_embeddings = None
    device = None

    if FACENET_AVAILABLE:
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            from PIL import Image
            from torchvision import transforms
            device = ('cuda' if torch.cuda.is_available() else 'cpu')
            mtcnn = MTCNN(keep_all=True, device=device)
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

            print("[FACENET INIT] Building known embeddings from images in:", My_Images_Path)
            imgs_paths = []
            names = []
            for root, _, files in os.walk(My_Images_Path):
                for fn in sorted(files):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path = os.path.join(root, fn)
                        base = os.path.splitext(fn)[0]
                        imgs_paths.append(path)
                        names.append(base)

            emb_list = []
            keep_names = []
            for idx, path in enumerate(imgs_paths):
                try:
                    pil = Image.open(path).convert('RGB')
                    # get face tensors aligned from mtcnn
                    face_tensors = mtcnn(pil)  # could be None, Tensor (3,160,160) or (n,3,160,160)
                    if face_tensors is None:
                        print(f"[FACENET INIT] no face tensor for {path}, skipping")
                        continue
                    # normalize shape to (n,3,160,160)
                    if face_tensors.dim() == 3:
                        face_tensors = face_tensors.unsqueeze(0)
                    # take the first face found in each image
                    ft = face_tensors[0].unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = facenet_model(ft)  # (1,512)
                    emb_np = emb[0].cpu().numpy()
                    emb_list.append(emb_np)
                    keep_names.append(names[idx])
                    print(f"[FACENET INIT] embedding added for {names[idx]}")
                except Exception as e:
                    print("[FACENET INIT] error processing", path, e)
                    continue

            if emb_list:
                known_embeddings = np.vstack(emb_list)
                known_names = keep_names
                print(f"[FACENET INIT] built {len(known_names)} known embeddings.")
            else:
                known_embeddings = None
                known_names = []
                print("[FACENET INIT] no known embeddings built.")
        except Exception as e:
            # If facenet init fails, disable it and fall back.
            print("vision.py: facenet init error — falling back. Error:", e)
            FACENET_AVAILABLE = False
            mtcnn = None
            facenet_model = None
            known_embeddings = None
            known_names = []
            device = None

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
            top, right, bottom, left = face_loc
            x = int(left); y = int(top)
            w = int(max(1, right - left)); h = int(max(1, bottom - top))
            return (x, y, w, h)
        except Exception:
            try:
                x, y, w, h = face_loc
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

    DETECTION_INTERVAL = 12
    MIN_FACE_AREA = 30 * 30
    LOST_FRAMES_THRESHOLD = 6
    SMOOTH_ALPHA = 0.25
    SCALE_CHANGE_THRESHOLD = 0.6
    IOU_THRESHOLD = 0.35
    PRIMARY_CHECK_INTERVAL = 3.0
    SEND_INTERVAL = 0.08

    # --- TIGHTER thresholds (changed per your request) ---
    # Face-confidence / facenet specific settings
    FACE_CONFIDENCE_ENABLED = True and (FACENET_AVAILABLE or FACE_RECOG_AVAILABLE)
    FACE_MATCH_DISTANCE_THRESHOLD = 0.45  # stricter for face_recognition fallback
    FACE_CONFIDENCE_MIN = 0.52           # minimum mapped confidence (0..1) to accept a match (raised)
    FACENET_COSINE_THRESHOLD = 0.52      # cosine similarity threshold (higher is stricter)

    # If encoding comparison fails but face bounding box area is large enough, accept as "Unknown".
    MIN_FACE_AREA_KEEP_FOR_UNKNOWN = MIN_FACE_AREA

    # NO-DETECTION behavior: configurable timeout (seconds). Default 3 minutes (180s).
    try:
        NO_DETECTION_TIMEOUT = float(getattr(Alfred_config, "NO_DETECTION_TIMEOUT", 180.0))
    except Exception:
        NO_DETECTION_TIMEOUT = 180.0

    tracker = None
    tracking_active = False
    locked_face = None
    name = None
    locked_confidence = None
    last_speaker = None
    frame_idx = 0
    lost_frames = 0
    smoothed_center = None
    last_primary_check = time.time()
    last_send_time = 0.0

    # Track last time we had any detection (face detected OR tracking active)
    last_detection_time = time.time()

    # guard to prevent races when movement sequence is being executed
    movement_in_progress = False

    print("Vision_Who_InFront_Look_At_New: Tracker ready. Press 'q' to stop.")

    try:
        while True:

            # Respect app shutdown signal
            if app_shutdown_event.is_set():
                print("Vision_Who_InFront_Look_At_New: shutdown event received, exiting.")
                break
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(e)
                print("There is no Caption or Frame detected, Please check the camera output")
                pass
                
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
                            print("Primary recovered — switched back.")
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
                locked_confidence = None
                if current_speaker:
                    print(f"Vision: switching target to speaker: {current_speaker}")

            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
            detected_locations = []
            detected_names = []
            detected_confidences = []
            detected_probs = []  # MTCNN probabilities (if available)

            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
                # Preferred: use facenet-pytorch MTCNN + InceptionResnet for detection+embedding if available
                if FACENET_AVAILABLE and mtcnn is not None and facenet_model is not None:
                    try:
                        from PIL import Image
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(rgb)
                        # detect boxes + probs and get aligned face tensors
                        boxes, probs = mtcnn.detect(pil_im)
                        face_tensors = mtcnn(pil_im)  # returns None, or tensor (3,160,160) or (n,3,160,160)
                        if probs is None:
                            probs_list = None
                        else:
                            probs_list = list(probs)
                        if face_tensors is None:
                            # no face tensors found; ensure empty lists
                            detected_locations = []
                            detected_names = []
                            detected_confidences = []
                            detected_probs = []
                        else:
                            # normalize to (n,3,160,160)
                            if face_tensors.dim() == 3:
                                face_tensors = face_tensors.unsqueeze(0)
                            for i_ft, ft in enumerate(face_tensors):
                                try:
                                    # boxes may be None or shorter than tensors — protect indices
                                    if boxes is not None and i_ft < len(boxes):
                                        b = boxes[i_ft]
                                        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
                                    else:
                                        # fallback: skip if no box mapping
                                        continue

                                    w = x2 - x1; h = y2 - y1
                                    if w * h < MIN_FACE_AREA:
                                        continue

                                    with torch.no_grad():
                                        emb = facenet_model(ft.unsqueeze(0).to(device))
                                        emb_np = emb[0].cpu().numpy()

                                    matched_name = "Unknown"
                                    conf_val = 0.0
                                    if known_embeddings is not None and known_embeddings.shape[0] > 0:
                                        e_norm = emb_np / (np.linalg.norm(emb_np) + 1e-8)
                                        known_norms = known_embeddings / (np.linalg.norm(known_embeddings, axis=1, keepdims=True) + 1e-8)
                                        sims = np.dot(known_norms, e_norm)  # cosine sims
                                        best_idx = int(np.argmax(sims))
                                        best_sim = float(sims[best_idx])
                                        # compute a 0..1 confidence when > threshold
                                        if best_sim >= FACENET_COSINE_THRESHOLD:
                                            conf_val = float((best_sim - FACENET_COSINE_THRESHOLD) / (1.0 - FACENET_COSINE_THRESHOLD))
                                            matched_name = known_names[best_idx]
                                        else:
                                            conf_val = 0.0
                                            matched_name = "Unknown"
                                        if DEBUG_OVERLAY:
                                            print(f"[FACENET MATCH] face#{i_ft} best_sim={best_sim:.3f} -> name={matched_name} conf={conf_val:.3f}")
                                    else:
                                        conf_val = 0.0
                                        matched_name = "Unknown"

                                    # record in same format used by your face pipeline (top,right,bottom,left)
                                    detected_locations.append((y1, x2, y2, x1))
                                    detected_names.append(matched_name)
                                    detected_confidences.append(conf_val)
                                    detected_probs.append(float(probs_list[i_ft]) if (probs_list is not None and i_ft < len(probs_list)) else None)
                                except Exception as e:
                                    if DEBUG_OVERLAY:
                                        print("[FACENET DETECT] per-face error:", e)
                                    continue
                    except Exception as e:
                        if DEBUG_OVERLAY:
                            print("[FACENET DETECT] error, falling back to SimpleFacerec:", e)
                else:
                    # fallback: use existing SimpleFacerec detection
                    try:
                        try:
                            detected_locations, detected_names = sfr.detect_known_faces(frame)
                        except Exception:
                            rgb_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            detected_locations, detected_names = sfr.detect_known_faces(rgb_tmp)
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
                                detected_names = [str(n) for n in detected_names]
                            except Exception:
                                detected_names = []
                    except Exception:
                        detected_locations, detected_names = [], []

                    # If face_recognition is available, compute confidences as before
                    if FACE_RECOG_AVAILABLE and getattr(sfr, 'known_face_encodings', None):
                        try:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            known_encodings = getattr(sfr, 'known_face_encodings', None)
                            for i, loc in enumerate(detected_locations):
                                nm = None
                                try:
                                    nm = detected_names[i] if i < len(detected_names) else None
                                except Exception:
                                    nm = None
                                conf_val = 1.0 if (nm and str(nm).strip().lower() not in ("unknown", "")) else 0.0
                                bb = face_loc_to_bbox(loc)
                                if bb is None:
                                    detected_confidences.append(conf_val)
                                    detected_probs.append(None)
                                    continue
                                x, y, w, h = bb
                                try:
                                    crop = rgb[y:y+h, x:x+w]
                                    encs = face_recognition.face_encodings(crop)
                                    if not encs:
                                        top = y; left = x; bottom = y + h; right = x + w
                                        full_encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
                                        if full_encs:
                                            enc = full_encs[0]
                                        else:
                                            enc = None
                                    else:
                                        enc = encs[0]
                                    if enc is None:
                                        detected_confidences.append(conf_val)
                                        detected_probs.append(None)
                                        continue
                                    distances = face_recognition.face_distance(known_encodings, enc)
                                    if distances is None or len(distances) == 0:
                                        detected_confidences.append(conf_val)
                                        detected_probs.append(None)
                                        continue
                                    min_idx = int(np.argmin(distances))
                                    min_dist = float(distances[min_idx])
                                    mapped_conf = max(0.0, min(1.0, (FACE_MATCH_DISTANCE_THRESHOLD - min_dist) / FACE_MATCH_DISTANCE_THRESHOLD))
                                    conf_val = mapped_conf
                                except Exception:
                                    conf_val = conf_val
                                detected_confidences.append(conf_val)
                                detected_probs.append(None)
                        except Exception:
                            # default confidences
                            detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
                            detected_probs = [None] * len(detected_confidences)
                    else:
                        detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
                        detected_probs = [None] * len(detected_confidences)

                # --- Filter out very low-confidence detections (to reduce false positives) ---
                filtered_locations = []
                filtered_names = []
                filtered_confidences = []
                filtered_probs = []
                for i, loc in enumerate(detected_locations):
                    bb = face_loc_to_bbox(loc)
                    area = 0
                    if bb is not None:
                        area = bb[2] * bb[3]
                    conf_val = detected_confidences[i] if i < len(detected_confidences) else 0.0
                    nm = detected_names[i] if i < len(detected_names) else None
                    prob = detected_probs[i] if i < len(detected_probs) else None

                    accept = False
                    if (FACENET_AVAILABLE and known_embeddings is not None):
                        # use facenet confidence rules
                        if conf_val >= FACE_CONFIDENCE_MIN:
                            accept = True
                        else:
                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
                                accept = True
                    elif FACE_RECOG_AVAILABLE:
                        if conf_val >= FACE_CONFIDENCE_MIN:
                            accept = True
                        else:
                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
                                accept = True
                    else:
                        if nm and str(nm).strip().lower() not in ("unknown", ""):
                            accept = True
                        elif area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
                            accept = True
                        else:
                            accept = False

                    if accept:
                        filtered_locations.append(loc)
                        filtered_names.append(nm if nm is not None else "Unknown")
                        filtered_confidences.append(conf_val)
                        filtered_probs.append(prob)
                    else:
                        if DEBUG_OVERLAY:
                            # debug print of rejected detection
                            try:
                                print(f"[DETECT FILTER] rejected loc area={area} conf={conf_val:.3f} name={nm}")
                            except Exception:
                                pass
                        pass

                detected_locations = filtered_locations
                detected_names = filtered_names
                detected_confidences = filtered_confidences
                detected_probs = filtered_probs

                if detected_confidences is None:
                    detected_confidences = [1.0] * len(detected_locations)
                    detected_probs = [None] * len(detected_locations)
                elif len(detected_confidences) != len(detected_locations):
                    detected_confidences = (detected_confidences + [0.0] * len(detected_locations))[:len(detected_locations)]
                    detected_probs = (detected_probs + [None] * len(detected_locations))[:len(detected_locations)]

            # Update last_detection_time whenever we have detection results to avoid the no-detection trigger
            try:
                if (detected_locations and len(detected_locations) > 0) or tracking_active or locked_face:
                    last_detection_time = time.time()
            except Exception:
                pass

            target_bbox = None
            target_name = None
            target_confidence = None

            # If there's no current speaker, try to pick one from face recognition results
            if not current_speaker:
                best_idx = None
                best_area = 0
                for i, loc in enumerate(detected_locations):
                    bb = face_loc_to_bbox(loc)
                    if bb is None:
                        continue
                    area = bb[2] * bb[3]
                    nm = None
                    try:
                        nm = detected_names[i] if i < len(detected_names) else None
                    except Exception:
                        nm = None
                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
                    score = (100000 if is_known else 0) + area
                    if score > best_area:
                        best_area = score
                        best_idx = i

                if best_idx is not None:
                    chosen = detected_locations[best_idx]
                    bb = face_loc_to_bbox(chosen)
                    if bb is not None:
                        target_bbox = bb
                    try:
                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
                    except Exception:
                        target_name = None

                    try:
                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
                    except Exception:
                        target_confidence = None

                    if target_name and str(target_name).strip().lower() not in ("unknown", ""):
                        try:
                            _set_current_speaker(target_name)
                            last_speaker = target_name
                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name}")
                        except Exception:
                            last_speaker = target_name
                    else:
                        if target_name is None:
                            target_name = "Unknown"

            # SAFE selection fallback
            if target_bbox is None and len(detected_locations) > 0:
                bboxes = []
                for loc in detected_locations:
                    b = face_loc_to_bbox(loc)
                    bboxes.append(b)

                norm_speaker = normalize_name(current_speaker)
                best_idx = None
                if norm_speaker:
                    for i, nm in enumerate(detected_names):
                        if nm and normalize_name(nm) == norm_speaker:
                            best_idx = i
                            break
                    if best_idx is None:
                        for i, nm in enumerate(detected_names):
                            if nm and norm_speaker in normalize_name(nm):
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
                        if area > best_area and area >= MIN_FACE_AREA:
                            best_area = area
                            best_idx = i

                if best_idx is not None and bboxes[best_idx] is not None:
                    target_bbox = bboxes[best_idx]
                    try:
                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
                    except Exception:
                        target_name = None
                    try:
                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
                    except Exception:
                        target_confidence = None

            # --- tracker init / update / draw / send logic ---
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
                                locked_confidence = target_confidence if target_confidence is not None else (1.0 if name and str(name).strip().lower() not in ("unknown", "") else 0.0)
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
                        if current_iou < IOU_THRESHOLD or scale_change > SCALE_CHANGE_THRESHOLD:
                            tr = create_csrt()
                            if tr is not None:
                                try:
                                    ok = tr.init(frame, (int(fx), int(fy), int(fw), int(fh)))
                                    if ok:
                                        tracker = tr
                                        locked_face = (int(fx), int(fy), int(fw), int(fh))
                                        name = target_name or name or "Unknown"
                                        locked_confidence = target_confidence if target_confidence is not None else locked_confidence
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
                        sx = int(round(smoothed_center[0] * (1.0 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA))
                        sy = int(round(smoothed_center[1] * (1.0 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA))
                        smoothed_center = (sx, sy)
                else:
                    lost_frames += 1
                    if lost_frames > LOST_FRAMES_THRESHOLD:
                        tracking_active = False
                        tracker = None
                        locked_face = None
                        name = None
                        smoothed_center = None
                        locked_confidence = None
                        lost_frames = 0

            # If we have a locked_face (tracking) consider that a detection too
            try:
                if locked_face:
                    last_detection_time = time.time()
            except Exception:
                pass

            # If no detections for longer than NO_DETECTION_TIMEOUT → perform the movement sequence
            try:
                current_detection_time = (time.time() - last_detection_time)
                no_detections_now = (not detected_locations) and (not tracking_active) and (locked_face is None)
                if current_detection_time > NO_DETECTION_TIMEOUT and no_detections_now and (not movement_in_progress):
                    if arduino is not None:
                        movement_in_progress = True
                        try:
                            # Movement commands (adjust as needed)
                            data_front_UD = f"p"   # changed in your snippet
                            data_front_LR = f"f"

                            print(f"[NO-DETECT] data_front_UD : {data_front_UD}")
                            print(f"[NO-DETECT] data_front_LR : {data_front_LR}")
                            for _ in range(1):
                                for i in range(10):
                                    try:
                                        if hasattr(arduino, "send_arduino"):
                                            try:
                                                arduino.send_arduino(data_front_UD)
                                            except Exception:
                                                if hasattr(arduino, "write"):
                                                    try:
                                                        arduino.write((data_front_UD + "\n").encode())
                                                    except Exception:
                                                        pass
                                        else:
                                            if hasattr(arduino, "write"):
                                                try:
                                                    arduino.write((data_front_UD + "\n").encode())
                                                except Exception:
                                                    pass
                                    except Exception as e:
                                        print("[NO-DETECT] exception while sending UD:", e)
                                    print(f"[NO-DETECT] sent UD i : {i+1}")
                                    try:
                                        time.sleep(0.02)
                                    except Exception:
                                        pass

                                try:
                                    time.sleep(2.5)  # your original pause between sequences
                                except Exception:
                                    pass

                                for i in range(10):
                                    try:
                                        if hasattr(arduino, "send_arduino"):
                                            try:
                                                arduino.send_arduino(data_front_LR)
                                            except Exception:
                                                if hasattr(arduino, "write"):
                                                    try:
                                                        arduino.write((data_front_LR + "\n").encode())
                                                    except Exception:
                                                        pass
                                        else:
                                            if hasattr(arduino, "write"):
                                                try:
                                                    arduino.write((data_front_LR + "\n").encode())
                                                except Exception:
                                                    pass
                                    except Exception as e:
                                        print("[NO-DETECT] exception while sending LR:", e)
                                    print(f"[NO-DETECT] sent LR i : {i+1}")
                                    try:
                                        time.sleep(0.02)
                                    except Exception:
                                        pass

                            print("[NO-DETECT] movement sequence complete.")
                        finally:
                            movement_in_progress = False

                    last_detection_time = time.time()
            except Exception:
                pass

            # Draw overlay for detections & debug info
            try:
                # draw detected (filtered) faces
                for i, loc in enumerate(detected_locations):
                    bb = face_loc_to_bbox(loc)
                    if bb is None:
                        continue
                    x, y, w, h = bb
                    conf_val = detected_confidences[i] if i < len(detected_confidences) else 0.0
                    nm = detected_names[i] if i < len(detected_names) else "Unknown"
                    prob = detected_probs[i] if i < len(detected_probs) else None

                    # color coding based on confidence
                    if conf_val >= FACE_CONFIDENCE_MIN:
                        color = (0, 200, 0)  # green
                    elif conf_val >= 0.4:
                        color = (0, 200, 200)  # yellow-ish (cyan)
                    else:
                        color = (0, 0, 200)  # red-ish (blue) for low confidence

                    # draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{nm} {int(conf_val*100):d}%"
                    if prob is not None:
                        label += f" p:{prob:.2f}"
                    # background for text for readability
                    (tx, ty), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x, max(0, y - 18)), (x + tx + 6, y), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x + 2, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # if tracking target locked, draw it bigger / distinct
                if locked_face:
                    x, y, w, h = locked_face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
                    if name:
                        label = str(name)
                        try:
                            if locked_confidence is not None:
                                pct = int(round(max(0.0, min(1.0, locked_confidence)) * 100.0))
                                label = f"{label} {pct}%"
                        except Exception:
                            pass
                        (tx, ty), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x, max(0, y - 22)), (x + tx + 8, y), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x + 2, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # HUD: thresholds & counts
                if DEBUG_OVERLAY:
                    hud_lines = [
                        f"detections: {len(detected_locations)}  tracking: {tracking_active}",
                        f"FACENET_th={FACENET_COSINE_THRESHOLD:.2f} CONF_min={FACE_CONFIDENCE_MIN:.2f}",
                        f"face_recog_th={FACE_MATCH_DISTANCE_THRESHOLD:.2f}",
                        f"no_detect_s:{int(current_detection_time if 'current_detection_time' in locals() else time.time()-last_detection_time)}",
                    ]
                    x0, y0 = 8, 18
                    for idx, ln in enumerate(hud_lines):
                        cv2.putText(frame, ln, (x0, y0 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            except Exception:
                pass

            if locked_face:
                x, y, w, h = locked_face
                try:
                    # small green rectangle already drawn above; keep existing name text behavior too
                    if smoothed_center:
                        send_x, send_y = smoothed_center
                    else:
                        send_x = x + w // 2
                        send_y = y + h // 2
                except Exception:
                    send_x = x + w // 2
                    send_y = y + h // 2

                if arduino is not None:
                    if time.time() - last_send_time > SEND_INTERVAL:
                        last_send_time = time.time()
                        data = f"X{int(send_x)}Y{int(send_y)}Z"
                        try:
                            arduino.send_arduino(data)
                        except Exception:
                            try:
                                if hasattr(arduino, "write"):
                                    arduino.write((data + "\n").encode())
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














##
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
##    global current_detection_time, FACENET_AVAILABLE
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
##    # --- FACENET initialization & known embeddings (only if available) ---
##    mtcnn = None
##    facenet_model = None
##    known_names = []
##    known_embeddings = None
##    device = None
##
##    if FACENET_AVAILABLE:
##        try:
##            device = ('cuda' if torch.cuda.is_available() else 'cpu')
##            mtcnn = MTCNN(keep_all=True, device=device)
##            facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
##
##            print("[FACENET INIT] Building known embeddings from images in:", My_Images_Path)
##            imgs_paths = []
##            names = []
##            for root, _, files in os.walk(My_Images_Path):
##                for fn in sorted(files):
##                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
##                        path = os.path.join(root, fn)
##                        base = os.path.splitext(fn)[0]
##                        imgs_paths.append(path)
##                        names.append(base)
##
##            emb_list = []
##            keep_names = []
##            for idx, path in enumerate(imgs_paths):
##                try:
##                    pil = Image.open(path).convert('RGB')
##                    # get face tensors aligned from mtcnn
##                    face_tensors = mtcnn(pil)  # could be None, Tensor (3,160,160) or (n,3,160,160)
##                    if face_tensors is None:
##                        print(f"[FACENET INIT] no face tensor for {path}, skipping")
##                        continue
##                    # normalize shape to (n,3,160,160)
##                    if face_tensors.dim() == 3:
##                        face_tensors = face_tensors.unsqueeze(0)
##                    # take the first face found in each image
##                    ft = face_tensors[0].unsqueeze(0).to(device)
##                    with torch.no_grad():
##                        emb = facenet_model(ft)  # (1,512)
##                    emb_np = emb[0].cpu().numpy()
##                    emb_list.append(emb_np)
##                    keep_names.append(names[idx])
##                    print(f"[FACENET INIT] embedding added for {names[idx]}")
##                except Exception as e:
##                    print("[FACENET INIT] error processing", path, e)
##                    continue
##
##            if emb_list:
##                known_embeddings = np.vstack(emb_list)
##                known_names = keep_names
##                print(f"[FACENET INIT] built {len(known_names)} known embeddings.")
##            else:
##                known_embeddings = None
##                known_names = []
##                print("[FACENET INIT] no known embeddings built.")
##        except Exception as e:
##            # If facenet init fails, disable it and fall back.
##            print("vision.py: facenet init error — falling back. Error:", e)
##            FACENET_AVAILABLE = False
##            mtcnn = None
##            facenet_model = None
##            known_embeddings = None
##            known_names = []
##            device = None
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
##    MIN_FACE_AREA = 30 * 30
##    LOST_FRAMES_THRESHOLD = 6
##    SMOOTH_ALPHA = 0.25
##    SCALE_CHANGE_THRESHOLD = 0.6
##    IOU_THRESHOLD = 0.35
##    PRIMARY_CHECK_INTERVAL = 3.0
##    SEND_INTERVAL = 0.08
##
##    # --- TIGHTER thresholds (changed per your request) ---
##    # Face-confidence / facenet specific settings
##    FACE_CONFIDENCE_ENABLED = True and (FACENET_AVAILABLE or FACE_RECOG_AVAILABLE)
##    FACE_MATCH_DISTANCE_THRESHOLD = 0.45  # stricter for face_recognition fallback
##    FACE_CONFIDENCE_MIN = 0.52           # minimum mapped confidence (0..1) to accept a match (raised)
##    FACENET_COSINE_THRESHOLD = 0.52      # cosine similarity threshold (higher is stricter)
##
##    # If encoding comparison fails but face bounding box area is large enough, accept as "Unknown".
##    MIN_FACE_AREA_KEEP_FOR_UNKNOWN = MIN_FACE_AREA
##
##    # NO-DETECTION behavior: configurable timeout (seconds). Default 3 minutes (180s).
##    try:
##        NO_DETECTION_TIMEOUT = float(getattr(Alfred_config, "NO_DETECTION_TIMEOUT", 180.0))
##    except Exception:
##        NO_DETECTION_TIMEOUT = 180.0
##
##    tracker = None
##    tracking_active = False
##    locked_face = None
##    name = None
##    locked_confidence = None
##    last_speaker = None
##    frame_idx = 0
##    lost_frames = 0
##    smoothed_center = None
##    last_primary_check = time.time()
##    last_send_time = 0.0
##
##    # Track last time we had any detection (face detected OR tracking active)
##    last_detection_time = time.time()
##
##    # guard to prevent races when movement sequence is being executed
##    movement_in_progress = False
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
##                            print("Primary recovered — switched back.")
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
##                locked_confidence = None
##                if current_speaker:
##                    print(f"Vision: switching target to speaker: {current_speaker}")
##
##            # --- DETECTION: convert arrays -> lists to avoid ambiguous truth tests ---
##            detected_locations = []
##            detected_names = []
##            detected_confidences = []
##            detected_probs = []  # MTCNN probabilities (if available)
##
##            if (not tracking_active) or (frame_idx % DETECTION_INTERVAL == 0):
##                # Preferred: use facenet-pytorch MTCNN + InceptionResnet for detection+embedding if available
##                if FACENET_AVAILABLE and mtcnn is not None and facenet_model is not None:
##                    try:
##                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                        pil_im = Image.fromarray(rgb)
##                        # detect boxes + probs and get aligned face tensors
##                        boxes, probs = mtcnn.detect(pil_im)
##                        face_tensors = mtcnn(pil_im)  # returns None, or tensor (3,160,160) or (n,3,160,160)
##                        if probs is None:
##                            probs_list = None
##                        else:
##                            probs_list = list(probs)
####                        print(f"[FACENET DETECT] boxes={None if boxes is None else len(boxes)}, probs={None if probs is None else ['{:.2f}'.format(p) for p in probs_list]}")
##                        if face_tensors is None:
##                            # no face tensors found; ensure empty lists
##                            detected_locations = []
##                            detected_names = []
##                            detected_confidences = []
##                            detected_probs = []
##                        else:
##                            # normalize to (n,3,160,160)
##                            if face_tensors.dim() == 3:
##                                face_tensors = face_tensors.unsqueeze(0)
##                            for i_ft, ft in enumerate(face_tensors):
##                                try:
##                                    # boxes may be None or shorter than tensors — protect indices
##                                    if boxes is not None and i_ft < len(boxes):
##                                        b = boxes[i_ft]
##                                        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
##                                    else:
##                                        # fallback: skip if no box mapping
##                                        continue
##
##                                    w = x2 - x1; h = y2 - y1
##                                    if w * h < MIN_FACE_AREA:
##                                        continue
##
##                                    with torch.no_grad():
##                                        emb = facenet_model(ft.unsqueeze(0).to(device))
##                                        emb_np = emb[0].cpu().numpy()
##
##                                    matched_name = "Unknown"
##                                    conf_val = 0.0
##                                    if known_embeddings is not None and known_embeddings.shape[0] > 0:
##                                        e_norm = emb_np / (np.linalg.norm(emb_np) + 1e-8)
##                                        known_norms = known_embeddings / (np.linalg.norm(known_embeddings, axis=1, keepdims=True) + 1e-8)
##                                        sims = np.dot(known_norms, e_norm)  # cosine sims
##                                        best_idx = int(np.argmax(sims))
##                                        best_sim = float(sims[best_idx])
##                                        # compute a 0..1 confidence when > threshold
##                                        if best_sim >= FACENET_COSINE_THRESHOLD:
##                                            conf_val = float((best_sim - FACENET_COSINE_THRESHOLD) / (1.0 - FACENET_COSINE_THRESHOLD))
##                                            matched_name = known_names[best_idx]
##                                        else:
##                                            conf_val = 0.0
##                                            matched_name = "Unknown"
##                                        if DEBUG_OVERLAY:
##                                            print(f"[FACENET MATCH] face#{i_ft} best_sim={best_sim:.3f} -> name={matched_name} conf={conf_val:.3f}")
##                                    else:
##                                        conf_val = 0.0
##                                        matched_name = "Unknown"
##
##                                    # record in same format used by your face pipeline (top,right,bottom,left)
##                                    detected_locations.append((y1, x2, y2, x1))
##                                    detected_names.append(matched_name)
##                                    detected_confidences.append(conf_val)
##                                    detected_probs.append(float(probs_list[i_ft]) if (probs_list is not None and i_ft < len(probs_list)) else None)
##                                except Exception as e:
##                                    if DEBUG_OVERLAY:
##                                        print("[FACENET DETECT] per-face error:", e)
##                                    continue
##                    except Exception as e:
##                        if DEBUG_OVERLAY:
##                            print("[FACENET DETECT] error, falling back to SimpleFacerec:", e)
####                        # fall back to SimpleFacerec if facenet detection fails mid-loop
####                        try:
####                            detected_locations, detected_names = sfr.detect_known_faces(frame)
####                        except Exception:
####                            try:
####                                rgb_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
####                                detected_locations, detected_names = sfr.detect_known_faces(rgb_tmp)
####                            except Exception:
####                                detected_locations, detected_names = [], []
####                        # confidences will be produced below
##                else:
##                    # fallback: use existing SimpleFacerec detection
##                    try:
##                        try:
##                            detected_locations, detected_names = sfr.detect_known_faces(frame)
##                        except Exception:
##                            rgb_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                            detected_locations, detected_names = sfr.detect_known_faces(rgb_tmp)
##                        if detected_locations is None:
##                            detected_locations = []
##                        elif isinstance(detected_locations, np.ndarray):
##                            try:
##                                detected_locations = detected_locations.tolist()
##                            except Exception:
##                                detected_locations = list(detected_locations)
##                        else:
##                            try:
##                                detected_locations = list(detected_locations)
##                            except Exception:
##                                detected_locations = []
##
##                        if detected_names is None:
##                            detected_names = []
##                        elif isinstance(detected_names, np.ndarray):
##                            try:
##                                detected_names = [str(n) for n in detected_names.tolist()]
##                            except Exception:
##                                detected_names = [str(n) for n in detected_names]
##                        else:
##                            try:
##                                detected_names = [str(n) for n in detected_names]
##                            except Exception:
##                                detected_names = []
##                    except Exception:
##                        detected_locations, detected_names = [], []
##
##                    # If face_recognition is available, compute confidences as before
##                    if FACE_RECOG_AVAILABLE and getattr(sfr, 'known_face_encodings', None):
##                        try:
##                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
##                            known_encodings = getattr(sfr, 'known_face_encodings', None)
##                            for i, loc in enumerate(detected_locations):
##                                nm = None
##                                try:
##                                    nm = detected_names[i] if i < len(detected_names) else None
##                                except Exception:
##                                    nm = None
##                                conf_val = 1.0 if (nm and str(nm).strip().lower() not in ("unknown", "")) else 0.0
##                                bb = face_loc_to_bbox(loc)
##                                if bb is None:
##                                    detected_confidences.append(conf_val)
##                                    detected_probs.append(None)
##                                    continue
##                                x, y, w, h = bb
##                                try:
##                                    crop = rgb[y:y+h, x:x+w]
##                                    encs = face_recognition.face_encodings(crop)
##                                    if not encs:
##                                        top = y; left = x; bottom = y + h; right = x + w
##                                        full_encs = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])
##                                        if full_encs:
##                                            enc = full_encs[0]
##                                        else:
##                                            enc = None
##                                    else:
##                                        enc = encs[0]
##                                    if enc is None:
##                                        detected_confidences.append(conf_val)
##                                        detected_probs.append(None)
##                                        continue
##                                    distances = face_recognition.face_distance(known_encodings, enc)
##                                    if distances is None or len(distances) == 0:
##                                        detected_confidences.append(conf_val)
##                                        detected_probs.append(None)
##                                        continue
##                                    min_idx = int(np.argmin(distances))
##                                    min_dist = float(distances[min_idx])
##                                    mapped_conf = max(0.0, min(1.0, (FACE_MATCH_DISTANCE_THRESHOLD - min_dist) / FACE_MATCH_DISTANCE_THRESHOLD))
##                                    conf_val = mapped_conf
##                                except Exception:
##                                    conf_val = conf_val
##                                detected_confidences.append(conf_val)
##                                detected_probs.append(None)
##                        except Exception:
##                            # default confidences
##                            detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
##                            detected_probs = [None] * len(detected_confidences)
##                    else:
##                        detected_confidences = [1.0 if (n and str(n).strip().lower() not in ("unknown", "")) else 0.0 for n in detected_names]
##                        detected_probs = [None] * len(detected_confidences)
##
##                # --- Filter out very low-confidence detections (to reduce false positives) ---
##                filtered_locations = []
##                filtered_names = []
##                filtered_confidences = []
##                filtered_probs = []
##                for i, loc in enumerate(detected_locations):
##                    bb = face_loc_to_bbox(loc)
##                    area = 0
##                    if bb is not None:
##                        area = bb[2] * bb[3]
##                    conf_val = detected_confidences[i] if i < len(detected_confidences) else 0.0
##                    nm = detected_names[i] if i < len(detected_names) else None
##                    prob = detected_probs[i] if i < len(detected_probs) else None
##
##                    accept = False
##                    if (FACENET_AVAILABLE and known_embeddings is not None):
##                        # use facenet confidence rules
##                        if conf_val >= FACE_CONFIDENCE_MIN:
##                            accept = True
##                        else:
##                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                                accept = True
##                    elif FACE_RECOG_AVAILABLE:
##                        if conf_val >= FACE_CONFIDENCE_MIN:
##                            accept = True
##                        else:
##                            if area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                                accept = True
##                    else:
##                        if nm and str(nm).strip().lower() not in ("unknown", ""):
##                            accept = True
##                        elif area >= MIN_FACE_AREA_KEEP_FOR_UNKNOWN:
##                            accept = True
##                        else:
##                            accept = False
##
##                    if accept:
##                        filtered_locations.append(loc)
##                        filtered_names.append(nm if nm is not None else "Unknown")
##                        filtered_confidences.append(conf_val)
##                        filtered_probs.append(prob)
##                    else:
##                        if DEBUG_OVERLAY:
##                            # debug print of rejected detection
##                            try:
##                                print(f"[DETECT FILTER] rejected loc area={area} conf={conf_val:.3f} name={nm}")
##                            except Exception:
##                                pass
##                        pass
##
##                detected_locations = filtered_locations
##                detected_names = filtered_names
##                detected_confidences = filtered_confidences
##                detected_probs = filtered_probs
##
##                if detected_confidences is None:
##                    detected_confidences = [1.0] * len(detected_locations)
##                    detected_probs = [None] * len(detected_locations)
##                elif len(detected_confidences) != len(detected_locations):
##                    detected_confidences = (detected_confidences + [0.0] * len(detected_locations))[:len(detected_locations)]
##                    detected_probs = (detected_probs + [None] * len(detected_locations))[:len(detected_locations)]
##
##            # Update last_detection_time whenever we have detection results to avoid the no-detection trigger
##            try:
##                if (detected_locations and len(detected_locations) > 0) or tracking_active or locked_face:
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            target_bbox = None
##            target_name = None
##            target_confidence = None
##
##            # If there's no current speaker, try to pick one from face recognition results
##            if not current_speaker:
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
##                    is_known = bool(nm and str(nm).strip().lower() not in ("unknown", ""))
##                    score = (100000 if is_known else 0) + area
##                    if score > best_area:
##                        best_area = score
##                        best_idx = i
##
##                if best_idx is not None:
##                    chosen = detected_locations[best_idx]
##                    bb = face_loc_to_bbox(chosen)
##                    if bb is not None:
##                        target_bbox = bb
##                    try:
##                        target_name = detected_names[best_idx] if best_idx < len(detected_names) else None
##                    except Exception:
##                        target_name = None
##
##                    try:
##                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
##                    except Exception:
##                        target_confidence = None
##
##                    if target_name and str(target_name).strip().lower() not in ("unknown", ""):
##                        try:
##                            _set_current_speaker(target_name)
##                            last_speaker = target_name
##                            print(f"Vision: No prior speaker → set speaker to detected person: {target_name}")
##                        except Exception:
##                            last_speaker = target_name
##                    else:
##                        if target_name is None:
##                            target_name = "Unknown"
##
##            # SAFE selection fallback
##            if target_bbox is None and len(detected_locations) > 0:
##                bboxes = []
##                for loc in detected_locations:
##                    b = face_loc_to_bbox(loc)
##                    bboxes.append(b)
##
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
##                    try:
##                        target_confidence = detected_confidences[best_idx] if best_idx < len(detected_confidences) else None
##                    except Exception:
##                        target_confidence = None
##
##            # --- tracker init / update / draw / send logic ---
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
##                                locked_confidence = target_confidence if target_confidence is not None else (1.0 if name and str(name).strip().lower() not in ("unknown", "") else 0.0)
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
##                                        locked_confidence = target_confidence if target_confidence is not None else locked_confidence
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
##                        name = None
##                        smoothed_center = None
##                        locked_confidence = None
##                        lost_frames = 0
##
##            # If we have a locked_face (tracking) consider that a detection too
##            try:
##                if locked_face:
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            # If no detections for longer than NO_DETECTION_TIMEOUT → perform the movement sequence
##            try:
##                current_detection_time = (time.time() - last_detection_time)
##                no_detections_now = (not detected_locations) and (not tracking_active) and (locked_face is None)
##                if current_detection_time > NO_DETECTION_TIMEOUT and no_detections_now and (not movement_in_progress):
##                    if arduino is not None:
##                        movement_in_progress = True
##                        try:
##                            # Movement commands (adjust as needed)
##                            data_front_UD = f"p"   # changed in your snippet
##                            data_front_LR = f"f"
##
##                            print(f"[NO-DETECT] data_front_UD : {data_front_UD}")
##                            print(f"[NO-DETECT] data_front_LR : {data_front_LR}")
##                            for _ in range(1):
##                                for i in range(10):
##                                    try:
##                                        if hasattr(arduino, "send_arduino"):
##                                            try:
##                                                arduino.send_arduino(data_front_UD)
##                                            except Exception:
##                                                if hasattr(arduino, "write"):
##                                                    try:
##                                                        arduino.write((data_front_UD + "\n").encode())
##                                                    except Exception:
##                                                        pass
##                                        else:
##                                            if hasattr(arduino, "write"):
##                                                try:
##                                                    arduino.write((data_front_UD + "\n").encode())
##                                                except Exception:
##                                                    pass
##                                    except Exception as e:
##                                        print("[NO-DETECT] exception while sending UD:", e)
##                                    print(f"[NO-DETECT] sent UD i : {i+1}")
##                                    try:
##                                        time.sleep(0.02)
##                                    except Exception:
##                                        pass
##
##                                try:
##                                    time.sleep(2.5)  # your original pause between sequences
##                                except Exception:
##                                    pass
##
##                                for i in range(10):
##                                    try:
##                                        if hasattr(arduino, "send_arduino"):
##                                            try:
##                                                arduino.send_arduino(data_front_LR)
##                                            except Exception:
##                                                if hasattr(arduino, "write"):
##                                                    try:
##                                                        arduino.write((data_front_LR + "\n").encode())
##                                                    except Exception:
##                                                        pass
##                                        else:
##                                            if hasattr(arduino, "write"):
##                                                try:
##                                                    arduino.write((data_front_LR + "\n").encode())
##                                                except Exception:
##                                                    pass
##                                    except Exception as e:
##                                        print("[NO-DETECT] exception while sending LR:", e)
##                                    print(f"[NO-DETECT] sent LR i : {i+1}")
##                                    try:
##                                        time.sleep(0.02)
##                                    except Exception:
##                                        pass
##
##                            print("[NO-DETECT] movement sequence complete.")
##                        finally:
##                            movement_in_progress = False
##
##                    last_detection_time = time.time()
##            except Exception:
##                pass
##
##            # Draw overlay for detections & debug info
##            try:
##                # draw detected (filtered) faces
##                for i, loc in enumerate(detected_locations):
##                    bb = face_loc_to_bbox(loc)
##                    if bb is None:
##                        continue
##                    x, y, w, h = bb
##                    conf_val = detected_confidences[i] if i < len(detected_confidences) else 0.0
##                    nm = detected_names[i] if i < len(detected_names) else "Unknown"
##                    prob = detected_probs[i] if i < len(detected_probs) else None
##
##                    # color coding based on confidence
##                    if conf_val >= FACE_CONFIDENCE_MIN:
##                        color = (0, 200, 0)  # green
##                    elif conf_val >= 0.4:
##                        color = (0, 200, 200)  # yellow-ish (cyan)
##                    else:
##                        color = (0, 0, 200)  # red-ish (blue) for low confidence
##
##                    # draw rectangle and label
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
##                    label = f"{nm} {int(conf_val*100):d}%"
##                    if prob is not None:
##                        label += f" p:{prob:.2f}"
##                    # background for text for readability
##                    (tx, ty), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
##                    cv2.rectangle(frame, (x, max(0, y - 18)), (x + tx + 6, y), (0, 0, 0), -1)
##                    cv2.putText(frame, label, (x + 2, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
##
##                # if tracking target locked, draw it bigger / distinct
##                if locked_face:
##                    x, y, w, h = locked_face
##                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), 2)
##                    if name:
##                        label = str(name)
##                        try:
##                            if locked_confidence is not None:
##                                pct = int(round(max(0.0, min(1.0, locked_confidence)) * 100.0))
##                                label = f"{label} {pct}%"
##                        except Exception:
##                            pass
##                        (tx, ty), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
##                        cv2.rectangle(frame, (x, max(0, y - 22)), (x + tx + 8, y), (0, 0, 0), -1)
##                        cv2.putText(frame, label, (x + 2, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
##
##                # HUD: thresholds & counts
##                if DEBUG_OVERLAY:
##                    hud_lines = [
##                        f"detections: {len(detected_locations)}  tracking: {tracking_active}",
##                        f"FACENET_th={FACENET_COSINE_THRESHOLD:.2f} CONF_min={FACE_CONFIDENCE_MIN:.2f}",
##                        f"face_recog_th={FACE_MATCH_DISTANCE_THRESHOLD:.2f}",
##                        f"no_detect_s:{int(current_detection_time if 'current_detection_time' in locals() else time.time()-last_detection_time)}",
##                    ]
##                    x0, y0 = 8, 18
##                    for idx, ln in enumerate(hud_lines):
##                        cv2.putText(frame, ln, (x0, y0 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
##
##            except Exception:
##                pass
##
##            if locked_face:
##                x, y, w, h = locked_face
##                try:
##                    # small green rectangle already drawn above; keep existing name text behavior too
##                    if smoothed_center:
##                        send_x, send_y = smoothed_center
##                    else:
##                        send_x = x + w // 2
##                        send_y = y + h // 2
##                except Exception:
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
##                            try:
##                                if hasattr(arduino, "write"):
##                                    arduino.write((data + "\n").encode())
##                            except Exception:
##                                pass
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
##




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


                        data_front_UD = f"D{240}Z"
                        data_front_LR = f"F{320}Z"
                                
                        print(f"data_front_UD : {data_front_UD}")
                        print(f"data_front_LR : {data_front_LR}")

                        while True:

                            for i in range(10):
                                i = i + 1
                                arduino.send_arduino(data_front_UD)
                                print(f" i : {i}")

                            for i in range(10):
                                i = i + 1
                                arduino.send_arduino(data_front_LR)
                                print(f" i : {i}")

                            break





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







