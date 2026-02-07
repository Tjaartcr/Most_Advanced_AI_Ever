



### ---------------------------
### SimpleFacerec (improved detect_known_faces fallbacks)
### ---------------------------
##import face_recognition
##import cv2
##import os
##import glob
##import numpy as np
##import pyttsx3  # pip install pyttsx3
##import speech_recognition as sr
##import fnmatch
##import Alfred_config
##
##from speech import speech
##
##from PIL import Image, ImageOps  # pip install pillow
##
##Number_Images = str(0)
##
##def AlfredSpeak_Start(audio):
##    engine = pyttsx3.init('sapi5')
##    engine.setProperty('rate', 190)
##    voices = engine.getProperty('voices')
##    engine.setProperty('voice', voices[0].id)
##    engine.setProperty('volume', 1)
##    print('engine: ' + str(audio), end = "\r")
##    print('\033c', end = '')
##    engine.say(audio)
##    engine.runAndWait()
##
##
##class SimpleFacerec:
##
##    def __init__(self):
##        self.known_face_encodings = []
##        self.known_face_names = []
##
##        # Resize frame for a faster speed
##        self.frame_resizing = 1
##
##        # Option: maximum image dimension to avoid extremely large images slowing detection.
##        self.max_image_dim = 1600
##
##        # Haar cascade for fallback face localization (OpenCV).
##        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
##
##        # Debug output folder for failing images and attempt images
##        self.debug_fail_dir = os.path.join(os.getcwd(), "face_debug_failures")
##        os.makedirs(self.debug_fail_dir, exist_ok=True)
##
##    def _ensure_uint8_contiguous(self, arr):
##        """
##        Ensure array is a numpy ndarray, C-contiguous and dtype uint8 (RGB order).
##        This fixes dlib / face_recognition 'Unsupported image type' errors.
##        """
##        if arr is None:
##            return arr
##        # Convert non-ndarray inputs to ndarray
##        if not isinstance(arr, np.ndarray):
##            try:
##                arr = np.asarray(arr)
##            except Exception:
##                # fallback: return as-is; caller will surface an error
##                return arr
##        # Safe check for C contiguous (flagsobj doesn't have .get)
##        try:
##            is_c_contig = bool(getattr(arr.flags, "c_contiguous", False))
##        except Exception:
##            # final fallback - try numpy function
##            is_c_contig = arr.flags['C_CONTIGUOUS'] if 'C_CONTIGUOUS' in arr.flags else False
##
##        if arr.dtype != np.uint8 or not is_c_contig:
##            arr = np.ascontiguousarray(arr, dtype=np.uint8)
##        return arr
##
##    def _open_as_rgb_numpy(self, img_path):
##        try:
##            pil_img = Image.open(img_path)
##        except Exception as e:
##            raise RuntimeError(f"Could not open image {img_path}: {e}")
##
##        try:
##            pil_img = ImageOps.exif_transpose(pil_img)
##        except Exception:
##            pass
##
##        try:
##            pil_img = pil_img.convert('RGB')
##        except Exception as e:
##            raise RuntimeError(f"Could not convert image {img_path} to RGB: {e}")
##
##        if self.max_image_dim is not None:
##            w, h = pil_img.size
##            max_dim = max(w, h)
##            if max_dim > self.max_image_dim:
##                scale = self.max_image_dim / float(max_dim)
##                new_w = int(w * scale)
##                new_h = int(h * scale)
##                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
##
##        arr = np.asarray(pil_img)
##        if arr.dtype != np.uint8:
##            arr = arr.astype(np.uint8)
##
##        # ensure contiguous uint8 (important for dlib)
##        arr = self._ensure_uint8_contiguous(arr)
##        return arr
##
##    # -------------------------
##    # Extra aggressive preprocess helpers
##    # -------------------------
##    def _apply_clahe_rgb(self, rgb, clip=2.0):
##        try:
##            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
##            y, cr, cb = cv2.split(ycrcb)
##            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
##            y2 = clahe.apply(y)
##            merged = cv2.merge((y2, cr, cb))
##            out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
##            return out
##        except Exception:
##            return rgb
##
##    def _equalize_rgb(self, rgb):
##        try:
##            channels = cv2.split(rgb)
##            eq = [cv2.equalizeHist(ch) for ch in channels]
##            return cv2.merge(eq)
##        except Exception:
##            return rgb
##
##    def _histogram_stretch(self, rgb):
##        try:
##            out = np.zeros_like(rgb)
##            for i in range(3):
##                ch = rgb[:,:,i]
##                p2, p98 = np.percentile(ch, (2, 98))
##                if p98 - p2 > 0:
##                    out[:,:,i] = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
##                else:
##                    out[:,:,i] = ch
##            return out
##        except Exception:
##            return rgb
##
##    def _gamma_correction(self, rgb, gamma):
##        inv = 1.0 / float(gamma)
##        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
##        return cv2.LUT(rgb, table)
##
##    def _denoise(self, rgb):
##        try:
##            return cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
##        except Exception:
##            return rgb
##
##    def _sharpen(self, rgb):
##        try:
##            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
##            return cv2.filter2D(rgb, -1, kernel)
##        except Exception:
##            return rgb
##
##    def _try_rotations_and_flips(self, rgb):
##        """Yield rotated and flipped variants that may reveal faces."""
##        variants = []
##        variants.append(rgb)  # original
##        variants.append(cv2.flip(rgb, 1))  # horizontal flip
##        for angle in (-15, -10, 10, 15):
##            (h, w) = rgb.shape[:2]
##            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
##            rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
##            variants.append(rotated)
##            variants.append(cv2.flip(rotated, 1))
##        return variants
##
##    def _haar_detect_and_crop(self, rgb_img, padding_ratio=0.25, scaleFactor=1.1, minNeighbors=4):
##        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
##        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(24, 24))
##        crops = []
##        h, w = gray.shape
##        for (x, y, fw, fh) in faces:
##            pad_w = int(fw * padding_ratio)
##            pad_h = int(fh * padding_ratio)
##            x1 = max(0, x - pad_w)
##            y1 = max(0, y - pad_h)
##            x2 = min(w, x + fw + pad_w)
##            y2 = min(h, y + fh + pad_h)
##            crop = rgb_img[y1:y2, x1:x2]
##            if crop.size != 0:
##                crop = self._ensure_uint8_contiguous(crop)
##                crops.append(crop)
##        return crops
##
##    def _save_debug_thumbnail_and_raise(self, img_path, rgb_img, reason, attempts=None):
##        basename = os.path.basename(img_path)
##        name, _ = os.path.splitext(basename)
##        out_path = os.path.join(self.debug_fail_dir, f"{name}_fail_thumb.jpg")
##        try:
##            pil = Image.fromarray(rgb_img)
##            pil.thumbnail((256,256))
##            pil.save(out_path, "JPEG", quality=85)
##        except Exception:
##            out_path = None
##        # Save attempts images for inspection (if provided)
##        if attempts:
##            att_dir = os.path.join(self.debug_fail_dir, f"{name}_attempts")
##            os.makedirs(att_dir, exist_ok=True)
##            for i, (label, arr) in enumerate(attempts):
##                try:
##                    p = os.path.join(att_dir, f"{i:02d}_{label}.jpg")
##                    Image.fromarray(arr).save(p, "JPEG", quality=85)
##                except Exception:
##                    pass
##        msg = f"No face could be detected in image {img_path}. Reason: {reason}."
##        if out_path:
##            msg += f" A thumbnail was saved to: {out_path}"
##        raise RuntimeError(msg)
##
##    # -------------------------
##    # Main loader (more aggressive)
##    # -------------------------
##    def load_encoding_images(self, images_path):
##        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")
##
##        images_path_list = []
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpg")))
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpeg")))
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.png")))
##        images_path_list = sorted(images_path_list)
##
##        Number_Images = len(images_path_list)
##
##        print(f" There are {Number_Images} faces found for the facial recognition system.")
##        speech.AlfredSpeak(f"There are {Number_Images} faces found for the facial recognition system.")
####        AlfredSpeak_Start(f"There are {Number_Images} faces found for the facial recognition system.")
##
##        for img_path in images_path_list:
##            rgb_img = self._open_as_rgb_numpy(img_path)
##            rgb_img = self._ensure_uint8_contiguous(rgb_img)  # <-- ensure contiguous here too
##            basename = os.path.basename(img_path)
##            (filename, ext) = os.path.splitext(basename)
##
##            # Keep track of attempts for debug saving if all fail
##            attempts = [("original", rgb_img.copy())]
##
##            # 1) Try direct encoding
##            encodings = []
##            try:
##                encodings = face_recognition.face_encodings(self._ensure_uint8_contiguous(rgb_img))
##                if encodings:
##                    # success
##                    pass
##            except Exception:
##                encodings = []
##
##            # 2) Try earlier preprocessing set (CLAHE, equalize, gamma)
##            if not encodings:
##                candidates = []
##                candidates.append(self._apply_clahe_rgb(rgb_img, clip=2.0))
##                candidates.append(self._equalize_rgb(rgb_img))
##                candidates.append(self._histogram_stretch(rgb_img))
##                for g in (0.8, 1.2, 1.5):
##                    candidates.append(self._gamma_correction(rgb_img, g))
##                # keep also sharpened and denoised versions
##                candidates.append(self._sharpen(rgb_img))
##                candidates.append(self._denoise(rgb_img))
##
##                for i, cand in enumerate(candidates):
##                    # ensure contiguous for each candidate
##                    cand = self._ensure_uint8_contiguous(cand)
##                    attempts.append((f"pre{i}", cand.copy()))
##                    try:
##                        enc = face_recognition.face_encodings(cand)
##                        if enc:
##                            encodings = enc
##                            rgb_img = cand
##                            break
##                    except Exception:
##                        continue
##
##            # 3) Try rotations/flips on original and on best preprocessed candidate
##            if not encodings:
##                rotations = self._try_rotations_and_flips(rgb_img)
##                for j, r in enumerate(rotations):
##                    r = self._ensure_uint8_contiguous(r)
##                    attempts.append((f"rot{j}", r.copy()))
##                    try:
##                        enc = face_recognition.face_encodings(r)
##                        if enc:
##                            encodings = enc
##                            rgb_img = r
##                            break
##                    except Exception:
##                        continue
##
##            # 4) Try Haar with relaxed parameters and cropping; also on variants
##            if not encodings:
##                # try stronger CLAHE and histogram stretch on original to feed Haar
##                stronger = self._apply_clahe_rgb(rgb_img, clip=4.0)
##                stronger = self._ensure_uint8_contiguous(stronger)
##                attempts.append(("clahe_strong", stronger.copy()))
##                crops = self._haar_detect_and_crop(stronger, padding_ratio=0.4, scaleFactor=1.05, minNeighbors=3)
##                # also try on histogram stretched original
##                stretched = self._histogram_stretch(rgb_img)
##                stretched = self._ensure_uint8_contiguous(stretched)
##                attempts.append(("stretched", stretched.copy()))
##                crops += self._haar_detect_and_crop(stretched, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
##                # expand: look at rotated variants too
##                for idx, variant in enumerate(self._try_rotations_and_flips(rgb_img)):
##                    variant = self._ensure_uint8_contiguous(variant)
##                    attempts.append((f"rot_haar{idx}", variant.copy()))
##                    crops += self._haar_detect_and_crop(variant, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
##                # Evaluate each crop and some preprocess variants of crop
##                for ci, crop in enumerate(crops):
##                    crop = self._ensure_uint8_contiguous(crop)
##                    attempts.append((f"crop{ci}", crop.copy()))
##                    # try crop as-is
##                    try:
##                        enc = face_recognition.face_encodings(crop)
##                        if enc:
##                            encodings = enc
##                            rgb_img = crop
##                            break
##                    except Exception:
##                        pass
##                    # try processed crops
##                    for pc in (self._apply_clahe_rgb(crop, clip=2.5),
##                               self._equalize_rgb(crop),
##                               self._histogram_stretch(crop),
##                               self._sharpen(crop)):
##                        pc = self._ensure_uint8_contiguous(pc)
##                        attempts.append((f"crop{ci}_proc", pc.copy()))
##                        try:
##                            enc = face_recognition.face_encodings(pc)
##                            if enc:
##                                encodings = enc
##                                rgb_img = pc
##                                break
##                        except Exception:
##                            continue
##                    if encodings:
##                        break
##
##            # 5) Final: try cnn detector if available
##            if not encodings:
##                try:
##                    fl = face_recognition.face_locations(self._ensure_uint8_contiguous(rgb_img), model='cnn')
##                    if fl:
##                        enc = face_recognition.face_encodings(self._ensure_uint8_contiguous(rgb_img), fl)
##                        if enc:
##                            encodings = enc
##                except Exception:
##                    pass
##
##            # If still nothing: save attempts and raise
##            if not encodings:
##                reason = "attempted raw -> preprocessing (CLAHE/equalize/stretch/gamma/sharpen/denoise) -> rotations/flips -> Haar crop (relaxed) -> cnn detector"
##                # cap attempts saved to avoid huge disk use
##                limited_attempts = attempts[:40]
##                self._save_debug_thumbnail_and_raise(img_path, rgb_img, reason, attempts=limited_attempts)
##
##            # Use the first encoding found
##            img_encoding = encodings[0]
##            self.known_face_encodings.append(img_encoding)
##            self.known_face_names.append(filename)
##            print(f"[OK] Encoded {filename}")
##
##        print(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
##        speech.AlfredSpeak(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
####        AlfredSpeak_Start(f"There were a total of {Number_Images} faces encoded and loaded for the facial recognition system.")
##
##
##    def detect_known_faces(self, frame):
##        """
##        Returns (face_locations, face_names)
##        - face_locations = numpy array of (top, right, bottom, left) in FULL image coordinates (ints)
##        - face_names = list of matched names in same order
##        This function will:
##          * try face_recognition with 'hog' first (fast),
##          * fallback to 'cnn' if nothing found,
##          * finally try Haar cascade as a fallback to produce candidate boxes.
##        """
##        # Resize frame for speed if requested
##        try:
##            small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing, interpolation=cv2.INTER_LINEAR)
##        except Exception:
##            small_frame = frame
##
##        # Convert BGR->RGB if necessary (we expect BGR input typically from cv2)
##        try:
##            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
##        except Exception:
##            # If conversion fails, assume already RGB
##            rgb_small_frame = small_frame
##
##        # Ensure contiguous uint8
##        rgb_small_frame = self._ensure_uint8_contiguous(rgb_small_frame)
##
##        face_locations = []
##        try:
##            # Try 'hog' (fast). If nothing found, try 'cnn' (more accurate but slower).
##            try:
##                face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
##            except Exception:
##                face_locations = []
##
##            if not face_locations:
##                try:
##                    face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
##                except Exception:
##                    # If cnn unavailable, keep empty
##                    face_locations = []
##
##        except Exception:
##            face_locations = []
##
##        # If still no detections, try Haar cascade to generate candidate boxes
##        if not face_locations:
##            try:
##                gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
##                haar_faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
##                if len(haar_faces) > 0:
##                    # convert (x,y,w,h) to (top, right, bottom, left)
##                    face_locations = []
##                    for (x, y, w, h) in haar_faces:
##                        top = int(y)
##                        left = int(x)
##                        bottom = int(y + h)
##                        right = int(x + w)
##                        face_locations.append((top, right, bottom, left))
##            except Exception:
##                pass
##
##        # Compute encodings for the detections we have (if any)
##        face_encodings = []
##        try:
##            if face_locations:
##                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
##            else:
##                face_encodings = []
##        except Exception:
##            # If encoding fails, try a safer fallback: full-image encodings (slower)
##            try:
##                face_encodings = face_recognition.face_encodings(rgb_small_frame)
##            except Exception:
##                face_encodings = []
##
##        # Build names list by comparing distances to known encodings
##        face_names = []
##        for encoding in face_encodings:
##            name = "Unknown"
##            try:
##                if self.known_face_encodings:
##                    matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
##                    face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
##                    best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
##                    if best_match_index is not None and matches and matches[best_match_index]:
##                        name = self.known_face_names[best_match_index]
##            except Exception:
##                name = "Unknown"
##            face_names.append(name)
##
##        # If we have fewer names than locations (rare), pad with "Unknown"
##        try:
##            if len(face_names) < len(face_locations):
##                face_names = face_names + ["Unknown"] * (len(face_locations) - len(face_names))
##        except Exception:
##            pass
##
##        # Convert face_locations list (top,right,bottom,left) to numpy array and scale to original frame size
##        try:
##            fl_arr = np.asarray(face_locations, dtype=float)
##            if fl_arr.size == 0:
##                # empty
##                return np.array([]), face_names
##            # Scale up if we downscaled previously
##            if self.frame_resizing and self.frame_resizing != 1:
##                fl_arr = fl_arr / float(self.frame_resizing)
##            fl_arr = fl_arr.astype(int)
##            return fl_arr, face_names
##        except Exception:
##            # Last resort: attempt manual conversion
##            out_locs = []
##            for loc in face_locations:
##                try:
##                    top, right, bottom, left = loc
##                    if self.frame_resizing and self.frame_resizing != 1:
##                        top = int(round(top / self.frame_resizing))
##                        right = int(round(right / self.frame_resizing))
##                        bottom = int(round(bottom / self.frame_resizing))
##                        left = int(round(left / self.frame_resizing))
##                    out_locs.append((int(top), int(right), int(bottom), int(left)))
##                except Exception:
##                    pass
##            return np.asarray(out_locs, dtype=int), face_names



















####import face_recognition
####import cv2
####import os
####import glob
####import numpy as np
####import pyttsx3  # pip install pyttsx3
####import speech_recognition as sr
####import fnmatch
####import Alfred_config
####
####from speech import speech
####
####from PIL import Image, ImageOps  # pip install pillow
####
####Number_Images = str(0)
####
####def AlfredSpeak_Start(audio):
####    engine = pyttsx3.init('sapi5')
####    engine.setProperty('rate', 190)
####    voices = engine.getProperty('voices')
####    engine.setProperty('voice', voices[0].id)
####    engine.setProperty('volume', 1)
####    print('engine: ' + str(audio), end = "\r")
####    print('\033c', end = '')
####    engine.say(audio)
####    engine.runAndWait()
####
####
####class SimpleFacerec:
####
####    def __init__(self):
####        self.known_face_encodings = []
####        self.known_face_names = []
####
####        # Resize frame for a faster speed
####        self.frame_resizing = 1
####
####        # Option: maximum image dimension to avoid extremely large images slowing detection.
####        self.max_image_dim = 1600
####
####        # Haar cascade for fallback face localization (OpenCV).
####        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
####
####        # Debug output folder for failing images and attempt images
####        self.debug_fail_dir = os.path.join(os.getcwd(), "face_debug_failures")
####        os.makedirs(self.debug_fail_dir, exist_ok=True)
####
####    def _ensure_uint8_contiguous(self, arr):
####        """
####        Ensure array is a numpy ndarray, C-contiguous and dtype uint8 (RGB order).
####        This fixes dlib / face_recognition 'Unsupported image type' errors.
####        """
####        if arr is None:
####            return arr
####        # Convert non-ndarray inputs to ndarray
####        if not isinstance(arr, np.ndarray):
####            try:
####                arr = np.asarray(arr)
####            except Exception:
####                # fallback: return as-is; caller will surface an error
####                return arr
####        # Safe check for C contiguous (flagsobj doesn't have .get)
####        try:
####            is_c_contig = bool(getattr(arr.flags, "c_contiguous", False))
####        except Exception:
####            # final fallback - try numpy function
####            is_c_contig = arr.flags['C_CONTIGUOUS'] if 'C_CONTIGUOUS' in arr.flags else False
####
####        if arr.dtype != np.uint8 or not is_c_contig:
####            arr = np.ascontiguousarray(arr, dtype=np.uint8)
####        return arr
####
####    def _open_as_rgb_numpy(self, img_path):
####        try:
####            pil_img = Image.open(img_path)
####        except Exception as e:
####            raise RuntimeError(f"Could not open image {img_path}: {e}")
####
####        try:
####            pil_img = ImageOps.exif_transpose(pil_img)
####        except Exception:
####            pass
####
####        try:
####            pil_img = pil_img.convert('RGB')
####        except Exception as e:
####            raise RuntimeError(f"Could not convert image {img_path} to RGB: {e}")
####
####        if self.max_image_dim is not None:
####            w, h = pil_img.size
####            max_dim = max(w, h)
####            if max_dim > self.max_image_dim:
####                scale = self.max_image_dim / float(max_dim)
####                new_w = int(w * scale)
####                new_h = int(h * scale)
####                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
####
####        arr = np.asarray(pil_img)
####        if arr.dtype != np.uint8:
####            arr = arr.astype(np.uint8)
####
####        # ensure contiguous uint8 (important for dlib)
####        arr = self._ensure_uint8_contiguous(arr)
####        return arr
####
####    # -------------------------
####    # Extra aggressive preprocess helpers
####    # -------------------------
####    def _apply_clahe_rgb(self, rgb, clip=2.0):
####        try:
####            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
####            y, cr, cb = cv2.split(ycrcb)
####            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
####            y2 = clahe.apply(y)
####            merged = cv2.merge((y2, cr, cb))
####            out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
####            return out
####        except Exception:
####            return rgb
####
####    def _equalize_rgb(self, rgb):
####        try:
####            channels = cv2.split(rgb)
####            eq = [cv2.equalizeHist(ch) for ch in channels]
####            return cv2.merge(eq)
####        except Exception:
####            return rgb
####
####    def _histogram_stretch(self, rgb):
####        try:
####            out = np.zeros_like(rgb)
####            for i in range(3):
####                ch = rgb[:,:,i]
####                p2, p98 = np.percentile(ch, (2, 98))
####                if p98 - p2 > 0:
####                    out[:,:,i] = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
####                else:
####                    out[:,:,i] = ch
####            return out
####        except Exception:
####            return rgb
####
####    def _gamma_correction(self, rgb, gamma):
####        inv = 1.0 / float(gamma)
####        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
####        return cv2.LUT(rgb, table)
####
####    def _denoise(self, rgb):
####        try:
####            return cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
####        except Exception:
####            return rgb
####
####    def _sharpen(self, rgb):
####        try:
####            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
####            return cv2.filter2D(rgb, -1, kernel)
####        except Exception:
####            return rgb
####
####    def _try_rotations_and_flips(self, rgb):
####        """Yield rotated and flipped variants that may reveal faces."""
####        variants = []
####        variants.append(rgb)  # original
####        variants.append(cv2.flip(rgb, 1))  # horizontal flip
####        for angle in (-15, -10, 10, 15):
####            (h, w) = rgb.shape[:2]
####            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
####            rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
####            variants.append(rotated)
####            variants.append(cv2.flip(rotated, 1))
####        return variants
####
####    def _haar_detect_and_crop(self, rgb_img, padding_ratio=0.25, scaleFactor=1.1, minNeighbors=4):
####        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
####        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(24, 24))
####        crops = []
####        h, w = gray.shape
####        for (x, y, fw, fh) in faces:
####            pad_w = int(fw * padding_ratio)
####            pad_h = int(fh * padding_ratio)
####            x1 = max(0, x - pad_w)
####            y1 = max(0, y - pad_h)
####            x2 = min(w, x + fw + pad_w)
####            y2 = min(h, y + fh + pad_h)
####            crop = rgb_img[y1:y2, x1:x2]
####            if crop.size != 0:
####                crop = self._ensure_uint8_contiguous(crop)
####                crops.append(crop)
####        return crops
####
####    def _save_debug_thumbnail_and_raise(self, img_path, rgb_img, reason, attempts=None):
####        basename = os.path.basename(img_path)
####        name, _ = os.path.splitext(basename)
####        out_path = os.path.join(self.debug_fail_dir, f"{name}_fail_thumb.jpg")
####        try:
####            pil = Image.fromarray(rgb_img)
####            pil.thumbnail((256,256))
####            pil.save(out_path, "JPEG", quality=85)
####        except Exception:
####            out_path = None
####        # Save attempts images for inspection (if provided)
####        if attempts:
####            att_dir = os.path.join(self.debug_fail_dir, f"{name}_attempts")
####            os.makedirs(att_dir, exist_ok=True)
####            for i, (label, arr) in enumerate(attempts):
####                try:
####                    p = os.path.join(att_dir, f"{i:02d}_{label}.jpg")
####                    Image.fromarray(arr).save(p, "JPEG", quality=85)
####                except Exception:
####                    pass
####        msg = f"No face could be detected in image {img_path}. Reason: {reason}."
####        if out_path:
####            msg += f" A thumbnail was saved to: {out_path}"
####        raise RuntimeError(msg)
####
####    # -------------------------
####    # Main loader (more aggressive)
####    # -------------------------
####    def load_encoding_images(self, images_path):
####        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")
####
####        images_path_list = []
####        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpg")))
####        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpeg")))
####        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.png")))
####        images_path_list = sorted(images_path_list)
####
####        Number_Images = len(images_path_list)
####
####        print(f" There are {Number_Images} faces found for the facial recognition system.")
####        speech.AlfredSpeak(f"There are {Number_Images} faces found for the facial recognition system.")
######        AlfredSpeak_Start(f"There are {Number_Images} faces found for the facial recognition system.")
####
####        for img_path in images_path_list:
####            rgb_img = self._open_as_rgb_numpy(img_path)
####            rgb_img = self._ensure_uint8_contiguous(rgb_img)  # <-- ensure contiguous here too
####            basename = os.path.basename(img_path)
####            (filename, ext) = os.path.splitext(basename)
####
####            # Keep track of attempts for debug saving if all fail
####            attempts = [("original", rgb_img.copy())]
####
####            # 1) Try direct encoding
####            encodings = []
####            try:
####                encodings = face_recognition.face_encodings(self._ensure_uint8_contiguous(rgb_img))
####                if encodings:
####                    # success
####                    pass
####            except Exception:
####                encodings = []
####
####            # 2) Try earlier preprocessing set (CLAHE, equalize, gamma)
####            if not encodings:
####                candidates = []
####                candidates.append(self._apply_clahe_rgb(rgb_img, clip=2.0))
####                candidates.append(self._equalize_rgb(rgb_img))
####                candidates.append(self._histogram_stretch(rgb_img))
####                for g in (0.8, 1.2, 1.5):
####                    candidates.append(self._gamma_correction(rgb_img, g))
####                # keep also sharpened and denoised versions
####                candidates.append(self._sharpen(rgb_img))
####                candidates.append(self._denoise(rgb_img))
####
####                for i, cand in enumerate(candidates):
####                    # ensure contiguous for each candidate
####                    cand = self._ensure_uint8_contiguous(cand)
####                    attempts.append((f"pre{i}", cand.copy()))
####                    try:
####                        enc = face_recognition.face_encodings(cand)
####                        if enc:
####                            encodings = enc
####                            rgb_img = cand
####                            break
####                    except Exception:
####                        continue
####
####            # 3) Try rotations/flips on original and on best preprocessed candidate
####            if not encodings:
####                rotations = self._try_rotations_and_flips(rgb_img)
####                for j, r in enumerate(rotations):
####                    r = self._ensure_uint8_contiguous(r)
####                    attempts.append((f"rot{j}", r.copy()))
####                    try:
####                        enc = face_recognition.face_encodings(r)
####                        if enc:
####                            encodings = enc
####                            rgb_img = r
####                            break
####                    except Exception:
####                        continue
####
####            # 4) Try Haar with relaxed parameters and cropping; also on variants
####            if not encodings:
####                # try stronger CLAHE and histogram stretch on original to feed Haar
####                stronger = self._apply_clahe_rgb(rgb_img, clip=4.0)
####                stronger = self._ensure_uint8_contiguous(stronger)
####                attempts.append(("clahe_strong", stronger.copy()))
####                crops = self._haar_detect_and_crop(stronger, padding_ratio=0.4, scaleFactor=1.05, minNeighbors=3)
####                # also try on histogram stretched original
####                stretched = self._histogram_stretch(rgb_img)
####                stretched = self._ensure_uint8_contiguous(stretched)
####                attempts.append(("stretched", stretched.copy()))
####                crops += self._haar_detect_and_crop(stretched, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
####                # expand: look at rotated variants too
####                for idx, variant in enumerate(self._try_rotations_and_flips(rgb_img)):
####                    variant = self._ensure_uint8_contiguous(variant)
####                    attempts.append((f"rot_haar{idx}", variant.copy()))
####                    crops += self._haar_detect_and_crop(variant, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
####                # Evaluate each crop and some preprocess variants of crop
####                for ci, crop in enumerate(crops):
####                    crop = self._ensure_uint8_contiguous(crop)
####                    attempts.append((f"crop{ci}", crop.copy()))
####                    # try crop as-is
####                    try:
####                        enc = face_recognition.face_encodings(crop)
####                        if enc:
####                            encodings = enc
####                            rgb_img = crop
####                            break
####                    except Exception:
####                        pass
####                    # try processed crops
####                    for pc in (self._apply_clahe_rgb(crop, clip=2.5),
####                               self._equalize_rgb(crop),
####                               self._histogram_stretch(crop),
####                               self._sharpen(crop)):
####                        pc = self._ensure_uint8_contiguous(pc)
####                        attempts.append((f"crop{ci}_proc", pc.copy()))
####                        try:
####                            enc = face_recognition.face_encodings(pc)
####                            if enc:
####                                encodings = enc
####                                rgb_img = pc
####                                break
####                        except Exception:
####                            continue
####                    if encodings:
####                        break
####
####            # 5) Final: try cnn detector if available
####            if not encodings:
####                try:
####                    fl = face_recognition.face_locations(self._ensure_uint8_contiguous(rgb_img), model='cnn')
####                    if fl:
####                        enc = face_recognition.face_encodings(self._ensure_uint8_contiguous(rgb_img), fl)
####                        if enc:
####                            encodings = enc
####                except Exception:
####                    pass
####
####            # If still nothing: save attempts and raise
####            if not encodings:
####                reason = "attempted raw -> preprocessing (CLAHE/equalize/stretch/gamma/sharpen/denoise) -> rotations/flips -> Haar crop (relaxed) -> cnn detector"
####                # cap attempts saved to avoid huge disk use
####                limited_attempts = attempts[:40]
####                self._save_debug_thumbnail_and_raise(img_path, rgb_img, reason, attempts=limited_attempts)
####
####            # Use the first encoding found
####            img_encoding = encodings[0]
####            self.known_face_encodings.append(img_encoding)
####            self.known_face_names.append(filename)
####            print(f"[OK] Encoded {filename}")
####
####        print(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
####        speech.AlfredSpeak(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
######        AlfredSpeak_Start(f"There were a total of {Number_Images} faces encoded and loaded for the facial recognition system.")
####
####
####    def detect_known_faces(self, frame):
####        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
####        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
####        rgb_small_frame = self._ensure_uint8_contiguous(rgb_small_frame)
####        face_locations = face_recognition.face_locations(rgb_small_frame)
####        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
####
####        face_names = []
####        for face_encoding in face_encodings:
####            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
####            name = "Unknown"
####            if self.known_face_encodings:
####                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
####                best_match_index = np.argmin(face_distances)
####                if matches[best_match_index]:
####                    name = self.known_face_names[best_match_index]
####            face_names.append(name)
####
####        face_locations = np.array(face_locations)
####        face_locations = face_locations / self.frame_resizing
####        return face_locations.astype(int), face_names
####
####
####





##import face_recognition
##import cv2
##import os
##import glob
##import numpy as np
##import pyttsx3  # pip install pyttsx3
##import speech_recognition as sr
##import fnmatch
##import Alfred_config
##
##from speech import speech
##
##from PIL import Image, ImageOps  # pip install pillow
##
##Number_Images = str(0)
##
##def AlfredSpeak_Start(audio):
##    engine = pyttsx3.init('sapi5')
##    engine.setProperty('rate', 190)
##    voices = engine.getProperty('voices')
##    engine.setProperty('voice', voices[1].id)
##    engine.setProperty('volume', 1)
##    print('engine: ' + str(audio), end = "\r")
##    print('\033c', end = '')
##    engine.say(audio)
##    engine.runAndWait()
##
##
##class SimpleFacerec:
##
##    def __init__(self):
##        self.known_face_encodings = []
##        self.known_face_names = []
##
##        # Resize frame for a faster speed
##        self.frame_resizing = 1
##
##        # Option: maximum image dimension to avoid extremely large images slowing detection.
##        self.max_image_dim = 1600
##
##        # Haar cascade for fallback face localization (OpenCV).
##        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
##
##        # Debug output folder for failing images and attempt images
##        self.debug_fail_dir = os.path.join(os.getcwd(), "face_debug_failures")
##        os.makedirs(self.debug_fail_dir, exist_ok=True)
##
##    def _open_as_rgb_numpy(self, img_path):
##        try:
##            pil_img = Image.open(img_path)
##        except Exception as e:
##            raise RuntimeError(f"Could not open image {img_path}: {e}")
##
##        try:
##            pil_img = ImageOps.exif_transpose(pil_img)
##        except Exception:
##            pass
##
##        try:
##            pil_img = pil_img.convert('RGB')
##        except Exception as e:
##            raise RuntimeError(f"Could not convert image {img_path} to RGB: {e}")
##
##        if self.max_image_dim is not None:
##            w, h = pil_img.size
##            max_dim = max(w, h)
##            if max_dim > self.max_image_dim:
##                scale = self.max_image_dim / float(max_dim)
##                new_w = int(w * scale)
##                new_h = int(h * scale)
##                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
##
##        arr = np.asarray(pil_img)
##        if arr.dtype != np.uint8:
##            arr = arr.astype(np.uint8)
##        return arr
##
##    # -------------------------
##    # Extra aggressive preprocess helpers
##    # -------------------------
##    def _apply_clahe_rgb(self, rgb, clip=2.0):
##        try:
##            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
##            y, cr, cb = cv2.split(ycrcb)
##            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
##            y2 = clahe.apply(y)
##            merged = cv2.merge((y2, cr, cb))
##            out = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
##            return out
##        except Exception:
##            return rgb
##
##    def _equalize_rgb(self, rgb):
##        try:
##            channels = cv2.split(rgb)
##            eq = [cv2.equalizeHist(ch) for ch in channels]
##            return cv2.merge(eq)
##        except Exception:
##            return rgb
##
##    def _histogram_stretch(self, rgb):
##        try:
##            out = np.zeros_like(rgb)
##            for i in range(3):
##                ch = rgb[:,:,i]
##                p2, p98 = np.percentile(ch, (2, 98))
##                if p98 - p2 > 0:
##                    out[:,:,i] = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
##                else:
##                    out[:,:,i] = ch
##            return out
##        except Exception:
##            return rgb
##
##    def _gamma_correction(self, rgb, gamma):
##        inv = 1.0 / float(gamma)
##        table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
##        return cv2.LUT(rgb, table)
##
##    def _denoise(self, rgb):
##        try:
##            return cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
##        except Exception:
##            return rgb
##
##    def _sharpen(self, rgb):
##        try:
##            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
##            return cv2.filter2D(rgb, -1, kernel)
##        except Exception:
##            return rgb
##
##    def _try_rotations_and_flips(self, rgb):
##        """Yield rotated and flipped variants that may reveal faces."""
##        variants = []
##        variants.append(rgb)  # original
##        variants.append(cv2.flip(rgb, 1))  # horizontal flip
##        for angle in (-15, -10, 10, 15):
##            (h, w) = rgb.shape[:2]
##            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
##            rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
##            variants.append(rotated)
##            variants.append(cv2.flip(rotated, 1))
##        return variants
##
##    def _haar_detect_and_crop(self, rgb_img, padding_ratio=0.25, scaleFactor=1.1, minNeighbors=4):
##        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
##        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(24, 24))
##        crops = []
##        h, w = gray.shape
##        for (x, y, fw, fh) in faces:
##            pad_w = int(fw * padding_ratio)
##            pad_h = int(fh * padding_ratio)
##            x1 = max(0, x - pad_w)
##            y1 = max(0, y - pad_h)
##            x2 = min(w, x + fw + pad_w)
##            y2 = min(h, y + fh + pad_h)
##            crop = rgb_img[y1:y2, x1:x2]
##            if crop.size != 0:
##                crops.append(crop)
##        return crops
##
##    def _save_debug_thumbnail_and_raise(self, img_path, rgb_img, reason, attempts=None):
##        basename = os.path.basename(img_path)
##        name, _ = os.path.splitext(basename)
##        out_path = os.path.join(self.debug_fail_dir, f"{name}_fail_thumb.jpg")
##        try:
##            pil = Image.fromarray(rgb_img)
##            pil.thumbnail((256,256))
##            pil.save(out_path, "JPEG", quality=85)
##        except Exception:
##            out_path = None
##        # Save attempts images for inspection (if provided)
##        if attempts:
##            att_dir = os.path.join(self.debug_fail_dir, f"{name}_attempts")
##            os.makedirs(att_dir, exist_ok=True)
##            for i, (label, arr) in enumerate(attempts):
##                try:
##                    p = os.path.join(att_dir, f"{i:02d}_{label}.jpg")
##                    Image.fromarray(arr).save(p, "JPEG", quality=85)
##                except Exception:
##                    pass
##        msg = f"No face could be detected in image {img_path}. Reason: {reason}."
##        if out_path:
##            msg += f" A thumbnail was saved to: {out_path}"
##        raise RuntimeError(msg)
##
##    # -------------------------
##    # Main loader (more aggressive)
##    # -------------------------
##    def load_encoding_images(self, images_path):
##        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")
##
##        images_path_list = []
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpg")))
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.jpeg")))
##        images_path_list.extend(glob.glob(os.path.join(My_Images_Path, "*.png")))
##        images_path_list = sorted(images_path_list)
##
##        Number_Images = len(images_path_list)
##
##        print(f" There are {Number_Images} faces found for the facial recognition system.")
##        speech.AlfredSpeak(f"There are {Number_Images} faces found for the facial recognition system.")
##
##        for img_path in images_path_list:
##            rgb_img = self._open_as_rgb_numpy(img_path)
##            basename = os.path.basename(img_path)
##            (filename, ext) = os.path.splitext(basename)
##
##            # Keep track of attempts for debug saving if all fail
##            attempts = [("original", rgb_img.copy())]
##
##            # 1) Try direct encoding
##            encodings = []
##            try:
##                encodings = face_recognition.face_encodings(rgb_img)
##                if encodings:
##                    # success
##                    pass
##            except Exception:
##                encodings = []
##
##            # 2) Try earlier preprocessing set (CLAHE, equalize, gamma)
##            if not encodings:
##                candidates = []
##                candidates.append(self._apply_clahe_rgb(rgb_img, clip=2.0))
##                candidates.append(self._equalize_rgb(rgb_img))
##                candidates.append(self._histogram_stretch(rgb_img))
##                for g in (0.8, 1.2, 1.5):
##                    candidates.append(self._gamma_correction(rgb_img, g))
##                # keep also sharpened and denoised versions
##                candidates.append(self._sharpen(rgb_img))
##                candidates.append(self._denoise(rgb_img))
##
##                for i, cand in enumerate(candidates):
##                    attempts.append((f"pre{i}", cand.copy()))
##                    try:
##                        enc = face_recognition.face_encodings(cand)
##                        if enc:
##                            encodings = enc
##                            rgb_img = cand
##                            break
##                    except Exception:
##                        continue
##
##            # 3) Try rotations/flips on original and on best preprocessed candidate
##            if not encodings:
##                rotations = self._try_rotations_and_flips(rgb_img)
##                for j, r in enumerate(rotations):
##                    attempts.append((f"rot{j}", r.copy()))
##                    try:
##                        enc = face_recognition.face_encodings(r)
##                        if enc:
##                            encodings = enc
##                            rgb_img = r
##                            break
##                    except Exception:
##                        continue
##
##            # 4) Try Haar with relaxed parameters and cropping; also on variants
##            if not encodings:
##                # try stronger CLAHE and histogram stretch on original to feed Haar
##                stronger = self._apply_clahe_rgb(rgb_img, clip=4.0)
##                attempts.append(("clahe_strong", stronger.copy()))
##                crops = self._haar_detect_and_crop(stronger, padding_ratio=0.4, scaleFactor=1.05, minNeighbors=3)
##                # also try on histogram stretched original
##                stretched = self._histogram_stretch(rgb_img)
##                attempts.append(("stretched", stretched.copy()))
##                crops += self._haar_detect_and_crop(stretched, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
##                # expand: look at rotated variants too
##                for idx, variant in enumerate(self._try_rotations_and_flips(rgb_img)):
##                    attempts.append((f"rot_haar{idx}", variant.copy()))
##                    crops += self._haar_detect_and_crop(variant, padding_ratio=0.35, scaleFactor=1.05, minNeighbors=3)
##                # Evaluate each crop and some preprocess variants of crop
##                for ci, crop in enumerate(crops):
##                    attempts.append((f"crop{ci}", crop.copy()))
##                    # try crop as-is
##                    try:
##                        enc = face_recognition.face_encodings(crop)
##                        if enc:
##                            encodings = enc
##                            rgb_img = crop
##                            break
##                    except Exception:
##                        pass
##                    # try processed crops
##                    for pc in (self._apply_clahe_rgb(crop, clip=2.5),
##                               self._equalize_rgb(crop),
##                               self._histogram_stretch(crop),
##                               self._sharpen(crop)):
##                        attempts.append((f"crop{ci}_proc", pc.copy()))
##                        try:
##                            enc = face_recognition.face_encodings(pc)
##                            if enc:
##                                encodings = enc
##                                rgb_img = pc
##                                break
##                        except Exception:
##                            continue
##                    if encodings:
##                        break
##
##            # 5) Final: try cnn detector if available
##            if not encodings:
##                try:
##                    fl = face_recognition.face_locations(rgb_img, model='cnn')
##                    if fl:
##                        enc = face_recognition.face_encodings(rgb_img, fl)
##                        if enc:
##                            encodings = enc
##                except Exception:
##                    pass
##
##            # If still nothing: save attempts and raise
##            if not encodings:
##                reason = "attempted raw -> preprocessing (CLAHE/equalize/stretch/gamma/sharpen/denoise) -> rotations/flips -> Haar crop (relaxed) -> cnn detector"
##                # cap attempts saved to avoid huge disk use
##                limited_attempts = attempts[:40]
##                self._save_debug_thumbnail_and_raise(img_path, rgb_img, reason, attempts=limited_attempts)
##
##            # Use the first encoding found
##            img_encoding = encodings[0]
##            self.known_face_encodings.append(img_encoding)
##            self.known_face_names.append(filename)
##            print(f"[OK] Encoded {filename}")
##
##        print(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
##        speech.AlfredSpeak(f"There were a total of {len(self.known_face_encodings)} faces encoded and loaded for the facial recognition system.")
##
##
##    def detect_known_faces(self, frame):
##        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
##        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
##        face_locations = face_recognition.face_locations(rgb_small_frame)
##        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
##
##        face_names = []
##        for face_encoding in face_encodings:
##            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
##            name = "Unknown"
##            if self.known_face_encodings:
##                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
##                best_match_index = np.argmin(face_distances)
##                if matches[best_match_index]:
##                    name = self.known_face_names[best_match_index]
##            face_names.append(name)
##
##        face_locations = np.array(face_locations)
##        face_locations = face_locations / self.frame_resizing
##        return face_locations.astype(int), face_names
##





import face_recognition
import cv2
import os
import glob
import numpy as np
import pyttsx3  #pip install pyttsx3
import speech_recognition as sr
import fnmatch
import Alfred_config

from speech import speech

Number_Images = str(0)

def AlfredSpeak_Start(audio):
    engine = pyttsx3.init('sapi5')
    engine.setProperty('rate', 190)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('volume', 1)
##    engine.setProperty('pitch', 0.1)
    print('engine: ' + str(audio), end = "\r")
    print('\033c', end = '') 
    engine.say(audio)
    engine.runAndWait()


class SimpleFacerec:
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
##        self.frame_resizing = 0.5
        self.frame_resizing = 1

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """

        My_Images_Path = (Alfred_config.DRIVE_LETTER+"Python_Env//New_Virtual_Env//Personal//Facial_Recognition_Folder//New_Images")
        
        # Load Images
        images_path = glob.glob(os.path.join(My_Images_Path, "*.jpg"))

        Number_Images = len(images_path)
##        Number_Images = len(fnmatch.filter(os.listdir(images_path), '*.*'))


        print(f" There are {Number_Images} faces found for the facial recognition system.")
##        AlfredSpeak_Start(f"There are {Number_Images} faces found for encoding.")
        speech.AlfredSpeak(f"There are {Number_Images} faces found for the facial recognition system.")
                           
        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            
        print(f"There were a total of {Number_Images} faces encoded and loaded for the facial recognition system.")
##        AlfredSpeak_Start(f"There were a total of {Number_Images} faces encoded and loaded for the facial recognition system.")
        speech.AlfredSpeak(f"There were a total of {Number_Images} faces encoded and loaded for the facial recognition system.")


    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            try:
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
            except Exception(e):
                print(f"No Matches are found : {e}")

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

