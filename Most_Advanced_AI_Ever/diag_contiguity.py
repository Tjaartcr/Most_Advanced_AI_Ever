# diag_contiguity.py
import glob, os
from PIL import Image, ImageOps
import numpy as np
import face_recognition
import cv2

IMAGES_DIR = r"D:\Python_Env\New_Virtual_Env\Personal\Facial_Recognition_Folder\New_Images"
paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) + glob.glob(os.path.join(IMAGES_DIR, "*.png")))
if not paths:
    raise SystemExit("No images found in: " + IMAGES_DIR)

p = paths[0]
print("Testing image:", p)
pil = Image.open(p)
pil = ImageOps.exif_transpose(pil)
pil = pil.convert("RGB")
arr = np.asarray(pil)

def info(name, a):
    print(f"\n== {name} ==")
    print("dtype:", a.dtype, "ndim:", a.ndim, "shape:", a.shape)
    print("flags:", a.flags)
    print("strides:", a.strides)
    print("min/max:", a.min(), a.max())
    # memory address
    try:
        print("data pointer:", a.__array_interface__['data'][0])
    except Exception:
        pass

info("original (PIL->np.asarray)", arr)

# Try face_recognition on original
try:
    print("\nCalling face_recognition.face_locations(original, model='hog') ...")
    locs = face_recognition.face_locations(arr, model='hog')
    print("OK - hog found", len(locs))
except Exception as e:
    print("ERROR on original:", repr(e))

# Make contiguous copy
contig = np.ascontiguousarray(arr, dtype=np.uint8)
info("contiguous copy (np.ascontiguousarray)", contig)
try:
    print("\nCalling face_recognition.face_locations(contiguous, model='hog') ...")
    locs2 = face_recognition.face_locations(contig, model='hog')
    print("OK - hog found", len(locs2))
except Exception as e:
    print("ERROR on contiguous:", repr(e))

# Also try cv2.imread variant
cvimg = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
info("cv2.imread -> RGB", cvimg)
try:
    print("\nCalling face_recognition.face_locations(cvimg, model='hog') ...")
    locs3 = face_recognition.face_locations(cvimg, model='hog')
    print("OK - hog found", len(locs3))
except Exception as e:
    print("ERROR on cvimg:", repr(e))

print("\nDone. If the contiguous copy works but original fails, add np.ascontiguousarray(...) before calling face_recognition.")
