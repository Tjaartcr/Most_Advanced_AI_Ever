# dlib_test.py
import numpy as np
from PIL import Image, ImageOps
import face_recognition

p = r"D:\Python_Env\New_Virtual_Env\Personal\Facial_Recognition_Folder\New_Images\Chante'.jpg"
pil = Image.open(p)
pil = ImageOps.exif_transpose(pil).convert("RGB")
arr = np.asarray(pil)
print("arr:", arr.shape, arr.dtype, "C_CONTIGUOUS:", arr.flags['C_CONTIGUOUS'])
locs = face_recognition.face_locations(arr, model='hog')
print("face_locations (hog):", locs)
encs = face_recognition.face_encodings(arr, locs)
print("encodings count:", len(encs))
