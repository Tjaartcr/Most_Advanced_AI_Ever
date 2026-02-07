# diagnostic_face_check.py
# Place next to your project and run with the same Python interpreter you use for your app.
# Example: python diagnostic_face_check.py

# diagnostic_face_check.py
# Standalone — does NOT import Alfred_config or other project modules

import os
import glob
import csv
import cv2
import numpy as np
from PIL import Image, ImageOps
import face_recognition

# === SET THIS PATH MANUALLY ===
MY_IMAGES_PATH = r"D:\Python_Env\New_Virtual_Env\Personal\Facial_Recognition_Folder\New_Images"

# === output directory ===
OUT_DIR = os.path.join(os.getcwd(), "face_debug_failures", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"Scanning images in: {MY_IMAGES_PATH}")
img_paths = sorted(
    glob.glob(os.path.join(MY_IMAGES_PATH, "*.jpg")) +
    glob.glob(os.path.join(MY_IMAGES_PATH, "*.jpeg")) +
    glob.glob(os.path.join(MY_IMAGES_PATH, "*.png"))
)

print(f"Found {len(img_paths)} images under: {MY_IMAGES_PATH}")
report_rows = []
for idx, p in enumerate(img_paths):
    name = os.path.basename(p)
    print(f"\n[{idx+1}/{len(img_paths)}] Inspecting: {name}")
    row = {"file": p, "basename": name, "ok_load": False, "shape": None, "dtype": None,
           "mean_brightness": None, "std_brightness": None,
           "hog_count": 0, "cnn_count": "n/a", "haar_count": 0, "notes": ""}

    # 1) Load with Pillow (same as loader)
    try:
        pil = Image.open(p)
        pil = ImageOps.exif_transpose(pil)   # fix orientation
        pil = pil.convert("RGB")
        img_np = np.asarray(pil)
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        row["ok_load"] = True
        row["shape"] = str(img_np.shape)
        row["dtype"] = str(img_np.dtype)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        row["mean_brightness"] = float(np.mean(gray))
        row["std_brightness"] = float(np.std(gray))
        # Save the canonical loaded image for your inspection
        loaded_out = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_loaded.jpg")
        Image.fromarray(img_np).save(loaded_out, "JPEG", quality=90)
        print(f"  loaded -> saved canonical image to: {loaded_out}")
    except Exception as e:
        row["notes"] = f"Load failed: {e}"
        print("  ERROR loading image:", e)
        report_rows.append(row)
        continue

    # Prepare a canvas to draw detections
    debug_canvas = img_np.copy()

    # 2) face_recognition hog detector
    try:
        hog_locs = face_recognition.face_locations(img_np, model="hog")
        row["hog_count"] = len(hog_locs)
        print(f"  HOG detector found: {len(hog_locs)}")
        for (top, right, bottom, left) in hog_locs:
            cv2.rectangle(debug_canvas, (left, top), (right, bottom), (0,255,0), 2)
    except Exception as e:
        row["notes"] += f" HOG error:{e}"
        print("  HOG detector error:", e)

    # 3) face_recognition cnn detector (may raise if dlib without cnn support)
    try:
        cnn_locs = face_recognition.face_locations(img_np, model="cnn")
        row["cnn_count"] = len(cnn_locs)
        print(f"  CNN detector found: {len(cnn_locs)}")
        for (top, right, bottom, left) in cnn_locs:
            cv2.rectangle(debug_canvas, (left, top), (right, bottom), (255,0,0), 2)
    except Exception as e:
        row["cnn_count"] = "error"
        row["notes"] += f" CNN error:{e}"
        print("  CNN detector error (this is expected if dlib isn't built with cnn):", e)

    # 4) Haar cascade
    try:
        gray_for_haar = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        haar = HAAR.detectMultiScale(gray_for_haar, scaleFactor=1.1, minNeighbors=3, minSize=(24,24))
        row["haar_count"] = len(haar)
        print(f"  Haar cascade found: {len(haar)}")
        for (x,y,w,h) in haar:
            cv2.rectangle(debug_canvas, (x, y), (x+w, y+h), (0,0,255), 2)
    except Exception as e:
        row["notes"] += f" Haar error:{e}"
        print("  Haar cascade error:", e)

    # 5) face_recognition encodings (report how many encodings are returned by default call)
    try:
        encs = face_recognition.face_encodings(img_np)
        enc_count = len(encs)
        print(f"  face_encodings(...) returned: {enc_count}")
        row["encodings_count"] = enc_count
    except Exception as e:
        row["encodings_count"] = "error"
        row["notes"] += f" enc_error:{e}"
        print("  face_encodings error:", e)

    # Save a debug annotated image showing all detections
    try:
        annotated_out = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}_annotated.jpg")
        # convert RGB->BGR for cv2.imwrite
        cv2.imwrite(annotated_out, cv2.cvtColor(debug_canvas, cv2.COLOR_RGB2BGR))
        print(f"  annotated image saved to: {annotated_out}")
    except Exception as e:
        print("  Failed to save annotated image:", e)

    report_rows.append(row)

# Save CSV summary
csv_path = os.path.join(OUT_DIR, "diagnostic_report.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["file","basename","ok_load","shape","dtype","mean_brightness","std_brightness",
                  "hog_count","cnn_count","haar_count","encodings_count","notes"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in report_rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

print("\nDiagnostic run complete.")
print("Summary CSV:", csv_path)
print("Check the 'loaded' and 'annotated' images inside:", OUT_DIR)
print("Paste the CSV or the printed output here if you want me to analyze the results and suggest the exact fix.")
