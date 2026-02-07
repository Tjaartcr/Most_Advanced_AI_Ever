# whisper_gender.py
import whisper
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import joblib
import sys
import sounddevice as sd
from scipy.io.wavfile import write

# ----------------------------
# Directories
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "wave")
MALE_DIR = os.path.join(DATA_DIR, "males")
FEMALE_DIR = os.path.join(DATA_DIR, "females")

SCALER_FILE = os.path.join(BASE_DIR, "gender_scaler.pkl")
CLF_FILE = os.path.join(BASE_DIR, "gender_clf.pkl")

# ----------------------------
# STEP 1: Feature extraction
# ----------------------------
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# ----------------------------
# STEP 2: Train gender classifier
# ----------------------------
def train_gender_classifier():
    X = []
    y = []

    dataset_paths = [(0, MALE_DIR), (1, FEMALE_DIR)]  # 0=male, 1=female
    for label, folder in dataset_paths:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(".wav"):
                mfcc = extract_mfcc(os.path.join(folder, f))
                X.append(mfcc)
                y.append(label)

    if not X:
        print("⚠️ No training data found in wave/males or wave/females.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_scaled, y)

    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(clf, CLF_FILE)
    print("✅ Gender classifier trained and saved.")

# ----------------------------
# STEP 3: Load Whisper + classifier
# ----------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")  # tiny/base/small/medium/large

if os.path.exists(SCALER_FILE) and os.path.exists(CLF_FILE):
    print("Loading existing gender classifier...")
    scaler = joblib.load(SCALER_FILE)
    clf = joblib.load(CLF_FILE)
else:
    print("No classifier found → training a new one...")
    train_gender_classifier()
    scaler = joblib.load(SCALER_FILE)
    clf = joblib.load(CLF_FILE)

# ----------------------------
# STEP 4: Transcribe and classify
# ----------------------------
def transcribe_and_classify(audio_file):
    print(f"🎙️ Transcribing {audio_file} ...")
    result = whisper_model.transcribe(audio_file)
    transcription = result["text"]

    mfcc = extract_mfcc(audio_file).reshape(1, -1)
    mfcc_scaled = scaler.transform(mfcc)

    gender_label = clf.predict(mfcc_scaled)[0]
    gender_prob = clf.predict_proba(mfcc_scaled)[0]

    gender_str = "Male" if gender_label == 0 else "Female"
    confidence = gender_prob[gender_label]

    print(f"📝 Transcription: {transcription}")
    print(f"🚻 Predicted Gender: {gender_str} (confidence: {confidence:.2f})")

    return transcription, gender_str, confidence

# ----------------------------
# STEP 5: Record your own voice
# ----------------------------
def record_and_test(duration=5, samplerate=16000, filename="recorded.wav"):
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()  # wait until recording is finished
    write(filename, samplerate, recording)
    print(f"✅ Saved recording to {filename}")

    return transcribe_and_classify(filename)

# ----------------------------
# STEP 6: CLI usage
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper transcription + gender classification")
    parser.add_argument("audio_file", type=str, nargs="?", help="Path to audio file (.wav, .m4a, etc.)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain gender classifier")
    parser.add_argument("--record", action="store_true", help="Record your voice and test classification")
    parser.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")
    args = parser.parse_args()

    if args.retrain:
        train_gender_classifier()
        scaler = joblib.load(SCALER_FILE)
        clf = joblib.load(CLF_FILE)

    if args.record:
        record_and_test(duration=args.duration)
    elif args.audio_file:
        transcribe_and_classify(args.audio_file)
    else:
        print("⚠️ Provide either an audio file or use --record to record your voice.")


