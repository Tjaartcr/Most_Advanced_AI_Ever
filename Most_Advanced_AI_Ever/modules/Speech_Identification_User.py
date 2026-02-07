

"""
Speech_Identification_User.py
Robust speaker identification utilities using SpeechBrain (ecapa-voxceleb).
- Uses LocalStrategy.COPY to avoid Windows symlink issues.
- Records audio via sounddevice, saves 16kHz mono int16 WAVs.
- Computes normalized embeddings using SpeechBrain SpeakerRecognition.
- Stores normalized embeddings in a JSON DB (atomic saves).
- Enrollment / update / interactive identification CLI included.
"""

# --- IMPORTANT: environment variables to avoid Windows symlink problems ---
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ------------------ torchaudio compatibility shim ------------------
# Some torchaudio versions removed `torchaudio.list_audio_backends`.
# Provide a small compatibility shim so imports that call that function won't crash.
try:
    import torchaudio
    # If missing, attempt to alias a sensible replacement
    if not hasattr(torchaudio, "list_audio_backends"):
        if hasattr(torchaudio, "utils") and hasattr(torchaudio.utils, "list_audio_backends"):
            # new location
            torchaudio.list_audio_backends = torchaudio.utils.list_audio_backends  # type: ignore
        else:
            # Create a tiny fallback that returns the active backend if possible
            def _fallback_list_audio_backends():
                backends = []
                try:
                    if hasattr(torchaudio, "get_audio_backend"):
                        backend = torchaudio.get_audio_backend()
                        if backend is not None:
                            backends.append(backend)
                except Exception:
                    pass
                return backends
            torchaudio.list_audio_backends = _fallback_list_audio_backends  # type: ignore
except Exception:
    # If torchaudio import fails entirely, don't crash here — the rest of the module
    # may still work (we only need torchaudio for optional features).
    torchaudio = None  # type: ignore

# Standard libs
import time
import json
from pathlib import Path
import shutil
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import torch

# SpeechBrain imports (new inference path)
from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ------------------ Configuration ------------------
# Adjust these to taste
CONFIDENCE_THRESHOLD = 0.60   # cosine similarity threshold (0.6-0.85 typical)
SAMPLE_RATE = 16000           # target sampling rate for SpeechBrain model
DEFAULT_RECORD_SECONDS = 5
VOICE_SAMPLES_DIR = Path("voice_samples")
TEST_WAV_DIR = Path("test_waves")
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
TEST_WAV_DIR.mkdir(parents=True, exist_ok=True)
ROUND_ROBIN_FILE = TEST_WAV_DIR / "last_index.txt"
SAVEDIR = Path("pretrained_model")  # HF model cache target (LocalStrategy.COPY)

# DB location (Windows-friendly). If you have Alfred_config with DRIVE_LETTER, prefer that.
try:
    import Alfred_config
    DRIVE = getattr(Alfred_config, "DRIVE_LETTER", "")
    if DRIVE:
        DB_FILE = Path(DRIVE) / "Python_Env" / "New_Virtual_Env" / "Alfred_Offline_New_GUI" / \
                  "2025_08_30A_WEBUI_MODEL_New_TRY_OLD" / "New_V2_Home_Head_Movement_Smoothing" / \
                  "modules" / "voice_db.json"
    else:
        DB_FILE = Path("voice_db.json")
except Exception:
    DB_FILE = Path("voice_db.json")

# make sure directories exist
DB_FILE.parent.mkdir(parents=True, exist_ok=True)
SAVEDIR.mkdir(parents=True, exist_ok=True)

# ------------------ Utilities ------------------
def get_next_index(limit: int = 20) -> int:
    """Return the next index for round-robin overwriting (1..limit)."""
    try:
        if ROUND_ROBIN_FILE.exists():
            try:
                last = int(ROUND_ROBIN_FILE.read_text().strip())
            except Exception:
                last = 0
        else:
            last = 0
        new_index = (last % limit) + 1
        ROUND_ROBIN_FILE.write_text(str(new_index))
        return new_index
    except Exception:
        # fallback safe behavior
        return 1

def _atomic_write_text(path: Path, text: str):
    """Write text atomically: write to tmp file then replace target."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        # If replace fails, attempt a safe fallback
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise e

def load_db() -> dict:
    """Load JSON DB (name -> embedding list). Returns {} if missing/invalid."""
    try:
        if DB_FILE.exists():
            text = DB_FILE.read_text(encoding="utf-8")
            if not text:
                return {}
            return json.loads(text)
    except Exception as e:
        print(f"[speech_id] Warning: failed to load DB ({e}), starting fresh.")
    return {}

def save_db(db: dict):
    """Save JSON DB using atomic write to reduce corruption risk."""
    try:
        _atomic_write_text(DB_FILE, json.dumps(db, ensure_ascii=False))
    except Exception as e:
        print(f"[speech_id] Error saving DB atomically: {e}; falling back to direct write.")
        try:
            DB_FILE.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
        except Exception as e2:
            print(f"[speech_id] FATAL: DB save failed: {e2}")

# ------------------ Model loading ------------------
def _try_set_torchaudio_backend():
    """
    Try to set a torchaudio backend from a short list of known-good options.
    This is a best-effort helper to reduce SpeechBrain import-time warnings.
    """
    try:
        import importlib
        ta = importlib.import_module("torchaudio")
    except Exception:
        return False, "torchaudio not importable"

    # prefer soundfile (pysoundfile) or sox_io if available
    candidates = ["soundfile", "sox_io", "sox"]
    for backend in candidates:
        try:
            if hasattr(ta, "set_audio_backend"):
                ta.set_audio_backend(backend)
                # verify it took
                if hasattr(ta, "get_audio_backend") and ta.get_audio_backend() == backend:
                    return True, backend
                # some torchaudio versions may accept set without get; still consider success
                return True, backend
        except Exception:
            continue
    # if list_audio_backends is available, report available backends
    try:
        if hasattr(ta, "list_audio_backends"):
            b = ta.list_audio_backends()
            return False, f"no candidate backend set; available backends: {b}"
    except Exception:
        pass
    return False, "no backend set and none reported"

def load_speaker_model(savedir: Path = SAVEDIR) -> SpeakerRecognition:
    """
    Load the SpeechBrain SpeakerRecognition model with LocalStrategy.COPY (no symlinks).
    Returns the loaded model and places it on GPU if available.
    """
    # Try to set torchaudio backend first (best-effort)
    ok, info = _try_set_torchaudio_backend()
    if ok:
        print(f"[speech_id] torchaudio backend set to: {info}")
    else:
        print(f"[speech_id] Could not auto-set torchaudio backend: {info}")
        print("[speech_id] If you want SpeechBrain's internal audio loaders to work,")
        print("  install a backend like soundfile: `pip install soundfile` and then restart.")
        print("  (Or install torchaudio with appropriate wheels for your platform.)")

    savedir = Path(savedir)
    print("[speech_id] Loading SpeechBrain speaker model (LocalStrategy.COPY). This may download ~100MB.")
    try:
        model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            local_strategy=LocalStrategy.COPY,
        )
    except Exception as e:
        print(f"[speech_id] Failed to load model from HF: {e}")
        raise

    # prefer GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Many SpeechBrain wrappers handle device internally; try to move if possible
    try:
        if hasattr(model, "to"):
            try:
                model.to(device)
            except Exception:
                pass
        # Some wrappers expose `.device` attribute; set it if present.
        try:
            setattr(model, "device", device)
        except Exception:
            pass
    except Exception:
        pass

    print(f"[speech_id] Model loaded and placed on device: {device}")
    return model

# load model (singleton)
try:
    verification = load_speaker_model(SAVEDIR)
except Exception as exc:
    print(f"[speech_id] ERROR: speaker model failed to load: {exc}")
    verification = None

# ------------------ Audio utilities ------------------
def record_audio(filename: str, duration: int = DEFAULT_RECORD_SECONDS, fs: int = SAMPLE_RATE):
    """
    Record audio from default microphone and save as a WAV file (mono int16).
    Uses sounddevice; returns after the file is written.
    """
    print(f"🎙️ Recording {duration}s to '{filename}' — speak now...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
    except Exception as e:
        print(f"[speech_id] Error recording audio: {e}")
        raise

    # sounddevice returns shape (N, channels) — flatten to 1-D
    audio_np = np.asarray(audio)
    if audio_np.ndim > 1:
        audio_1d = audio_np.reshape(-1)
    else:
        audio_1d = audio_np

    if audio_1d.dtype != np.int16:
        # cast to int16 safely (clip)
        audio_1d = np.clip(audio_1d, -32768, 32767).astype(np.int16)

    try:
        wav.write(filename, fs, audio_1d)
        print(f"✅ Saved recording to {filename}")
    except Exception as e:
        print(f"[speech_id] Error saving recording {filename}: {e}")
        raise

def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample 1-D numpy array to target_sr. Returns float32 array."""
    if sr == target_sr:
        return data.astype(np.float32)
    from scipy.signal import resample
    num = int(len(data) * target_sr / sr)
    if num <= 1:
        raise ValueError(f"Invalid resample length {num} (orig sr={sr}, len={len(data)})")
    return resample(data, num).astype(np.float32)

def get_embedding(filepath: str) -> np.ndarray:
    """
    Compute a normalized embedding for the WAV file using the loaded SpeechBrain model.
    Returns a 1-D numpy float array (unit norm).
    """
    if verification is None:
        raise RuntimeError("Speaker verification model not loaded.")
    sr, data = wav.read(filepath)

    # Ensure mono
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    # Convert to float32 in range [-1,1] if integer dtypes
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        # guard: int16 -> divide by 32767 to get approx -1..1
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    if sr != SAMPLE_RATE:
        data = _resample_if_needed(data, sr, SAMPLE_RATE)

    if data.size == 0:
        raise ValueError("Audio file contains no data after preprocessing.")

    # Create tensor shape (batch, samples)
    tensor = torch.from_numpy(data).unsqueeze(0).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tensor = tensor.to(device)
    except Exception:
        tensor = tensor.cpu()

    # compute embedding (inference mode)
    try:
        with torch.no_grad():
            emb = verification.encode_batch(tensor)
    except Exception as e:
        # fallback to CPU if GPU fails
        try:
            tensor_cpu = tensor.to("cpu")
            with torch.no_grad():
                emb = verification.encode_batch(tensor_cpu)
        except Exception as e2:
            print(f"[speech_id] Embedding error: {e} / fallback: {e2}")
            raise

    # Convert to numpy safely and normalize
    try:
        emb = emb.squeeze().detach().cpu().numpy().astype(float)
    except Exception:
        emb = np.array(emb).squeeze().astype(float)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

# ------------------ DB / Identity helpers ------------------
def identify_user(embedding: np.ndarray):
    """Identify the closest enrolled speaker given a precomputed embedding."""
    db = load_db()
    if not db:
        print("⚠️ Voice DB empty — enroll users first.")
        return None, None

    test_emb = np.array(embedding, dtype=float)
    if test_emb.size == 0:
        return None, None
    test_norm = np.linalg.norm(test_emb)
    if test_norm == 0:
        return None, None
    test_emb = test_emb / test_norm

    best_score = -1.0
    best_name = None
    for name, emb in db.items():
        emb = np.array(emb, dtype=float)
        if emb.size == 0:
            continue
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            continue
        emb = emb / emb_norm
        score = float(np.dot(test_emb, emb))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < CONFIDENCE_THRESHOLD:
        return None, best_score
    return best_name, best_score

def identify_from_file(wav_path: str, limit: int = 20):
    """
    Copy wav into test_waves (round-robin) and identify speaker, but only if audio is valid (non-silent).
    Returns (name, score) or (None, score).
    """
    try:
        sr, data = wav.read(wav_path)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)

        if len(data) == 0:
            print(f"[speech_id] Skipping {wav_path} — empty file.")
            return None, None

        # compute normalized RMS for silence check
        rms = np.sqrt(np.mean(np.square(data.astype(float))))
        if np.issubdtype(data.dtype, np.integer):
            rms = rms / float(np.iinfo(data.dtype).max)
        # threshold: 0.01 works for normalized [-1,1] floats
        if rms < 0.01:
            print(f"[speech_id] Skipping {wav_path} — silence/low energy (rms={rms:.6f}).")
            return None, None

        # Passed checks → save into round-robin slot
        idx = get_next_index(limit)
        dest = TEST_WAV_DIR / f"test_{idx}.wav"
        shutil.copy(wav_path, dest)
        print(f"[speech_id] Saved test file → {dest}")

        emb = get_embedding(str(dest))
        return identify_user(emb)

    except Exception as e:
        print(f"[speech_id] identify_from_file error: {e}")
        # fallback: try to compute embedding directly and identify
        try:
            emb = get_embedding(wav_path)
            return identify_user(emb)
        except Exception as e2:
            print(f"[speech_id] identify_from_file fallback error: {e2}")
            return None, None

# ------------------ Enrollment / Update ------------------
def enroll_user(name: str, samples: int = 3):
    """Enroll a new user by recording `samples` audio files and averaging embeddings."""
    db = load_db()
    user_dir = VOICE_SAMPLES_DIR / name
    user_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    for i in range(samples):
        filename = user_dir / f"{name}_enroll_{int(time.time())}_{i+1}.wav"
        print(f"🔴 Sample {i+1}/{samples} for {name}")
        record_audio(str(filename))
        try:
            emb = get_embedding(str(filename))
        except Exception as e:
            print(f"[speech_id] Warning: embedding for sample {i+1} failed: {e}")
            continue
        embeddings.append(emb)

    if not embeddings:
        print(f"❌ No valid embeddings collected for {name}; enrollment aborted.")
        return

    avg_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = (avg_emb / norm).tolist()
    else:
        avg_emb = avg_emb.tolist()

    db[name] = avg_emb
    save_db(db)
    print(f"✅ Enrolled {name} with {len(embeddings)} samples (DB saved to {DB_FILE})")

def update_user(name: str, new_samples: int = 2):
    """Update an existing user's embedding by adding new samples and averaging."""
    db = load_db()
    if name not in db:
        print(f"⚠️ User '{name}' not found in DB — call enroll instead.")
        return

    user_dir = VOICE_SAMPLES_DIR / name
    user_dir.mkdir(parents=True, exist_ok=True)

    try:
        existing_emb = np.array(db[name], dtype=float)
    except Exception:
        existing_emb = None

    embeddings = []
    if existing_emb is not None and existing_emb.size > 0:
        embeddings.append(existing_emb)

    for i in range(new_samples):
        filename = user_dir / f"{name}_update_{int(time.time())}_{i+1}.wav"
        print(f"🔁 New sample {i+1}/{new_samples} for {name}")
        record_audio(str(filename))
        try:
            emb = get_embedding(str(filename))
        except Exception as e:
            print(f"[speech_id] Warning: embedding for update sample {i+1} failed: {e}")
            continue
        embeddings.append(emb)

    if not embeddings:
        print(f"❌ No valid embeddings collected for {name}; update aborted.")
        return

    avg_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = (avg_emb / norm).tolist()
    else:
        avg_emb = avg_emb.tolist()

    db[name] = avg_emb
    save_db(db)
    print(f"✅ Updated {name} with {len(embeddings) - (1 if existing_emb is not None else 0)} new samples (DB saved)")

# ------------------ Convenience wrappers ------------------
def identify_speaker_from_file(testfile: str):
    """Identify a speaker from a WAV file path; returns (name, score)."""
    db = load_db()
    if not db:
        print("⚠️ Voice DB empty — enroll users first.")
        return None, None
    try:
        test_emb = get_embedding(testfile)
    except Exception as e:
        print(f"[speech_id] Failed to compute embedding for test file: {e}")
        return None, None
    return identify_user(test_emb)

def identify_speaker_interactive(limit: int = 20):
    """Record a test wave (round robin), run validation, and attempt identification."""
    idx = get_next_index(limit)
    testfile = TEST_WAV_DIR / f"test_{idx}.wav"
    record_audio(str(testfile))

    try:
        sr, data = wav.read(str(testfile))
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if len(data) == 0:
            print(f"[speech_id] Skipping testfile {testfile} — empty recording.")
            testfile.unlink(missing_ok=True)
            return
        rms = np.sqrt(np.mean(np.square(data.astype(float))))
        if np.issubdtype(data.dtype, np.integer):
            rms = rms / float(np.iinfo(data.dtype).max)
        if rms < 0.01:
            print(f"[speech_id] Skipping testfile {testfile} — silence detected (rms={rms:.6f}).")
            testfile.unlink(missing_ok=True)
            return
    except Exception as e:
        print(f"[speech_id] Error validating recorded testfile {testfile}: {e}")
        testfile.unlink(missing_ok=True)
        return

    name, score = identify_speaker_from_file(str(testfile))
    if name:
        print(f"🔊 Identified as: {name} (score={score:.3f})")
    else:
        print(f"⚠️ Unknown speaker (best score={score:.3f})")

# ------------------ DB migration helper ------------------
def migrate_normalize_db():
    """Normalize all embeddings in DB to unit vectors (creates a .bak copy first)."""
    if not DB_FILE.exists():
        print(f"[speech_id] No DB file found at {DB_FILE}; nothing to migrate.")
        return
    backup = DB_FILE.with_suffix(DB_FILE.suffix + ".bak")
    shutil.copy2(DB_FILE, backup)
    print(f"[speech_id] Backed up DB to {backup}")

    db = load_db()
    changed = 0
    for k, v in list(db.items()):
        a = np.array(v, dtype=float)
        n = np.linalg.norm(a)
        if n > 0:
            db[k] = (a / n).tolist()
            changed += 1
        else:
            print(f"[speech_id] Warning: {k} has zero-norm embedding; leaving as-is.")
    save_db(db)
    print(f"[speech_id] Normalized {changed} embeddings in {DB_FILE}")

# ------------------ CLI ------------------
def _cli():
    while True:
        print()
        print("--- Alfred: Speech Identification ---")
        print("1. Enroll new user")
        print("2. Update existing user")
        print("3. Identify (record test now)")
        print("4. Identify from WAV file")
        print("5. List enrolled users")
        print("6. Normalize/migrate DB (backup first)")
        print("7. Exit")
        choice = input("Choose: ").strip()

        if choice == "1":
            name = input("Enter new user name: ").strip()
            samples = input("How many samples? (default=3): ").strip()
            samples = int(samples) if samples else 3
            enroll_user(name, samples)

        elif choice == "2":
            name = input("Enter existing user name to update: ").strip()
            samples = input("How many new samples? (default=2): ").strip()
            samples = int(samples) if samples else 2
            update_user(name, samples)

        elif choice == "3":
            identify_speaker_interactive()

        elif choice == "4":
            path = input("Path to wav file: ").strip()
            if not path:
                print("No path given.")
            else:
                name, score = identify_from_file(path)
                if name:
                    print(f"Identified: {name} (score={score:.3f})")
                else:
                    print(f"Unknown (best score={score})")

        elif choice == "5":
            db = load_db()
            if db:
                print("Enrolled users:")
                for u in db.keys():
                    print(" -", u)
            else:
                print("(none)")

        elif choice == "6":
            confirm = input("This will backup and normalize your DB. Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                migrate_normalize_db()
            else:
                print("Migration canceled.")

        elif choice == "7":
            break

        else:
            print("Invalid option — choose again.")

# Run CLI if module executed directly
if __name__ == "__main__":
    print("Speech ID module starting. Model loaded:", verification is not None)
    try:
        _cli()
    except KeyboardInterrupt:
        print("\nExiting.")


##########"""
##########Speech_Identification_User.py
##########Robust speaker identification utilities using SpeechBrain (ecapa-voxceleb).
##########- Uses LocalStrategy.COPY to avoid Windows symlink issues.
##########- Records audio via sounddevice, saves 16kHz mono int16 WAVs.
##########- Computes normalized embeddings using SpeechBrain SpeakerRecognition.
##########- Stores normalized embeddings in a JSON DB (atomic saves).
##########- Enrollment / update / interactive identification CLI included.
##########"""
##########
########### --- IMPORTANT: environment variables to avoid Windows symlink problems ---
##########import os
##########os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
##########os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
##########
########### ------------------ torchaudio compatibility shim ------------------
########### Some torchaudio versions removed `torchaudio.list_audio_backends`.
########### Provide a small compatibility shim so imports that call that function won't crash.
##########try:
##########    import torchaudio
##########    # If missing, attempt to alias a sensible replacement
##########    if not hasattr(torchaudio, "list_audio_backends"):
##########        if hasattr(torchaudio, "utils") and hasattr(torchaudio.utils, "list_audio_backends"):
##########            # new location
##########            torchaudio.list_audio_backends = torchaudio.utils.list_audio_backends  # type: ignore
##########        else:
##########            # Create a tiny fallback that returns the active backend if possible
##########            def _fallback_list_audio_backends():
##########                backends = []
##########                try:
##########                    if hasattr(torchaudio, "get_audio_backend"):
##########                        backend = torchaudio.get_audio_backend()
##########                        if backend is not None:
##########                            backends.append(backend)
##########                except Exception:
##########                    pass
##########                return backends
##########            torchaudio.list_audio_backends = _fallback_list_audio_backends  # type: ignore
##########except Exception:
##########    # If torchaudio import fails entirely, don't crash here — the rest of the module
##########    # may still work (we only need torchaudio for optional features).
##########    pass
##########
########### Standard libs
##########import time
##########import json
##########from pathlib import Path
##########import shutil
##########import numpy as np
##########import sounddevice as sd
##########import scipy.io.wavfile as wav
##########import torch
##########
########### SpeechBrain imports (new inference path)
##########from speechbrain.inference import SpeakerRecognition
##########from speechbrain.utils.fetching import LocalStrategy
##########
########### ------------------ Configuration ------------------
########### Adjust these to taste
##########CONFIDENCE_THRESHOLD = 0.60   # cosine similarity threshold (0.6-0.85 typical)
##########SAMPLE_RATE = 16000           # target sampling rate for SpeechBrain model
##########DEFAULT_RECORD_SECONDS = 5
##########VOICE_SAMPLES_DIR = Path("voice_samples")
##########TEST_WAV_DIR = Path("test_waves")
##########VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
##########TEST_WAV_DIR.mkdir(parents=True, exist_ok=True)
##########ROUND_ROBIN_FILE = TEST_WAV_DIR / "last_index.txt"
##########SAVEDIR = Path("pretrained_model")  # HF model cache target (LocalStrategy.COPY)
##########
########### DB location (Windows-friendly). If you have Alfred_config with DRIVE_LETTER, prefer that.
##########try:
##########    import Alfred_config
##########    DRIVE = getattr(Alfred_config, "DRIVE_LETTER", "")
##########    if DRIVE:
##########        DB_FILE = Path(DRIVE) / "Python_Env" / "New_Virtual_Env" / "Alfred_Offline_New_GUI" / \
##########                  "2025_08_30A_WEBUI_MODEL_New_TRY_OLD" / "New_V2_Home_Head_Movement_Smoothing" / \
##########                  "modules" / "voice_db.json"
##########    else:
##########        DB_FILE = Path("voice_db.json")
##########except Exception:
##########    DB_FILE = Path("voice_db.json")
##########
########### make sure directories exist
##########DB_FILE.parent.mkdir(parents=True, exist_ok=True)
##########SAVEDIR.mkdir(parents=True, exist_ok=True)
##########
########### ------------------ Utilities ------------------
##########def get_next_index(limit: int = 20) -> int:
##########    """Return the next index for round-robin overwriting (1..limit)."""
##########    try:
##########        if ROUND_ROBIN_FILE.exists():
##########            try:
##########                last = int(ROUND_ROBIN_FILE.read_text().strip())
##########            except Exception:
##########                last = 0
##########        else:
##########            last = 0
##########        new_index = (last % limit) + 1
##########        ROUND_ROBIN_FILE.write_text(str(new_index))
##########        return new_index
##########    except Exception:
##########        # fallback safe behavior
##########        return 1
##########
##########def _atomic_write_text(path: Path, text: str):
##########    """Write text atomically: write to tmp file then replace target."""
##########    tmp = path.with_suffix(path.suffix + ".tmp")
##########    try:
##########        tmp.write_text(text, encoding="utf-8")
##########        tmp.replace(path)
##########    except Exception as e:
##########        # If replace fails, attempt a safe fallback
##########        try:
##########            tmp.unlink(missing_ok=True)
##########        except Exception:
##########            pass
##########        raise e
##########
##########def load_db() -> dict:
##########    """Load JSON DB (name -> embedding list). Returns {} if missing/invalid."""
##########    try:
##########        if DB_FILE.exists():
##########            text = DB_FILE.read_text(encoding="utf-8")
##########            if not text:
##########                return {}
##########            return json.loads(text)
##########    except Exception as e:
##########        print(f"[speech_id] Warning: failed to load DB ({e}), starting fresh.")
##########    return {}
##########
##########def save_db(db: dict):
##########    """Save JSON DB using atomic write to reduce corruption risk."""
##########    try:
##########        _atomic_write_text(DB_FILE, json.dumps(db, ensure_ascii=False))
##########    except Exception as e:
##########        print(f"[speech_id] Error saving DB atomically: {e}; falling back to direct write.")
##########        try:
##########            DB_FILE.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
##########        except Exception as e2:
##########            print(f"[speech_id] FATAL: DB save failed: {e2}")
##########
########### ------------------ Model loading ------------------
##########def load_speaker_model(savedir: Path = SAVEDIR) -> SpeakerRecognition:
##########    """
##########    Load the SpeechBrain SpeakerRecognition model with LocalStrategy.COPY (no symlinks).
##########    Returns the loaded model and places it on GPU if available.
##########    """
##########    savedir = Path(savedir)
##########    print("[speech_id] Loading SpeechBrain speaker model (LocalStrategy.COPY). This may download ~100MB.")
##########    try:
##########        model = SpeakerRecognition.from_hparams(
##########            source="speechbrain/spkrec-ecapa-voxceleb",
##########            savedir=str(savedir),
##########            local_strategy=LocalStrategy.COPY,
##########        )
##########    except Exception as e:
##########        print(f"[speech_id] Failed to load model from HF: {e}")
##########        raise
##########
##########    # prefer GPU if available
##########    device = "cuda" if torch.cuda.is_available() else "cpu"
##########    # Many SpeechBrain wrappers handle device internally; try to move if possible
##########    try:
##########        if hasattr(model, "to"):
##########            try:
##########                model.to(device)
##########            except Exception:
##########                pass
##########        # Some wrappers expose `.device` attribute; set it if present.
##########        try:
##########            setattr(model, "device", device)
##########        except Exception:
##########            pass
##########    except Exception:
##########        pass
##########
##########    print(f"[speech_id] Model loaded and placed on device: {device}")
##########    return model
##########
########### load model (singleton)
##########try:
##########    verification = load_speaker_model(SAVEDIR)
##########except Exception as exc:
##########    print(f"[speech_id] ERROR: speaker model failed to load: {exc}")
##########    verification = None
##########
########### ------------------ Audio utilities ------------------
##########def record_audio(filename: str, duration: int = DEFAULT_RECORD_SECONDS, fs: int = SAMPLE_RATE):
##########    """
##########    Record audio from default microphone and save as a WAV file (mono int16).
##########    Uses sounddevice; returns after the file is written.
##########    """
##########    print(f"🎙️ Recording {duration}s to '{filename}' — speak now...")
##########    try:
##########        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
##########        sd.wait()
##########    except Exception as e:
##########        print(f"[speech_id] Error recording audio: {e}")
##########        raise
##########
##########    # sounddevice returns shape (N, channels) — flatten to 1-D
##########    audio_np = np.asarray(audio)
##########    if audio_np.ndim > 1:
##########        audio_1d = audio_np.reshape(-1)
##########    else:
##########        audio_1d = audio_np
##########
##########    if audio_1d.dtype != np.int16:
##########        # cast to int16 safely (clip)
##########        audio_1d = np.clip(audio_1d, -32768, 32767).astype(np.int16)
##########
##########    try:
##########        wav.write(filename, fs, audio_1d)
##########        print(f"✅ Saved recording to {filename}")
##########    except Exception as e:
##########        print(f"[speech_id] Error saving recording {filename}: {e}")
##########        raise
##########
##########def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
##########    """Resample 1-D numpy array to target_sr. Returns float32 array."""
##########    if sr == target_sr:
##########        return data.astype(np.float32)
##########    from scipy.signal import resample
##########    num = int(len(data) * target_sr / sr)
##########    if num <= 1:
##########        raise ValueError(f"Invalid resample length {num} (orig sr={sr}, len={len(data)})")
##########    return resample(data, num).astype(np.float32)
##########
##########def get_embedding(filepath: str) -> np.ndarray:
##########    """
##########    Compute a normalized embedding for the WAV file using the loaded SpeechBrain model.
##########    Returns a 1-D numpy float array (unit norm).
##########    """
##########    if verification is None:
##########        raise RuntimeError("Speaker verification model not loaded.")
##########    sr, data = wav.read(filepath)
##########
##########    # Ensure mono
##########    if getattr(data, "ndim", 1) > 1:
##########        data = data.mean(axis=1)
##########
##########    # Convert to float32 in range [-1,1] if integer dtypes
##########    if np.issubdtype(data.dtype, np.integer):
##########        max_val = float(np.iinfo(data.dtype).max)
##########        # guard: int16 -> divide by 32767 to get approx -1..1
##########        data = data.astype(np.float32) / max_val
##########    else:
##########        data = data.astype(np.float32)
##########
##########    if sr != SAMPLE_RATE:
##########        data = _resample_if_needed(data, sr, SAMPLE_RATE)
##########
##########    if data.size == 0:
##########        raise ValueError("Audio file contains no data after preprocessing.")
##########
##########    # Create tensor shape (batch, samples)
##########    tensor = torch.from_numpy(data).unsqueeze(0).float()
##########    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##########    try:
##########        tensor = tensor.to(device)
##########    except Exception:
##########        tensor = tensor.cpu()
##########
##########    # compute embedding (inference mode)
##########    try:
##########        with torch.no_grad():
##########            emb = verification.encode_batch(tensor)
##########    except Exception as e:
##########        # fallback to CPU if GPU fails
##########        try:
##########            tensor_cpu = tensor.to("cpu")
##########            with torch.no_grad():
##########                emb = verification.encode_batch(tensor_cpu)
##########        except Exception as e2:
##########            print(f"[speech_id] Embedding error: {e} / fallback: {e2}")
##########            raise
##########
##########    # Convert to numpy safely and normalize
##########    try:
##########        emb = emb.squeeze().detach().cpu().numpy().astype(float)
##########    except Exception:
##########        emb = np.array(emb).squeeze().astype(float)
##########    norm = np.linalg.norm(emb)
##########    if norm > 0:
##########        emb = emb / norm
##########    return emb
##########
########### ------------------ DB / Identity helpers ------------------
##########def identify_user(embedding: np.ndarray):
##########    """Identify the closest enrolled speaker given a precomputed embedding."""
##########    db = load_db()
##########    if not db:
##########        print("⚠️ Voice DB empty — enroll users first.")
##########        return None, None
##########
##########    test_emb = np.array(embedding, dtype=float)
##########    if test_emb.size == 0:
##########        return None, None
##########    test_norm = np.linalg.norm(test_emb)
##########    if test_norm == 0:
##########        return None, None
##########    test_emb = test_emb / test_norm
##########
##########    best_score = -1.0
##########    best_name = None
##########    for name, emb in db.items():
##########        emb = np.array(emb, dtype=float)
##########        if emb.size == 0:
##########            continue
##########        emb_norm = np.linalg.norm(emb)
##########        if emb_norm == 0:
##########            continue
##########        emb = emb / emb_norm
##########        score = float(np.dot(test_emb, emb))
##########        if score > best_score:
##########            best_score = score
##########            best_name = name
##########
##########    if best_score < CONFIDENCE_THRESHOLD:
##########        return None, best_score
##########    return best_name, best_score
##########
##########def identify_from_file(wav_path: str, limit: int = 20):
##########    """
##########    Copy wav into test_waves (round-robin) and identify speaker, but only if audio is valid (non-silent).
##########    Returns (name, score) or (None, score).
##########    """
##########    try:
##########        sr, data = wav.read(wav_path)
##########        if getattr(data, "ndim", 1) > 1:
##########            data = data.mean(axis=1)
##########
##########        if len(data) == 0:
##########            print(f"[speech_id] Skipping {wav_path} — empty file.")
##########            return None, None
##########
##########        # compute normalized RMS for silence check
##########        rms = np.sqrt(np.mean(np.square(data.astype(float))))
##########        if np.issubdtype(data.dtype, np.integer):
##########            rms = rms / float(np.iinfo(data.dtype).max)
##########        # threshold: 0.01 works for normalized [-1,1] floats
##########        if rms < 0.01:
##########            print(f"[speech_id] Skipping {wav_path} — silence/low energy (rms={rms:.6f}).")
##########            return None, None
##########
##########        # Passed checks → save into round-robin slot
##########        idx = get_next_index(limit)
##########        dest = TEST_WAV_DIR / f"test_{idx}.wav"
##########        shutil.copy(wav_path, dest)
##########        print(f"[speech_id] Saved test file → {dest}")
##########
##########        emb = get_embedding(str(dest))
##########        return identify_user(emb)
##########
##########    except Exception as e:
##########        print(f"[speech_id] identify_from_file error: {e}")
##########        # fallback: try to compute embedding directly and identify
##########        try:
##########            emb = get_embedding(wav_path)
##########            return identify_user(emb)
##########        except Exception as e2:
##########            print(f"[speech_id] identify_from_file fallback error: {e2}")
##########            return None, None
##########
########### ------------------ Enrollment / Update ------------------
##########def enroll_user(name: str, samples: int = 3):
##########    """Enroll a new user by recording `samples` audio files and averaging embeddings."""
##########    db = load_db()
##########    user_dir = VOICE_SAMPLES_DIR / name
##########    user_dir.mkdir(parents=True, exist_ok=True)
##########
##########    embeddings = []
##########    for i in range(samples):
##########        filename = user_dir / f"{name}_enroll_{int(time.time())}_{i+1}.wav"
##########        print(f"🔴 Sample {i+1}/{samples} for {name}")
##########        record_audio(str(filename))
##########        try:
##########            emb = get_embedding(str(filename))
##########        except Exception as e:
##########            print(f"[speech_id] Warning: embedding for sample {i+1} failed: {e}")
##########            continue
##########        embeddings.append(emb)
##########
##########    if not embeddings:
##########        print(f"❌ No valid embeddings collected for {name}; enrollment aborted.")
##########        return
##########
##########    avg_emb = np.mean(embeddings, axis=0)
##########    norm = np.linalg.norm(avg_emb)
##########    if norm > 0:
##########        avg_emb = (avg_emb / norm).tolist()
##########    else:
##########        avg_emb = avg_emb.tolist()
##########
##########    db[name] = avg_emb
##########    save_db(db)
##########    print(f"✅ Enrolled {name} with {len(embeddings)} samples (DB saved to {DB_FILE})")
##########
##########def update_user(name: str, new_samples: int = 2):
##########    """Update an existing user's embedding by adding new samples and averaging."""
##########    db = load_db()
##########    if name not in db:
##########        print(f"⚠️ User '{name}' not found in DB — call enroll instead.")
##########        return
##########
##########    user_dir = VOICE_SAMPLES_DIR / name
##########    user_dir.mkdir(parents=True, exist_ok=True)
##########
##########    try:
##########        existing_emb = np.array(db[name], dtype=float)
##########    except Exception:
##########        existing_emb = None
##########
##########    embeddings = []
##########    if existing_emb is not None and existing_emb.size > 0:
##########        embeddings.append(existing_emb)
##########
##########    for i in range(new_samples):
##########        filename = user_dir / f"{name}_update_{int(time.time())}_{i+1}.wav"
##########        print(f"🔁 New sample {i+1}/{new_samples} for {name}")
##########        record_audio(str(filename))
##########        try:
##########            emb = get_embedding(str(filename))
##########        except Exception as e:
##########            print(f"[speech_id] Warning: embedding for update sample {i+1} failed: {e}")
##########            continue
##########        embeddings.append(emb)
##########
##########    if not embeddings:
##########        print(f"❌ No valid embeddings collected for {name}; update aborted.")
##########        return
##########
##########    avg_emb = np.mean(embeddings, axis=0)
##########    norm = np.linalg.norm(avg_emb)
##########    if norm > 0:
##########        avg_emb = (avg_emb / norm).tolist()
##########    else:
##########        avg_emb = avg_emb.tolist()
##########
##########    db[name] = avg_emb
##########    save_db(db)
##########    print(f"✅ Updated {name} with {len(embeddings) - (1 if existing_emb is not None else 0)} new samples (DB saved)")
##########
########### ------------------ Convenience wrappers ------------------
##########def identify_speaker_from_file(testfile: str):
##########    """Identify a speaker from a WAV file path; returns (name, score)."""
##########    db = load_db()
##########    if not db:
##########        print("⚠️ Voice DB empty — enroll users first.")
##########        return None, None
##########    try:
##########        test_emb = get_embedding(testfile)
##########    except Exception as e:
##########        print(f"[speech_id] Failed to compute embedding for test file: {e}")
##########        return None, None
##########    return identify_user(test_emb)
##########
##########def identify_speaker_interactive(limit: int = 20):
##########    """Record a test wave (round robin), run validation, and attempt identification."""
##########    idx = get_next_index(limit)
##########    testfile = TEST_WAV_DIR / f"test_{idx}.wav"
##########    record_audio(str(testfile))
##########
##########    try:
##########        sr, data = wav.read(str(testfile))
##########        if getattr(data, "ndim", 1) > 1:
##########            data = data.mean(axis=1)
##########        if len(data) == 0:
##########            print(f"[speech_id] Skipping testfile {testfile} — empty recording.")
##########            testfile.unlink(missing_ok=True)
##########            return
##########        rms = np.sqrt(np.mean(np.square(data.astype(float))))
##########        if np.issubdtype(data.dtype, np.integer):
##########            rms = rms / float(np.iinfo(data.dtype).max)
##########        if rms < 0.01:
##########            print(f"[speech_id] Skipping testfile {testfile} — silence detected (rms={rms:.6f}).")
##########            testfile.unlink(missing_ok=True)
##########            return
##########    except Exception as e:
##########        print(f"[speech_id] Error validating recorded testfile {testfile}: {e}")
##########        testfile.unlink(missing_ok=True)
##########        return
##########
##########    name, score = identify_speaker_from_file(str(testfile))
##########    if name:
##########        print(f"🔊 Identified as: {name} (score={score:.3f})")
##########    else:
##########        print(f"⚠️ Unknown speaker (best score={score:.3f})")
##########
########### ------------------ DB migration helper ------------------
##########def migrate_normalize_db():
##########    """Normalize all embeddings in DB to unit vectors (creates a .bak copy first)."""
##########    if not DB_FILE.exists():
##########        print(f"[speech_id] No DB file found at {DB_FILE}; nothing to migrate.")
##########        return
##########    backup = DB_FILE.with_suffix(DB_FILE.suffix + ".bak")
##########    shutil.copy2(DB_FILE, backup)
##########    print(f"[speech_id] Backed up DB to {backup}")
##########
##########    db = load_db()
##########    changed = 0
##########    for k, v in list(db.items()):
##########        a = np.array(v, dtype=float)
##########        n = np.linalg.norm(a)
##########        if n > 0:
##########            db[k] = (a / n).tolist()
##########            changed += 1
##########        else:
##########            print(f"[speech_id] Warning: {k} has zero-norm embedding; leaving as-is.")
##########    save_db(db)
##########    print(f"[speech_id] Normalized {changed} embeddings in {DB_FILE}")
##########
########### ------------------ CLI ------------------
##########def _cli():
##########    while True:
##########        print()
##########        print("--- Alfred: Speech Identification ---")
##########        print("1. Enroll new user")
##########        print("2. Update existing user")
##########        print("3. Identify (record test now)")
##########        print("4. Identify from WAV file")
##########        print("5. List enrolled users")
##########        print("6. Normalize/migrate DB (backup first)")
##########        print("7. Exit")
##########        choice = input("Choose: ").strip()
##########
##########        if choice == "1":
##########            name = input("Enter new user name: ").strip()
##########            samples = input("How many samples? (default=3): ").strip()
##########            samples = int(samples) if samples else 3
##########            enroll_user(name, samples)
##########
##########        elif choice == "2":
##########            name = input("Enter existing user name to update: ").strip()
##########            samples = input("How many new samples? (default=2): ").strip()
##########            samples = int(samples) if samples else 2
##########            update_user(name, samples)
##########
##########        elif choice == "3":
##########            identify_speaker_interactive()
##########
##########        elif choice == "4":
##########            path = input("Path to wav file: ").strip()
##########            if not path:
##########                print("No path given.")
##########            else:
##########                name, score = identify_from_file(path)
##########                if name:
##########                    print(f"Identified: {name} (score={score:.3f})")
##########                else:
##########                    print(f"Unknown (best score={score})")
##########
##########        elif choice == "5":
##########            db = load_db()
##########            if db:
##########                print("Enrolled users:")
##########                for u in db.keys():
##########                    print(" -", u)
##########            else:
##########                print("(none)")
##########
##########        elif choice == "6":
##########            confirm = input("This will backup and normalize your DB. Continue? (y/n): ").strip().lower()
##########            if confirm == 'y':
##########                migrate_normalize_db()
##########            else:
##########                print("Migration canceled.")
##########
##########        elif choice == "7":
##########            break
##########
##########        else:
##########            print("Invalid option — choose again.")
##########
########### Run CLI if module executed directly
##########if __name__ == "__main__":
##########    print("Speech ID module starting. Model loaded:", verification is not None)
##########    try:
##########        _cli()
##########    except KeyboardInterrupt:
##########        print("\nExiting.")







########"""
########Speech_Identification_User.py
########Robust speaker identification utilities using SpeechBrain (ecapa-voxceleb).
########- Uses LocalStrategy.COPY to avoid Windows symlink issues.
########- Records audio via sounddevice, saves 16kHz mono int16 WAVs.
########- Computes normalized embeddings using SpeechBrain SpeakerRecognition.
########- Stores normalized embeddings in a JSON DB (atomic saves).
########- Enrollment / update / interactive identification CLI included.
########"""
########
######### --- IMPORTANT: environment variables to avoid Windows symlink problems ---
########import os
########os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
########os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
########
######### Standard libs
########import time
########import json
########from pathlib import Path
########import shutil
########import numpy as np
########import sounddevice as sd
########import scipy.io.wavfile as wav
########import torch
########
######### SpeechBrain imports (new inference path)
########from speechbrain.inference import SpeakerRecognition
########from speechbrain.utils.fetching import LocalStrategy
########
######### ------------------ Configuration ------------------
######### Adjust these to taste
########CONFIDENCE_THRESHOLD = 0.60   # cosine similarity threshold (0.6-0.85 typical)
########SAMPLE_RATE = 16000           # target sampling rate for SpeechBrain model
########DEFAULT_RECORD_SECONDS = 5
########VOICE_SAMPLES_DIR = Path("voice_samples")
########TEST_WAV_DIR = Path("test_waves")
########VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
########TEST_WAV_DIR.mkdir(parents=True, exist_ok=True)
########ROUND_ROBIN_FILE = TEST_WAV_DIR / "last_index.txt"
########SAVEDIR = Path("pretrained_model")  # HF model cache target (LocalStrategy.COPY)
########
######### DB location (Windows-friendly). If you have Alfred_config with DRIVE_LETTER, prefer that.
########try:
########    import Alfred_config
########    DRIVE = getattr(Alfred_config, "DRIVE_LETTER", "")
########    if DRIVE:
########        DB_FILE = Path(DRIVE) / "Python_Env" / "New_Virtual_Env" / "Alfred_Offline_New_GUI" / \
########                  "2025_08_30A_WEBUI_MODEL_New_TRY_OLD" / "New_V2_Home_Head_Movement_Smoothing" / \
########                  "modules" / "voice_db.json"
########    else:
########        DB_FILE = Path("voice_db.json")
########except Exception:
########    DB_FILE = Path("voice_db.json")
########
######### make sure directories exist
########DB_FILE.parent.mkdir(parents=True, exist_ok=True)
########SAVEDIR.mkdir(parents=True, exist_ok=True)
########
######### ------------------ Utilities ------------------
########def get_next_index(limit: int = 20) -> int:
########    """Return the next index for round-robin overwriting (1..limit)."""
########    try:
########        if ROUND_ROBIN_FILE.exists():
########            try:
########                last = int(ROUND_ROBIN_FILE.read_text().strip())
########            except Exception:
########                last = 0
########        else:
########            last = 0
########        new_index = (last % limit) + 1
########        ROUND_ROBIN_FILE.write_text(str(new_index))
########        return new_index
########    except Exception:
########        # fallback safe behavior
########        return 1
########
########def _atomic_write_text(path: Path, text: str):
########    """Write text atomically: write to tmp file then replace target."""
########    tmp = path.with_suffix(path.suffix + ".tmp")
########    try:
########        tmp.write_text(text, encoding="utf-8")
########        tmp.replace(path)
########    except Exception as e:
########        # If replace fails, attempt a safe fallback
########        try:
########            tmp.unlink(missing_ok=True)
########        except Exception:
########            pass
########        raise e
########
########def load_db() -> dict:
########    """Load JSON DB (name -> embedding list). Returns {} if missing/invalid."""
########    try:
########        if DB_FILE.exists():
########            text = DB_FILE.read_text(encoding="utf-8")
########            if not text:
########                return {}
########            return json.loads(text)
########    except Exception as e:
########        print(f"[speech_id] Warning: failed to load DB ({e}), starting fresh.")
########    return {}
########
########def save_db(db: dict):
########    """Save JSON DB using atomic write to reduce corruption risk."""
########    try:
########        _atomic_write_text(DB_FILE, json.dumps(db, ensure_ascii=False))
########    except Exception as e:
########        print(f"[speech_id] Error saving DB atomically: {e}; falling back to direct write.")
########        try:
########            DB_FILE.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")
########        except Exception as e2:
########            print(f"[speech_id] FATAL: DB save failed: {e2}")
########
######### ------------------ Model loading ------------------
########def load_speaker_model(savedir: Path = SAVEDIR) -> SpeakerRecognition:
########    """
########    Load the SpeechBrain SpeakerRecognition model with LocalStrategy.COPY (no symlinks).
########    Returns the loaded model and places it on GPU if available.
########    """
########    savedir = Path(savedir)
########    print("[speech_id] Loading SpeechBrain speaker model (LocalStrategy.COPY). This may download ~100MB.")
########    try:
########        model = SpeakerRecognition.from_hparams(
########            source="speechbrain/spkrec-ecapa-voxceleb",
########            savedir=str(savedir),
########            local_strategy=LocalStrategy.COPY,
########        )
########    except Exception as e:
########        print(f"[speech_id] Failed to load model from HF: {e}")
########        raise
########
########    # prefer GPU if available
########    device = "cuda" if torch.cuda.is_available() else "cpu"
########    # Many SpeechBrain wrappers handle device internally; try to move if possible
########    try:
########        if hasattr(model, "to"):
########            try:
########                model.to(device)
########            except Exception:
########                pass
########        # Some wrappers expose `.device` attribute; set it if present.
########        try:
########            setattr(model, "device", device)
########        except Exception:
########            pass
########    except Exception:
########        pass
########
########    print(f"[speech_id] Model loaded and placed on device: {device}")
########    return model
########
######### load model (singleton)
########try:
########    verification = load_speaker_model(SAVEDIR)
########except Exception as exc:
########    print(f"[speech_id] ERROR: speaker model failed to load: {exc}")
########    verification = None
########
######### ------------------ Audio utilities ------------------
########def record_audio(filename: str, duration: int = DEFAULT_RECORD_SECONDS, fs: int = SAMPLE_RATE):
########    """
########    Record audio from default microphone and save as a WAV file (mono int16).
########    Uses sounddevice; returns after the file is written.
########    """
########    print(f"🎙️ Recording {duration}s to '{filename}' — speak now...")
########    try:
########        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
########        sd.wait()
########    except Exception as e:
########        print(f"[speech_id] Error recording audio: {e}")
########        raise
########
########    # sounddevice returns shape (N, channels) — flatten to 1-D
########    audio_np = np.asarray(audio)
########    if audio_np.ndim > 1:
########        audio_1d = audio_np.reshape(-1)
########    else:
########        audio_1d = audio_np
########
########    if audio_1d.dtype != np.int16:
########        # cast to int16 safely (clip)
########        audio_1d = np.clip(audio_1d, -32768, 32767).astype(np.int16)
########
########    try:
########        wav.write(filename, fs, audio_1d)
########        print(f"✅ Saved recording to {filename}")
########    except Exception as e:
########        print(f"[speech_id] Error saving recording {filename}: {e}")
########        raise
########
########def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
########    """Resample 1-D numpy array to target_sr. Returns float32 array."""
########    if sr == target_sr:
########        return data.astype(np.float32)
########    from scipy.signal import resample
########    num = int(len(data) * target_sr / sr)
########    if num <= 1:
########        raise ValueError(f"Invalid resample length {num} (orig sr={sr}, len={len(data)})")
########    return resample(data, num).astype(np.float32)
########
########def get_embedding(filepath: str) -> np.ndarray:
########    """
########    Compute a normalized embedding for the WAV file using the loaded SpeechBrain model.
########    Returns a 1-D numpy float array (unit norm).
########    """
########    if verification is None:
########        raise RuntimeError("Speaker verification model not loaded.")
########    sr, data = wav.read(filepath)
########
########    # Ensure mono
########    if getattr(data, "ndim", 1) > 1:
########        data = data.mean(axis=1)
########
########    # Convert to float32 in range [-1,1] if integer dtypes
########    if np.issubdtype(data.dtype, np.integer):
########        max_val = float(np.iinfo(data.dtype).max)
########        # guard: int16 -> divide by 32767 to get approx -1..1
########        data = data.astype(np.float32) / max_val
########    else:
########        data = data.astype(np.float32)
########
########    if sr != SAMPLE_RATE:
########        data = _resample_if_needed(data, sr, SAMPLE_RATE)
########
########    if data.size == 0:
########        raise ValueError("Audio file contains no data after preprocessing.")
########
########    # Create tensor shape (batch, samples)
########    tensor = torch.from_numpy(data).unsqueeze(0).float()
########    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########    try:
########        tensor = tensor.to(device)
########    except Exception:
########        tensor = tensor.cpu()
########
########    # compute embedding (inference mode)
########    try:
########        with torch.no_grad():
########            emb = verification.encode_batch(tensor)
########    except Exception as e:
########        # fallback to CPU if GPU fails
########        try:
########            tensor_cpu = tensor.to("cpu")
########            with torch.no_grad():
########                emb = verification.encode_batch(tensor_cpu)
########        except Exception as e2:
########            print(f"[speech_id] Embedding error: {e} / fallback: {e2}")
########            raise
########
########    # Convert to numpy safely and normalize
########    try:
########        emb = emb.squeeze().detach().cpu().numpy().astype(float)
########    except Exception:
########        emb = np.array(emb).squeeze().astype(float)
########    norm = np.linalg.norm(emb)
########    if norm > 0:
########        emb = emb / norm
########    return emb
########
######### ------------------ DB / Identity helpers ------------------
########def identify_user(embedding: np.ndarray):
########    """Identify the closest enrolled speaker given a precomputed embedding."""
########    db = load_db()
########    if not db:
########        print("⚠️ Voice DB empty — enroll users first.")
########        return None, None
########
########    test_emb = np.array(embedding, dtype=float)
########    if test_emb.size == 0:
########        return None, None
########    test_norm = np.linalg.norm(test_emb)
########    if test_norm == 0:
########        return None, None
########    test_emb = test_emb / test_norm
########
########    best_score = -1.0
########    best_name = None
########    for name, emb in db.items():
########        emb = np.array(emb, dtype=float)
########        if emb.size == 0:
########            continue
########        emb_norm = np.linalg.norm(emb)
########        if emb_norm == 0:
########            continue
########        emb = emb / emb_norm
########        score = float(np.dot(test_emb, emb))
########        if score > best_score:
########            best_score = score
########            best_name = name
########
########    if best_score < CONFIDENCE_THRESHOLD:
########        return None, best_score
########    return best_name, best_score
########
########def identify_from_file(wav_path: str, limit: int = 20):
########    """
########    Copy wav into test_waves (round-robin) and identify speaker, but only if audio is valid (non-silent).
########    Returns (name, score) or (None, score).
########    """
########    try:
########        sr, data = wav.read(wav_path)
########        if getattr(data, "ndim", 1) > 1:
########            data = data.mean(axis=1)
########
########        if len(data) == 0:
########            print(f"[speech_id] Skipping {wav_path} — empty file.")
########            return None, None
########
########        # compute normalized RMS for silence check
########        rms = np.sqrt(np.mean(np.square(data.astype(float))))
########        if np.issubdtype(data.dtype, np.integer):
########            rms = rms / float(np.iinfo(data.dtype).max)
########        # threshold: 0.01 works for normalized [-1,1] floats
########        if rms < 0.01:
########            print(f"[speech_id] Skipping {wav_path} — silence/low energy (rms={rms:.6f}).")
########            return None, None
########
########        # Passed checks → save into round-robin slot
########        idx = get_next_index(limit)
########        dest = TEST_WAV_DIR / f"test_{idx}.wav"
########        shutil.copy(wav_path, dest)
########        print(f"[speech_id] Saved test file → {dest}")
########
########        emb = get_embedding(str(dest))
########        return identify_user(emb)
########
########    except Exception as e:
########        print(f"[speech_id] identify_from_file error: {e}")
########        # fallback: try to compute embedding directly and identify
########        try:
########            emb = get_embedding(wav_path)
########            return identify_user(emb)
########        except Exception as e2:
########            print(f"[speech_id] identify_from_file fallback error: {e2}")
########            return None, None
########
######### ------------------ Enrollment / Update ------------------
########def enroll_user(name: str, samples: int = 3):
########    """Enroll a new user by recording `samples` audio files and averaging embeddings."""
########    db = load_db()
########    user_dir = VOICE_SAMPLES_DIR / name
########    user_dir.mkdir(parents=True, exist_ok=True)
########
########    embeddings = []
########    for i in range(samples):
########        filename = user_dir / f"{name}_enroll_{int(time.time())}_{i+1}.wav"
########        print(f"🔴 Sample {i+1}/{samples} for {name}")
########        record_audio(str(filename))
########        try:
########            emb = get_embedding(str(filename))
########        except Exception as e:
########            print(f"[speech_id] Warning: embedding for sample {i+1} failed: {e}")
########            continue
########        embeddings.append(emb)
########
########    if not embeddings:
########        print(f"❌ No valid embeddings collected for {name}; enrollment aborted.")
########        return
########
########    avg_emb = np.mean(embeddings, axis=0)
########    norm = np.linalg.norm(avg_emb)
########    if norm > 0:
########        avg_emb = (avg_emb / norm).tolist()
########    else:
########        avg_emb = avg_emb.tolist()
########
########    db[name] = avg_emb
########    save_db(db)
########    print(f"✅ Enrolled {name} with {len(embeddings)} samples (DB saved to {DB_FILE})")
########
########def update_user(name: str, new_samples: int = 2):
########    """Update an existing user's embedding by adding new samples and averaging."""
########    db = load_db()
########    if name not in db:
########        print(f"⚠️ User '{name}' not found in DB — call enroll instead.")
########        return
########
########    user_dir = VOICE_SAMPLES_DIR / name
########    user_dir.mkdir(parents=True, exist_ok=True)
########
########    try:
########        existing_emb = np.array(db[name], dtype=float)
########    except Exception:
########        existing_emb = None
########
########    embeddings = []
########    if existing_emb is not None and existing_emb.size > 0:
########        embeddings.append(existing_emb)
########
########    for i in range(new_samples):
########        filename = user_dir / f"{name}_update_{int(time.time())}_{i+1}.wav"
########        print(f"🔁 New sample {i+1}/{new_samples} for {name}")
########        record_audio(str(filename))
########        try:
########            emb = get_embedding(str(filename))
########        except Exception as e:
########            print(f"[speech_id] Warning: embedding for update sample {i+1} failed: {e}")
########            continue
########        embeddings.append(emb)
########
########    if not embeddings:
########        print(f"❌ No valid embeddings collected for {name}; update aborted.")
########        return
########
########    avg_emb = np.mean(embeddings, axis=0)
########    norm = np.linalg.norm(avg_emb)
########    if norm > 0:
########        avg_emb = (avg_emb / norm).tolist()
########    else:
########        avg_emb = avg_emb.tolist()
########
########    db[name] = avg_emb
########    save_db(db)
########    print(f"✅ Updated {name} with {len(embeddings) - (1 if existing_emb is not None else 0)} new samples (DB saved)")
########
######### ------------------ Convenience wrappers ------------------
########def identify_speaker_from_file(testfile: str):
########    """Identify a speaker from a WAV file path; returns (name, score)."""
########    db = load_db()
########    if not db:
########        print("⚠️ Voice DB empty — enroll users first.")
########        return None, None
########    try:
########        test_emb = get_embedding(testfile)
########    except Exception as e:
########        print(f"[speech_id] Failed to compute embedding for test file: {e}")
########        return None, None
########    return identify_user(test_emb)
########
########def identify_speaker_interactive(limit: int = 20):
########    """Record a test wave (round robin), run validation, and attempt identification."""
########    idx = get_next_index(limit)
########    testfile = TEST_WAV_DIR / f"test_{idx}.wav"
########    record_audio(str(testfile))
########
########    try:
########        sr, data = wav.read(str(testfile))
########        if getattr(data, "ndim", 1) > 1:
########            data = data.mean(axis=1)
########        if len(data) == 0:
########            print(f"[speech_id] Skipping testfile {testfile} — empty recording.")
########            testfile.unlink(missing_ok=True)
########            return
########        rms = np.sqrt(np.mean(np.square(data.astype(float))))
########        if np.issubdtype(data.dtype, np.integer):
########            rms = rms / float(np.iinfo(data.dtype).max)
########        if rms < 0.01:
########            print(f"[speech_id] Skipping testfile {testfile} — silence detected (rms={rms:.6f}).")
########            testfile.unlink(missing_ok=True)
########            return
########    except Exception as e:
########        print(f"[speech_id] Error validating recorded testfile {testfile}: {e}")
########        testfile.unlink(missing_ok=True)
########        return
########
########    name, score = identify_speaker_from_file(str(testfile))
########    if name:
########        print(f"🔊 Identified as: {name} (score={score:.3f})")
########    else:
########        print(f"⚠️ Unknown speaker (best score={score:.3f})")
########
######### ------------------ DB migration helper ------------------
########def migrate_normalize_db():
########    """Normalize all embeddings in DB to unit vectors (creates a .bak copy first)."""
########    if not DB_FILE.exists():
########        print(f"[speech_id] No DB file found at {DB_FILE}; nothing to migrate.")
########        return
########    backup = DB_FILE.with_suffix(DB_FILE.suffix + ".bak")
########    shutil.copy2(DB_FILE, backup)
########    print(f"[speech_id] Backed up DB to {backup}")
########
########    db = load_db()
########    changed = 0
########    for k, v in list(db.items()):
########        a = np.array(v, dtype=float)
########        n = np.linalg.norm(a)
########        if n > 0:
########            db[k] = (a / n).tolist()
########            changed += 1
########        else:
########            print(f"[speech_id] Warning: {k} has zero-norm embedding; leaving as-is.")
########    save_db(db)
########    print(f"[speech_id] Normalized {changed} embeddings in {DB_FILE}")
########
######### ------------------ CLI ------------------
########def _cli():
########    while True:
########        print()
########        print("--- Alfred: Speech Identification ---")
########        print("1. Enroll new user")
########        print("2. Update existing user")
########        print("3. Identify (record test now)")
########        print("4. Identify from WAV file")
########        print("5. List enrolled users")
########        print("6. Normalize/migrate DB (backup first)")
########        print("7. Exit")
########        choice = input("Choose: ").strip()
########
########        if choice == "1":
########            name = input("Enter new user name: ").strip()
########            samples = input("How many samples? (default=3): ").strip()
########            samples = int(samples) if samples else 3
########            enroll_user(name, samples)
########
########        elif choice == "2":
########            name = input("Enter existing user name to update: ").strip()
########            samples = input("How many new samples? (default=2): ").strip()
########            samples = int(samples) if samples else 2
########            update_user(name, samples)
########
########        elif choice == "3":
########            identify_speaker_interactive()
########
########        elif choice == "4":
########            path = input("Path to wav file: ").strip()
########            if not path:
########                print("No path given.")
########            else:
########                name, score = identify_from_file(path)
########                if name:
########                    print(f"Identified: {name} (score={score:.3f})")
########                else:
########                    print(f"Unknown (best score={score})")
########
########        elif choice == "5":
########            db = load_db()
########            if db:
########                print("Enrolled users:")
########                for u in db.keys():
########                    print(" -", u)
########            else:
########                print("(none)")
########
########        elif choice == "6":
########            confirm = input("This will backup and normalize your DB. Continue? (y/n): ").strip().lower()
########            if confirm == 'y':
########                migrate_normalize_db()
########            else:
########                print("Migration canceled.")
########
########        elif choice == "7":
########            break
########
########        else:
########            print("Invalid option — choose again.")
########
######### Run CLI if module executed directly
########if __name__ == "__main__":
########    print("Speech ID module starting. Model loaded:", verification is not None)
########    try:
########        _cli()
########    except KeyboardInterrupt:
########        print("\nExiting.")
########







##"""
##Speech_Identification_User.py
##Robust speaker identification utilities using SpeechBrain (ecapa-voxceleb).
##- Uses LocalStrategy.COPY to avoid Windows symlink issues.
##- Records audio via sounddevice, saves 16kHz mono int16 WAVs.
##- Computes normalized embeddings using SpeechBrain SpeakerRecognition.
##- Stores normalized embeddings in a JSON DB (atomic saves).
##- Enrollment / update / interactive identification CLI included.
##"""
##
### --- IMPORTANT: environment variables to avoid Windows symlink problems ---
##import os
##os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
##os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
##
### Standard libs
##import time
##import json
##from pathlib import Path
##import shutil
##import numpy as np
##import sounddevice as sd
##import scipy.io.wavfile as wav
##import torch
##
### SpeechBrain imports (new inference path)
##from speechbrain.inference import SpeakerRecognition
##from speechbrain.utils.fetching import LocalStrategy
##
### ------------------ Configuration ------------------
### Adjust these to taste
##CONFIDENCE_THRESHOLD = 0.60   # cosine similarity threshold (0.6-0.85 typical)
##SAMPLE_RATE = 16000           # target sampling rate for SpeechBrain model
##DEFAULT_RECORD_SECONDS = 5
##VOICE_SAMPLES_DIR = Path("voice_samples")
##TEST_WAV_DIR = Path("test_waves")
##VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
##TEST_WAV_DIR.mkdir(parents=True, exist_ok=True)
##ROUND_ROBIN_FILE = TEST_WAV_DIR / "last_index.txt"
##SAVEDIR = Path("pretrained_model")  # HF model cache target (LocalStrategy.COPY)
##
### DB location (Windows-friendly). If you have Alfred_config with DRIVE_LETTER, prefer that.
##try:
##    import Alfred_config
##    DRIVE = getattr(Alfred_config, "DRIVE_LETTER", "")
##    if DRIVE:
##        DB_FILE = Path(DRIVE) / "Python_Env" / "New_Virtual_Env" / "Alfred_Offline_New_GUI" / \
##                  "2025_08_30A_WEBUI_MODEL_New_TRY_OLD" / "New_V2_Home_Head_Movement_Smoothing" / \
##                  "modules" / "voice_db.json"
##    else:
##        DB_FILE = Path("voice_db.json")
##except Exception:
##    DB_FILE = Path("voice_db.json")
##
### make sure directories exist
##DB_FILE.parent.mkdir(parents=True, exist_ok=True)
##SAVEDIR.mkdir(parents=True, exist_ok=True)
##
### ------------------ Utilities ------------------
##def get_next_index(limit: int = 20) -> int:
##    """Return the next index for round-robin overwriting (1..limit)."""
##    try:
##        if ROUND_ROBIN_FILE.exists():
##            try:
##                last = int(ROUND_ROBIN_FILE.read_text().strip())
##            except Exception:
##                last = 0
##        else:
##            last = 0
##        new_index = (last % limit) + 1
##        ROUND_ROBIN_FILE.write_text(str(new_index))
##        return new_index
##    except Exception:
##        # fallback safe behavior
##        return 1
##
##def _atomic_write_text(path: Path, text: str):
##    """Write text atomically: write to tmp file then replace target."""
##    tmp = path.with_suffix(path.suffix + ".tmp")
##    tmp.write_text(text)
##    tmp.replace(path)
##
##def load_db() -> dict:
##    """Load JSON DB (name -> embedding list). Returns {} if missing/invalid."""
##    try:
##        if DB_FILE.exists():
##            text = DB_FILE.read_text()
##            if not text:
##                return {}
##            return json.loads(text)
##    except Exception as e:
##        print(f"[speech_id] Warning: failed to load DB ({e}), starting fresh.")
##    return {}
##
##def save_db(db: dict):
##    """Save JSON DB using atomic write to reduce corruption risk."""
##    try:
##        _atomic_write_text(DB_FILE, json.dumps(db))
##    except Exception as e:
##        print(f"[speech_id] Error saving DB atomically: {e}; falling back to direct write.")
##        try:
##            DB_FILE.write_text(json.dumps(db))
##        except Exception as e2:
##            print(f"[speech_id] FATAL: DB save failed: {e2}")
##
### ------------------ Model loading ------------------
##def load_speaker_model(savedir: Path = SAVEDIR) -> SpeakerRecognition:
##    """
##    Load the SpeechBrain SpeakerRecognition model with LocalStrategy.COPY (no symlinks).
##    Returns the loaded model and places it on GPU if available.
##    """
##    savedir = Path(savedir)
##    print("[speech_id] Loading SpeechBrain speaker model (LocalStrategy.COPY). This may download ~100MB.")
##    try:
##        model = SpeakerRecognition.from_hparams(
##            source="speechbrain/spkrec-ecapa-voxceleb",
##            savedir=str(savedir),
##            local_strategy=LocalStrategy.COPY,
##        )
##    except Exception as e:
##        print(f"[speech_id] Failed to load model from HF: {e}")
##        raise
##
##    # prefer GPU if available
##    device = "cuda" if torch.cuda.is_available() else "cpu"
##    # Many SpeechBrain wrappers handle device internally; try to move if possible
##    try:
##        if hasattr(model, "to"):
##            try:
##                model.to(device)
##            except Exception:
##                pass
##        # Some wrappers expose `.device` attribute; set it if present.
##        try:
##            setattr(model, "device", device)
##        except Exception:
##            pass
##    except Exception:
##        pass
##
##    print(f"[speech_id] Model loaded and placed on device: {device}")
##    return model
##
### load model (singleton)
##try:
##    verification = load_speaker_model(SAVEDIR)
##except Exception as exc:
##    print(f"[speech_id] ERROR: speaker model failed to load: {exc}")
##    verification = None
##
### ------------------ Audio utilities ------------------
##def record_audio(filename: str, duration: int = DEFAULT_RECORD_SECONDS, fs: int = SAMPLE_RATE):
##    """
##    Record audio from default microphone and save as a WAV file (mono int16).
##    Uses sounddevice; returns after the file is written.
##    """
##    print(f"🎙️ Recording {duration}s to '{filename}' — speak now...")
##    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
##    sd.wait()
##    # sounddevice returns shape (N, channels) — flatten to 1-D
##    audio_1d = np.asarray(audio).reshape(-1)
##    try:
##        wav.write(filename, fs, audio_1d)
##        print(f"✅ Saved recording to {filename}")
##    except Exception as e:
##        print(f"[speech_id] Error saving recording {filename}: {e}")
##        raise
##
##def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
##    """Resample 1-D numpy array to target_sr. Returns float32 array."""
##    if sr == target_sr:
##        return data.astype(np.float32)
##    from scipy.signal import resample
##    num = int(len(data) * target_sr / sr)
##    if num <= 1:
##        raise ValueError(f"Invalid resample length {num} (orig sr={sr}, len={len(data)})")
##    return resample(data, num).astype(np.float32)
##
##def get_embedding(filepath: str) -> np.ndarray:
##    """
##    Compute a normalized embedding for the WAV file using the loaded SpeechBrain model.
##    Returns a 1-D numpy float array (unit norm).
##    """
##    if verification is None:
##        raise RuntimeError("Speaker verification model not loaded.")
##    sr, data = wav.read(filepath)
##
##    # Ensure mono
##    if getattr(data, "ndim", 1) > 1:
##        data = data.mean(axis=1)
##
##    # Convert to float32 in range [-1,1] if integer dtypes
##    if np.issubdtype(data.dtype, np.integer):
##        max_val = float(np.iinfo(data.dtype).max)
##        data = data.astype(np.float32) / max_val
##    else:
##        data = data.astype(np.float32)
##
##    if sr != SAMPLE_RATE:
##        data = _resample_if_needed(data, sr, SAMPLE_RATE)
##
##    if data.size == 0:
##        raise ValueError("Audio file contains no data after preprocessing.")
##
##    # Create tensor shape (batch, samples)
##    tensor = torch.from_numpy(data).unsqueeze(0)
##    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##    tensor = tensor.to(device)
##
##    # compute embedding (inference mode)
##    try:
##        with torch.no_grad():
##            emb = verification.encode_batch(tensor)
##    except Exception as e:
##        # fallback to CPU if GPU fails
##        try:
##            tensor_cpu = tensor.to("cpu")
##            with torch.no_grad():
##                emb = verification.encode_batch(tensor_cpu)
##        except Exception as e2:
##            print(f"[speech_id] Embedding error: {e} / fallback: {e2}")
##            raise
##
##    # Convert to numpy safely and normalize
##    try:
##        emb = emb.squeeze().detach().cpu().numpy().astype(float)
##    except Exception:
##        emb = np.array(emb).squeeze().astype(float)
##    norm = np.linalg.norm(emb)
##    if norm > 0:
##        emb = emb / norm
##    return emb
##
### ------------------ DB / Identity helpers ------------------
##def identify_user(embedding: np.ndarray):
##    """Identify the closest enrolled speaker given a precomputed embedding."""
##    db = load_db()
##    if not db:
##        print("⚠️ Voice DB empty — enroll users first.")
##        return None, None
##
##    test_emb = np.array(embedding, dtype=float)
##    if test_emb.size == 0:
##        return None, None
##    test_norm = np.linalg.norm(test_emb)
##    if test_norm == 0:
##        return None, None
##    test_emb = test_emb / test_norm
##
##    best_score = -1.0
##    best_name = None
##    for name, emb in db.items():
##        emb = np.array(emb, dtype=float)
##        if emb.size == 0:
##            continue
##        emb_norm = np.linalg.norm(emb)
##        if emb_norm == 0:
##            continue
##        emb = emb / emb_norm
##        score = float(np.dot(test_emb, emb))
##        if score > best_score:
##            best_score = score
##            best_name = name
##
##    if best_score < CONFIDENCE_THRESHOLD:
##        return None, best_score
##    return best_name, best_score
##
##def identify_from_file(wav_path: str, limit: int = 20):
##    """
##    Copy wav into test_waves (round-robin) and identify speaker, but only if audio is valid (non-silent).
##    Returns (name, score) or (None, score).
##    """
##    try:
##        sr, data = wav.read(wav_path)
##        if getattr(data, "ndim", 1) > 1:
##            data = data.mean(axis=1)
##
##        if len(data) == 0:
##            print(f"[speech_id] Skipping {wav_path} — empty file.")
##            return None, None
##
##        # compute normalized RMS for silence check
##        rms = np.sqrt(np.mean(np.square(data.astype(float))))
##        if np.issubdtype(data.dtype, np.integer):
##            rms = rms / float(np.iinfo(data.dtype).max)
##        # threshold: 0.01 works for normalized [-1,1] floats
##        if rms < 0.01:
##            print(f"[speech_id] Skipping {wav_path} — silence/low energy (rms={rms:.6f}).")
##            return None, None
##
##        # Passed checks → save into round-robin slot
##        idx = get_next_index(limit)
##        dest = TEST_WAV_DIR / f"test_{idx}.wav"
##        shutil.copy(wav_path, dest)
##        print(f"[speech_id] Saved test file → {dest}")
##
##        emb = get_embedding(str(dest))
##        return identify_user(emb)
##
##    except Exception as e:
##        print(f"[speech_id] identify_from_file error: {e}")
##        # fallback: try to compute embedding directly and identify
##        try:
##            emb = get_embedding(wav_path)
##            return identify_user(emb)
##        except Exception as e2:
##            print(f"[speech_id] identify_from_file fallback error: {e2}")
##            return None, None
##
### ------------------ Enrollment / Update ------------------
##def enroll_user(name: str, samples: int = 3):
##    """Enroll a new user by recording `samples` audio files and averaging embeddings."""
##    db = load_db()
##    user_dir = VOICE_SAMPLES_DIR / name
##    user_dir.mkdir(parents=True, exist_ok=True)
##
##    embeddings = []
##    for i in range(samples):
##        filename = user_dir / f"{name}_enroll_{int(time.time())}_{i+1}.wav"
##        print(f"🔴 Sample {i+1}/{samples} for {name}")
##        record_audio(str(filename))
##        try:
##            emb = get_embedding(str(filename))
##        except Exception as e:
##            print(f"[speech_id] Warning: embedding for sample {i+1} failed: {e}")
##            continue
##        embeddings.append(emb)
##
##    if not embeddings:
##        print(f"❌ No valid embeddings collected for {name}; enrollment aborted.")
##        return
##
##    avg_emb = np.mean(embeddings, axis=0)
##    norm = np.linalg.norm(avg_emb)
##    if norm > 0:
##        avg_emb = (avg_emb / norm).tolist()
##    else:
##        avg_emb = avg_emb.tolist()
##
##    db[name] = avg_emb
##    save_db(db)
##    print(f"✅ Enrolled {name} with {len(embeddings)} samples (DB saved to {DB_FILE})")
##
##def update_user(name: str, new_samples: int = 2):
##    """Update an existing user's embedding by adding new samples and averaging."""
##    db = load_db()
##    if name not in db:
##        print(f"⚠️ User '{name}' not found in DB — call enroll instead.")
##        return
##
##    user_dir = VOICE_SAMPLES_DIR / name
##    user_dir.mkdir(parents=True, exist_ok=True)
##
##    try:
##        existing_emb = np.array(db[name], dtype=float)
##    except Exception:
##        existing_emb = None
##
##    embeddings = []
##    if existing_emb is not None and existing_emb.size > 0:
##        embeddings.append(existing_emb)
##
##    for i in range(new_samples):
##        filename = user_dir / f"{name}_update_{int(time.time())}_{i+1}.wav"
##        print(f"🔁 New sample {i+1}/{new_samples} for {name}")
##        record_audio(str(filename))
##        try:
##            emb = get_embedding(str(filename))
##        except Exception as e:
##            print(f"[speech_id] Warning: embedding for update sample {i+1} failed: {e}")
##            continue
##        embeddings.append(emb)
##
##    if not embeddings:
##        print(f"❌ No valid embeddings collected for {name}; update aborted.")
##        return
##
##    avg_emb = np.mean(embeddings, axis=0)
##    norm = np.linalg.norm(avg_emb)
##    if norm > 0:
##        avg_emb = (avg_emb / norm).tolist()
##    else:
##        avg_emb = avg_emb.tolist()
##
##    db[name] = avg_emb
##    save_db(db)
##    print(f"✅ Updated {name} with {len(embeddings) - (1 if existing_emb is not None else 0)} new samples (DB saved)")
##
### ------------------ Convenience wrappers ------------------
##def identify_speaker_from_file(testfile: str):
##    """Identify a speaker from a WAV file path; returns (name, score)."""
##    db = load_db()
##    if not db:
##        print("⚠️ Voice DB empty — enroll users first.")
##        return None, None
##    try:
##        test_emb = get_embedding(testfile)
##    except Exception as e:
##        print(f"[speech_id] Failed to compute embedding for test file: {e}")
##        return None, None
##    return identify_user(test_emb)
##
##def identify_speaker_interactive(limit: int = 20):
##    """Record a test wave (round robin), run validation, and attempt identification."""
##    idx = get_next_index(limit)
##    testfile = TEST_WAV_DIR / f"test_{idx}.wav"
##    record_audio(str(testfile))
##
##    try:
##        sr, data = wav.read(str(testfile))
##        if getattr(data, "ndim", 1) > 1:
##            data = data.mean(axis=1)
##        if len(data) == 0:
##            print(f"[speech_id] Skipping testfile {testfile} — empty recording.")
##            testfile.unlink(missing_ok=True)
##            return
##        rms = np.sqrt(np.mean(np.square(data.astype(float))))
##        if np.issubdtype(data.dtype, np.integer):
##            rms = rms / float(np.iinfo(data.dtype).max)
##        if rms < 0.01:
##            print(f"[speech_id] Skipping testfile {testfile} — silence detected (rms={rms:.6f}).")
##            testfile.unlink(missing_ok=True)
##            return
##    except Exception as e:
##        print(f"[speech_id] Error validating recorded testfile {testfile}: {e}")
##        testfile.unlink(missing_ok=True)
##        return
##
##    name, score = identify_speaker_from_file(str(testfile))
##    if name:
##        print(f"🔊 Identified as: {name} (score={score:.3f})")
##    else:
##        print(f"⚠️ Unknown speaker (best score={score:.3f})")
##
### ------------------ DB migration helper ------------------
##def migrate_normalize_db():
##    """Normalize all embeddings in DB to unit vectors (creates a .bak copy first)."""
##    if not DB_FILE.exists():
##        print(f"[speech_id] No DB file found at {DB_FILE}; nothing to migrate.")
##        return
##    backup = DB_FILE.with_suffix(DB_FILE.suffix + ".bak")
##    shutil.copy(DB_FILE, backup)
##    print(f"[speech_id] Backed up DB to {backup}")
##
##    db = load_db()
##    changed = 0
##    for k, v in list(db.items()):
##        a = np.array(v, dtype=float)
##        n = np.linalg.norm(a)
##        if n > 0:
##            db[k] = (a / n).tolist()
##            changed += 1
##        else:
##            print(f"[speech_id] Warning: {k} has zero-norm embedding; leaving as-is.")
##    save_db(db)
##    print(f"[speech_id] Normalized {changed} embeddings in {DB_FILE}")
##
### ------------------ CLI ------------------
##def _cli():
##    while True:
##        print()
##        print("--- Alfred: Speech Identification ---")
##        print("1. Enroll new user")
##        print("2. Update existing user")
##        print("3. Identify (record test now)")
##        print("4. Identify from WAV file")
##        print("5. List enrolled users")
##        print("6. Normalize/migrate DB (backup first)")
##        print("7. Exit")
##        choice = input("Choose: ").strip()
##
##        if choice == "1":
##            name = input("Enter new user name: ").strip()
##            samples = input("How many samples? (default=3): ").strip()
##            samples = int(samples) if samples else 3
##            enroll_user(name, samples)
##
##        elif choice == "2":
##            name = input("Enter existing user name to update: ").strip()
##            samples = input("How many new samples? (default=2): ").strip()
##            samples = int(samples) if samples else 2
##            update_user(name, samples)
##
##        elif choice == "3":
##            identify_speaker_interactive()
##
##        elif choice == "4":
##            path = input("Path to wav file: ").strip()
##            if not path:
##                print("No path given.")
##            else:
##                name, score = identify_from_file(path)
##                if name:
##                    print(f"Identified: {name} (score={score:.3f})")
##                else:
##                    print(f"Unknown (best score={score})")
##
##        elif choice == "5":
##            db = load_db()
##            if db:
##                print("Enrolled users:")
##                for u in db.keys():
##                    print(" -", u)
##            else:
##                print("(none)")
##
##        elif choice == "6":
##            confirm = input("This will backup and normalize your DB. Continue? (y/n): ").strip().lower()
##            if confirm == 'y':
##                migrate_normalize_db()
##            else:
##                print("Migration canceled.")
##
##        elif choice == "7":
##            break
##
##        else:
##            print("Invalid option — choose again.")
##
### Run CLI if module executed directly
##if __name__ == "__main__":
##    print("Speech ID module starting. Model loaded:", verification is not None)
##    try:
##        _cli()
##    except KeyboardInterrupt:
##        print("\nExiting.") 
##
##
##
##
