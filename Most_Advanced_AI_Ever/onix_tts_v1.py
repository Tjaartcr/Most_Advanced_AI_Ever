


import os
import subprocess
import sys
import tempfile

# =========================
# CONFIGURATION
# =========================

VOICE_DIR = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Alfred_Offline_Venv\AlfredOffline\voices"
ONNX_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx")
CONFIG_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx.json")
OUTPUT_WAV = os.path.join(os.getcwd(), "hello_af.wav")
TEXT_TO_SPEAK = "Hallo, hoe gaan dit vandag met jou my seun , Sebastiaan en dogters Selena en Dalinya?"
DEVICE = "cpu"
VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

# =========================
# CHECK FILES
# =========================
for f in [ONNX_FILE, CONFIG_FILE]:
    if not os.path.isfile(f):
        print(f"Error: file not found: {f}")
        sys.exit(1)

if not os.path.isfile(VLC_PATH):
    print(f"Error: VLC not found at {VLC_PATH}")
    sys.exit(1)

# =========================
# WRITE TEMP TEXT FILE
# =========================
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
    tmp_file.write(TEXT_TO_SPEAK)
    tmp_filename = tmp_file.name

# =========================
# SYNTHESIZE USING PIPER
# =========================
print("Synthesizing...")
cmd_synthesize = [
    sys.executable, "-m", "piper",
    "-m", ONNX_FILE,
    "-c", CONFIG_FILE,
    "-i", tmp_filename,
    "-f", OUTPUT_WAV,
    "--device", DEVICE
]

try:
    subprocess.run(cmd_synthesize, check=True)
finally:
    os.remove(tmp_filename)  # clean up temp file

# =========================
# PLAY AUDIO VIA VLC
# =========================
print("Playing audio via VLC...")
subprocess.run([VLC_PATH, "--play-and-exit", OUTPUT_WAV])




##      # STILL NOT BAD
##
##import os
##import subprocess
##import sys
##
### =========================
### CONFIGURATION
### =========================
##
### Paths to your local Piper voice files
##VOICE_DIR = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Alfred_Offline_Venv\AlfredOffline\voices"
##ONNX_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx")
##CONFIG_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx.json")
##
### Output WAV file
##OUTPUT_WAV = os.path.join(os.getcwd(), "hello_af.wav")
##
### Text to synthesize
##TEXT_TO_SPEAK = "Hallo, hoe gaan dit vandag met jou?"
##
### Device (CPU)
##DEVICE = "cpu"
##
### Path to VLC executable (adjust if needed)
##VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
##
### =========================
### CHECK FILES
### =========================
##for f in [ONNX_FILE, CONFIG_FILE]:
##    if not os.path.isfile(f):
##        print(f"Error: file not found: {f}")
##        sys.exit(1)
##
##if not os.path.isfile(VLC_PATH):
##    print(f"Error: VLC not found at {VLC_PATH}")
##    sys.exit(1)
##
### =========================
### SYNTHESIZE USING PIPER
### =========================
##print("Synthesizing...")
##cmd_synthesize = [
##    sys.executable, "-m", "piper",
##    "-m", ONNX_FILE,
##    "-c", CONFIG_FILE,
##    "-f", OUTPUT_WAV,
##    "--text", TEXT_TO_SPEAK,
##    "--device", DEVICE
##]
##
##try:
##    subprocess.run(cmd_synthesize, check=True)
##except subprocess.CalledProcessError as e:
##    print("Error during synthesis:", e)
##    sys.exit(1)
##
### =========================
### PLAY AUDIO VIA VLC
### =========================
##print("Playing audio via VLC...")
##subprocess.run([VLC_PATH, "--play-and-exit", OUTPUT_WAV])




##      # Best so far
##
##import os
##import sys
##import subprocess
##import urllib.request
##
### ---------- CONFIG ----------
##
### Text to speak
##TEXT = "Hallo, hoe gaan dit vandag met jou my seun, Sebastiaan?"
##
### Paths
##VOICE_DIR = os.path.join(os.getcwd(), "voices")
##OUTPUT_WAV = os.path.join(os.getcwd(), "hello_af.wav")
##
### Example small voice (you can switch to af_ZA if available)
##ONNX_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx")
##CONFIG_FILE = os.path.join(VOICE_DIR, "nl_NL-ronnie-medium.onnx.json")
##
### URLs for downloading the voice if missing
##ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium.onnx"
##CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_NL/ronnie/medium/nl_NL-ronnie-medium.onnx.json"
##
### Path to VLC executable
##VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"  # adjust if your path differs
##
### ----------------------------
##
### Ensure voices folder exists
##os.makedirs(VOICE_DIR, exist_ok=True)
##
### Download ONNX if missing
##if not os.path.isfile(ONNX_FILE):
##    print("Downloading ONNX model...")
##    urllib.request.urlretrieve(ONNX_URL, ONNX_FILE)
##
### Download config if missing
##if not os.path.isfile(CONFIG_FILE):
##    print("Downloading config...")
##    urllib.request.urlretrieve(CONFIG_URL, CONFIG_FILE)
##
### Synthesize speech using Piper
##print("Synthesizing...")
##cmd_synthesize = [
##    sys.executable, "-m", "piper",
##    "-m", ONNX_FILE,
##    "-c", CONFIG_FILE,
##    "-f", OUTPUT_WAV,
##    "--text", TEXT,
##    "--device", "cpu"
##]
##
##try:
##    subprocess.run(cmd_synthesize, check=True)
##    print(f"Saved synthesized speech to {OUTPUT_WAV}")
##except subprocess.CalledProcessError as e:
##    print("Error during synthesis:", e)
##    sys.exit(1)
##
### Play the WAV via VLC
##if os.path.isfile(VLC_PATH):
##    print("Playing audio with VLC...")
##    subprocess.run([VLC_PATH, "--play-and-exit", OUTPUT_WAV])
##else:
##    print("VLC not found. Falling back to playsound...")
##    try:
##        from playsound import playsound
##        playsound(OUTPUT_WAV)
##    except ImportError:
##        print("Install playsound via 'pip install playsound' to play audio.")



##import os
##import urllib.request
##import subprocess
##
### ------------------------------
### Configuration
### ------------------------------
##
### Piper Dutch voice (runs on CPU, small)
##VOICE_NAME = "nl_NL-ronnie-medium"
##VOICE_URL_BASE = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/nl/nl_NL/ronnie/medium/"
##
##VOICE_DIR = os.path.join(os.getcwd(), "voices")
##os.makedirs(VOICE_DIR, exist_ok=True)
##
##ONNX_FILE = os.path.join(VOICE_DIR, f"{VOICE_NAME}.onnx")
##CONFIG_FILE = os.path.join(VOICE_DIR, f"{VOICE_NAME}.onnx.json")
##
### Output WAV
##OUTPUT_WAV = os.path.join(os.getcwd(), "hello_af.wav")
##
### Text to synthesize
##TEXT = "Hallo, hoe gaan dit vandag met jou?"
##
### VLC path (adjust if different)
##VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
##
### ------------------------------
### Download voice if missing
### ------------------------------
##
##if not os.path.exists(ONNX_FILE):
##    print("Downloading ONNX model...")
##    urllib.request.urlretrieve(VOICE_URL_BASE + f"{VOICE_NAME}.onnx", ONNX_FILE)
##
##if not os.path.exists(CONFIG_FILE):
##    print("Downloading voice config...")
##    urllib.request.urlretrieve(VOICE_URL_BASE + f"{VOICE_NAME}.onnx.json", CONFIG_FILE)
##
### ------------------------------
### Synthesize text
### ------------------------------
##
##cmd_synthesize = [
##    "python", "-m", "piper",
##    "-m", ONNX_FILE,
##    "-c", CONFIG_FILE,
##    "-f", OUTPUT_WAV,
##    "--text", TEXT,
##    "--device", "cpu"
##]
##
##print("Synthesizing...")
##subprocess.run(cmd_synthesize, check=True)
##print(f"Saved speech to {OUTPUT_WAV}")
##
### ------------------------------
### Play with VLC
### ------------------------------
##
##print("Playing audio...")
##subprocess.run([VLC_PATH, "--play-and-exit", OUTPUT_WAV])




######import os
######import urllib.request
######
####### -----------------------------
####### 1️⃣ Download Afrikaans ONNX voice (once)
####### -----------------------------
######voice_dir = r"D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Alfred_Offline_Venv\AlfredOffline\voices"
######os.makedirs(voice_dir, exist_ok=True)
######
######onnx_file = os.path.join(voice_dir, "af_ZA_small.onnx")
######json_file = os.path.join(voice_dir, "af_ZA_small.onnx.json")
######
####### Replace these URLs with a real small Afrikaans model if you have one
######onnx_url = "https://huggingface.co/your-afrikaans-onnx/resolve/main/af_ZA_small.onnx"
######json_url = "https://huggingface.co/your-afrikaans-onnx/resolve/main/af_ZA_small.onnx.json"
######
######if not os.path.exists(onnx_file):
######    print("Downloading ONNX model...")
######    urllib.request.urlretrieve(onnx_url, onnx_file)
######if not os.path.exists(json_file):
######    print("Downloading config file...")
######    urllib.request.urlretrieve(json_url, json_file)
######
####### -----------------------------
####### 2️⃣ Generate TTS with Piper
####### -----------------------------
######text_to_speak = "Hallo, hoe gaan dit vandag met jou?"
######output_wav = os.path.join(voice_dir, "hello_af.wav")
######
######cmd = f'python -m piper -m "{onnx_file}" -c "{json_file}" -f "{output_wav}" --text "{text_to_speak}" --device cpu'
######print("Generating speech...")
######os.system(cmd)
######
####### -----------------------------
####### 3️⃣ Play WAV with VLC
####### -----------------------------
######vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"  # adjust if needed
######play_cmd = f'"{vlc_path}" --play-and-exit "{output_wav}"'
######os.system(play_cmd)
######
######print("Done!")
