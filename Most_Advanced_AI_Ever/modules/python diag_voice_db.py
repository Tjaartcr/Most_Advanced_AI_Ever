# diag_voice_db.py
import os, json, sys
from pathlib import Path

print("=== Diagnostic: voice DB ===")
print("Current working directory:", os.getcwd())
print("Python executable:", sys.executable)

# Try to inspect voice_db.json in cwd
cwd_file = Path("voice_db.json")
print("\nCWD voice_db.json exists?:", cwd_file.exists(), " ->", cwd_file.resolve() if cwd_file.exists() else "")

# Try to import your module and inspect DB_FILE if present
try:
    import importlib
    m = importlib.import_module("AAAAA_Speech_Identification_User")
    DB_FILE = getattr(m, "DB_FILE", None)
    print("\nAAAAA_Speech_Identification_User.DB_FILE:", DB_FILE)
    if DB_FILE:
        p = Path(str(DB_FILE))
        print("Resolved DB_FILE absolute:", p.resolve())
        print("Exists?:", p.exists())
        try:
            text = p.read_text()
            print("DB file size (bytes):", len(text))
            try:
                db = json.loads(text)
                print("Loaded DB keys:", list(db.keys()))
            except Exception as e:
                print("Failed to parse DB JSON:", e)
        except Exception as e:
            print("Could not read DB_FILE:", e)
except Exception as e:
    print("Could not import AAAAA_Speech_Identification_User:", e)

# If voice_db.json isn't in cwd, list files for debugging
print("\nListing files in cwd for context (top 50):")
for i, f in enumerate(os.listdir(".")):
    if i >= 50: break
    print(" -", f)
