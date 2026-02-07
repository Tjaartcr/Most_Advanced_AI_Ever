# inspect_voice_db.py
import json, numpy as np
db = json.load(open("voice_db.json"))
for name, vec in db.items():
    v = np.array(vec, dtype=float)
    print(f"{name}: shape={v.shape}, norm={np.linalg.norm(v):.6f}")
