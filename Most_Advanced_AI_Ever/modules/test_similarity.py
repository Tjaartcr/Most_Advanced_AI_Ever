# test_similarity.py
import json, numpy as np
from AAAAA_Speech_Identification_User import get_embedding, load_db, CONFIDENCE_THRESHOLD

# path to your test wav
test_wav = "test_waves/test_1756597208.wav"   # replace with actual path

test_emb = get_embedding(test_wav)      # this will be normalized if you apply patch below
if test_emb is None:
    raise SystemExit("get_embedding failed")

db = load_db()
if not db:
    print("Voice DB empty")
    raise SystemExit

scores = []
for name, vec in db.items():
    emb = np.array(vec, dtype=float)
    # if not normalized, normalize here:
    if abs(np.linalg.norm(emb) - 1.0) > 1e-6:
        emb = emb / np.linalg.norm(emb)
    sim = float(np.dot(test_emb, emb))  # dot==cosine when both unit-norm
    scores.append((name, sim))

scores.sort(key=lambda x: x[1], reverse=True)
print("Top matches:")
for n, s in scores[:5]:
    print(f" - {n}: {s:.4f}")

best_name, best_score = scores[0]
print("Best:", best_name, best_score)
print("Threshold:", CONFIDENCE_THRESHOLD)
print("Match?" , best_score >= CONFIDENCE_THRESHOLD)
