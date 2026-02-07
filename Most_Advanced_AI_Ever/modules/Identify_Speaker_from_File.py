def identify_speaker_from_file(testfile: str):
    db = load_db()
    if not db:
        print("⚠️ Voice DB empty — enroll users first.")
        return None, None

    test_emb = get_embedding(testfile)
    best_score = -1.0
    best_name = None

    for name, emb in db.items():
        emb = np.array(emb)
        score = float(np.dot(test_emb, emb) / (np.linalg.norm(test_emb) * np.linalg.norm(emb)))
        print(f"DEBUG: {name} similarity = {score:.4f}")  
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < CONFIDENCE_THRESHOLD:
        print(f"DEBUG: best_score {best_score:.4f} < threshold {CONFIDENCE_THRESHOLD}")
        return None, best_score
    return best_name, best_score
