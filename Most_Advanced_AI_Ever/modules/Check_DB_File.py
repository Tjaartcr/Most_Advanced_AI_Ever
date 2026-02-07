import json
from pathlib import Path

db_file = Path("voice_db.json")
db = json.loads(db_file.read_text())
print(db.keys())       # should print: dict_keys(['Tjaart'])
print(len(db["Tjaart"]))  # should print: 192 (embedding size)
