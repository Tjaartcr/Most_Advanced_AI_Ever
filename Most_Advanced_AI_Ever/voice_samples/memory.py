
# memory.py
"""
MemoryManager: simple persistent conversation memory.

Features:
- load/save to memory.json located next to this module (avoids cwd issues)
- add_to_memory(entry) for raw entries
- store_speaker_context(speaker, text, score=None, meta=None) for speaker-aware entries
- get_history(), get_by_speaker(), clear_memory()
- thread-safe save/load using a simple threading.Lock
"""

import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

# Ensure memory file lives next to this module (avoids cwd/import mismatches)
MEMORY_FILE: Path = Path(__file__).resolve().parent / "memory.json"
_MAX_MEMORY = 100  # keep latest N entries

_lock = threading.Lock()

class MemoryManager:
    def __init__(self, file_path: Path = MEMORY_FILE, max_entries: int = _MAX_MEMORY):
        self.file_path = Path(file_path)
        self.max_entries = int(max_entries)
        self.memory: List[Dict[str, Any]] = []
        # ensure file and folder exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_memory()

    def load_memory(self) -> None:
        """Load memory from disk into self.memory (thread-safe)."""
        with _lock:
            if self.file_path.exists():
                try:
                    data = json.loads(self.file_path.read_text(encoding="utf-8") or "[]")
                    if isinstance(data, list):
                        self.memory = data
                    else:
                        # migrate non-list format to empty list
                        self.memory = []
                except (json.JSONDecodeError, OSError):
                    # corrupted file — reset memory
                    print("❌ Error loading memory file. Resetting memory.")
                    self.memory = []
                    self.save_memory()
            else:
                # No file yet — start empty
                self.memory = []
                self.save_memory()

    def save_memory(self) -> None:
        """Persist current memory to disk (thread-safe)."""
        with _lock:
            try:
                # write atomically: write to temp then replace
                tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
                tmp.write_text(json.dumps(self.memory, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.file_path)
            except Exception as e:
                # fallback: try direct write
                try:
                    self.file_path.write_text(json.dumps(self.memory, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as e2:
                    print(f"[memory] Failed to save memory: {e2}")

    def add_to_memory(self, entry: Dict[str, Any]) -> None:
        """Add a raw dictionary entry to memory (latest first)."""
        if not isinstance(entry, dict):
            raise TypeError("entry must be a dict")
        with _lock:
            # insert at beginning (latest first)
            self.memory.insert(0, entry)
            # trim older entries
            if len(self.memory) > self.max_entries:
                self.memory = self.memory[: self.max_entries]
            self.save_memory()

    def store_speaker_context(
        self,
        speaker: Optional[str],
        text: str,
        score: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience: store a speaker-text pair with timestamp.
        Example entry:
        {
          "timestamp": 1693497600.123,
          "iso": "2025-08-31T03:00:00Z",
          "speaker": "Tjaart",
          "score": 0.82,
          "text": "what is your name",
          "meta": { ... }    # optional additional info
        }
        """
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
        entry = {
            "timestamp": ts,
            "iso": iso,
            "speaker": speaker,
            "score": float(score) if score is not None else None,
            "text": str(text) if text is not None else "",
            "meta": meta or {},
        }
        self.add_to_memory(entry)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the memory list (latest first). Optionally limit the number returned."""
        with _lock:
            if limit is None:
                return list(self.memory)
            return list(self.memory[: int(limit)])

    def get_by_speaker(self, speaker_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return up to `limit` recent entries matching speaker_name (case-insensitive)."""
        with _lock:
            results = [e for e in self.memory if e.get("speaker") and str(e.get("speaker")).lower() == str(speaker_name).lower()]
            return results[:int(limit)]

    def clear_memory(self) -> None:
        """Clear memory and persist empty file."""
        with _lock:
            self.memory = []
            self.save_memory()

# Create singleton instance used by other modules
memory = MemoryManager()




### memory.py
##
##
##import json
##import os
##import time
##
##
##
##MEMORY_FILE = "memory.json"
##
##class MemoryManager:
##    def __init__(self):
##        """Initialize memory and load existing history."""
##        self.memory = []
##        self.load_memory()
##
##    def load_memory(self):
##        """Load memory from a JSON file."""
##        if os.path.exists(MEMORY_FILE):
##            try:
##                with open(MEMORY_FILE, "r") as file:
##                    self.memory = json.load(file)
##            except json.JSONDecodeError:
##                print("❌ Error loading memory file. Resetting memory.")
##                self.memory = []
##                self.save_memory()
##
##    def save_memory(self):
##        """Save current memory to a JSON file."""
##        with open(MEMORY_FILE, "w") as file:
##            json.dump(self.memory, file, indent=4)
##
##    def add_to_memory(self, entry):
##        """Add a conversation entry to memory."""
##        self.memory.insert(0, entry)  # Latest query on top
##        if len(self.memory) > 100:  # Limit memory size
##            self.memory.pop()  # Remove oldest
##        self.save_memory()
##
##    def clear_memory(self):
##        """Clear all stored memory."""
##        self.memory = []
##        self.save_memory()
##
##    def get_history(self):
##        """Return the conversation history."""
##        return self.memory
##
### ✅ Create a memory instance
##memory = MemoryManager()
##








