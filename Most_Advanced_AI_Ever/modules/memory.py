
# memory.py


import json
import os
import time
from typing import Any, Dict, List, Optional



MEMORY_FILE = "memory.json"

class MemoryManager:
    def __init__(self):
        """Initialize memory and load existing history."""
        self.memory = []
        self.load_memory()

    def load_memory(self):
        """Load memory from a JSON file."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as file:
                    self.memory = json.load(file)
            except json.JSONDecodeError:
                print("❌ Error loading memory file. Resetting memory.")
                self.memory = []
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



    def save_memory(self):
        """Save current memory to a JSON file."""
        with open(MEMORY_FILE, "w") as file:
            json.dump(self.memory, file, indent=4)

    def add_to_memory(self, entry):
        """Add a conversation entry to memory."""
        self.memory.insert(0, entry)  # Latest query on top
        if len(self.memory) > 100:  # Limit memory size
            self.memory.pop()  # Remove oldest
        self.save_memory()

    def clear_memory(self):
        """Clear all stored memory."""
        self.memory = []
        self.save_memory()

    def get_history(self):
        """Return the conversation history."""
        return self.memory

# ✅ Create a memory instance
memory = MemoryManager()









