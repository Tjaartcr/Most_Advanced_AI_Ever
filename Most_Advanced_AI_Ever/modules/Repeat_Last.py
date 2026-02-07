# Repeat_Last.py
# Repeat_Last.py



import json
import os
import time


REPEAT_FILE = "Last_Repeat.json"

class RepeatLast:
    def __init__(self):
        """Initialize repeat storage and load the last stored response."""
        self.repeat = []
        self.load_last_repeat()

    def load_last_repeat(self):
        """Load the last repeated response from a JSON file."""
        if os.path.exists(REPEAT_FILE):
            try:
                with open(REPEAT_FILE, "r") as file:
                    self.repeat = json.load(file)
            except json.JSONDecodeError:
                print("❌ Error loading repeat file. Resetting repeat.")
                self.repeat = []
                self.save_last()

    def save_last(self):
        """Save the latest repeated response to the JSON file."""
        with open(REPEAT_FILE, "w") as file:
            json.dump(self.repeat, file, indent=4)

    def add_to_repeat(self, entry):
        """Save the latest query-response pair for repetition."""
        self.repeat = [entry]  # Always keep only the last entry
        self.save_last()

    def clear_repeat(self):
        """Clear the stored last repeated response."""
        self.repeat = []
        self.save_last()

    def get_last(self):
        """Return the last stored response."""
        return self.repeat[0] if self.repeat else None

# ✅ Create a global instance
repeat = RepeatLast()






