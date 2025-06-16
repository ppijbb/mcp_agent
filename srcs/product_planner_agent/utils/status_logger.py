import json
from pathlib import Path
from typing import List, Dict, Any, Literal

STATUS_FILE = Path(__file__).parent.parent / "status.json"

Status = Literal["pending", "in_progress", "completed", "failed"]

class StatusLogger:
    def __init__(self, steps: List[str]):
        self.steps = steps
        self.statuses: Dict[str, Status] = {step: "pending" for step in self.steps}
        self._initialize_status_file()

    def _initialize_status_file(self):
        """Initializes the status file with all steps pending."""
        self.statuses = {step: "pending" for step in self.steps}
        self._write_status()

    def _write_status(self):
        """Writes the current statuses to the JSON file."""
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.statuses, f, ensure_ascii=False, indent=4)

    def update_status(self, step: str, status: Status):
        """Updates the status of a specific step and writes to the file."""
        if step in self.statuses:
            self.statuses[step] = status
            self._write_status()
        else:
            print(f"Warning: Step '{step}' not found in status logger.")

    def get_status(self) -> Dict[str, Status]:
        """Returns the current status of all steps."""
        return self.statuses

    @staticmethod
    def read_status() -> Dict[str, Status]:
        """Reads the status from the file."""
        if not STATUS_FILE.exists():
            return {}
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def complete_all(self):
        """Marks all steps as completed."""
        for step in self.steps:
            if self.statuses[step] == "pending" or self.statuses[step] == "in_progress":
                self.statuses[step] = "completed"
        self._write_status() 