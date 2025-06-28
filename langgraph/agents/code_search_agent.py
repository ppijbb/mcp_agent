from autogen import ConversableAgent
from typing import Dict, Any
import os

class CodeSearchAgent(ConversableAgent):
    def __init__(self, name="CodeSearcher", work_dir=".", **kwargs):
        super().__init__(
            name=name,
            system_message="""You are a CodeSearcher. Your role is to search for files and code snippets in the given directory.
You will be given a file name or a function/class name to search for.
You should return the full content of the file or the relevant code block.
You have access to the file system.
""",
            **kwargs,
        )
        self.work_dir = work_dir

    def search_file(self, filename: str) -> str:
        filepath = os.path.join(self.work_dir, filename)
        try:
            with open(filepath, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{filename}' not found in '{self.work_dir}'."

    def run(self, message: str) -> Dict[str, Any]:
        # A simple implementation: assumes the message is the filename to search for.
        content = self.search_file(message)
        return {"content": content} # Returning a dict to be compatible with other agents, though not directly conversable 