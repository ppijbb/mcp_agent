from autogen import ConversableAgent
from typing import Dict, Any

class CodeWriterAgent(ConversableAgent):
    def __init__(self, name="CodeWriter", **kwargs):
        super().__init__(
            name=name,
            system_message="""You are a CodeWriter. You are an expert Python programmer.
Your role is to receive a coding task with specific requirements and write clean, efficient, and correct Python code.
You will be given the context or existing code by other agents.
You should only output the code block in markdown format. Do not add any explanation.
""",
            **kwargs,
        )

    def run(self, message: str) -> Dict[str, Any]:
        # This agent will typically receive a message containing the code to write or modify.
        # Its reply will be the generated code.
        return self.generate_reply(messages=[{"role": "user", "content": message}]) 