from autogen import ConversableAgent
from typing import Dict, Any

class PlannerAgent(ConversableAgent):
    def __init__(self, name="Planner", **kwargs):
        super().__init__(
            name=name,
            system_message="""You are a Planner. Your role is to receive a complex task, break it down into a sequence of actionable steps, and coordinate the execution of these steps by other agents.
At each step, clearly state the goal and nominate the next agent to perform the task.
Your output should be a clear, numbered plan.
""",
            **kwargs,
        )

    def run(self, message: str) -> Dict[str, Any]:
        # This is a simplified run method.
        # In a real scenario, this would involve more complex logic to generate a plan.
        return self.generate_reply(messages=[{"role": "user", "content": message}]) 