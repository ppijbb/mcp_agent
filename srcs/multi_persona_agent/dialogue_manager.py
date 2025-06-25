"""
Manages the dialogue flow between different personas.
"""
import asyncio
from typing import List, Dict, Any, Tuple

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams

class DialogueTurn:
    """Represents a single turn in the dialogue."""
    def __init__(self, persona_name: str, content: str):
        self.persona_name = persona_name
        self.content = content
        
    def __str__(self):
        return f"**{self.persona_name}**: {self.content}"

class DialogueManager:
    """Orchestrates the conversation between personas."""
    
    def __init__(self, topic: str, personas: List[Agent]):
        self.topic = topic
        self.personas = {p.name: p for p in personas}
        self.history: List[DialogueTurn] = []

    async def run_dialogue_round(self) -> DialogueTurn:
        """Runs a single round of dialogue where one persona speaks."""
        
        if not self.history:
            next_persona_name = "Advocate"
        else:
            persona_sequence = ["Critic", "Skeptic"]
            last_speaker_index = -1 
            
            if self.history[-1].persona_name in persona_sequence:
                last_speaker_index = persona_sequence.index(self.history[-1].persona_name)

            if last_speaker_index != -1 and last_speaker_index < len(persona_sequence) - 1:
                 next_persona_name = persona_sequence[last_speaker_index + 1]
            else:
                 next_persona_name = "Advocate"

        persona_agent = self.personas[next_persona_name]
        
        prompt = self._construct_prompt(persona_agent.name)
        
        # Directly call the agent's LLM. The agent must be running within an MCPApp context.
        # Combine instruction and prompt into a single message.
        full_prompt = f"{persona_agent.instruction}\n\n{prompt}"
        
        response_content = await persona_agent.llm.generate_str(
            message=full_prompt,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.7),
        )
        
        new_turn = DialogueTurn(persona_agent.name, response_content)
        self.history.append(new_turn)
        return new_turn

    def _construct_prompt(self, current_speaker: str) -> str:
        """Constructs the prompt for the current speaker."""
        
        history_str = "\n".join(str(turn) for turn in self.history)
        
        prompt = f"""
        **Discussion Topic**: {self.topic}
        **Dialogue History**:
        {history_str}
        **Your Turn**:
        You are the **{current_speaker}**. Based on the dialogue so far, provide your contribution.
        Follow your persona's instructions strictly. Your response should be concise and impactful.
        """
        return prompt.strip()

    async def get_summary(self) -> str:
        """Asks the Synthesizer to summarize the discussion."""
        synthesizer = self.personas.get("Synthesizer")
        if not synthesizer:
            return "Error: Synthesizer persona not found."
            
        history_str = "\n".join(str(turn) for turn in self.history)
        
        prompt = f"""
        **Discussion Topic**: {self.topic}
        **Complete Dialogue History**:
        {history_str}
        **Your Task**:
        As the Synthesizer, create a balanced, nuanced, and comprehensive summary.
        - Acknowledge the valid points from all sides (Advocate, Critic, Skeptic).
        - Resolve contradictions where possible, or present them as open questions.
        - Identify the key takeaways and the final, integrated perspective.
        - The summary should be objective and reflect the collective intelligence of the group.
        """
        
        full_summary_prompt = f"{synthesizer.instruction}\n\n{prompt}"
        summary = await synthesizer.llm.generate_str(
            message=full_summary_prompt,
            request_params=RequestParams(model="gemini-2.5-pro-preview-06-17", temperature=0.5),
        )
        return summary

    async def get_meta_commentary(self) -> str:
        """Asks the Meta-Observer to comment on the process."""
        observer = self.personas.get("MetaObserver")
        if not observer:
            return ""
            
        history_str = "\n".join(str(turn) for turn in self.history)
        prompt = f"Observing the following dialogue about '{self.topic}':\n{history_str}\n\nProvide your meta-level analysis of the conversation process itself."
        
        full_meta_prompt = f"{observer.instruction}\n\n{prompt}"
        commentary = await observer.llm.generate_str(
            message=full_meta_prompt,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-17", temperature=0.3),
        )
        return commentary 