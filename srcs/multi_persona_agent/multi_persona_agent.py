"""
Main entry point for the Multi-Persona Dialogue Agent system.
"""
import asyncio
from typing import List, Dict, Any

from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

from .personas import PERSONA_BUILDERS
from .dialogue_manager import DialogueManager

class MultiPersonaDialogueAgent:
    """
    An agent system that uses a dialogue between multiple personas 
    to explore a topic from different angles and produce a comprehensive result.
    """
    def __init__(self, app: MCPApp = None):
        self.app = app or MCPApp(
            name="multi_persona_dialogue_agent",
            settings=get_settings("configs/mcp_agent.config.yaml"),
        )
        self.all_personas = [builder() for builder in PERSONA_BUILDERS.values()]

    async def run_dialogue(self, topic: str, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Conducts a multi-persona dialogue on a given topic.
        """
        async with self.app.run() as app_context:
            logger = app_context.logger
            logger.info(f"Starting multi-persona dialogue on topic: '{topic}'")

            # Attach the app's LLM to each persona agent
            llm = GoogleAugmentedLLM(app=self.app)
            for persona in self.all_personas:
                persona.llm = llm
            
            dialogue_manager = DialogueManager(
                topic=topic,
                personas=self.all_personas,
            )

            # Run Dialogue Rounds
            for i in range(max_rounds * 3):
                logger.info(f"--- Dialogue Turn {i+1} ---")
                turn = await dialogue_manager.run_dialogue_round()
                print(f"{turn}\n")
            
            logger.info("--- Dialogue Concluded ---")

            # Get Meta-Observer Commentary
            meta_commentary = await dialogue_manager.get_meta_commentary()
            if meta_commentary:
                logger.info(f"Meta-Observer's Commentary:\n{meta_commentary}")

            # Get Final Synthesized Summary
            logger.info("Generating final summary...")
            final_summary = await dialogue_manager.get_summary()
            
            logger.info("Dialogue process complete.")

            return {
                "topic": topic,
                "history": [turn.__dict__ for turn in dialogue_manager.history],
                "meta_commentary": meta_commentary,
                "summary": final_summary,
            }

async def main():
    """A simple runner for the MultiPersonaDialogueAgent."""
    agent = MultiPersonaDialogueAgent()
    
    topic = "Is a fully autonomous, self-replicating AI a net-positive for humanity's future?"
    
    result = await agent.run_dialogue(topic, max_rounds=2)
    
    print("\n\n" + "="*80)
    print("                      FINAL REPORT")
    print("="*80)
    print(f"TOPIC: {result['topic']}\n")
    print("-" * 30 + " DIALOGUE HISTORY " + "-" * 30)
    for turn in result['history']:
        print(f"**{turn['persona_name']}**: {turn['content']}\n")
    
    if result['meta_commentary']:
        print("-" * 30 + " META COMMENTARY " + "-" * 30)
        print(f"{result['meta_commentary']}\n")
        
    print("-" * 30 + " FINAL SUMMARY " + "-" * 30)
    print(result['summary'])
    print("="*80)


if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(main()) 