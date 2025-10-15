"""
Main entry point for the Multi-Persona Dialogue Agent system.
"""
import asyncio
import json
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.utils import setup_agent_app
# from srcs.core.agent.base import BaseAgent  # Temporarily disabled due to settings issue
from .multi_persona_config import config, get_persona_config, get_dialogue_config, get_llm_config
from .personas import PERSONA_INSTRUCTIONS
from .dialogue_manager import DialogueManager

class MultiPersonaDialogueAgent:
    """
    An agent system that uses a dialogue between multiple personas 
    to explore a topic from different angles and produce a comprehensive result.
    """
    def __init__(self):
        # Load configurations
        self.config = config
        self.persona_config = get_persona_config()
        self.dialogue_config = get_dialogue_config()
        self.llm_config = get_llm_config()
        
        # Initialize MCPApp
        self.app = setup_agent_app("multi_persona_agent")
        self.name = "multi_persona_agent"
        self.instruction = "Multi-persona dialogue system for comprehensive topic analysis"
        self.server_names = ["g-search", "fetch"]
        
        # Initialize persona agents
        self.persona_agents = {}
        self._initialize_personas()

    def _initialize_personas(self):
        """Initialize persona instructions from configuration."""
        for name in self.persona_config.enabled_personas:
            if name in PERSONA_INSTRUCTIONS:
                self.persona_agents[name] = {
                    "name": name,
                    "instruction": PERSONA_INSTRUCTIONS[name]
                }

    async def run_workflow(self, topic: str, max_rounds: int = None) -> Dict[str, Any]:
        """Main workflow method."""
        if max_rounds is None:
            max_rounds = self.dialogue_config.max_rounds
        
        return await self.run_dialogue(topic, max_rounds)
    
    async def run_dialogue(self, topic: str, max_rounds: int) -> Dict[str, Any]:
        """
        Conducts a multi-persona dialogue on a given topic using real LLM.
        """
        async with self.app.run() as app_context:
            logger = app_context.logger
            logger.info(f"Starting multi-persona dialogue on topic: '{topic}'")
            logger.info(f"Using {len(self.persona_agents)} personas: {list(self.persona_agents.keys())}")
            logger.info(f"Configuration: max_rounds={max_rounds}, turns_per_round={self.dialogue_config.turns_per_round}")

            # Create Agent instances within MCPApp context
            agents = {}
            for name, persona_info in self.persona_agents.items():
                agent = Agent(
                    name=persona_info["name"],
                    instruction=persona_info["instruction"],
                    server_names=self.server_names
                )
                # Set the LLM using GoogleAugmentedLLM
                agent.llm = GoogleAugmentedLLM(app=self.app)
                agents[name] = agent
            
            dialogue_manager = DialogueManager(
                topic=topic,
                personas=list(agents.values()),
                llm_config=self.llm_config
            )

            # Run Dialogue Rounds
            for i in range(max_rounds * self.dialogue_config.turns_per_round):
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
    import sys
    
    agent = MultiPersonaDialogueAgent()
    
    # Get topic from command line argument or use default
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "The impact of artificial intelligence on modern society"
    
    result = await agent.run_workflow(topic, max_rounds=2)
    
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