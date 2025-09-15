"""
GraphRAG Agent - Main Agent Class

This module contains the main GraphRAG Agent class that orchestrates the workflow.
"""

import logging
from typing import Optional
from config import GraphRAGConfig, AgentConfig
from .workflow import GraphRAGWorkflow


class GraphRAGAgent:
    """Main GraphRAG Agent class"""
    
    def __init__(self, config: GraphRAGConfig):
        """Initialize the GraphRAG Agent"""
        self.config = config
        self.logger = self._setup_logging()
        self.coordinator = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_agent_logging(self):
        """Setup logging for agents"""
        level = logging.INFO
        if self.config.debug:
            level = logging.DEBUG
        
        logging.getLogger('agents').setLevel(level)
    
    def _initialize_coordinator(self):
        """Initialize the LangGraph workflow"""
        # Convert our config to agent config
        agent_config = AgentConfig(
            openai_api_key=self.config.agent.openai_api_key,
            model_name=self.config.agent.model_name,
            max_search_results=self.config.agent.max_search_results,
            context_window_size=self.config.agent.context_window_size
        )
        
        self.coordinator = GraphRAGWorkflow(agent_config)
    
    async def run(self) -> bool:
        """
        Run the GraphRAG Agent
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("GraphRAG Agent initialized successfully")
            
            # Initialize the coordinator
            self._initialize_coordinator()
            
            # Prepare initial state
            initial_state = {
                "mode": self.config.mode,
                "data_file": self.config.data_file,
                "graph_path": self.config.graph_path,
                "output_path": self.config.output_path,
                "query": self.config.query
            }
            
            # Run the workflow
            result = await self.coordinator.run(initial_state)
            
            if result:
                self.logger.info("✅ Operation completed successfully")
                return True
            else:
                self.logger.error("❌ Operation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Operation failed: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get system status"""
        return {
            "status": "ready",
            "config": {
                "mode": self.config.mode,
                "model": self.config.agent.model_name,
                "visualization_enabled": self.config.visualization.enabled,
                "optimization_enabled": self.config.optimization.enabled
            }
        }
