"""
Simple GraphRAG Workflow

This module defines a simple workflow without LangGraph dependencies.
"""

from typing import Dict, Any
from models.types import GraphRAGState
from config import AgentConfig
from .graph_generator import GraphGeneratorNode
from .rag_agent import RAGAgentNode
from utils.visualization import VisualizationNode
from utils.optimization import OptimizationNode


class GraphRAGWorkflow:
    """Simple GraphRAG workflow"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.graph_generator = GraphGeneratorNode(config)
        self.rag_agent = RAGAgentNode(config)
        self.visualizer = VisualizationNode(config)
        self.optimizer = OptimizationNode(config)
    
    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow with initial state"""
        try:
            # Convert dict to GraphRAGState
            state = GraphRAGState(**initial_state)
            
            # Route based on mode
            mode = state.get("mode", "create")
            
            if mode == "create":
                # Generate graph
                state = self.graph_generator(state)
                if state.get("status") != "completed":
                    return dict(state)
                
                # Save graph if output_path is specified
                if state.get("output_path"):
                    import pickle
                    with open(state["output_path"], "wb") as f:
                        pickle.dump(state["knowledge_graph"], f)
                    state["graph_path"] = state["output_path"]
                
                # Visualize if enabled
                if self.config.enable_visualization:
                    state = self.visualizer(state)
                    if state.get("status") != "completed":
                        return dict(state)
                
                # Optimize if enabled
                if self.config.enable_optimization:
                    state = self.optimizer(state)
                    if state.get("status") != "completed":
                        return dict(state)
                
                # Mark as completed if all steps succeeded
                state["status"] = "completed"
                
            elif mode == "query":
                # Load existing graph and query
                if state.get("graph_path"):
                    import pickle
                    with open(state["graph_path"], "rb") as f:
                        state["knowledge_graph"] = pickle.load(f)
                
                state = self.rag_agent(state)
                if state.get("status") != "completed":
                    return dict(state)
                
            elif mode == "visualize":
                # Load existing graph and visualize
                if state.get("graph_path"):
                    import pickle
                    with open(state["graph_path"], "rb") as f:
                        state["knowledge_graph"] = pickle.load(f)
                
                state = self.visualizer(state)
                if state.get("status") != "completed":
                    return dict(state)
                
            elif mode == "optimize":
                # Load existing graph and optimize
                if state.get("graph_path"):
                    import pickle
                    with open(state["graph_path"], "rb") as f:
                        state["knowledge_graph"] = pickle.load(f)
                
                state = self.optimizer(state)
                if state.get("status") != "completed":
                    return dict(state)
                
            elif mode == "status":
                # Return status information
                state["status"] = "completed"
                state["response"] = "System is running normally"
                state["system_info"] = {
                    "config": {
                        "model": self.config.model_name,
                        "visualization_enabled": self.config.enable_visualization,
                        "optimization_enabled": self.config.enable_optimization
                    }
                }
            
            else:
                state["error"] = f"Unknown mode: {mode}"
                state["status"] = "error"
            
            return dict(state)
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_sync(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow synchronously"""
        import asyncio
        return asyncio.run(self.run(initial_state))