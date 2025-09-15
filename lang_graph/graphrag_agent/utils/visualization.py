"""
LangGraph-optimized Visualization Node
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any
from typing import Callable
from models.types import GraphRAGState
from config import AgentConfig


class VisualizationNode:
    """LangGraph node for creating graph visualizations"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = None
    
    def __call__(self, state: GraphRAGState) -> GraphRAGState:
        """Create visualizations for the knowledge graph"""
        try:
            if not state.get("knowledge_graph"):
                state["error"] = "No knowledge graph available for visualization"
                state["status"] = "error"
                return state
            
            if not self.config.enable_visualization:
                state["status"] = "skipped"
                return state
            
            # Create visualizations
            visualizations = self._create_visualizations(state["knowledge_graph"])
            
            # Update state
            state["visualizations"] = visualizations
            state["status"] = "completed"
            
            if self.logger:
                self.logger.info(f"Visualizations created: {len(visualizations)} files")
            
        except Exception as e:
            state["error"] = f"Visualization failed: {str(e)}"
            state["status"] = "error"
            if self.logger:
                self.logger.error(f"Visualization error: {e}")
        
        return state
    
    def _create_visualizations(self, graph: nx.Graph) -> Dict[str, Any]:
        """Create various visualizations of the graph"""
        visualizations = {}
        
        try:
            # Basic network visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                                 node_size=500, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(graph, pos, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(graph, pos, font_size=8)
            
            plt.title("Knowledge Graph Visualization")
            plt.axis('off')
            
            # Save visualization
            viz_path = "knowledge_graph_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations["network"] = {
                "path": viz_path,
                "format": "png",
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges()
            }
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Visualization creation failed: {e}")
        
        return visualizations
