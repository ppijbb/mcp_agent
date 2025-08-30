"""
Multi-Agent Coordinator

Graph Generator Agent와 RAG Agent를 조율하는 코디네이터
"""

import asyncio
import pickle
from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
from pathlib import Path

from .agents.graph_generator_agent import GraphGeneratorAgent, GraphGeneratorConfig
from .agents.rag_agent import RAGAgent, RAGAgentConfig
from .agents.graph_visualization_agent import GraphVisualizationAgent, GraphVisualizationConfig
from .agents.graph_optimization_agent import GraphOptimizationAgent, GraphOptimizationConfig
from pydantic import BaseModel, Field, validator


class MultiAgentConfig(BaseModel):
    """Configuration for Multi-Agent System"""
    openai_api_key: str = Field(..., description="OpenAI API key")
    graph_model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="Model for graph generation")
    rag_model_name: str = Field(default="gemini-2.5-flash-lite-preview-06-07", description="Model for RAG responses")
    max_search_results: int = Field(default=5, description="Max search results for RAG")
    context_window_size: int = Field(default=4000, description="Context window size")
    enable_visualization: bool = Field(default=True, description="Enable graph visualization")
    enable_optimization: bool = Field(default=True, description="Enable graph optimization")
    visualization_output_dir: str = Field(default="./graph_visualizations", description="Visualization output directory")
    optimization_quality_threshold: float = Field(default=0.8, description="Graph quality threshold for optimization")
    
    @validator('max_search_results')
    def validate_max_search_results(cls, v):
        if v < 1 or v > 20:
            raise ValueError('max_search_results must be between 1 and 20')
        return v
    
    @validator('context_window_size')
    def validate_context_window_size(cls, v):
        if v < 1000 or v > 32000:
            raise ValueError('context_window_size must be between 1000 and 32000')
        return v
    
    @validator('optimization_quality_threshold')
    def validate_optimization_quality_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('optimization_quality_threshold must be between 0.0 and 1.0')
        return v


class MultiAgentCoordinator:
    """멀티 에이전트 시스템 코디네이터"""

    def __init__(self, config: MultiAgentConfig):
        """
        Initialize Multi-Agent Coordinator
        
        Args:
            config: MultiAgentConfig object with all necessary parameters
        """
        self.config = config
        self._setup_logging()
        self._initialize_agents()
        self.logger.info("Multi-Agent Coordinator initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('graphrag_coordinator.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_agents(self):
        """Initialize the specialized agents"""
        try:
            # Initialize Graph Generator Agent
            graph_config = GraphGeneratorConfig(
                openai_api_key=self.config.openai_api_key,
                model_name=self.config.graph_model_name,
            )
            self.graph_generator = GraphGeneratorAgent(graph_config)
            self.logger.info("GraphGeneratorAgent initialized successfully")
            
            # Initialize RAG Agent
            rag_config = RAGAgentConfig(
                openai_api_key=self.config.openai_api_key,
                model_name=self.config.rag_model_name,
                max_search_results=self.config.max_search_results,
                context_window_size=self.config.context_window_size
            )
            self.rag_agent = RAGAgent(rag_config)
            self.logger.info("RAGAgent initialized successfully")
            
            # Initialize Graph Visualization Agent (if enabled)
            if self.config.enable_visualization:
                viz_config = GraphVisualizationConfig(
                    output_directory=self.config.visualization_output_dir
                )
                self.graph_visualizer = GraphVisualizationAgent(viz_config)
                self.logger.info("GraphVisualizationAgent initialized successfully")
            else:
                self.graph_visualizer = None
                self.logger.info("Graph visualization disabled")
            
            # Initialize Graph Optimization Agent (if enabled)
            if self.config.enable_optimization:
                opt_config = GraphOptimizationConfig(
                    quality_threshold=self.config.optimization_quality_threshold
                )
                self.graph_optimizer = GraphOptimizationAgent(opt_config)
                self.logger.info("GraphOptimizationAgent initialized successfully")
            else:
                self.graph_optimizer = None
                self.logger.info("Graph optimization disabled")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}")

    async def create_knowledge_graph(self, data_file_path: str, output_path: str, 
                                   enable_visualization: bool = None, 
                                   enable_optimization: bool = None) -> Dict[str, Any]:
        """
        Loads data, generates a knowledge graph, and saves it to a file.
        Optionally creates visualizations and optimizes the graph.
        
        Args:
            data_file_path: Path to the input data file
            output_path: Path where the knowledge graph will be saved
            enable_visualization: Override config for visualization (optional)
            enable_optimization: Override config for optimization (optional)
            
        Returns:
            Dict containing status and results
        """
        self.logger.info(f"Starting knowledge graph creation from {data_file_path}")
        
        # Use override values or config defaults
        should_visualize = enable_visualization if enable_visualization is not None else self.config.enable_visualization
        should_optimize = enable_optimization if enable_optimization is not None else self.config.enable_optimization
        
        # Validate input file
        if not Path(data_file_path).exists():
            error_msg = f"Data file not found: {data_file_path}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        
        # Load and validate data
        try:
            df = pd.read_csv(data_file_path)
            required_columns = ["id", "document_id", "text_unit"]
            
            if df.empty:
                raise ValueError("Data file is empty")
                
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Validate data quality
            if df['text_unit'].isna().any():
                self.logger.warning("Found empty text units, removing them")
                df = df.dropna(subset=['text_unit'])
                
            if df.empty:
                raise ValueError("No valid text units found after cleaning")
                
            self.logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            error_msg = f"Data loading failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # Generate knowledge graph
        try:
            self.logger.info("Delegating to GraphGeneratorAgent...")
            result = await self.graph_generator.process_text_units(df)
            
            if result["status"] != "completed":
                error_msg = result.get("error", "Unknown error during graph generation")
                self.logger.error(error_msg)
                return {"status": "error", "error": error_msg}
            
            knowledge_graph = result["knowledge_graph"]
            self.logger.info(f"Graph generated successfully: {result.get('stats')}")
            
        except Exception as e:
            error_msg = f"Graph generation failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # Optimize graph quality (if enabled)
        optimization_result = None
        if should_optimize and self.graph_optimizer:
            try:
                self.logger.info("Starting graph optimization...")
                graph_name = Path(output_path).stem
                optimization_result = await self.graph_optimizer.optimize_graph_quality(
                    knowledge_graph, graph_name
                )
                
                if optimization_result["status"] == "completed":
                    self.logger.info(f"Graph optimization completed. Quality score: {optimization_result['overall_quality']:.3f}")
                else:
                    self.logger.warning(f"Graph optimization failed: {optimization_result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"Graph optimization failed: {e}")
                optimization_result = {"status": "error", "error": str(e)}

        # Create visualizations (if enabled)
        visualization_result = None
        if should_visualize and self.graph_visualizer:
            try:
                self.logger.info("Creating graph visualizations...")
                graph_name = Path(output_path).stem
                visualization_result = await self.graph_visualizer.create_comprehensive_visualization(
                    knowledge_graph, graph_name
                )
                
                if visualization_result["status"] == "completed":
                    self.logger.info("Graph visualizations created successfully")
                else:
                    self.logger.warning(f"Graph visualization failed: {visualization_result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"Graph visualization failed: {e}")
                visualization_result = {"status": "error", "error": str(e)}

        # Save knowledge graph
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                pickle.dump(knowledge_graph, f)
                
            self.logger.info(f"Knowledge graph saved to {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to save knowledge graph: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        return {
            "status": "completed",
            "graph_path": str(output_path),
            "stats": result.get("stats"),
            "data_stats": {
                "total_rows": len(df),
                "columns": list(df.columns)
            },
            "optimization": optimization_result,
            "visualization": visualization_result
        }

    async def query_knowledge_graph(self, user_query: str, graph_path: str) -> Dict[str, Any]:
        """
        Loads a knowledge graph and queries it using the RAG agent.
        
        Args:
            user_query: The user's question
            graph_path: Path to the knowledge graph file
            
        Returns:
            Dict containing status and response
        """
        self.logger.info(f"Loading knowledge graph from {graph_path} for query: '{user_query}'")

        # Validate query
        if not user_query or not user_query.strip():
            error_msg = "Empty or invalid query provided"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Please provide a valid question."}

        # Load knowledge graph
        try:
            if not Path(graph_path).exists():
                error_msg = f"Knowledge graph file not found: {graph_path}"
                self.logger.error(error_msg)
                return {"status": "error", "error": error_msg, "response": "Could not find the necessary knowledge graph to answer."}
                
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
                
            if not knowledge_graph or not hasattr(knowledge_graph, 'nodes') or not knowledge_graph.nodes:
                error_msg = "Knowledge graph is empty or invalid"
                self.logger.error(error_msg)
                return {"status": "error", "error": error_msg, "response": "The knowledge base is empty or corrupted."}
                
            self.logger.info(f"Knowledge graph loaded successfully with {len(knowledge_graph.nodes)} nodes")
            
        except FileNotFoundError:
            error_msg = f"Knowledge graph file not found: {graph_path}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Could not find the necessary knowledge graph to answer."}
        except Exception as e:
            error_msg = f"Failed to load knowledge graph: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "Failed to load the knowledge graph."}

        # Process query
        try:
            self.logger.info("Delegating to RAGAgent...")
            result = await self.rag_agent.query_knowledge_graph(
                user_query=user_query,
                knowledge_graph=knowledge_graph
            )
            self.logger.info("RAGAgent finished processing successfully")
            return result
            
        except Exception as e:
            error_msg = f"Query processing failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg, "response": "An error occurred while answering the question."}

    async def create_graph_visualizations(self, graph_path: str, output_name: str = None) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for an existing knowledge graph
        
        Args:
            graph_path: Path to the knowledge graph file
            output_name: Custom name for the visualizations (optional)
            
        Returns:
            Dict containing visualization results
        """
        if not self.graph_visualizer:
            return {"status": "error", "error": "Graph visualization is not enabled"}
        
        try:
            # Load knowledge graph
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
            
            # Create visualizations
            graph_name = output_name or Path(graph_path).stem
            result = await self.graph_visualizer.create_comprehensive_visualization(
                knowledge_graph, graph_name
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Visualization creation failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    async def optimize_existing_graph(self, graph_path: str, graph_name: str = None) -> Dict[str, Any]:
        """
        Optimize an existing knowledge graph
        
        Args:
            graph_path: Path to the knowledge graph file
            graph_name: Custom name for the optimization (optional)
            
        Returns:
            Dict containing optimization results
        """
        if not self.graph_optimizer:
            return {"status": "error", "error": "Graph optimization is not enabled"}
        
        try:
            # Load knowledge graph
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
            
            # Optimize graph
            graph_name = graph_name or Path(graph_path).stem
            result = await self.graph_optimizer.optimize_graph_quality(
                knowledge_graph, graph_name
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Graph optimization failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    async def export_graph_data(self, graph_path: str, export_format: str = "json", 
                               output_name: str = None) -> Dict[str, Any]:
        """
        Export graph data in various formats
        
        Args:
            graph_path: Path to the knowledge graph file
            export_format: Export format (json, csv, graphml)
            output_name: Custom name for the export (optional)
            
        Returns:
            Dict containing export results
        """
        if not self.graph_visualizer:
            return {"status": "error", "error": "Graph visualization is not enabled"}
        
        try:
            # Load knowledge graph
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
            
            # Export data
            graph_name = output_name or Path(graph_path).stem
            result = await self.graph_visualizer.export_graph_data(
                knowledge_graph, graph_name, export_format
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Graph export failed: {e}"
            self.logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        try:
            status = {
                "graph_generator": "ready" if hasattr(self, 'graph_generator') else "not_initialized",
                "rag_agent": "ready" if hasattr(self, 'rag_agent') else "not_initialized",
                "graph_visualizer": "ready" if hasattr(self, 'graph_visualizer') and self.graph_visualizer else "disabled",
                "graph_optimizer": "ready" if hasattr(self, 'graph_optimizer') and self.graph_optimizer else "disabled",
                "coordinator": "ready",
                "config": {
                    "graph_model": self.config.graph_model_name,
                    "rag_model": self.config.rag_model_name,
                    "max_search_results": self.config.max_search_results,
                    "context_window_size": self.config.context_window_size,
                    "visualization_enabled": self.config.enable_visualization,
                    "optimization_enabled": self.config.enable_optimization
                }
            }
            return status
        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the system"""
        try:
            status = self.get_agent_status()
            
            # Test API connectivity
            try:
                # Simple test to verify API key works
                test_result = await self.rag_agent.test_connectivity()
                api_status = "healthy" if test_result else "unhealthy"
            except Exception as e:
                api_status = f"error: {e}"
            
            # Test visualization agent if enabled
            viz_status = "not_available"
            if self.graph_visualizer:
                try:
                    viz_status = "ready"
                except Exception as e:
                    viz_status = f"error: {e}"
            
            # Test optimization agent if enabled
            opt_status = "not_available"
            if self.graph_optimizer:
                try:
                    opt_status = "ready"
                except Exception as e:
                    opt_status = f"error: {e}"
            
            return {
                "status": "healthy",
                "agents": status,
                "api_connectivity": api_status,
                "visualization_status": viz_status,
                "optimization_status": opt_status,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
