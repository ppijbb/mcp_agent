"""
GraphRAG Agent - Main Entry Point

A production-ready agent that operates based on configuration files and environment variables.
No more CLI arguments - everything is managed through config.yaml and environment variables.

Usage:
    # Set environment variables or use config.yaml
    export OPENAI_API_KEY="your_key_here"
    export MODE="create"
    export DATA_FILE="data.csv"
    export OUTPUT_PATH="graph.pkl"
    
    # Run the agent
    python main.py
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from config_manager import ConfigManager, GraphRAGConfig
from multi_agent_coordinator import MultiAgentCoordinator


class GraphRAGAgent:
    """Main GraphRAG Agent class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize GraphRAG Agent
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_manager = ConfigManager(config_path)
        self.config = None
        self.coordinator = None
        self.logger = None
        
    def initialize(self) -> bool:
        """
        Initialize the agent with configuration
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load configuration
            self.config = self.config_manager.load_config()
            
            # Validate configuration
            if not self.config_manager.validate_config(self.config):
                return False
            
            # Setup logging
            self._setup_logging()
            
            # Initialize coordinator
            self._initialize_coordinator()
            
            self.logger.info("GraphRAG Agent initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize GraphRAG Agent: {e}")
            return False
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.logging
        
        # Configure logging level
        level = getattr(logging, log_config.level)
        
        # Create formatter
        formatter = logging.Formatter(log_config.format)
        
        # Setup handlers
        handlers = [logging.StreamHandler()]
        
        # Add file handler if specified
        if log_config.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_config.file_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Set specific logger levels
        logging.getLogger('multi_agent_coordinator').setLevel(level)
        logging.getLogger('agents').setLevel(level)
    
    def _initialize_coordinator(self):
        """Initialize the multi-agent coordinator"""
        from multi_agent_coordinator import MultiAgentConfig
        
        # Convert our config to coordinator config
        coordinator_config = MultiAgentConfig(
            openai_api_key=self.config.agent.openai_api_key,
            graph_model_name=self.config.agent.model_name,
            rag_model_name=self.config.agent.model_name,
            max_search_results=self.config.agent.max_search_results,
            context_window_size=self.config.agent.context_window_size,
            enable_visualization=self.config.visualization.enabled,
            enable_optimization=self.config.optimization.enabled,
            visualization_output_dir=self.config.visualization.output_directory,
            optimization_quality_threshold=self.config.optimization.quality_threshold,
            enable_domain_specialization=self.config.graph.enable_domain_specialization,
            domain_type=self.config.graph.domain_type,
            enable_security_privacy=self.config.graph.enable_security_privacy,
            enable_query_optimization=self.config.graph.enable_query_optimization,
            default_data_classification=self.config.graph.default_data_classification
        )
        
        self.coordinator = MultiAgentCoordinator(coordinator_config)
    
    async def run(self) -> bool:
        """
        Run the agent based on configuration
        
        Returns:
            bool: True if operation successful, False otherwise
        """
        if not self.config:
            self.logger.error("Agent not initialized")
            return False
        
        try:
            mode = self.config.mode.lower()
            
            if mode == "create":
                return await self._create_graph()
            elif mode == "query":
                return await self._query_graph()
            elif mode == "visualize":
                return await self._visualize_graph()
            elif mode == "optimize":
                return await self._optimize_graph()
            elif mode == "export":
                return await self._export_graph()
            elif mode == "status":
                return await self._check_status()
            else:
                self.logger.error(f"Unknown mode: {mode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during operation: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    async def _create_graph(self) -> bool:
        """Create knowledge graph from data"""
        if not self.config.data_file:
            self.logger.error("Data file not specified")
            return False
        
        if not self.config.output_path:
            self.logger.error("Output path not specified")
            return False
        
        if not Path(self.config.data_file).exists():
            self.logger.error(f"Data file not found: {self.config.data_file}")
            return False
        
        self.logger.info(f"Creating knowledge graph from {self.config.data_file}")
        
        result = await self.coordinator.create_knowledge_graph(
            self.config.data_file,
            self.config.output_path,
            enable_visualization=self.config.visualization.enabled,
            enable_optimization=self.config.optimization.enabled
        )
        
        if result["status"] == "completed":
            self.logger.info(f"✅ Knowledge graph created successfully: {result['graph_path']}")
            
            if self.config.verbose and "stats" in result:
                stats = result["stats"]
                self.logger.info(f"Graph Stats: {stats.get('nodes')} nodes, {stats.get('edges')} edges")
            
            return True
        else:
            self.logger.error(f"❌ Graph creation failed: {result.get('error')}")
            return False
    
    async def _query_graph(self) -> bool:
        """Query knowledge graph"""
        if not self.config.graph_path:
            self.logger.error("Graph path not specified")
            return False
        
        if not self.config.query:
            self.logger.error("Query not specified")
            return False
        
        if not Path(self.config.graph_path).exists():
            self.logger.error(f"Graph file not found: {self.config.graph_path}")
            return False
        
        self.logger.info(f"Querying graph: {self.config.query}")
        
        result = await self.coordinator.query_knowledge_graph(
            self.config.query,
            self.config.graph_path
        )
        
        if result["status"] == "completed":
            print("\n" + "="*60)
            print("📝 Response:")
            print(result["response"])
            
            if self.config.verbose and "context" in result:
                print("\n" + "-"*20 + " Context " + "-"*20)
                print(result["context"])
            
            print("="*60)
            return True
        else:
            self.logger.error(f"❌ Query failed: {result.get('error')}")
            return False
    
    async def _visualize_graph(self) -> bool:
        """Create graph visualizations"""
        if not self.config.graph_path:
            self.logger.error("Graph path not specified")
            return False
        
        if not Path(self.config.graph_path).exists():
            self.logger.error(f"Graph file not found: {self.config.graph_path}")
            return False
        
        self.logger.info(f"Creating visualizations for {self.config.graph_path}")
        
        result = await self.coordinator.create_graph_visualizations(
            self.config.graph_path,
            self.config.output_path
        )
        
        if result["status"] == "completed":
            self.logger.info("✅ Visualizations created successfully")
            return True
        else:
            self.logger.error(f"❌ Visualization failed: {result.get('error')}")
            return False
    
    async def _optimize_graph(self) -> bool:
        """Optimize graph quality"""
        if not self.config.graph_path:
            self.logger.error("Graph path not specified")
            return False
        
        if not Path(self.config.graph_path).exists():
            self.logger.error(f"Graph file not found: {self.config.graph_path}")
            return False
        
        self.logger.info(f"Optimizing graph: {self.config.graph_path}")
        
        result = await self.coordinator.optimize_existing_graph(
            self.config.graph_path,
            self.config.output_path
        )
        
        if result["status"] == "completed":
            self.logger.info(f"✅ Graph optimization completed. Quality: {result['overall_quality']:.3f}")
            return True
        else:
            self.logger.error(f"❌ Optimization failed: {result.get('error')}")
            return False
    
    async def _export_graph(self) -> bool:
        """Export graph data"""
        if not self.config.graph_path:
            self.logger.error("Graph path not specified")
            return False
        
        if not Path(self.config.graph_path).exists():
            self.logger.error(f"Graph file not found: {self.config.graph_path}")
            return False
        
        export_format = getattr(self.config, 'export_format', 'json')
        
        self.logger.info(f"Exporting graph in {export_format} format")
        
        result = await self.coordinator.export_graph_data(
            self.config.graph_path,
            export_format,
            self.config.output_path
        )
        
        if result["status"] == "completed":
            self.logger.info("✅ Graph export completed successfully")
            return True
        else:
            self.logger.error(f"❌ Export failed: {result.get('error')}")
            return False
    
    async def _check_status(self) -> bool:
        """Check system status"""
        self.logger.info("Checking system status...")
        
        # Get agent status
        agent_status = self.coordinator.get_agent_status()
        print("\n📋 Agent Status:")
        for agent, status in agent_status.items():
            if agent != "config":
                print(f"   - {agent}: {status}")
        
        # Get system health
        health = await self.coordinator.health_check()
        print(f"\n🏥 System Health: {health['status']}")
        print(f"   - API Connectivity: {health['api_connectivity']}")
        print(f"   - Visualization: {health['visualization_status']}")
        print(f"   - Optimization: {health['optimization_status']}")
        print(f"   - Timestamp: {health['timestamp']}")
        
        if self.config.verbose and "agents" in health and "config" in health["agents"]:
            print("\n⚙️  Configuration:")
            config = health["agents"]["config"]
            for key, value in config.items():
                print(f"   - {key}: {value}")
        
        return True
    
    def create_sample_data(self) -> Optional[str]:
        """Create sample data file for testing"""
        try:
            import pandas as pd
            
            sample_data = {
                "id": [1, 2, 3, 4, 5],
                "document_id": ["doc_1", "doc_1", "doc_2", "doc_2", "doc_3"],
                "text_unit": [
                    "Apple Inc. is a technology company based in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
                    "The company is known for innovative products like the iPhone, iPad, and Mac computers. Tim Cook is the current CEO of Apple.",
                    "Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. It was founded by Bill Gates and Paul Allen in 1975.",
                    "Microsoft is best known for its Windows operating systems, Office productivity suite, and Azure cloud computing platform. Satya Nadella is the current CEO.",
                    "Google LLC is an American multinational technology company that specializes in Internet-related services and products. It was founded by Larry Page and Sergey Brin while they were PhD students at Stanford University."
                ]
            }
            
            df = pd.DataFrame(sample_data)
            sample_file = "sample_data.csv"
            df.to_csv(sample_file, index=False)
            
            if self.logger:
                self.logger.info(f"Sample data created: {sample_file}")
            else:
                print(f"✅ Sample data created: {sample_file}")
            
            return sample_file
            
        except ImportError:
            error_msg = "pandas is required. Install with: pip install pandas"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"❌ Error: {error_msg}")
            return None
        except Exception as e:
            error_msg = f"Error creating sample data: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"❌ Error: {error_msg}")
            return None


async def main():
    """Main entry point"""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        return
    
    # Initialize agent
    agent = GraphRAGAgent()
    
    if not agent.initialize():
        sys.exit(1)
    
    # Run the agent
    success = await agent.run()
    
    if not success:
        sys.exit(1)


def print_help():
    """Print help information"""
    help_text = """
GraphRAG Agent - Configuration-based Knowledge Graph Agent

This agent operates based on configuration files and environment variables.
No CLI arguments needed - everything is managed through config.yaml and environment variables.

Configuration:
  - config.yaml: Main configuration file
  - Environment variables: Override config.yaml settings
  - See env.example for available environment variables

Usage Examples:

1. Create a knowledge graph:
   export OPENAI_API_KEY="your_key_here"
   export MODE="create"
   export DATA_FILE="data.csv"
   export OUTPUT_PATH="graph.pkl"
   python main.py

2. Query a knowledge graph:
   export OPENAI_API_KEY="your_key_here"
   export MODE="query"
   export GRAPH_PATH="graph.pkl"
   export QUERY="What is the main topic?"
   python main.py

3. Create visualizations:
   export OPENAI_API_KEY="your_key_here"
   export MODE="visualize"
   export GRAPH_PATH="graph.pkl"
   python main.py

4. Optimize graph quality:
   export OPENAI_API_KEY="your_key_here"
   export MODE="optimize"
   export GRAPH_PATH="graph.pkl"
   python main.py

5. Export graph data:
   export OPENAI_API_KEY="your_key_here"
   export MODE="export"
   export GRAPH_PATH="graph.pkl"
   export EXPORT_FORMAT="json"
   python main.py

6. Check system status:
   export OPENAI_API_KEY="your_key_here"
   export MODE="status"
   python main.py

Configuration File:
  The agent looks for config.yaml in the current directory or the same directory as main.py.
  You can specify a custom config file by setting CONFIG_PATH environment variable.

Environment Variables:
  See env.example for a complete list of available environment variables.
  Environment variables override settings in config.yaml.

Required:
  - OPENAI_API_KEY: Your OpenAI API key

Optional:
  - All other settings can be configured via config.yaml or environment variables
"""
    print(help_text)


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
