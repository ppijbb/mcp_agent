#!/usr/bin/env python3
"""
GraphRAG Agent - Main Entry Point

This is the main entry point for the GraphRAG Agent application.
It handles configuration loading, agent initialization, and workflow execution.
"""

import asyncio
import sys
from config import ConfigManager
from agents import GraphRAGAgent
from utils.sample_data import create_sample_data


async def main():
    """Main entry point"""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_help()
        return
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        if not config:
            print("❌ Failed to load configuration")
            return
        
        # Validate configuration
        if not config_manager.validate_config(config):
            print("❌ Configuration validation failed")
            return
        
        # Initialize agent
        agent = GraphRAGAgent(config)
        
        # Create sample data if needed
        if config.mode == "create" and not config.data_file:
            sample_file = create_sample_data(agent.logger)
            if sample_file:
                config.data_file = sample_file
            else:
                print("❌ Failed to create sample data")
                return
        
        # Run the agent
        success = await agent.run()
        
        if success:
            print("✅ GraphRAG Agent completed successfully")
        else:
            print("❌ GraphRAG Agent failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to initialize GraphRAG Agent: {e}")
        return


def print_help():
    """Print help information"""
    help_text = """
GraphRAG Agent - Knowledge Graph Generation and Querying

Usage:
    python main.py [options]

Options:
    -h, --help          Show this help message

Environment Variables:
    OPENAI_API_KEY      OpenAI API key (required)
    MODE                Operation mode: create, query, visualize, optimize, status
    DATA_FILE           Path to input data file (for create mode)
    GRAPH_PATH          Path to knowledge graph file (for query/visualize/optimize modes)
    OUTPUT_PATH         Path for output files
    QUERY               Query string (for query mode)
    
    # Optional configuration overrides
    LOG_LEVEL           Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    MAX_SEARCH_RESULTS  Max search results for RAG (1-20)
    CONTEXT_WINDOW_SIZE Context window size (1000-32000)
    ENABLE_VISUALIZATION Enable graph visualization (true/false)
    ENABLE_OPTIMIZATION Enable graph optimization (true/false)
    OPTIMIZATION_QUALITY_THRESHOLD Quality threshold (0.0-1.0)
    MAX_OPTIMIZATION_ITERATIONS Max iterations (1-100)

Examples:
    # Create a knowledge graph
    export OPENAI_API_KEY="your_key"
    export MODE="create"
    export DATA_FILE="data.csv"
    python main.py
    
    # Query the knowledge graph
    export MODE="query"
    export GRAPH_PATH="graph.pkl"
    export QUERY="What is Apple?"
    python main.py
    
    # Visualize the knowledge graph
    export MODE="visualize"
    export GRAPH_PATH="graph.pkl"
    python main.py
    
    # Check system status
    export MODE="status"
    python main.py

For more information, see the README.md file.
"""
    print(help_text)


if __name__ == "__main__":
    asyncio.run(main())