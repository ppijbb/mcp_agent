#!/usr/bin/env python3
"""
GraphRAG Agent - Main Entry Point

This is the main entry point for the GraphRAG Agent application.
It handles configuration loading, agent initialization, and workflow execution.
"""

import asyncio
import sys
import argparse
from config import ConfigManager
from agents import GraphRAGAgent
from utils.sample_data import create_sample_data

# Import standalone agent with error handling
try:
    from standalone_agent import StandaloneGraphRAGAgent
    STANDALONE_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: StandaloneGraphRAGAgent not available: {e}")
    STANDALONE_AGENT_AVAILABLE = False


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GraphRAG Agent - Knowledge Graph Management')
    parser.add_argument('--standalone', '-s', action='store_true',
                       help='Run in standalone mode (no API server)')
    parser.add_argument('--agent-id', '-a', type=str, default='graphrag_agent',
                       help='Agent ID for standalone mode')
    parser.add_argument('--a2a-server', action='store_true',
                       help='Start A2A server for inter-agent communication')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Start interactive CLI mode')
    parser.add_argument('--command', '-c', type=str, 
                       help='Execute a single natural language command')
    parser.add_argument('--mode', '-m', type=str, 
                       choices=['create', 'query', 'visualize', 'optimize', 'status'],
                       help='Operation mode (legacy)')
    parser.add_argument('--data-file', '-d', type=str, 
                       help='Input data file')
    parser.add_argument('--graph-path', '-g', type=str, 
                       help='Graph file path')
    parser.add_argument('--output-path', '-o', type=str, 
                       help='Output file path')
    parser.add_argument('--query', '-q', type=str, 
                       help='Query string')
    
    args = parser.parse_args()
    
    # Check for help flag
    if len(sys.argv) == 1:
        print_help()
        return
    
    try:
        # Standalone mode
        if args.standalone:
            if not STANDALONE_AGENT_AVAILABLE:
                print("❌ Standalone agent not available. Please check dependencies.")
                sys.exit(1)
                
            agent = StandaloneGraphRAGAgent(agent_id=args.agent_id)
            
            if args.a2a_server:
                agent.start_a2a_server()
            
            success = await agent.run_standalone(
                command=args.command,
                interactive=args.interactive
            )
            
            if not success:
                sys.exit(1)
            return
        
        # Legacy mode
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
        
        # Handle different execution modes
        if args.interactive:
            # Start interactive CLI
            from interactive_cli import InteractiveCLI
            cli = InteractiveCLI()
            await cli.run()
            return
        
        elif args.command:
            # Execute single natural language command
            await execute_natural_language_command(config, args.command)
            return
        
        else:
            # Legacy mode execution
            await execute_legacy_mode(config, args)
            
    except Exception as e:
        print(f"❌ Failed to initialize GraphRAG Agent: {e}")
        return


async def execute_natural_language_command(config, command: str):
    """Execute a single natural language command"""
    try:
        from agents.natural_language_agent import NaturalLanguageAgent
        
        # Initialize natural language agent
        nl_agent = NaturalLanguageAgent(config)
        
        # Parse and execute command
        parsed_command = nl_agent.parse_command(command)
        response = nl_agent.execute_command(parsed_command, {})
        
        # Print response
        if response.get("status") == "completed":
            print(f"✅ {response.get('message', '완료되었습니다.')}")
            
            # If it's a graph creation command, actually create the graph
            if response.get("action") == "create_graph":
                print(f"🎯 사용자 의도: {parsed_command.user_intent}")
                await execute_legacy_mode(config, {
                    "mode": "create",
                    "data_file": response.get("data_file"),
                    "user_intent": parsed_command.user_intent
                })
        else:
            print(f"❌ {response.get('message', '오류가 발생했습니다.')}")
            
    except Exception as e:
        print(f"❌ 명령 실행 실패: {e}")


async def execute_legacy_mode(config, args):
    """Execute in legacy mode"""
    try:
        # Initialize agent
        agent = GraphRAGAgent(config)
        
        # Override config with command line arguments
        if hasattr(args, 'mode') and args.mode:
            config.mode = args.mode
        if hasattr(args, 'data_file') and args.data_file:
            config.data_file = args.data_file
        if hasattr(args, 'graph_path') and args.graph_path:
            config.graph_path = args.graph_path
        if hasattr(args, 'output_path') and args.output_path:
            config.output_path = args.output_path
        if hasattr(args, 'query') and args.query:
            config.query = args.query
        if hasattr(args, 'user_intent') and args.user_intent:
            config.user_intent = args.user_intent
        
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
        print(f"❌ Legacy mode execution failed: {e}")
        return


def print_help():
    """Print help information"""
    help_text = """
🧠 GraphRAG Agent - Knowledge Graph Management System

Usage:
    python main.py [options]

Standalone Mode (Recommended):
    --standalone, -s            Run in standalone mode (no API server)
    --agent-id, -a <id>         Agent ID for standalone mode (default: graphrag_agent)
    --a2a-server                Start A2A server for inter-agent communication

Interactive Mode:
    --interactive, -i           Start interactive CLI mode
    --command, -c <command>     Execute a single natural language command

Legacy Mode:
    --mode, -m <mode>           Legacy mode: create, query, visualize, optimize, status

Standalone Mode (Recommended):
    python main.py --standalone --interactive
    python main.py --standalone --command "그래프 생성해줘"
    python main.py --standalone --a2a-server

Interactive Mode:
    python main.py --interactive
    
    Example commands in interactive mode:
    - "그래프 생성해줘"
    - "Apple을 그래프에 추가해줘"
    - "Apple에 대해 알려줘"
    - "그래프 시각화해줘"
    - "도움말"

Single Command Mode:
    python main.py --command "Apple을 그래프에 추가해줘"
    python main.py --command "그래프 시각화해줘"

Legacy Mode:
    python main.py --mode create --data-file data.csv
    python main.py --mode query --graph-path graph.pkl --query "What is Apple?"

Environment Variables:
    OPENAI_API_KEY              OpenAI API key (required)
    
    # Optional configuration overrides
    LOG_LEVEL                   Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    MAX_SEARCH_RESULTS          Max search results for RAG (1-20)
    CONTEXT_WINDOW_SIZE         Context window size (1000-32000)
    ENABLE_VISUALIZATION        Enable graph visualization (true/false)
    ENABLE_OPTIMIZATION         Enable graph optimization (true/false)
    OPTIMIZATION_QUALITY_THRESHOLD Quality threshold (0.0-1.0)
    MAX_OPTIMIZATION_ITERATIONS Max iterations (1-100)

Examples:
    # Interactive mode (recommended)
    export OPENAI_API_KEY="your_key"
    python main.py --interactive
    
    # Single command
    export OPENAI_API_KEY="your_key"
    python main.py --command "Apple과 Microsoft의 관계를 그래프에 추가해줘"
    
    # Legacy mode
    export OPENAI_API_KEY="your_key"
    python main.py --mode create --data-file data.csv

For more information, see the README.md file.
"""
    print(help_text)


if __name__ == "__main__":
    asyncio.run(main())