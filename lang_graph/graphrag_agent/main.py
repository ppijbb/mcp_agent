"""
Multi-Agent Graph RAG System - Main Entry Point

Usage:
  # 1. Create a knowledge graph from your data
  python main.py create-graph --data-file path/to/your/data.csv --output-path knowledge_graph.pkl

  # 2. Query the generated knowledge graph
  python main.py query --graph-path knowledge_graph.pkl --query "Your question here"

  # To create and use sample data:
  python main.py create-graph --create-sample-data --output-path sample_graph.pkl
  python main.py query --graph-path sample_graph.pkl --query "Who is the CEO of Apple?"
"""

import asyncio
import argparse
import os
from pathlib import Path

from multi_agent_coordinator import MultiAgentCoordinator, MultiAgentConfig


def create_sample_data() -> str:
    """Create sample data file for testing and return its path."""
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
    print(f"✅ Sample data created: {sample_file}")
    return sample_file


async def main():
    """Main function to handle CLI commands."""
    parser = argparse.ArgumentParser(description="Multi-Agent Graph RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Global arguments ---
    parser.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # --- `create-graph` command ---
    parser_create = subparsers.add_parser("create-graph", help="Create a knowledge graph from data")
    parser_create.add_argument("--data-file", "-d", help="Path to data CSV file")
    parser_create.add_argument("--create-sample-data", action="store_true", help="Create and use a sample data file")
    parser_create.add_argument("--output-path", "-o", required=True, help="Path to save the generated knowledge graph")

    # --- `query` command ---
    parser_query = subparsers.add_parser("query", help="Query a knowledge graph")
    parser_query.add_argument("--graph-path", "-g", required=True, help="Path to the knowledge graph file")
    parser_query.add_argument("--query", "-q", required=True, help="Query to process")

    args = parser.parse_args()
    
    # --- API Key validation ---
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OpenAI API key is required. Set it via --openai-api-key or the OPENAI_API_KEY environment variable.")
        return

    # --- Initialize Coordinator ---
    config = MultiAgentConfig(
        openai_api_key=openai_api_key,
        graph_model_name="gemini-2.5-flash-lite-preview-06-07",
        rag_model_name="gemini-2.5-flash-lite-preview-06-07"
    )
    coordinator = MultiAgentCoordinator(config)

    # --- Command handling ---
    if args.command == "create-graph":
        data_file = args.data_file
        if args.create_sample_data:
            data_file = create_sample_data()
        
        if not data_file:
            print("❌ Error: --data-file or --create-sample-data is required for create-graph.")
            return
        
        if not Path(data_file).exists():
            print(f"❌ Error: Data file not found: {data_file}")
            return
        
        print("🚀 Starting knowledge graph creation...")
        result = await coordinator.create_knowledge_graph(data_file, args.output_path)
        
        if result["status"] == "completed":
            print(f"\n🎉 Knowledge Graph created successfully!")
            print(f"   - Saved to: {result['graph_path']}")
            if args.verbose and "stats" in result:
                stats = result["stats"]
                print(f"   - Graph Stats: {stats.get('nodes')} nodes, {stats.get('edges')} edges")
        else:
            print(f"\n❌ Error during graph creation: {result.get('error')}")

    elif args.command == "query":
        if not Path(args.graph_path).exists():
            print(f"❌ Error: Knowledge graph file not found: {args.graph_path}")
            return
            
        print(f"🔍 Processing query against {args.graph_path}...")
        result = await coordinator.query_knowledge_graph(args.query, args.graph_path)

        print("\n" + "="*60)
        if result["status"] == "completed":
            print("📝 Response:")
            print(result["response"])
            if args.verbose and "context" in result:
                print("\n" + "-"*20 + " Context " + "-"*20)
                print(result["context"])
        else:
            print(f"❌ Error during query processing: {result.get('error')}")
            if "response" in result:
                print(f"   - Assistant says: {result['response']}")
        print("="*60)


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
