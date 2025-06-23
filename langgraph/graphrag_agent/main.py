"""
Multi-Agent Graph RAG System - Main Entry Point

Usage:
    python main.py --query "Your question here" --data-file path/to/data.csv
"""

import asyncio
import argparse
import os
from pathlib import Path
from typing import Optional

from multi_agent_coordinator import MultiAgentCoordinator, MultiAgentConfig


def create_sample_data():
    """Create sample data file for testing"""
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
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-Agent Graph RAG System")
    parser.add_argument("--query", "-q", required=True, help="Query to process")
    parser.add_argument("--data-file", "-d", help="Path to data CSV file")
    parser.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample data file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --openai-api-key")
        return
    
    # Create sample data if requested
    if args.create_sample_data:
        sample_file = create_sample_data()
        if not args.data_file:
            args.data_file = sample_file
    
    # Validate data file
    if not args.data_file:
        print("❌ Error: Data file is required. Use --data-file or --create-sample-data")
        return
    
    if not Path(args.data_file).exists():
        print(f"❌ Error: Data file not found: {args.data_file}")
        return
    
    # Configure multi-agent system
    config = MultiAgentConfig(
        openai_api_key=openai_api_key,
        data_file_path=args.data_file,
        graph_model_name="gpt-4o-mini",
        rag_model_name="gpt-4o-mini",
        max_search_results=5,
        context_window_size=4000
    )
    
    # Initialize coordinator
    print("🚀 Initializing Multi-Agent Graph RAG System...")
    coordinator = MultiAgentCoordinator(config)
    
    # Check agent status
    if args.verbose:
        status = coordinator.get_agent_status()
        print(f"📊 Agent Status: {status}")
    
    print(f"🔍 Processing query: {args.query}")
    print(f"📁 Using data file: {args.data_file}")
    print("=" * 60)
    
    try:
        # Process the query
        result = await coordinator.process_query(
            user_query=args.query,
            thread_id=f"session_{asyncio.get_event_loop().time()}"
        )
        
        # Display results
        print("\n🎉 Processing Complete!")
        print("=" * 60)
        print("📝 Response:")
        print(result["response"])
        
        if args.verbose:
            print("\n📊 System Statistics:")
            print(f"Status: {result['status']}")
            
            if result.get("knowledge_graph_stats"):
                stats = result["knowledge_graph_stats"]
                print(f"Knowledge Graph: {stats['nodes']} nodes, {stats['edges']} edges")
                print(f"Entity Types: {stats['entity_types']}, Relationship Types: {stats['relationship_types']}")
            
            if result.get("agent_communications"):
                print(f"\n🤝 Agent Communications ({len(result['agent_communications'])} messages):")
                for comm in result["agent_communications"]:
                    print(f"  {comm['from']} → {comm['to']}: {comm['message']}")
        
        if result.get("error"):
            print(f"\n⚠️  Errors encountered: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Enable better error handling for asyncio
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 