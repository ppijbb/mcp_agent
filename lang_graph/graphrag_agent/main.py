"""
Multi-Agent Graph RAG System - Main Entry Point

Usage:
  # 1. Create a knowledge graph from your data
  python main.py create-graph --data-file path/to/your/data.csv --output-path knowledge_graph.pkl

  # 2. Query the generated knowledge graph
  python main.py query --graph-path knowledge_graph.pkl --query "Your question here"

  # 3. Create visualizations for existing graph
  python main.py visualize --graph-path knowledge_graph.pkl

  # 4. Optimize existing graph quality
  python main.py optimize --graph-path knowledge_graph.pkl

  # 5. Export graph data in various formats
  python main.py export --graph-path knowledge_graph.pkl --format json

  # To create and use sample data:
  python main.py create-graph --create-sample-data --output-path sample_graph.pkl
  python main.py query --graph-path sample_graph.pkl --query "Who is the CEO of Apple?"
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from multi_agent_coordinator import MultiAgentCoordinator, MultiAgentConfig


def create_sample_data() -> str:
    """Create sample data file for testing and return its path."""
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
        print(f"✅ Sample data created: {sample_file}")
        return sample_file
    except ImportError:
        print("❌ Error: pandas is required. Install with: pip install pandas")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        sys.exit(1)


def validate_api_key(api_key: Optional[str]) -> str:
    """Validate and return the API key."""
    if not api_key:
        print("❌ Error: OpenAI API key is required. Set it via --openai-api-key or the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    return api_key


async def main():
    """Main function to handle CLI commands."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Graph RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py create-graph --create-sample-data --output-path sample_graph.pkl
  python main.py query --graph-path sample_graph.pkl --query "Who is the CEO of Apple?"
  python main.py visualize --graph-path sample_graph.pkl
  python main.py optimize --graph-path sample_graph.pkl
  python main.py export --graph-path sample_graph.pkl --format json
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Global arguments ---
    parser.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    # --- `create-graph` command ---
    parser_create = subparsers.add_parser("create-graph", help="Create a knowledge graph from data")
    parser_create.add_argument("--data-file", "-d", help="Path to data CSV file")
    parser_create.add_argument("--create-sample-data", action="store_true", help="Create and use a sample data file")
    parser_create.add_argument("--output-path", "-o", required=True, help="Path to save the generated knowledge graph")
    parser_create.add_argument("--no-visualization", action="store_true", help="Disable graph visualization")
    parser_create.add_argument("--no-optimization", action="store_true", help="Disable graph optimization")

    # --- `query` command ---
    parser_query = subparsers.add_parser("query", help="Query a knowledge graph")
    parser_query.add_argument("--graph-path", "-g", required=True, help="Path to the knowledge graph file")
    parser_query.add_argument("--query", "-q", required=True, help="Query to process")

    # --- `visualize` command ---
    parser_visualize = subparsers.add_parser("visualize", help="Create visualizations for existing graph")
    parser_visualize.add_argument("--graph-path", "-g", required=True, help="Path to the knowledge graph file")
    parser_visualize.add_argument("--output-name", help="Custom name for visualizations")

    # --- `optimize` command ---
    parser_optimize = subparsers.add_parser("optimize", help="Optimize existing graph quality")
    parser_optimize.add_argument("--graph-path", "-g", required=True, help="Path to the knowledge graph file")
    parser_optimize.add_argument("--graph-name", help="Custom name for optimization process")

    # --- `export` command ---
    parser_export = subparsers.add_parser("export", help="Export graph data in various formats")
    parser_export.add_argument("--graph-path", "-g", required=True, help="Path to the knowledge graph file")
    parser_export.add_argument("--format", "-f", choices=["json", "csv", "graphml"], default="json", help="Export format")
    parser_export.add_argument("--output-name", help="Custom name for exported files")

    # --- `status` command ---
    parser_status = subparsers.add_parser("status", help="Check system status and agent health")

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
    try:
        if args.command == "create-graph":
            await handle_create_graph(args, coordinator)
        elif args.command == "query":
            await handle_query(args, coordinator)
        elif args.command == "visualize":
            await handle_visualize(args, coordinator)
        elif args.command == "optimize":
            await handle_optimize(args, coordinator)
        elif args.command == "export":
            await handle_export(args, coordinator)
        elif args.command == "status":
            await handle_status(args, coordinator)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def handle_create_graph(args, coordinator):
    """Handle the create-graph command."""
    data_file = args.data_file
    if args.create_sample_data:
        data_file = create_sample_data()
    
    if not data_file:
        print("❌ Error: --data-file or --create-sample-data is required for create-graph.")
        sys.exit(1)
    
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)
    
    # Determine visualization and optimization settings
    enable_visualization = not args.no_visualization
    enable_optimization = not args.no_optimization
    
    print("🚀 Starting knowledge graph creation...")
    if enable_visualization:
        print("   📊 Graph visualization: enabled")
    if enable_optimization:
        print("   ⚡ Graph optimization: enabled")
    
    result = await coordinator.create_knowledge_graph(
        data_file, 
        args.output_path,
        enable_visualization=enable_visualization,
        enable_optimization=enable_optimization
    )
    
    if result["status"] == "completed":
        print(f"\n🎉 Knowledge Graph created successfully!")
        print(f"   - Saved to: {result['graph_path']}")
        
        if args.verbose and "stats" in result:
            stats = result["stats"]
            print(f"   - Graph Stats: {stats.get('nodes')} nodes, {stats.get('edges')} edges")
            if 'entity_types' in stats:
                print(f"   - Entity Types: {stats.get('entity_types')}")
                print(f"   - Relationship Types: {stats.get('relationship_types')}")
        
        # Show optimization results
        if "optimization" in result and result["optimization"]:
            opt_result = result["optimization"]
            if opt_result["status"] == "completed":
                print(f"   - Optimization Quality Score: {opt_result['overall_quality']:.3f}")
                if opt_result.get('meets_threshold'):
                    print("   - ✅ Graph meets quality threshold")
                else:
                    print("   - ⚠️  Graph below quality threshold")
        
        # Show visualization results
        if "visualization" in result and result["visualization"]:
            viz_result = result["visualization"]
            if viz_result["status"] == "completed":
                print(f"   - Visualizations created: {len(viz_result['visualizations'])} types")
                for viz_type, viz_data in viz_result["visualizations"].items():
                    if "paths" in viz_data:
                        print(f"     - {viz_type}: {len(viz_data['paths'])} files")
    else:
        print(f"\n❌ Error during graph creation: {result.get('error')}")
        sys.exit(1)


async def handle_query(args, coordinator):
    """Handle the query command."""
    if not Path(args.graph_path).exists():
        print(f"❌ Error: Knowledge graph file not found: {args.graph_path}")
        sys.exit(1)
        
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
        sys.exit(1)
    print("="*60)


async def handle_visualize(args, coordinator):
    """Handle the visualize command."""
    if not Path(args.graph_path).exists():
        print(f"❌ Error: Knowledge graph file not found: {args.graph_path}")
        sys.exit(1)
    
    print(f"📊 Creating visualizations for {args.graph_path}...")
    result = await coordinator.create_graph_visualizations(args.graph_path, args.output_name)
    
    if result["status"] == "completed":
        print("✅ Visualizations created successfully!")
        print(f"   - Graph: {result['graph_name']}")
        print(f"   - Timestamp: {result['timestamp']}")
        
        if "visualizations" in result:
            for viz_type, viz_data in result["visualizations"].items():
                print(f"   - {viz_type}:")
                if "paths" in viz_data:
                    for fmt, path in viz_data["paths"].items():
                        print(f"     - {fmt.upper()}: {path}")
                if "node_count" in viz_data:
                    print(f"     - Nodes: {viz_data['node_count']}, Edges: {viz_data['edges_count']}")
    else:
        print(f"❌ Error during visualization: {result.get('error')}")
        sys.exit(1)


async def handle_optimize(args, coordinator):
    """Handle the optimize command."""
    if not Path(args.graph_path).exists():
        print(f"❌ Error: Knowledge graph file not found: {args.graph_path}")
        sys.exit(1)
    
    print(f"⚡ Optimizing graph quality for {args.graph_path}...")
    result = await coordinator.optimize_existing_graph(args.graph_path, args.graph_name)
    
    if result["status"] == "completed":
        print("✅ Graph optimization completed successfully!")
        print(f"   - Overall Quality Score: {result['overall_quality']:.3f}")
        
        if result.get('meets_threshold'):
            print("   - ✅ Graph meets quality threshold")
        else:
            print("   - ⚠️  Graph below quality threshold")
        
        # Show detailed results
        if "optimization_results" in result:
            for opt_type, opt_data in result["optimization_results"].items():
                print(f"   - {opt_type}: {opt_data.get('quality_score', 0):.3f}")
        
        # Show recommendations
        if "optimization_report" in result and "summary" in result["optimization_report"]:
            report_path = result["optimization_report"]["path"]
            print(f"   - Detailed report: {report_path}")
    else:
        print(f"❌ Error during optimization: {result.get('error')}")
        sys.exit(1)


async def handle_export(args, coordinator):
    """Handle the export command."""
    if not Path(args.graph_path).exists():
        print(f"❌ Error: Knowledge graph file not found: {args.graph_path}")
        sys.exit(1)
    
    print(f"📤 Exporting graph data from {args.graph_path} in {args.format.upper()} format...")
    result = await coordinator.export_graph_data(args.graph_path, args.format, args.output_name)
    
    if result["status"] == "completed":
        print("✅ Graph export completed successfully!")
        print(f"   - Format: {result['format'].upper()}")
        print(f"   - Nodes: {result['node_count']}")
        print(f"   - Edges: {result['edge_count']}")
        
        if "path" in result:
            print(f"   - Output: {result['path']}")
        elif "paths" in result:
            print("   - Output files:")
            for file_type, file_path in result["paths"].items():
                print(f"     - {file_type}: {file_path}")
    else:
        print(f"❌ Error during export: {result.get('error')}")
        sys.exit(1)


async def handle_status(args, coordinator):
    """Handle the status command."""
    print("🔍 Checking system status...")
    
    # Get agent status
    agent_status = coordinator.get_agent_status()
    print("\n📋 Agent Status:")
    for agent, status in agent_status.items():
        if agent != "config":
            print(f"   - {agent}: {status}")
    
    # Get system health
    health = await coordinator.health_check()
    print(f"\n🏥 System Health: {health['status']}")
    print(f"   - API Connectivity: {health['api_connectivity']}")
    print(f"   - Visualization: {health['visualization_status']}")
    print(f"   - Optimization: {health['optimization_status']}")
    print(f"   - Timestamp: {health['timestamp']}")
    
    if args.verbose and "agents" in health and "config" in health["agents"]:
        print("\n⚙️  Configuration:")
        config = health["agents"]["config"]
        for key, value in config.items():
            print(f"   - {key}: {value}")


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
