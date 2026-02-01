#!/usr/bin/env python3
"""
Main entry point for the Genome Agent

This module provides the main interface for running genome analysis workflows
and managing the genome agent system.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))

from genome_agent import GenomeAgentMCP
from config import get_config


async def main():
    """Main function for the Genome Agent"""

    print("🧬 Genome Agent - Starting up...")
    print("=" * 50)

    try:
        # Load configuration
        config = get_config()
        print(f"✅ Configuration loaded: {config.agent_name} v{config.version}")
        print(f"🌍 Environment: {config.environment}")
        print(f"📁 Output directory: {config.analysis.default_output_dir}")

        # Create agent instance
        agent = GenomeAgentMCP(
            output_dir=config.analysis.default_output_dir,
            enable_mcp=True
        )
        print("✅ Genome agent created successfully")

        # Initialize MCP connections
        await agent._initialize_mcp_connections()
        print("✅ MCP connections initialized")

        # Display agent capabilities
        print("\n🔬 Agent Capabilities:")
        print(f"   • Supported databases: {len(config.get_active_databases())}")
        print(f"   • Supported tools: {len(config.get_active_tools())}")
        print(f"   • MCP servers: {len(config.get_active_mcp_servers())}")

        # Example analysis
        print("\n🚀 Running example analysis...")
        example_request = "Analyze genetic variants in human genome for disease risk assessment"

        result = await agent.run_workflow(
            analysis_request=example_request,
            enable_research=True,
            execute_plan=False
        )

        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
        else:
            print("✅ Example analysis completed successfully!")
            print(f"📋 Generated plan ID: {result.get('plan', {}).get('plan_id', 'N/A')}")

        # Cleanup
        await agent.cleanup()
        print("✅ Genome agent shutdown completed")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
