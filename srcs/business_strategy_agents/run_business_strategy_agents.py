#!/usr/bin/env python3
"""
Business Strategy Agents Runner
--------------------------------
Execute all business strategy MCPAgents for comprehensive business intelligence.
This script provides unified access to all business strategy analysis capabilities.
"""

import asyncio
import sys
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from .strategy_planner_agent import StrategyPlanner
from .unified_business_strategy_agent import UnifiedBusinessStrategy

 


class BusinessStrategyRunner:
    """
    Runner for all business strategy MCPAgents.
    Provides unified access to comprehensive business intelligence capabilities.
    """
    
    def __init__(self, 
                 google_drive_mcp_url: str = "http://localhost:3001",
                 data_sourcing_mcp_url: str = "http://localhost:3005"):
        self.google_drive_mcp_url = google_drive_mcp_url
        self.data_sourcing_mcp_url = data_sourcing_mcp_url
        
        # Initialize dependency-light agents
        self.planner = StrategyPlanner(
            google_drive_mcp_url=self.google_drive_mcp_url,
            data_sourcing_mcp_url=self.data_sourcing_mcp_url
        )
        self.unifier = UnifiedBusinessStrategy(
            google_drive_mcp_url=self.google_drive_mcp_url,
            data_sourcing_mcp_url=self.data_sourcing_mcp_url
        )

    async def run_agents(self, 
                       industry: str,
                       company_profile: str,
                       competitors: List[str],
                       tech_trends: List[str]) -> Dict[str, Any]:
        """
        Runs the sequence of business strategy agents.
        """
        print("Running Strategy Planner Agent...")
        planner_result = await self.planner.analyze_market_and_business(
            industry=industry,
            company_info=company_profile,
            competitors=competitors
        )
        
        # In a real scenario, you might have more complex logic to pass
        # results from one agent to the next.
        
        print("\nRunning Unified Business Strategy Agent...")
        unifier_result = await self.unifier.develop_strategy(
            industry=industry,
            company_profile=company_profile,
            competitors=competitors,
            tech_trends=tech_trends
        )
        
        # Save individual reports
        await self.planner.save_report(planner_result, f"strategy_plan_{industry}.json")
        await self.unifier.save_report(unifier_result, f"unified_strategy_{industry}.json")

        final_summary = {
            "planner_output": planner_result,
            "unifier_output": unifier_result
        }
        
        # Save the final summary report
        await self.save_summary_report(final_summary, f"final_summary_{industry}.json")
        
        return final_summary

    async def save_summary_report(self, summary_data: Dict, file_name: str) -> str:
        """Save execution results to local JSON file in reports directory."""
        output_dir = "business_strategy_reports"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, file_name)
        report_data = {
            "execution_timestamp": datetime.now().isoformat(),
            "runner": "BusinessStrategyRunner",
            "results": summary_data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Execution report saved: {path}")
        return path


# CLI Functions
def print_usage():
    """Print usage instructions"""
    print("""
Usage: python run_business_strategy_agents.py <keywords> [options]

Arguments:
  keywords              Comma-separated keywords (required)
  
Options:
  --business-context    Business context description
  --objectives          Comma-separated objectives
  --regions            Comma-separated regions
  --time-horizon       Time horizon (3_months, 6_months, 12_months, 24_months)
  --mode               Execution mode (individual, unified, both)
  --mcp-url            MCP server URL (default: http://localhost:3001)

Examples:
  # Basic usage
  python run_business_strategy_agents.py "AI,fintech"
  
  # Full configuration
  python run_business_strategy_agents.py "AI,fintech,sustainability" \\
    --business-context "Tech startup in AI space" \\
    --objectives "growth,expansion,efficiency" \\
    --regions "North America,Europe" \\
    --time-horizon "12_months" \\
    --mode "unified" \\
    --mcp-url "http://your-mcp-server:3001"

Modes:
  individual    Run each MCPAgent separately
  unified       Run the unified MCPAgent (recommended)
  both          Run both individual and unified agents
    """)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    args = {
        "keywords": [k.strip() for k in sys.argv[1].split(',')],
        "business_context": None,
        "objectives": None,
        "regions": None,
        "time_horizon": "12_months",
        "mode": "unified",
        "mcp_url": "http://localhost:3001"
    }
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--business-context" and i + 1 < len(sys.argv):
            args["business_context"] = {"description": sys.argv[i + 1]}
            i += 2
        elif arg == "--objectives" and i + 1 < len(sys.argv):
            args["objectives"] = [o.strip() for o in sys.argv[i + 1].split(',')]
            i += 2
        elif arg == "--regions" and i + 1 < len(sys.argv):
            args["regions"] = [r.strip() for r in sys.argv[i + 1].split(',')]
            i += 2
        elif arg == "--time-horizon" and i + 1 < len(sys.argv):
            args["time_horizon"] = sys.argv[i + 1]
            i += 2
        elif arg == "--mode" and i + 1 < len(sys.argv):
            args["mode"] = sys.argv[i + 1]
            i += 2
        elif arg == "--mcp-url" and i + 1 < len(sys.argv):
            args["mcp_url"] = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    return args


# Main execution
async def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_args()
    
    # Validate mode
    valid_modes = ["individual", "unified", "both"]
    if args["mode"] not in valid_modes:
        print(f"❌ Invalid mode: {args['mode']}")
        print(f"Valid modes: {', '.join(valid_modes)}")
        sys.exit(1)
    
    # Validate time horizon
    valid_horizons = ["3_months", "6_months", "12_months", "24_months"]
    if args["time_horizon"] not in valid_horizons:
        print(f"❌ Invalid time horizon: {args['time_horizon']}")
        print(f"Valid horizons: {', '.join(valid_horizons)}")
        sys.exit(1)
    
    # Create runner and execute
    runner = BusinessStrategyRunner(google_drive_mcp_url=args["mcp_url"])
    
    try:
        results = await runner.run_agents(
            industry=args["keywords"][0] if args["keywords"] else "General",
            company_profile=args["business_context"]["description"] if args["business_context"] else "Business analysis",
            competitors=args["keywords"][1:] if len(args["keywords"]) > 1 else [],
            tech_trends=args["keywords"]
        )
        
        # Save execution report
        await runner.save_summary_report(results, f"final_summary_{args['keywords'][0] if args['keywords'] else 'general'}.json")
        
        print("\n🎉 All business strategy agents executed successfully!")
        sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import required modules
    import os
    
    # Run the main function
    asyncio.run(main()) 