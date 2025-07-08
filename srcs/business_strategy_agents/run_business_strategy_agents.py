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
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Import all business strategy MCPAgents
try:
    from .business_data_scout_agent import run_business_data_scout
    from .trend_analyzer_agent import run_trend_analysis  
    from .strategy_planner_agent import run_strategy_planning
    from .unified_business_strategy_agent import run_unified_business_strategy
except ImportError:
    # Direct imports if running as main
    import os
    sys.path.append(os.path.dirname(__file__))
    from business_data_scout_agent import run_business_data_scout
    from trend_analyzer_agent import run_trend_analysis
    from strategy_planner_agent import run_strategy_planning
    from unified_business_strategy_agent import run_unified_business_strategy
import aiohttp

# Helper function to create the HTTP client session
def get_http_session():
    return aiohttp.ClientSession()


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
        
        # Initialize agents with all required MCP URLs
        self.planner = StrategyPlannerMCPAgent(
            google_drive_mcp_url=self.google_drive_mcp_url,
            data_sourcing_mcp_url=self.data_sourcing_mcp_url
        )
        self.unifier = UnifiedBusinessStrategyMCPAgent(
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

    async def save_summary_report(self, summary_data: Dict, file_name: str):
        """Save execution results to JSON file on Google Drive via MCP"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"business_strategy_execution_{timestamp}.json"
        
        report_data = {
            "execution_timestamp": datetime.now().isoformat(),
            "agent_type": "Business Strategy MCPAgent",
            "architecture": "mcp_agent.app.MCPApp + mcp_agent.agents.agent.Agent",
            "results": self.results
        }
        
        report_content = json.dumps(report_data, indent=2, default=str)
        upload_url = f"{self.google_drive_mcp_url}/upload"
        payload = {"fileName": filename, "content": report_content}

        try:
            async with get_http_session() as session:
                async with session.post(upload_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            if result.get("success"):
                file_id = result.get('fileId')
                file_url = f"https://docs.google.com/document/d/{file_id}"
                print(f"📄 Execution report uploaded: {file_url}")
                return file_url
            else:
                raise Exception(f"MCP upload failed: {result.get('message')}")

        except Exception as e:
            print(f"❌ Failed to upload execution report: {e}")
            # Fallback or re-raise
            return f"upload_failed: {e}"


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
        results = await runner.run_full_suite(
            keywords=args["keywords"],
            business_context=args["business_context"],
            objectives=args["objectives"],
            regions=args["regions"],
            time_horizon=args["time_horizon"],
            mode=args["mode"]
        )
        
        # Save execution report
        await runner.save_execution_report()
        
        # Exit with appropriate code
        successful = results["summary"]["successful_agents"]
        total = results["summary"]["total_agents"]
        
        if successful == total:
            print("\n🎉 All business strategy agents executed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️  {total - successful} agent(s) had issues")
            sys.exit(1)
            
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