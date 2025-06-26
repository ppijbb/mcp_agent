import asyncio
import argparse
import json
from typing import List, Dict, Any

# Ensure the script can find the runner
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from business_strategy_agents.run_business_strategy_agents import BusinessStrategyRunner

def parse_args() -> Dict[str, Any]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Business Strategy Agents via script.")
    parser.add_argument("--keywords", required=True, help="Comma-separated keywords for analysis.")
    parser.add_argument("--business-context", type=str, default='{}', help="JSON string for business context.")
    parser.add_argument("--objectives", type=str, default='', help="Comma-separated objectives.")
    parser.add_argument("--regions", type=str, default='', help="Comma-separated regions.")
    parser.add_argument("--time-horizon", default="12_months", help="Time horizon for analysis.")
    parser.add_argument("--mode", default="unified", choices=["unified", "individual", "both"], help="Execution mode.")
    parser.add_argument("--output-dir", default="business_strategy_reports", help="Directory to save reports.")
    
    args = parser.parse_args()
    
    return {
        "keywords": [k.strip() for k in args.keywords.split(',') if k.strip()],
        "business_context": json.loads(args.business_context),
        "objectives": [o.strip() for o in args.objectives.split(',') if o.strip()] if args.objectives else None,
        "regions": [r.strip() for r in args.regions.split(',') if r.strip()] if args.regions else None,
        "time_horizon": args.time_horizon,
        "mode": args.mode,
        "output_dir": args.output_dir
    }

async def main():
    """Main function to run the business strategy analysis."""
    args = parse_args()
    
    runner = BusinessStrategyRunner(output_dir=args["output_dir"])
    
    print("🚀 Starting Business Strategy Analysis Script...")
    
    results = await runner.run_full_suite(
        keywords=args["keywords"],
        business_context=args["business_context"],
        objectives=args["objectives"],
        regions=args["regions"],
        time_horizon=args["time_horizon"],
        mode=args["mode"]
    )
    
    report_file = runner.save_execution_report()
    print(f"\n✅ Analysis complete. Final report saved to {report_file}")
    
if __name__ == "__main__":
    asyncio.run(main()) 