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


class BusinessStrategyRunner:
    """
    Runner for all business strategy MCPAgents.
    Provides unified access to comprehensive business intelligence capabilities.
    """
    
    def __init__(self, output_dir: str = "business_strategy_reports"):
        self.output_dir = output_dir
        self.results = {}
        
    async def run_individual_agents(self, 
                                  keywords: List[str],
                                  business_context: Dict[str, Any] = None,
                                  objectives: List[str] = None,
                                  regions: List[str] = None,
                                  time_horizon: str = "12_months") -> Dict[str, Any]:
        """Run each business strategy MCPAgent individually"""
        
        print("🚀 Running Individual Business Strategy Agents...")
        print("=" * 60)
        
        # 1. Business Data Scout
        print("\n📊 1. BUSINESS DATA SCOUT MCPAgent")
        print("-" * 40)
        try:
            scout_result = await run_business_data_scout(
                keywords=keywords,
                regions=regions,
                output_dir=self.output_dir
            )
            self.results["data_scout"] = scout_result
            
            if scout_result["success"]:
                print(f"✅ Data Scout completed: {scout_result['output_file']}")
            else:
                print(f"❌ Data Scout failed: {scout_result['error']}")
                
        except Exception as e:
            print(f"❌ Data Scout error: {e}")
            self.results["data_scout"] = {"success": False, "error": str(e)}
        
        # 2. Trend Analyzer
        print("\n📈 2. TREND ANALYZER MCPAgent")
        print("-" * 40)
        try:
            trend_result = await run_trend_analysis(
                focus_areas=keywords,
                time_horizon=time_horizon,
                output_dir=self.output_dir
            )
            self.results["trend_analyzer"] = trend_result
            
            if trend_result["success"]:
                print(f"✅ Trend Analyzer completed: {trend_result['output_file']}")
            else:
                print(f"❌ Trend Analyzer failed: {trend_result['error']}")
                
        except Exception as e:
            print(f"❌ Trend Analyzer error: {e}")
            self.results["trend_analyzer"] = {"success": False, "error": str(e)}
        
        # 3. Strategy Planner
        print("\n🎯 3. STRATEGY PLANNER MCPAgent")
        print("-" * 40)
        try:
            strategy_result = await run_strategy_planning(
                business_context=business_context or {"description": "General business"},
                objectives=objectives or ["growth", "efficiency"],
                output_dir=self.output_dir
            )
            self.results["strategy_planner"] = strategy_result
            
            if strategy_result["success"]:
                print(f"✅ Strategy Planner completed: {strategy_result['output_file']}")
            else:
                print(f"❌ Strategy Planner failed: {strategy_result['error']}")
                
        except Exception as e:
            print(f"❌ Strategy Planner error: {e}")
            self.results["strategy_planner"] = {"success": False, "error": str(e)}
        
        return self.results
    
    async def run_unified_agent(self,
                              keywords: List[str],
                              business_context: Dict[str, Any] = None,
                              objectives: List[str] = None,
                              regions: List[str] = None,
                              time_horizon: str = "12_months") -> Dict[str, Any]:
        """Run the unified business strategy MCPAgent"""
        
        print("\n🎉 UNIFIED BUSINESS STRATEGY MCPAgent")
        print("=" * 60)
        print("Complete integrated business intelligence analysis!")
        
        try:
            unified_result = await run_unified_business_strategy(
                keywords=keywords,
                business_context=business_context,
                objectives=objectives,
                regions=regions,
                time_horizon=time_horizon,
                output_dir=self.output_dir
            )
            self.results["unified_strategy"] = unified_result
            
            if unified_result["success"]:
                print(f"✅ Unified Strategy completed: {unified_result['output_file']}")
                print("🎉 SUCCESS: Business strategy analysis completed successfully!")
            else:
                print(f"❌ Unified Strategy failed: {unified_result['error']}")
                
        except Exception as e:
            print(f"❌ Unified Strategy error: {e}")
            self.results["unified_strategy"] = {"success": False, "error": str(e)}
        
        return unified_result
    
    async def run_full_suite(self,
                           keywords: List[str],
                           business_context: Dict[str, Any] = None,
                           objectives: List[str] = None,
                           regions: List[str] = None,
                           time_horizon: str = "12_months",
                           mode: str = "unified") -> Dict[str, Any]:
        """
        Run the complete business strategy MCPAgent suite
        
        Args:
            mode: "individual", "unified", or "both"
        """
        
        print("🎯 BUSINESS STRATEGY MCPAGENT SUITE")
        print("=" * 60)
        print(f"Keywords: {', '.join(keywords)}")
        print(f"Regions: {', '.join(regions) if regions else 'Global'}")
        print(f"Time Horizon: {time_horizon}")
        print(f"Execution Mode: {mode}")
        print(f"Output Directory: {self.output_dir}")
        print()
        
        start_time = datetime.now()
        
        if mode in ["individual", "both"]:
            await self.run_individual_agents(
                keywords, business_context, objectives, regions, time_horizon
            )
        
        if mode in ["unified", "both"]:
            await self.run_unified_agent(
                keywords, business_context, objectives, regions, time_horizon
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        
        successful_agents = sum(1 for result in self.results.values() 
                              if result.get("success", False))
        total_agents = len(self.results)
        
        print(f"✅ Successful Agents: {successful_agents}/{total_agents}")
        print(f"⏱️  Total Execution Time: {duration:.2f} seconds")
        print(f"📁 Output Directory: {self.output_dir}")
        
        # List all generated reports
        print("\n📄 Generated Reports:")
        for agent_name, result in self.results.items():
            if result.get("success") and "output_file" in result:
                print(f"  • {agent_name}: {result['output_file']}")
        
        if successful_agents == total_agents:
            print("\n🎉 ALL BUSINESS STRATEGY AGENTS EXECUTED SUCCESSFULLY!")
            print("✨ Complete business intelligence analysis finished!")
        else:
            print(f"\n⚠️  {total_agents - successful_agents} agent(s) had issues")
        
        return {
            "summary": {
                "successful_agents": successful_agents,
                "total_agents": total_agents,
                "execution_time": duration,
                "output_directory": self.output_dir
            },
            "results": self.results
        }
    
    def save_execution_report(self, filename: str = None) -> str:
        """Save execution results to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/business_strategy_execution_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report_data = {
            "execution_timestamp": datetime.now().isoformat(),
            "agent_type": "Business Strategy MCPAgent",
            "architecture": "mcp_agent.app.MCPApp + mcp_agent.agents.agent.Agent",
            "results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Execution report saved: {filename}")
        return filename


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
  --output-dir         Output directory for reports

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
    --output-dir "reports"

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
        "output_dir": "business_strategy_reports"
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
        elif arg == "--output-dir" and i + 1 < len(sys.argv):
            args["output_dir"] = sys.argv[i + 1]
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
    runner = BusinessStrategyRunner(output_dir=args["output_dir"])
    
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
        runner.save_execution_report()
        
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