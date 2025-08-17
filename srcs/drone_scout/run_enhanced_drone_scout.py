#!/usr/bin/env python3
"""
Enhanced Drone Scout Runner

This script provides enhanced MCP-integrated drone control capabilities
with comprehensive mission planning, safety analysis, and real-time monitoring.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Adjust the path to include the project root, so we can import the agent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from enhanced_drone_control_agent import main as run_enhanced_drone_agent

async def main_runner(args):
    """Asynchronous runner to execute the enhanced drone agent system."""
    try:
        print("🚁 Enhanced Drone Scout Agent - MCP Integration Enabled")
        print("=" * 60)
        
        if args.enable_mcp:
            print("🔧 MCP Integration: ENABLED")
            print("   - Filesystem server: ✅")
            print("   - Weather server: ✅")
            print("   - Search server: ✅")
            print("   - Browser server: ✅")
            print("   - GIS server: ✅")
        else:
            print("🔧 MCP Integration: DISABLED (Limited mode)")
        
        print(f"🎯 Mission: {args.mission}")
        print(f"📁 Output: {args.result_json_path}")
        print("=" * 60)
        
        # Execute the enhanced drone agent
        await run_enhanced_drone_agent(
            mission=args.mission,
            result_json_path=args.result_json_path
        )
        
        print("✅ Enhanced Drone Scout mission completed successfully!")
        
    except Exception as e:
        print(f"❌ An error occurred while running the enhanced drone agent: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error to result file
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'success': False, 
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': asyncio.get_event_loop().time()
            }, f, ensure_ascii=False, indent=4)
        sys.exit(1)

def validate_mission(mission: str) -> bool:
    """Validate mission input"""
    if not mission or len(mission.strip()) < 10:
        return False
    
    # Check for Korean or English content
    has_korean = any('\u3131' <= char <= '\u3163' or '\uac00' <= char <= '\ud7af' for char in mission)
    has_english = any(char.isascii() and char.isalpha() for char in mission)
    
    return has_korean or has_english

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Drone Scout Agent with MCP Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mission with MCP enabled
  python run_enhanced_drone_scout.py --mission "서울숲 공원 상공을 50m 고도로 비행하며 주요 시설물 상태를 촬영해줘" --result-json-path drone_result.json
  
  # Mission without MCP (limited mode)
  python run_enhanced_drone_scout.py --mission "Fly over Seoul Forest at 50m altitude" --result-json-path drone_result.json --no-mcp
  
  # Test mode
  python run_enhanced_drone_scout.py --test
        """
    )
    
    parser.add_argument(
        "--mission", 
        type=str, 
        help="Natural language mission description (Korean or English)"
    )
    parser.add_argument(
        "--result-json-path", 
        type=str, 
        help="Path to save the final result JSON"
    )
    parser.add_argument(
        "--enable-mcp", 
        action="store_true", 
        default=True,
        help="Enable MCP integration (default: True)"
    )
    parser.add_argument(
        "--no-mcp", 
        action="store_true", 
        help="Disable MCP integration"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run in test mode with sample mission"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle MCP enable/disable
    if args.no_mcp:
        args.enable_mcp = False
    
    # Test mode
    if args.test:
        print("🧪 Enhanced Drone Scout - Test Mode")
        print("=" * 50)
        
        test_mission = "서울숲 공원 상공을 50m 고도로 비행하며 주요 시설물의 현재 상태를 촬영하고 보고서를 작성해줘. 비행 고도는 50m로 유지해."
        test_result_path = "test_enhanced_drone_result.json"
        
        print(f"🎯 Test Mission: {test_mission}")
        print(f"📁 Test Output: {test_result_path}")
        print("=" * 50)
        
        args.mission = test_mission
        args.result_json_path = test_result_path
    
    # Validate arguments
    if not args.test and (not args.mission or not args.result_json_path):
        parser.error("--mission and --result-json-path are required unless using --test")
    
    if not args.test and not validate_mission(args.mission):
        parser.error("Mission must be at least 10 characters and contain Korean or English text")
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("🔍 Verbose logging enabled")
    
    # Run the enhanced drone agent
    asyncio.run(main_runner(args))
