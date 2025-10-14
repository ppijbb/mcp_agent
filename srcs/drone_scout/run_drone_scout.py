import argparse
import asyncio
import json
import sys
from pathlib import Path

# Adjust the path to include the project root, so we can import the agent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from drone_scout_agent import create_drone_scout_agent
from drone_config import get_drone_config

async def main_runner(args):
    """Asynchronous runner to execute the drone agent system."""
    try:
        # Get configuration
        drone_config = get_drone_config()
        
        print("🚁 Drone Scout Agent - Advanced Autonomous Drone Fleet Control")
        print("=" * 70)
        print(f"🎯 Mission: {args.mission}")
        print(f"📁 Output Directory: {drone_config.output_dir}")
        print(f"🚁 Fleet Size: {drone_config.default_fleet_size}")
        print(f"🏢 Mission Control: {drone_config.mission_control_center}")
        print("=" * 70)
        
        # Create and run drone scout agent
        agent = create_drone_scout_agent()
        
        # Prepare mission context
        context = {
            "mission": args.mission,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Execute mission
        result = await agent.run(context)
        
        # Save result to file
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print("✅ Drone Scout mission completed successfully!")
        print(f"📄 Results saved to: {args.result_json_path}")
        
    except Exception as e:
        print(f"❌ An error occurred while running the drone agent: {e}")
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
        description="Drone Scout Agent - Advanced Autonomous Drone Fleet Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mission
  python run_drone_scout.py --mission "서울숲 공원 상공을 50m 고도로 비행하며 주요 시설물 상태를 촬영해줘" --result-json-path drone_result.json
  
  # Test mode
  python run_drone_scout.py --test
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
    
    # Test mode
    if args.test:
        print("🧪 Drone Scout Agent - Test Mode")
        print("=" * 50)
        
        test_mission = "서울숲 공원 상공을 50m 고도로 비행하며 주요 시설물의 현재 상태를 촬영하고 보고서를 작성해줘. 비행 고도는 50m로 유지해."
        test_result_path = "test_drone_result.json"
        
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
    
    # Run the drone agent
    asyncio.run(main_runner(args)) 