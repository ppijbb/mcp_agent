import argparse
import asyncio
import json
import sys
from pathlib import Path

# Adjust the path to include the project root, so we can import the agent
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.drone_scout.drone_control_agent import main as run_drone_agent_system

async def main_runner(args):
    """Asynchronous runner to execute the drone agent system."""
    try:
        # We will refactor `run_drone_agent_system` to accept these args
        await run_drone_agent_system(
            mission=args.mission,
            result_json_path=args.result_json_path
        )
    except Exception as e:
        print(f"An error occurred while running the drone agent: {e}")
        # Save error to result file
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump({'success': False, 'error': str(e)}, f, ensure_ascii=False, indent=4)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Drone Scout Agent.")
    parser.add_argument("--mission", type=str, required=True, help="Natural language mission description.")
    parser.add_argument("--result-json-path", type=str, required=True, help="Path to save the final result JSON.")
    
    args = parser.parse_args()
    
    # The agent's main function is async, so we run it in an event loop.
    asyncio.run(main_runner(args)) 