import asyncio
import argparse
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

from product_planner_agent.product_planner_agent import run_agent_workflow

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Product Planner Agent workflow.")
    parser.add_argument("--figma-url", required=True, help="Full Figma URL including node-id.")
    parser.add_argument("--figma-api-key", required=True, help="Figma API Key.")
    
    args = parser.parse_args()
    return args

async def main():
    """Main async function to run the product planner workflow."""
    args = parse_args()

    print("🚀 Starting Product Planner Agent from script...")
    
    try:
        success = await run_agent_workflow(
            figma_url=args.figma_url,
            figma_api_key=args.figma_api_key
        )
        
        if success:
            print("\n✅ Workflow completed successfully via script.")
        else:
            print("\n❌ Workflow failed or completed with no result via script.")

    except Exception as e:
        print(f"💥 An unexpected error occurred during workflow execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 