import argparse
import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.product_planner_agent.coordinators.executive_coordinator import ExecutiveCoordinator

async def main():
    """Product Planner 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Product Planner with a Figma URL.")
    parser.add_argument("--figma-url", required=True, help="The Figma URL to analyze.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")
    
    args = parser.parse_args()

    print(f"🔄 Starting Product Planner...")
    print(f"   - Figma URL: {args.figma_url}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_result = {"success": False, "data": None, "error": None}

    try:
        coordinator = ExecutiveCoordinator()
        analysis_result = await coordinator.run_with_figma_url(figma_url=args.figma_url)
        
        print("✅ Agent finished successfully.")
        final_result["success"] = True
        final_result["data"] = analysis_result

    except Exception as e:
        import traceback
        error_msg = f"❌ An error occurred during agent execution: {e}\n{traceback.format_exc()}"
        print(error_msg)
        final_result["error"] = str(e)
    
    finally:
        print(f"💾 Saving final results to {result_json_path}...")
        try:
            with open(result_json_path, 'w', encoding='utf-8') as f:
                # The result can be complex, ensure default handler for non-serializable objects
                json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)
            print("🎉 Results saved.")
        except Exception as e:
            print(f"❌ Failed to save result JSON: {e}")
            final_result["success"] = False
            final_result["error"] = f"Failed to save result JSON: {e}"
        
        if not final_result["success"]:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 