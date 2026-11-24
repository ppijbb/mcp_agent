import argparse
import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.advanced_agents.decision_agent import run_simplified_decision_analysis  # noqa: E402


async def main():
    """Decision Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Decision Agent with simplified inputs.")
    parser.add_argument("--user-id", required=True, help="The user ID for the analysis.")
    parser.add_argument("--interaction-type", required=True, help="The type of interaction (e.g., '구매').")
    parser.add_argument("--context-json", required=True, help="The interaction context as a JSON string.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    print("🔄 Starting Decision Agent...")
    print(f"   - User ID: {args.user_id}")
    print(f"   - Interaction: {args.interaction_type}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    final_result = {"success": False, "data": None, "error": None}

    try:
        analysis_result = await run_simplified_decision_analysis(
            user_id=args.user_id,
            interaction_type=args.interaction_type,
            context_json=args.context_json
        )

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
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print("🎉 Results saved.")
        except Exception as e:
            print(f"❌ Failed to save result JSON: {e}")
            final_result["success"] = False
            final_result["error"] = f"Failed to save result JSON: {e}"

        if not final_result["success"]:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
