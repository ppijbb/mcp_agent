import argparse
import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.basic_agents.workflow_orchestration import run_workflow

async def main():
    """Workflow Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Workflow Orchestrator from the command line.")
    parser.add_argument("--task", required=True, help="The task description for the workflow.")
    parser.add_argument("--model", default="gpt-4o-mini", help="The model to use for the workflow.")
    parser.add_argument("--plan-type", default="full", choices=['full', 'step', 'none'], help="The planning type for the orchestrator.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")
    
    args = parser.parse_args()

    print(f"🔄 Starting Workflow Orchestrator...")
    print(f"   - Task: '{args.task[:50]}...'")
    print(f"   - Model: {args.model}")
    print(f"   - Plan Type: {args.plan_type}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_result = {"success": False, "result": None, "error": None}

    try:
        result_content = await run_workflow(
            task=args.task,
            model_name=args.model,
            plan_type=args.plan_type
        )
        
        print("✅ Workflow finished successfully.")
        final_result["success"] = True
        final_result["result"] = result_content

    except Exception as e:
        print(f"❌ An error occurred during workflow execution: {e}")
        final_result["error"] = str(e)
    
    finally:
        print(f"💾 Saving final results to {result_json_path}...")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print("🎉 Results saved.")
        if not final_result["success"]:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 