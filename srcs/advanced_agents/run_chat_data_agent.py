import argparse
import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.advanced_agents.enhanced_data_generator import SyntheticDataAgent

async def main():
    """Chat Data Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the SyntheticDataAgent from the command line for chat-based generation.")
    parser.add_argument("--data-type", required=True, help="The type of data to generate (e.g., 'customer').")
    parser.add_argument("--record-count", required=True, type=int, help="The number of records to generate.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")
    
    args = parser.parse_args()

    print(f"🔄 Starting Chat Data Agent...")
    print(f"   - Data Type: {args.data_type}")
    print(f"   - Record Count: {args.record_count}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 에이전트가 생성한 데이터가 저장될 디렉토리
    agent_output_dir = result_json_path.parent / "agent_generated_data"
    agent_output_dir.mkdir(exist_ok=True)

    final_result = {"success": False, "response": None, "error": None}

    try:
        # 에이전트 초기화 시 출력 디렉토리 지정
        agent = SyntheticDataAgent(output_dir=str(agent_output_dir))
        
        response_message = await agent.run(
            data_type=args.data_type,
            record_count=args.record_count
        )
        
        print(f"✅ Agent finished successfully.")
        print(f"   - Response: {response_message}")
        final_result["success"] = True
        final_result["response"] = response_message

    except Exception as e:
        error_msg = f"❌ An error occurred during agent execution: {e}"
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