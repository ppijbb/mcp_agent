import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.basic_agents.data_generator import AIDataGenerationAgent


def main():
    """Detailed Data Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the AIDataGenerationAgent with detailed configurations.")
    parser.add_argument("--agent-method", required=True,
                        choices=['generate_smart_data', 'create_custom_dataset', 'generate_customer_profiles', 'generate_timeseries_data'],
                        help="The method to call on the AIDataGenerationAgent.")
    parser.add_argument("--config-json", required=True, help="The configuration dictionary as a JSON string.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    print(f"🔄 Starting Detailed Data Agent...")
    print(f"   - Method: {args.agent_method}")
    print(f"   - Config: {args.config_json}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    final_result = {"success": False, "data": None, "error": None}

    try:
        # 에이전트 인스턴스 생성
        agent = AIDataGenerationAgent()

        # 호출할 메서드 가져오기
        method_to_call = getattr(agent, args.agent_method)

        # JSON 설정 파싱
        config = json.loads(args.config_json)

        # 메서드 호출
        # 이 메서드들은 내부에 asyncio.run을 포함한 동기 래퍼이므로 직접 호출
        result_data = method_to_call(config)

        if "error" in result_data:
            raise Exception(f"Agent reported an error: {result_data['error']}")

        print(f"✅ Agent finished successfully.")
        final_result["success"] = True
        final_result["data"] = result_data

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
    main()
