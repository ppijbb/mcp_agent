import argparse
import json
import sys
from pathlib import Path
import importlib
import asyncio
from typing import Dict, Any, Optional



def main():
    """
    범용 에이전트 실행 스크립트.
    모듈 경로, 클래스 이름, 메서드 이름을 인자로 받아 동적으로 에이전트를 실행합니다.
    """
    parser = argparse.ArgumentParser(description="Generic agent runner.")
    parser.add_argument("--module-path", required=True, help="Dot-separated path to the agent module (e.g., 'srcs.basic_agents.data_generator').")
    parser.add_argument("--class-name", required=True, help="Name of the agent class to instantiate.")
    parser.add_argument("--method-name", required=True, help="Name of the method to call on the agent instance.")
    parser.add_argument("--config-json", default='{}', help="Configuration dictionary as a JSON string for the method.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    # 프로젝트 루트를 동적으로 sys.path에 추가 (python -m으로 실행 시 필요)
    # Note: This might need adjustment based on execution context.
    # A simple approach is to assume execution from project root.
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"🔄 Starting Generic Agent Runner...")
    print(f"   - Module: {args.module_path}")
    print(f"   - Class: {args.class_name}")
    print(f"   - Method: {args.method_name}")
    print("-" * 30)

    result_json_path = Path(args.result_json_path)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)

    final_result = {"success": False, "data": None, "error": None}

    try:
        # 동적으로 모듈 임포트
        agent_module = importlib.import_module(args.module_path)

        # 클래스 가져오기
        AgentClass = getattr(agent_module, args.class_name)

        # 에이전트 인스턴스 생성
        # TODO: 생성자에 인자가 필요한 경우를 대비해 확장 필요
        agent_instance = AgentClass()

        # 호출할 메서드 가져오기
        method_to_call = getattr(agent_instance, args.method_name)

        # JSON 설정 파싱
        config = json.loads(args.config_json)

        # 메서드 호출 (비동기/동기 분기 처리)
        if asyncio.iscoroutinefunction(method_to_call):
            print("   - Running async method.")
            result_data = asyncio.run(method_to_call(**config))
        else:
            print("   - Running sync method.")
            result_data = method_to_call(**config)

        # 에이전트가 반환한 값에 error가 포함되어 있는지 확인
        if isinstance(result_data, dict) and "error" in result_data:
            from srcs.core.errors import WorkflowError
            raise WorkflowError(f"Agent reported an error: {result_data['error']}")

        print(f"✅ Agent method '{args.method_name}' finished successfully.")
        final_result["success"] = True
        final_result["data"] = result_data

    except (ImportError, AttributeError, json.JSONDecodeError) as e:
        import traceback
        error_msg = f"❌ An error occurred during agent execution: {e}"
        print(error_msg)
        final_result["error"] = str(error_msg)
    except Exception as e:
        import traceback
        # Check for custom errors with deferred imports
        try:
            from srcs.core.errors import MCPError, ConfigError, WorkflowError
            if isinstance(e, (MCPError, ConfigError, WorkflowError)):
                error_msg = f"❌ An error occurred during agent execution: {e}"
            else:
                error_msg = f"❌ Unexpected error during agent execution: {e}\n{traceback.format_exc()}"
        except ImportError:
            error_msg = f"❌ Unexpected error during agent execution: {e}\n{traceback.format_exc()}"
        print(error_msg)
        final_result["error"] = error_msg

    finally:
        print(f"💾 Saving final results to {result_json_path}...")
        try:
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print("🎉 Results saved.")
        except Exception as e:
            print(f"❌ Failed to save result JSON: {e}")
            final_result["success"] = False
            # Overwrite final_result to ensure the error is reported
            final_result["error"] = f"Failed to save result JSON: {e}"
            with open(result_json_path, 'w', encoding='utf-8') as f:
                 json.dump(final_result, f, indent=2, ensure_ascii=False)

        if not final_result["success"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
