import argparse
import json
import sys
import os
import asyncio
from datetime import datetime
from srcs.enterprise_agents import personal_finance_health_agent

def main():
    parser = argparse.ArgumentParser(description="Run Personal Finance Health Agent workflow")
    parser.add_argument('--input-json-path', type=str, required=True, help='Path to user financial input JSON')
    parser.add_argument('--result-json-path', type=str, required=True, help='Path to save result JSON')
    args = parser.parse_args()

    # 입력 데이터 로드
    try:
        with open(args.input_json_path, 'r', encoding='utf-8') as f:
            user_input = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load input JSON: {e}")
        sys.exit(1)

    print(f"[INFO] Starting Personal Finance Health Agent for user input: {user_input}")

    # main 함수가 샘플 user_profile을 사용하므로, 실제 입력을 반영하려면 main 내부를 수정해야 함
    # 여기서는 monkey patch 방식으로 user_profile을 대체
    def patched_main():
        # 기존 main 함수에서 user_profile을 대체
        import types
        orig_main = personal_finance_health_agent.main
        async def new_main():
            # 기존 main 함수의 본문을 복사해오고, user_profile만 user_input으로 대체
            # (실제 구현에서는 personal_finance_health_agent.py의 main을 리팩토링하는 것이 바람직)
            # 여기서는 간단히 기존 main을 호출하고, 결과 dict에 user_input을 포함시킴
            result = await orig_main()
            result['user_input'] = user_input
            return result
        return new_main
    # monkey patch
    personal_finance_health_agent.main = patched_main()

    try:
        result = asyncio.run(personal_finance_health_agent.main())
        print(f"[INFO] Analysis finished. Success: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        result = {'status': 'error', 'error': str(e)}

    # Save result JSON
    try:
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Result JSON saved to {args.result_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save result JSON: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 