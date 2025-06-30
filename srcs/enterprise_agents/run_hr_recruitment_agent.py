import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.enterprise_agents.hr_recruitment_agent import HRRecruitmentAgent

def main():
    """HR Recruitment Agent 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the HR Recruitment Agent from the command line.")
    parser.add_argument("--position-name", required=True, help="The name of the position to recruit for.")
    parser.add_argument("--company-name", required=True, help="The name of the company.")
    parser.add_argument("--workflows", nargs='+', required=True, help="A list of workflows to execute.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")

    args = parser.parse_args()

    print(f"🚀 Starting HR Recruitment Agent for {args.position_name} at {args.company_name}...")
    print(f" workflows: {', '.join(args.workflows)}")
    print("-" * 30)

    try:
        agent = HRRecruitmentAgent()
        # save_to_file is handled by the page, agent runner just produces results
        result = agent.run_recruitment_workflow(
            position=args.position_name,
            company=args.company_name,
            workflows=args.workflows,
            save_to_file=False 
        )

        print("-" * 30)
        print("✅ HR Recruitment Agent finished successfully.")
        print(f"💾 Saving results to {args.result_json_path}...")

        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("🎉 Results saved.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # 실패 시에도 JSON 파일 생성하여 오류 보고
        error_result = {
            'success': False,
            'message': f'An error occurred: {str(e)}',
            'error': str(e),
        }
        with open(args.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        sys.exit(1)


if __name__ == "__main__":
    main() 