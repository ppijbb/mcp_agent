import argparse
import json
import sys
from srcs.enterprise_agents.hr_recruitment_agent import HRRecruitmentAgent


def main():
    parser = argparse.ArgumentParser(description="Run HR Recruitment Agent workflow")
    parser.add_argument('--position', type=str, required=True, help='Position name')
    parser.add_argument('--company', type=str, required=True, help='Company name')
    parser.add_argument('--workflows', type=str, nargs='*', help='List of workflows to execute')
    parser.add_argument('--save-to-file', action='store_true', help='Save results to file')
    parser.add_argument('--result-json-path', type=str, required=True, help='Path to save result JSON')
    args = parser.parse_args()

    print(f"[INFO] Starting HR Recruitment Agent for position: {args.position}, company: {args.company}")
    agent = HRRecruitmentAgent()
    result = agent.run_recruitment_workflow(
        position=args.position,
        company=args.company,
        workflows=args.workflows,
        save_to_file=args.save_to_file
    )
    print(f"[INFO] Recruitment workflow finished. Success: {result.get('success')}")
    if result.get('success'):
        print(f"[INFO] Message: {result.get('message')}")
        if args.save_to_file:
            print(f"[INFO] Output directory: {result.get('output_dir')}")
        else:
            print("[INFO] Results returned for display (not saved to file)")
    else:
        print(f"[ERROR] {result.get('message')}")
        if 'error' in result:
            print(f"[ERROR] Details: {result['error']}")
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
