import asyncio
import argparse
import json
import sys
import os
from typing import List

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

from enterprise_agents.cybersecurity_infrastructure_agent import CybersecurityAgent

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Cybersecurity Agent workflow.")
    parser.add_argument("--company-name", required=True, help="Company name for assessment.")
    parser.add_argument("--assessment-type", required=True, help="Type of security assessment.")
    parser.add_argument("--frameworks", required=True, help="JSON string of compliance frameworks.")
    parser.add_argument("--save-to-file", action="store_true", help="Save results to files.")
    
    args = parser.parse_args()
    
    return {
        "company_name": args.company_name,
        "assessment_type": args.assessment_type,
        "frameworks": json.loads(args.frameworks),
        "save_to_file": args.save_to_file
    }

def main():
    """Main function to run the cybersecurity agent workflow."""
    args = parse_args()
    
    print("🚀 Initializing Cybersecurity Agent...")
    agent = CybersecurityAgent()
    
    print(f"🏢 Company: {args['company_name']}")
    print(f"🛡️ Assessment Type: {args['assessment_type']}")
    print(f"📋 Frameworks: {', '.join(args['frameworks'])}")
    print(f"💾 Save to file: {'Yes' if args['save_to_file'] else 'No'}")
    print("-" * 50)

    try:
        # CybersecurityAgent.run_cybersecurity_workflow는 동기 함수이므로 바로 호출
        result = agent.run_cybersecurity_workflow(
            company_name=args["company_name"],
            assessment_type=args["assessment_type"],
            frameworks=args["frameworks"],
            save_to_file=args["save_to_file"]
        )
        
        print("\n" + "=" * 50)
        if result and result.get('success'):
            print("✅ Cybersecurity workflow completed successfully!")
            if result.get('content'):
                # 결과가 너무 길 수 있으므로 일부만 출력
                print("\n--- Generated Content (Preview) ---")
                print(result['content'][:1000] + "...")
                print("-" * 50)
            if args['save_to_file']:
                print(f"📂 Reports saved in: {result.get('output_dir', 'N/A')}")
        else:
            print("❌ Cybersecurity workflow failed.")
            if result:
                print(f"Error: {result.get('message')}")

    except Exception as e:
        print(f"💥 An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 