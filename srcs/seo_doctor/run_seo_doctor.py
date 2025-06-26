import asyncio
import argparse
import sys
import os
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    DEFAULT_REPORTS_PATH = get_reports_path('seo_doctor')
except ImportError:
    DEFAULT_REPORTS_PATH = "reports/seo_doctor"

from seo_doctor.seo_doctor_agent import run_emergency_seo_diagnosis

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SEO Doctor Emergency Diagnosis.")
    parser.add_argument("--url", required=True, help="The URL of the website to analyze.")
    parser.add_argument("--no-competitors", action="store_true", help="Do not include competitor analysis.")
    parser.add_argument("--output-dir", default=DEFAULT_REPORTS_PATH, help="Directory to save the reports.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the final structured JSON result.")
    
    args = parser.parse_args()
    return args

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    JSON 인코더 확장: dataclass, datetime, Enum 등 처리
    """
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, 'value'): # Enum 처리
            return o.value
        return super().default(o)

async def main():
    """Main async function to run the SEO diagnosis."""
    args = parse_args()

    if not args.url.startswith(('http://', 'https://')):
        url = 'https://' + args.url
    else:
        url = args.url

    print("🚀 Initializing SEO Doctor...")
    print(f"🌐 Analyzing URL: {url}")
    print(f"🕵️ Competitor Analysis: {'No' if args.no_competitors else 'Yes'}")
    print(f"📁 Output Directory: {args.output_dir}")
    print("-" * 50)

    try:
        analysis_result = await run_emergency_seo_diagnosis(
            url=url,
            include_competitors=not args.no_competitors,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 50)
        if hasattr(analysis_result, 'emergency_level'):
            print("✅ SEO Diagnosis Completed Successfully!")
            print(f"Overall Score: {analysis_result.overall_score}")
            print(f"Emergency Level: {analysis_result.emergency_level.value}")
            
            # 결과를 JSON 파일로 저장
            try:
                with open(args.result_json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=2)
                print(f"📄 Structured result saved to: {args.result_json_path}")
            except Exception as e:
                print(f"❌ Failed to save JSON result: {e}")

        else:
            print("❌ SEO Diagnosis Failed.")
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                print(f"Error: {analysis_result['error']}")

    except Exception as e:
        print(f"💥 An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 