import argparse
import asyncio
import json
import sys
from pathlib import Path
import asyncio
import aiohttp

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.seo_doctor.seo_doctor_agent import SEODoctorAgent
from srcs.core.utils import EnhancedJSONEncoder

# Dataclass를 dict로 변환하기 위한 헬퍼
from dataclasses import asdict, is_dataclass
from datetime import datetime

async def upload_to_drive(session: aiohttp.ClientSession, mcp_url: str, file_name: str, content: str) -> dict:
    """Uploads content to Google Drive via MCP."""
    upload_url = f"{mcp_url}/upload"
    payload = {"fileName": file_name, "content": content}
    
    async with session.post(upload_url, json=payload) as response:
        response.raise_for_status()
        return await response.json()

async def run_agent(args):
    """SEO Doctor 에이전트의 핵심 로직을 실행합니다."""
    print(f"🔄 Starting SEO Doctor...")
    print(f"   - URL: {args.url}")
    print(f"   - Google Drive MCP: {args.google_drive_mcp_url}")
    print(f"   - SEO MCP: {args.seo_mcp_url}")
    print("-" * 30)
    
    final_result = {"success": False, "data": None, "error": None}
    agent_result = None
    agent = SEODoctorAgent()

    try:
        analysis_result = await agent.run(
            url=args.url,
            keywords=args.competitor_urls
        )
        
        print("✅ Agent finished successfully.")
        final_result["success"] = True
        agent_result = analysis_result

    except Exception as e:
        import traceback
        error_msg = f"❌ An error occurred during agent execution: {e}\n{traceback.format_exc()}"
        print(error_msg)
        final_result["error"] = str(e)
    
    finally:
        if agent_result:
            final_result["data"] = json.loads(json.dumps(agent_result, cls=EnhancedJSONEncoder))

        json_content_to_upload = json.dumps(final_result, indent=2, ensure_ascii=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file_name = f"seo_doctor_result_{timestamp}.json"

        print(f"💾 Uploading final results to Google Drive as {json_file_name}...")
        try:
            async with aiohttp.ClientSession() as session:
                upload_result = await upload_to_drive(session, args.google_drive_mcp_url, json_file_name, json_content_to_upload)
                if upload_result.get("success"):
                    file_id = upload_result.get("fileId")
                    print(f"🎉 Results uploaded successfully. File ID: {file_id}")
                else:
                    raise Exception(f"MCP upload failed: {upload_result.get('message')}")
        except Exception as e:
            print(f"❌ Failed to upload result JSON to Google Drive: {e}")
            print("--- FALLBACK: FINAL RESULT JSON ---")
            print(json_content_to_upload)
            print("------------------------------------")
            final_result["success"] = False
            final_result["error"] = f"Failed to upload result JSON: {e}"
        
        if not final_result["success"]:
            sys.exit(1)

def main():
    """명령줄 인자를 파싱하고 에이전트를 실행합니다."""
    parser = argparse.ArgumentParser(description="Run the SEO Doctor Agent.")
    parser.add_argument("--url", required=True, help="The URL to analyze.")
    parser.add_argument("--include-competitors", action='store_true', help="Include competitor analysis.")
    parser.add_argument("--competitor-urls", nargs='*', help="List of competitor URLs.")
    parser.add_argument(
        "--google-drive-mcp-url",
        default="http://localhost:3001",
        help="The URL for the Google Drive MCP server."
    )
    parser.add_argument(
        "--seo-mcp-url",
        default="http://localhost:3002",
        help="The URL for the SEO MCP server."
    )
    
    args = parser.parse_args()
    asyncio.run(run_agent(args))

if __name__ == "__main__":
    main() 