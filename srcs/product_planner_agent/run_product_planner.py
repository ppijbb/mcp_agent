import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import aiohttp

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from srcs.product_planner_agent.coordinators.executive_coordinator import ExecutiveCoordinator
from srcs.product_planner_agent.utils.json_encoder import EnhancedJSONEncoder

# Helper function to create the HTTP client session
def get_http_session():
    return aiohttp.ClientSession()

async def upload_to_drive(mcp_url: str, file_name: str, content: str) -> dict:
    """Uploads content to Google Drive via MCP."""
    upload_url = f"{mcp_url}/upload"
    payload = {"fileName": file_name, "content": content}
    
    async with get_http_session() as session:
        async with session.post(upload_url, json=payload) as response:
            response.raise_for_status()
            return await response.json()

async def main():
    """Product Planner 실행 스크립트"""
    parser = argparse.ArgumentParser(description="Run the Product Planner Agent workflow.")
    parser.add_argument("--product-concept", required=True, help="The high-level concept for the product.")
    parser.add_argument("--user-persona", required=True, help="A description of the target user persona.")
    parser.add_argument("--figma-file-id", help="The file ID of the Figma design.")
    parser.add_argument("--notion-page-id", help="The page ID of the Notion planning document.")
    parser.add_argument(
        "--google-drive-mcp-url",
        default="http://localhost:3001",
        help="URL for the Google Drive MCP server."
    )
    parser.add_argument(
        "--figma-mcp-url",
        default="http://localhost:3003",
        help="URL for the Figma Context MCP server."
    )
    parser.add_argument(
        "--notion-mcp-url",
        default="http://localhost:3004",
        help="URL for the Notion Context MCP server."
    )
    
    args = parser.parse_args()

    print("🚀 Starting Product Planner Workflow...")
    print(f"   - Product Concept: {args.product_concept[:100]}...")
    print(f"   - User Persona: {args.user_persona[:100]}...")
    print(f"   - Figma File ID: {args.figma_file_id}")
    print(f"   - Notion Page ID: {args.notion_page_id}")
    print(f"   - Google Drive MCP: {args.google_drive_mcp_url}")
    print(f"   - Figma MCP: {args.figma_mcp_url}")
    print(f"   - Notion MCP: {args.notion_mcp_url}")
    print("-" * 30)

    final_result = {"success": False, "data": None, "error": None}

    try:
        coordinator = ExecutiveCoordinator(
            google_drive_mcp_url=args.google_drive_mcp_url,
            figma_mcp_url=args.figma_mcp_url,
            notion_mcp_url=args.notion_mcp_url
        )
        
        workflow_result = await coordinator.run_product_planning_workflow(
            product_concept=args.product_concept,
            user_persona=args.user_persona,
            figma_file_id=args.figma_file_id,
            notion_page_id=args.notion_page_id
        )
        
        print("✅ Agent finished successfully.")
        final_result["success"] = True
        final_result["data"] = workflow_result

    except Exception as e:
        import traceback
        error_msg = f"❌ An error occurred during agent execution: {e}\n{traceback.format_exc()}"
        print(error_msg)
        final_result["error"] = str(e)
    
    finally:
        json_content_to_upload = json.dumps(final_result, indent=2, ensure_ascii=False, default=str)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file_name = f"product_planner_result_{timestamp}.json"
        
        print(f"💾 Uploading final results to Google Drive as {json_file_name}...")
        try:
            upload_result = await upload_to_drive(args.google_drive_mcp_url, json_file_name, json_content_to_upload)
            if upload_result.get("success"):
                file_id = upload_result.get("fileId")
                print(f"🎉 Results uploaded successfully. File ID: {file_id}")
            else:
                raise Exception(f"MCP upload failed: {upload_result.get('message')}")
        except Exception as e:
            print(f"❌ Failed to upload result JSON to Google Drive: {e}")
            # As a fallback, print to console
            print("--- FALLBACK: FINAL RESULT JSON ---")
            print(json_content_to_upload)
            print("------------------------------------")
            final_result["success"] = False # Mark as not fully successful
            final_result["error"] = f"Failed to upload result JSON: {e}"
        
        if not final_result["success"]:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 