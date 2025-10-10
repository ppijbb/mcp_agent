import argparse
import asyncio
import json
import sys
from pathlib import Path
import base64
from datetime import datetime, timedelta

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.travel_scout.mcp_browser_client import MCPBrowserClient
from srcs.travel_scout.travel_scout_agent import TravelScoutAgent

async def run_agent(args):
    """Travel Scout 에이전트의 핵심 로직을 실행합니다."""
    result_json_path = Path(args.result_json_path)
    output_dir = result_json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✈️ Starting Travel Scout Agent for task: {args.task}")
    print("-" * 30)

    client = MCPBrowserClient(headless=True, disable_gpu=True, screenshot_dir=str(output_dir))
    agent = TravelScoutAgent(browser_client=client)
    
    final_result = {"success": False, "data": None, "screenshots": [], "error": None}

    try:
        print("🔌 Connecting to MCP Server...")
        if not await client.connect_to_mcp_server():
            raise Exception("Failed to connect to MCP Server.")
        print("✅ MCP Server Connected.")

        data = None
        if args.task == 'search_hotels':
            if not all([args.destination, args.check_in, args.check_out, args.guests]):
                raise ValueError("Missing required arguments for hotel search.")
            print(f"🏨 Searching hotels in {args.destination}...")
            data = await agent.search_hotels(args.destination, args.check_in, args.check_out, args.guests)
        
        elif args.task == 'search_flights':
            if not all([args.origin, args.destination, args.departure_date, args.return_date]):
                raise ValueError("Missing required arguments for flight search.")
            print(f"✈️ Searching flights from {args.origin} to {args.destination}...")
            data = await agent.search_flights(args.origin, args.destination, args.departure_date, args.return_date)
        
        elif args.task == 'search_complete_travel':
            if not all([args.origin, args.destination, args.check_in, args.check_out, args.guests]):
                raise ValueError("Missing required arguments for complete travel search.")
            print(f"🧳 Searching complete travel package from {args.origin} to {args.destination}...")
            data = await agent.search_complete_travel(args.origin, args.destination, args.check_in, args.check_out, args.guests)

        print("✅ Task completed.")
        
        final_result["success"] = True
        final_result["data"] = data
        final_result["screenshots"] = client.screenshots

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        final_result["error"] = str(e)
    
    finally:
        print("🧹 Cleaning up browser instance...")
        await agent.cleanup()
        print("💾 Saving final results...")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"🎉 Results saved to {result_json_path}")
        if not final_result["success"]:
            sys.exit(1)

def main():
    """명령줄 인자를 파싱하고 에이전트를 실행합니다."""
    parser = argparse.ArgumentParser(description="Run the Travel Scout Agent from the command line.")
    parser.add_argument("--task", required=True, choices=['search_hotels', 'search_flights', 'search_complete_travel'], help="The task for the agent to perform.")
    parser.add_argument("--result-json-path", required=True, help="Path to save the JSON result file.")
    
    # 호텔 검색 인자
    parser.add_argument("--destination", help="Destination for hotel/flight search.")
    parser.add_argument("--check-in", help="Check-in date for hotels (YYYY-MM-DD).")
    parser.add_argument("--check-out", help="Check-out date for hotels (YYYY-MM-DD).")
    parser.add_argument("--guests", type=int, help="Number of guests for hotels.")
    
    # 항공편 검색 인자
    parser.add_argument("--origin", help="Origin for flight search.")
    parser.add_argument("--departure-date", help="Departure date for flights (YYYY-MM-DD).")
    parser.add_argument("--return-date", help="Return date for flights (YYYY-MM-DD).")
    
    args = parser.parse_args()
    asyncio.run(run_agent(args))

if __name__ == "__main__":
    main() 