import argparse
import json
import asyncio
from srcs.business_strategy_agents.run_business_strategy_agents import BusinessStrategyRunner

async def main():
    parser = argparse.ArgumentParser(description="Run the Business Strategy Agent Suite.")
    parser.add_argument("--industry", required=True, help="The industry to analyze.")
    parser.add_argument("--company-profile", required=True, help="A brief profile of the company.")
    parser.add_argument("--competitors", required=True, nargs='+', help="A list of competitor stock tickers.")
    parser.add_argument("--tech-trends", required=True, nargs='+', help="A list of relevant technology trends.")
    parser.add_argument(
        "--google-drive-mcp-url",
        default="http://localhost:3001",
        help="URL for the Google Drive MCP server."
    )
    parser.add_argument(
        "--data-sourcing-mcp-url",
        default="http://localhost:3005",
        help="URL for the Data Sourcing MCP server."
    )
    parser.add_argument("--output-json-path", help="Path to save the final JSON result.")
    
    args = parser.parse_args()

    print("🚀 Starting Business Strategy Agent Workflow...")
    print(f"   - Industry: {args.industry}")
    print(f"   - Company Profile: {args.company_profile[:100]}...")
    print(f"   - Competitors: {args.competitors}")
    print(f"   - Tech Trends: {args.tech_trends}")
    print(f"   - Google Drive MCP: {args.google_drive_mcp_url}")
    print(f"   - Data Sourcing MCP: {args.data_sourcing_mcp_url}")
    print("-" * 30)

    try:
        runner = BusinessStrategyRunner(
            google_drive_mcp_url=args.google_drive_mcp_url,
            data_sourcing_mcp_url=args.data_sourcing_mcp_url
        )

        final_result = await runner.run_agents(
            industry=args.industry,
            company_profile=args.company_profile,
            competitors=args.competitors,
            tech_trends=args.tech_trends
        )

        if args.output_json_path:
            with open(args.output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=4, ensure_ascii=False)
            print(f"\n✅ Final results saved to {args.output_json_path}")

        print("\n🎉 Workflow finished successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during the workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 