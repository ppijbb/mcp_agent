#!/usr/bin/env python3
import asyncio
from srcs.travel_scout.mcp_browser_client import MCPBrowserClient

async def main():
    client = MCPBrowserClient(debug=True)
    await client.connect_to_mcp_server()
    hotels = await client.search_hotels_incognito('Seoul', '2025-03-15', '2025-03-20')
    print(f"Found {len(hotels)} hotels (may be 0 due to selector mismatch)")
    await client.cleanup()

if __name__ == '__main__':
    asyncio.run(main()) 