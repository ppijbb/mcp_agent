import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Any

# MCP 서버 실행을 위한 설정
server_params = StdioServerParameters(
    command="python",
    args=["lang_graph/financial_agent/financial_mcp_server.py"],
    env=None,
)

async def _call_tool_async(tool_name: str, arguments: Dict) -> Any:
    """비동기적으로 MCP 서버에 연결하고 도구를 호출하는 헬퍼 함수"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # 결과가 JSON 문자열일 수 있으므로 파싱
            content = result.content[0].text
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content

def _run_async_call(tool_name: str, arguments: Dict) -> Any:
    """동기 함수에서 비동기 MCP 호출을 실행하기 위한 래퍼"""
    return asyncio.run(_call_tool_async(tool_name, arguments))

# --- 에이전트 노드에서 사용할 동기 함수 ---

def call_technical_indicators_tool(ticker: str) -> Dict:
    """'get_technical_indicators' MCP 도구를 동기적으로 호출합니다."""
    return _run_async_call("get_technical_indicators", {"ticker": ticker})

def call_market_news_tool(ticker: str) -> List[Dict]:
    """'get_market_news' MCP 도구를 동기적으로 호출합니다."""
    return _run_async_call("get_market_news", {"ticker": ticker}) 