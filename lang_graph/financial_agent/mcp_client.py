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


async def _call_tool_async(session: ClientSession, tool_name: str, arguments: Dict) -> Any:
    """단일 MCP 도구를 비동기적으로 호출하는 내부 헬퍼 함수"""
    result = await session.call_tool(tool_name, arguments=arguments)
    content = result.content[0].text
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content

async def _call_tools_concurrently_async(tool_name: str, tickers: List[str]) -> Dict[str, Any]:
    """여러 티커에 대해 단일 MCP 도구를 병렬로 호출하는 헬퍼 함수"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tasks = [
                _call_tool_async(session, tool_name, {"ticker": ticker})
                for ticker in tickers
            ]
            
            # asyncio.gather를 사용하여 모든 작업을 병렬로 실행합니다.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과를 티커와 매핑하여 딕셔너리로 만듭니다.
            ticker_results = {}
            for ticker, result in zip(tickers, results):
                if isinstance(result, Exception):
                    print(f"MCP tool '{tool_name}' for ticker '{ticker}' failed: {result}")
                    ticker_results[ticker] = {"error": str(result)}
                else:
                    ticker_results[ticker] = result
            return ticker_results

def _run_async_concurrent_calls(tool_name: str, tickers: List[str]) -> Dict[str, Any]:
    """동기 함수에서 비동기 병렬 MCP 호출을 실행하기 위한 래퍼"""
    return asyncio.run(_call_tools_concurrently_async(tool_name, tickers))



# --- 에이전트 노드에서 사용할 동기 함수 ---

def call_technical_indicators_tool(tickers: List[str]) -> Dict:
    """여러 티커에 대해 'get_technical_indicators' MCP 도구를 병렬로 동기적으로 호출합니다."""
    return _run_async_concurrent_calls("get_technical_indicators", tickers)

def call_market_news_tool(tickers: List[str]) -> Dict:
    """여러 티커에 대해 'get_market_news' MCP 도구를 병렬로 동기적으로 호출합니다."""
    return _run_async_concurrent_calls("get_market_news", tickers) 