import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Any
from .config import get_mcp_config

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
    config = get_mcp_config()
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tasks = [
                    _call_tool_async(session, tool_name, {"ticker": ticker})
                    for ticker in tickers
                ]
                
                # 설정된 타임아웃으로 병렬 실행
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=config.timeout
                )
                
                # 결과를 티커와 매핑하여 딕셔너리로 만듭니다.
                ticker_results = {}
                failed_tickers = []
                
                for ticker, result in zip(tickers, results):
                    if isinstance(result, Exception):
                        error_msg = f"MCP tool '{tool_name}' for ticker '{ticker}' failed: {result}"
                        print(f"❌ {error_msg}")
                        failed_tickers.append(ticker)
                        ticker_results[ticker] = {"error": str(result)}
                    else:
                        ticker_results[ticker] = result
                
                # 실패한 티커가 있으면 에러 발생
                if failed_tickers:
                    raise RuntimeError(f"MCP 도구 호출 실패 - 티커: {failed_tickers}")
                
                return ticker_results
                
    except asyncio.TimeoutError:
        error_msg = f"MCP 도구 호출 타임아웃 ({config.timeout}초 초과) - 도구: {tool_name}, 티커: {tickers}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"MCP 서버 연결 실패 - 도구: {tool_name}, 티커: {tickers}, 에러: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

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

def call_ohlcv_data_tool(tickers: List[str], period: str = None) -> Dict:
    """여러 티커에 대해 'get_ohlcv_data' MCP 도구를 병렬로 동기적으로 호출합니다."""
    if period:
        # period가 지정된 경우, 각 티커에 대해 period를 포함한 arguments 전달
        async def _call_with_period():
            config = get_mcp_config()
            try:
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tasks = [
                            _call_tool_async(session, "get_ohlcv_data", {"ticker": ticker, "period": period})
                            for ticker in tickers
                        ]
                        
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=config.timeout
                        )
                        
                        ticker_results = {}
                        failed_tickers = []
                        
                        for ticker, result in zip(tickers, results):
                            if isinstance(result, Exception):
                                error_msg = f"MCP tool 'get_ohlcv_data' for ticker '{ticker}' failed: {result}"
                                print(f"❌ {error_msg}")
                                failed_tickers.append(ticker)
                                ticker_results[ticker] = {"error": str(result)}
                            else:
                                ticker_results[ticker] = result
                        
                        if failed_tickers:
                            raise RuntimeError(f"MCP 도구 호출 실패 - 티커: {failed_tickers}")
                        
                        return ticker_results
            except asyncio.TimeoutError:
                error_msg = f"MCP 도구 호출 타임아웃 ({config.timeout}초 초과) - 도구: get_ohlcv_data, 티커: {tickers}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"MCP 서버 연결 실패 - 도구: get_ohlcv_data, 티커: {tickers}, 에러: {e}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        return asyncio.run(_call_with_period())
    else:
        return _run_async_concurrent_calls("get_ohlcv_data", tickers) 