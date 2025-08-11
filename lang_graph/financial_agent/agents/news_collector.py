from typing import Dict
from ..state import AgentState
from ..mcp_client import call_market_news_tool

def news_collector_node(state: AgentState) -> Dict:
    """
    뉴스 수집가 노드: MCP 서버를 통해 최신 뉴스를 수집합니다.
    """
    print("--- AGENT: News Collector (MCP-Powered) ---")
    log_message = "MCP를 통해 최신 시장 뉴스 수집을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    tickers = state.get("target_tickers", [])
    if not tickers:
        return {"news_data": {"news": {}}}

    # 동시 호출 지원: 한 번에 모든 티커를 요청
    print(f"Fetching news for {tickers} via MCP (concurrent)...")
    all_news = call_market_news_tool(tickers)
    
    log_message = f"{len(tickers)}개의 티커에 대한 최신 뉴스 수집 완료."
    state["log"].append(log_message)
    print(log_message)

    return {"news_data": {"news": all_news}} 