from typing import Dict
from ..state import AgentState
from ..mcp_client import call_market_news_tool

def news_sentiment_analyzer_node(state: AgentState) -> Dict:
    """
    뉴스 분석가 노드: MCP 서버를 통해 최신 뉴스를 가져옵니다. (현재는 수집만 수행)
    """
    print("--- AGENT: News Analyzer (MCP-Powered) ---")
    log_message = "MCP를 통해 최신 시장 뉴스 수집을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    tickers = state.get("target_tickers", [])
    if not tickers:
        return {"sentiment_analysis": {"news": {}}}

    all_news = {}
    for ticker in tickers:
        print(f"Fetching news for {ticker} via MCP...")
        all_news[ticker] = call_market_news_tool(ticker)
    
    log_message = f"{len(tickers)}개의 티커에 대한 최신 뉴스 수집 완료."
    state["log"].append(log_message)
    print(log_message)

    return {"sentiment_analysis": {"news": all_news}} 