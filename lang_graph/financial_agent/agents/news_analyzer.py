from typing import Dict
from ..state import AgentState
from ..tools.financial_tools import get_market_news

def news_sentiment_analyzer_node(state: AgentState) -> Dict:
    """
    뉴스 분석가 노드: yfinance를 사용하여 최신 뉴스를 가져옵니다.
    """
    print("--- AGENT: News Analyzer (Live) ---")
    log_message = "최신 시장 뉴스 수집을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    tickers = state.get("target_tickers", [])
    if not tickers:
        return {"sentiment_analysis": {"news": {}}}

    all_news = {}
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        all_news[ticker] = get_market_news(ticker)
    
    log_message = f"{len(tickers)}개의 티커에 대한 최신 뉴스 수집 완료."
    state["log"].append(log_message)
    print(log_message)

    return {"sentiment_analysis": {"news": all_news}} 