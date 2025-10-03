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
    
    try:
        all_news = call_market_news_tool(tickers)
        
        # 데이터 검증
        if not all_news:
            raise ValueError("뉴스 데이터가 비어있습니다.")
        
        # 각 티커의 뉴스 데이터 검증
        for ticker in tickers:
            if ticker not in all_news:
                raise ValueError(f"{ticker}의 뉴스 데이터가 없습니다.")
            
            ticker_news = all_news[ticker]
            # MCP 서버에서 딕셔너리 형태로 반환하므로 이를 리스트로 변환
            if isinstance(ticker_news, dict):
                # 딕셔너리인 경우 리스트로 변환
                if any(ticker_news.values()):  # 값이 있는 경우만
                    all_news[ticker] = [ticker_news]
                else:  # 값이 없는 경우 빈 리스트
                    all_news[ticker] = []
            elif not isinstance(ticker_news, list):
                raise ValueError(f"{ticker}의 뉴스 데이터 형식이 올바르지 않습니다.")
        
        log_message = f"{len(tickers)}개의 티커에 대한 최신 뉴스 수집 완료."
        state["log"].append(log_message)
        print(log_message)

        return {"news_data": {"news": all_news}}
    except Exception as e:
        error_message = f"뉴스 데이터 수집 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        raise ValueError(error_message) 