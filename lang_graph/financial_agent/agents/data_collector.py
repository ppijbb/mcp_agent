from typing import Dict
from ..state import AgentState
from ..tools.financial_tools import get_technical_indicators

def market_data_collector_node(state: AgentState) -> Dict:
    """
    시장 데이터 수집가 노드: yfinance를 사용하여 기술적 분석 데이터를 수집합니다.
    """
    print("--- AGENT: Market Data Collector (Live) ---")
    log_message = "실시간 시장 데이터 수집을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    tickers = state.get("target_tickers", [])
    if not tickers:
        error_message = "분석할 티커가 지정되지 않았습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

    all_technicals = {}
    for ticker in tickers:
        print(f"Fetching technical data for {ticker}...")
        all_technicals[ticker] = get_technical_indicators(ticker)

    log_message = f"{len(tickers)}개의 티커에 대한 기술적 분석 데이터 수집 완료."
    state["log"].append(log_message)
    print(log_message)
    
    return {"technical_analysis": all_technicals} 