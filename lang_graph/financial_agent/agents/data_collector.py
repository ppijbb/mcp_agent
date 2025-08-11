from typing import Dict
from ..state import AgentState
from ..mcp_client import call_technical_indicators_tool

def market_data_collector_node(state: AgentState) -> Dict:
    """
    시장 데이터 수집가 노드: MCP 서버를 통해 기술적 분석 데이터를 수집합니다.
    """
    print("--- AGENT: Market Data Collector (MCP-Powered) ---")
    log_message = "MCP를 통해 실시간 시장 데이터 수집을 시작합니다."
    state["log"].append(log_message)
    print(log_message)
    
    tickers = state.get("target_tickers", [])
    if not tickers:
        error_message = "분석할 티커가 지정되지 않았습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

    # 동시 호출 지원: 한 번에 모든 티커를 요청
    print(f"Fetching technical data for {tickers} via MCP (concurrent)...")
    all_technicals = call_technical_indicators_tool(tickers)

    log_message = f"{len(tickers)}개의 티커에 대한 기술적 분석 데이터 수집 완료."
    state["log"].append(log_message)
    print(log_message)
    
    return {"technical_analysis": all_technicals} 