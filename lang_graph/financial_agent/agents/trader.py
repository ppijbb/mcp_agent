import random
from typing import Dict

from ..state import AgentState

def trader_node(state: AgentState) -> Dict:
    """
    트레이더 노드: 수립된 투자 계획에 따라 모의 거래를 실행합니다.
    """
    print("--- AGENT: Trader ---")
    log_message = "투자 계획 실행을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    plan = state.get("investment_plan")

    if not plan or (not plan.get("buy") and not plan.get("sell")):
        log_message = "실행할 거래가 없습니다."
        print(log_message)
        state["log"].append(log_message)
        return {"trade_results": [], "daily_pnl": 0.0}

    trade_results = []
    total_pnl = 0.0
    
    # 모의 거래
    for ticker in plan.get("buy", []):
        price = round(random.uniform(100, 500), 2)
        trade_results.append({"ticker": ticker, "action": "buy", "price": price, "shares": 10})
        total_pnl -= price * 10
        print(f"BUY: {ticker} at ${price}")

    for ticker in plan.get("sell", []):
        price = round(random.uniform(100, 500), 2)
        trade_results.append({"ticker": ticker, "action": "sell", "price": price, "shares": 10})
        total_pnl += price * 10
        print(f"SELL: {ticker} at ${price}")

    log_message = f"거래 실행 완료. 총 손익: ${total_pnl:.2f}"
    print(log_message)
    state["log"].append(log_message)

    return {"trade_results": trade_results, "daily_pnl": total_pnl} 