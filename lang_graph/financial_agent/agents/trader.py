from typing import Dict
from ..state import AgentState
from ..mcp_client import call_technical_indicators_tool
from ..config import get_trading_config

def trader_node(state: AgentState) -> Dict:
    """
    트레이더 노드: 수립된 투자 계획에 따라 실제 시장 데이터를 기반으로 거래를 실행합니다.
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
    
    # 거래할 모든 티커 수집
    trade_tickers = list(set(plan.get("buy", []) + plan.get("sell", [])))
    
    if not trade_tickers:
        log_message = "거래할 티커가 없습니다."
        print(log_message)
        state["log"].append(log_message)
        return {"trade_results": [], "daily_pnl": 0.0}
    
    # 실제 시장 데이터에서 현재 가격 가져오기
    try:
        current_prices = call_technical_indicators_tool(trade_tickers)
        log_message = f"거래 대상 {len(trade_tickers)}개 티커의 현재 가격을 조회했습니다."
        print(log_message)
        state["log"].append(log_message)
    except Exception as e:
        error_message = f"현재 가격 조회 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        raise ValueError(error_message)
    
    # 거래 설정 가져오기
    trading_config = get_trading_config()
    default_shares = trading_config.default_shares
    max_trade_amount = trading_config.max_trade_amount
    
    # 매수 거래 실행
    for ticker in plan.get("buy", []):
        if ticker in current_prices and "price" in current_prices[ticker]:
            price = current_prices[ticker]["price"]
            if price is not None:
                # 최대 거래 금액을 고려하여 수량 계산
                shares = min(default_shares, int(max_trade_amount / price)) if price > 0 else default_shares
                trade_amount = price * shares
                
                if trade_amount > max_trade_amount:
                    error_message = f"{ticker} 거래 금액이 최대 거래 금액을 초과합니다. (${trade_amount:.2f} > ${max_trade_amount:.2f})"
                    print(error_message)
                    state["log"].append(error_message)
                    continue
                
                trade_results.append({"ticker": ticker, "action": "buy", "price": price, "shares": shares})
                total_pnl -= trade_amount
                print(f"BUY: {ticker} at ${price:.2f} (수량: {shares}, 금액: ${trade_amount:.2f})")
            else:
                error_message = f"{ticker}의 가격 데이터를 가져올 수 없습니다."
                print(error_message)
                state["log"].append(error_message)
        else:
            error_message = f"{ticker}의 시장 데이터를 찾을 수 없습니다."
            print(error_message)
            state["log"].append(error_message)

    # 매도 거래 실행
    for ticker in plan.get("sell", []):
        if ticker in current_prices and "price" in current_prices[ticker]:
            price = current_prices[ticker]["price"]
            if price is not None:
                # 최대 거래 금액을 고려하여 수량 계산
                shares = min(default_shares, int(max_trade_amount / price)) if price > 0 else default_shares
                trade_amount = price * shares
                
                if trade_amount > max_trade_amount:
                    error_message = f"{ticker} 거래 금액이 최대 거래 금액을 초과합니다. (${trade_amount:.2f} > ${max_trade_amount:.2f})"
                    print(error_message)
                    state["log"].append(error_message)
                    continue
                
                trade_results.append({"ticker": ticker, "action": "sell", "price": price, "shares": shares})
                total_pnl += trade_amount
                print(f"SELL: {ticker} at ${price:.2f} (수량: {shares}, 금액: ${trade_amount:.2f})")
            else:
                error_message = f"{ticker}의 가격 데이터를 가져올 수 없습니다."
                print(error_message)
                state["log"].append(error_message)
        else:
            error_message = f"{ticker}의 시장 데이터를 찾을 수 없습니다."
            print(error_message)
            state["log"].append(error_message)

    log_message = f"거래 실행 완료. 총 손익: ${total_pnl:.2f}"
    print(log_message)
    state["log"].append(log_message)

    return {"trade_results": trade_results, "daily_pnl": total_pnl} 