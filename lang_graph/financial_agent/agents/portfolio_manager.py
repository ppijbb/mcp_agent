import json
from typing import Dict

from ..state import AgentState
from ..llm_client import call_llm

def portfolio_manager_node(state: AgentState) -> Dict:
    """
    포트폴리오 관리자 노드 (LLM 기반): 시장 전망과 리스크 프로필에 따라 LLM을 호출하여 투자 계획을 수립합니다.
    """
    print("--- AGENT: Portfolio Manager (LLM-Powered) ---")
    log_message = "LLM을 사용하여 투자 계획 수립을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    outlook = state.get("market_outlook")
    risk_profile = state.get("risk_profile")
    tickers = state.get("target_tickers", [])

    if not outlook:
        error_message = "투자 계획 수립에 필요한 시장 전망 데이터가 없습니다."
        print(error_message)
        state["log"].append(error_message)
        return {"error_message": error_message}

    prompt = f"""
    당신은 포트폴리오 매니저입니다. 아래의 시장 전망과 투자자의 리스크 성향을 바탕으로 구체적인 투자 계획을 JSON 형식으로 제시해주세요.
    매수(buy), 매도(sell), 보유(hold)할 티커 목록을 명확히 구분하여 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.

    **시장 전망:**
    {outlook}

    **투자자 리스크 성향:** {risk_profile}

    **분석 대상 티커:** {tickers}

    **투자 계획 (JSON 형식):**
    """

    plan_str = call_llm(prompt)
    print(f"LLM이 생성한 투자 계획 (raw): {plan_str}")
    
    try:
        # LLM 응답이 JSON 형식인지 파싱 시도
        plan = json.loads(plan_str.strip().replace("```json", "").replace("```", ""))
    except (json.JSONDecodeError, AttributeError):
        error_message = f"LLM으로부터 유효한 JSON 형식의 투자 계획을 받지 못했습니다. 응답: {plan_str}"
        print(error_message)
        state["log"].append(error_message)
        # 폴백: 모든 자산을 보유하는 것으로 처리
        plan = {"buy": [], "sell": [], "hold": tickers}

    print(f"수립된 투자 계획: {plan}")
    state["log"].append(f"LLM 기반 투자 계획: {plan}")

    return {"investment_plan": plan} 