import json
from typing import Dict, List

from ..state import AgentState
from ..llm_client import call_llm

def portfolio_manager_node(state: AgentState) -> Dict:
    """
    포트폴리오 관리자 노드 (LLM 기반): 시장 전망과 리스크 프로필에 따라 LLM을 호출하여 투자 계획을 수립합니다.
    JSON 파싱 실패 시, 자가 수정을 시도합니다.
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
역할: 포트폴리오 매니저. 시장 전망, 리스크 성향, 대상 티커로 실행 가능한 계획을 산출하라.
제약:
- 응답은 오직 JSON.
- buy/sell/hold는 제공된 티커 집합의 부분집합이며 중복 금지.
- 각 티커의 조치 근거를 reason_by_ticker에 포함.
- 리스크 성향 규칙:
  - conservative: buy≤1, sell≥0, 신규 편입 최소화
  - moderate: buy≤2, 포지션 균형
  - aggressive: buy≤3, 공세적 포지셔닝 허용
- 불확실할 경우 hold로 분류.

입력:
- 시장 전망:
{outlook}

- 리스크 성향: {risk_profile}
- 대상 티커: {tickers}

출력(JSON only):
{{
  "buy": ["..."],
  "sell": ["..."],
  "hold": ["..."],
  "reason_by_ticker": {{
    "TICKER": "한국어 1문장 근거"
  }}
}}
"""

    # LLM 호출 및 응답 파싱
    plan_str = call_llm(prompt)
    print(f"LLM이 생성한 투자 계획 (raw): {plan_str}")
    
    try:
        # JSON 파싱 시도
        cleaned_response = plan_str.strip().replace("```json", "").replace("```", "")
        plan = json.loads(cleaned_response)
        
        # 필수 키 검증
        required_keys = ["buy", "sell", "hold", "reason_by_ticker"]
        missing_keys = [key for key in required_keys if key not in plan]
        if missing_keys:
            raise KeyError(f"필수 키가 누락되었습니다: {missing_keys}")
            
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        error_message = f"LLM 응답 파싱 실패: {e}. 응답: {plan_str}"
        print(error_message)
        state["log"].append(error_message)
        raise ValueError(error_message)

    print(f"수립된 투자 계획: {plan}")
    state["log"].append(f"LLM 기반 투자 계획: {plan}")

    return {"investment_plan": plan} 