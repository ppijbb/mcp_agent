import json
import numpy as np
from typing import Dict

from ..state import AgentState
from ..llm_client import call_llm

class NumpyEncoder(json.JSONEncoder):
    """ Numpy 데이터 타입을 JSON으로 변환하기 위한 커스텀 인코더 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def chief_strategist_node(state: AgentState) -> Dict:
    """
    수석 전략가 노드 (LLM 기반): 기술적 분석과 뉴스 감성 분석 결과를 바탕으로
    LLM을 호출하여 종합 시장 전망을 생성합니다.
    """
    print("--- AGENT: Chief Strategist (LLM-Powered) ---")
    log_message = "LLM을 사용하여 종합 시장 전망 분석을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    tech_analysis = state.get("technical_analysis")
    sentiment_analysis = state.get("sentiment_analysis")

    if not tech_analysis and not sentiment_analysis:
        error_message = "분석에 필요한 데이터가 부족합니다."
        state["log"].append(error_message)
        raise ValueError(error_message)

    prompt = f"""
역할: 수석 전략가. 아래 기술적 분석과 뉴스 감성 결과를 통합하여 시장 전망을 산출하라.
요구 사항:
1) 긍/부/중립 요인을 모두 고려해 단일 결론을 내린다(결정적).
2) 불확실성은 수치형 신뢰도(0~1)로 표현한다.
3) 출력은 오직 아래 JSON 스키마만 반환한다. 추가 텍스트 금지.

입력:
- 기술적 분석(JSON):
{json.dumps(tech_analysis, indent=2, cls=NumpyEncoder)}

- 뉴스 감성 분석(JSON):
{json.dumps(sentiment_analysis, indent=2, ensure_ascii=False)}

출력(JSON only):
{{
  "outlook": "bullish|bearish|neutral",
  "rationale": "핵심 근거를 한국어로 2~4문장",
  "risks": ["최대 3개 리스크"],
  "confidence": 0.0
}}
"""
    
    try:
        outlook = call_llm(prompt)
        print(f"LLM이 생성한 시장 전망:\n{outlook}")
        state["log"].append("LLM 기반 시장 전망 분석 완료.")
        
        # JSON 파싱 검증
        try:
            json.loads(outlook)
        except json.JSONDecodeError:
            error_message = f"LLM 응답이 유효한 JSON 형식이 아닙니다: {outlook}"
            print(error_message)
            state["log"].append(error_message)
            raise ValueError(error_message)
        
        return {"market_outlook": outlook}
    except Exception as e:
        error_message = f"시장 전망 분석 중 오류 발생: {e}"
        print(error_message)
        state["log"].append(error_message)
        raise ValueError(error_message)