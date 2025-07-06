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
    수석 전략가 노드 (LLM 기반): 수집된 데이터를 바탕으로 LLM을 호출하여 시장 전망을 생성합니다.
    """
    print("--- AGENT: Chief Strategist (LLM-Powered) ---")
    log_message = "LLM을 사용하여 종합 시장 전망 분석을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    tech_analysis = state.get("technical_analysis")
    news_analysis = state.get("sentiment_analysis")

    if not tech_analysis and not news_analysis:
        error_message = "분석에 필요한 데이터가 부족합니다."
        return {"error_message": error_message}

    prompt = f"""
    당신은 최고의 금융 분석가입니다. 아래의 기술적 분석 데이터와 최신 뉴스를 바탕으로, 시장 전체에 대한 명확하고 간결한 전망을 제시해주세요.
    긍정적, 부정적, 중립적 요인을 모두 고려하여 종합적인 결론을 내려주세요.

    **기술적 분석 데이터:**
    ```json
    {json.dumps(tech_analysis, indent=2, cls=NumpyEncoder)}
    ```

    **최신 뉴스 헤드라인:**
    ```json
    {json.dumps(news_analysis, indent=2, cls=NumpyEncoder)}
    ```

    **분석 결과 (시장 전망):**
    """
    
    outlook = call_llm(prompt)
    print(f"LLM이 생성한 시장 전망:\n{outlook}")
    state["log"].append("LLM 기반 시장 전망 분석 완료.")
    
    return {"market_outlook": outlook} 