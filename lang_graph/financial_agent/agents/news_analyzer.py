import json
from typing import Dict

from ..state import AgentState
from ..llm_client import call_llm

def news_analyzer_node(state: AgentState) -> Dict:
    """
    뉴스 분석가 노드 (LLM 기반): 수집된 뉴스를 LLM을 사용하여 분석하고,
    감성(sentiment)과 요약(summary)을 추출합니다.
    """
    print("--- AGENT: News Analyzer (LLM-Powered) ---")
    log_message = "LLM을 사용하여 수집된 뉴스 분석을 시작합니다."
    state["log"].append(log_message)
    print(log_message)

    news_data = state.get("news_data", {}).get("news", {})
    if not news_data:
        log_message = "분석할 뉴스 데이터가 없습니다."
        print(log_message)
        state["log"].append(log_message)
        return {"sentiment_analysis": {}}

    analyzed_results = {}
    for ticker, news_items in news_data.items():
        if not news_items or (len(news_items) == 1 and "No recent news found" in news_items[0]['title']):
            analyzed_results[ticker] = {"sentiment": "neutral", "summary": "뉴스 없음"}
            continue

        headlines = "- " + "\n- ".join([item['title'] for item in news_items if item.get('title')])
        
        prompt = f"""
        당신은 금융 뉴스 분석가입니다. 다음은 특정 주식 티커 '{ticker}'에 대한 최신 뉴스 헤드라인 목록입니다.
        이 헤드라인들을 종합하여, 해당 주식에 대한 전반적인 시장 감성(sentiment)을 'positive', 'negative', 'neutral' 중 하나로 평가하고,
        핵심 내용을 한 문장으로 요약해주세요.

        **뉴스 헤드라인:**
        {headlines}

        **분석 결과 (오직 JSON 객체만 응답):**
        {{
          "sentiment": "...",
          "summary": "..."
        }}
        """
        
        try:
            response_str = call_llm(prompt)
            analysis = json.loads(response_str)
            if "sentiment" not in analysis or "summary" not in analysis:
                raise KeyError("LLM 응답에 'sentiment' 또는 'summary' 키가 없습니다.")
            analyzed_results[ticker] = analysis
            print(f"'{ticker}' 뉴스 분석 완료: {analysis}")
        except Exception as e:
            error_message = f"'{ticker}' 뉴스 분석 중 심각한 오류 발생: {e}"
            print(error_message)
            state["log"].append(error_message)
            # 오류를 전파하여 워크플로우를 중단시킵니다.
            raise e

    log_message = "모든 티커에 대한 뉴스 감성 분석 완료."
    print(log_message)
    state["log"].append(log_message)
    
    return {"sentiment_analysis": analyzed_results} 