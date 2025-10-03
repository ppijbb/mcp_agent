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
역할: 뉴스 분석가. 아래 티커('{ticker}')의 최신 헤드라인을 평가하라.
요구 사항:
- sentiment 값은 ["positive","negative","neutral"] 중 하나.
- summary는 한국어 1문장(최대 120자).
- evidence는 상위 2~3개 근거(헤드라인 발췌).
- 출력은 오직 JSON. 추가 텍스트 금지.

입력 헤드라인:
{headlines}

출력(JSON only):
{{
  "sentiment": "positive|negative|neutral",
  "summary": "...",
  "evidence": ["...", "..."]
}}
"""
        
        try:
            response_str = call_llm(prompt)
            analysis = json.loads(response_str)
            
            # 필수 키 검증
            required_keys = ["sentiment", "summary", "evidence"]
            missing_keys = [key for key in required_keys if key not in analysis]
            if missing_keys:
                raise KeyError(f"LLM 응답에 필수 키가 누락되었습니다: {missing_keys}")
            
            # sentiment 값 검증
            valid_sentiments = ["positive", "negative", "neutral"]
            if analysis["sentiment"] not in valid_sentiments:
                raise ValueError(f"유효하지 않은 sentiment 값: {analysis['sentiment']}. 유효한 값: {valid_sentiments}")
            
            analyzed_results[ticker] = analysis
            print(f"'{ticker}' 뉴스 분석 완료: {analysis}")
        except Exception as e:
            error_message = f"'{ticker}' 뉴스 분석 중 오류 발생: {e}"
            print(error_message)
            state["log"].append(error_message)
            raise ValueError(error_message)

    log_message = "모든 티커에 대한 뉴스 감성 분석 완료."
    print(log_message)
    state["log"].append(log_message)
    
    return {"sentiment_analysis": analyzed_results} 