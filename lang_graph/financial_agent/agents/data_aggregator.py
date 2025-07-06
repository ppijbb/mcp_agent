from typing import Dict
from ..state import AgentState

def aggregator_node(state: AgentState) -> Dict:
    """
    병렬로 실행된 데이터 수집 노드들의 결과를 종합합니다.
    """
    print("--- AGENT: Data Aggregator ---")
    
    tech_analysis = state.get("technical_analysis", {})
    news_analysis = state.get("sentiment_analysis", {})
    
    tech_count = len(tech_analysis) if tech_analysis else 0
    news_count = len(news_analysis.get("news", {})) if news_analysis else 0

    log_message = f"데이터 집계 완료: 기술적 분석 {tech_count}개, 뉴스 분석 {news_count}개"
    print(log_message)
    state["log"].append(log_message)
    
    return {} 