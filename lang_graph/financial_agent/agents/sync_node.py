from typing import Dict
from ..state import AgentState

def sync_node(state: AgentState) -> Dict:
    """
    병렬로 실행된 데이터 수집 노드들의 완료를 동기화합니다.
    """
    print("--- AGENT: Data Sync ---")
    
    tech_analysis = state.get("technical_analysis", {})
    news_data = state.get("news_data", {})
    
    tech_count = len(tech_analysis) if tech_analysis else 0
    news_count = len(news_data.get("news", {})) if news_data else 0

    log_message = f"데이터 동기화 완료: 기술적 분석 {tech_count}개, 뉴스 {news_count}개 수집됨."
    print(log_message)
    state["log"].append(log_message)
    
    return {} 