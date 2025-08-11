from typing import List, TypedDict, Optional, Dict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

# Annotated와 add_messages를 사용하면 LangGraph가 메시지 목록을 더 잘 관리할 수 있습니다.
# 지금 당장 필수는 아니지만, 채팅 기록과 같은 것을 추가할 때 유용합니다.

class InvestmentPlan(TypedDict, total=False):
    """투자 계획 스키마"""
    buy: List[str]
    sell: List[str]
    hold: List[str]
    reason_by_ticker: Dict[str, str]


class AgentState(TypedDict):
    """
    자율 금융 에이전트의 상태를 정의합니다.
    그래프의 모든 노드(에이전트)가 이 상태 객체를 공유하고 업데이트합니다.
    """
    date: str
    risk_profile: str  # e.g., "conservative", "moderate", "aggressive"
    target_tickers: List[str]
    
    # 데이터 수집 단계 결과 (기본값을 빈 dict로 변경)
    technical_analysis: dict
    news_data: dict
    
    # 분석 및 전략 단계 결과
    sentiment_analysis: Optional[dict]
    market_outlook: Optional[str]
    investment_plan: Optional[InvestmentPlan]
    
    # 실행 및 감사 단계 결과
    trade_results: Optional[List[Dict[str, any]]]
    daily_pnl: Optional[float]
    
    # 에러 및 메타 정보
    error_message: Optional[str]
    log: List[str] 