from typing import List, TypedDict, Optional, Dict, Any
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
    user_id: Optional[str]  # 사용자 ID (재무 분석용)
    
    # 재무 분석 단계 결과
    financial_analysis: Optional[Dict[str, Any]]  # 소비 패턴, 예산, 저축 목표, 건강 점수
    budget_status: Optional[Dict[str, Any]]  # 예산 상태
    savings_progress: Optional[Dict[str, Any]]  # 저축 목표 진행률
    
    # 세금 최적화 단계 결과
    tax_optimization: Optional[Dict[str, Any]]  # 공제 항목, 세금 최적화 전략
    
    # 부채 관리 단계 결과
    debt_management: Optional[Dict[str, Any]]  # 대출 상환 전략, 이자 최소화 계획
    
    # 재무 목표 달성 단계 결과
    financial_goals: Optional[List[Dict[str, Any]]]  # 장기 재무 목표 목록
    goal_progress: Optional[Dict[str, Any]]  # 목표별 진행률
    
    # 데이터 수집 단계 결과 (기본값을 빈 dict로 변경)
    technical_analysis: dict
    news_data: dict
    
    # 차트 분석 단계 결과 (신규)
    ohlcv_data: Optional[Dict[str, List[Dict[str, Any]]]]  # 티커별 OHLCV 데이터
    chart_analysis: Optional[Dict[str, Dict[str, Any]]]  # 티커별 차트 분석 결과
    chart_images: Optional[Dict[str, str]]  # 티커별 차트 이미지 (base64 또는 파일 경로)
    technical_indicators_advanced: Optional[Dict[str, Dict[str, Any]]]  # 고급 기술적 지표
    
    # 분석 및 전략 단계 결과
    sentiment_analysis: Optional[dict]
    market_outlook: Optional[str]
    
    # 최종 지표 산출 및 매도시점 추측 (신규)
    synthesized_indicators: Optional[Dict[str, Dict[str, Any]]]  # 최종 통합 지표
    exit_point_predictions: Optional[Dict[str, Dict[str, Any]]]  # 티커별 매도시점 추측
    
    investment_plan: Optional[InvestmentPlan]
    
    # 실행 및 감사 단계 결과
    trade_results: Optional[List[Dict[str, any]]]
    daily_pnl: Optional[float]
    
    # 구조적 상업성 요소
    commission_rate: Optional[float]  # 거래 수수료율
    total_commission: Optional[float]  # 총 수수료
    affiliate_commission: Optional[float]  # 제휴 수수료
    
    # 에러 및 메타 정보
    error_message: Optional[str]
    log: List[str] 