"""
State Management for LangGraph Shopping Chain

LangGraph StateGraph에서 사용할 상태 정의
"""

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime


class ShoppingState(TypedDict):
    """
    스마트 쇼핑 어시스턴트 워크플로우 상태
    
    LangGraph StateGraph에서 사용하는 상태 구조
    """
    # 사용자 정보
    user_id: str
    query: str
    
    # 선호도 분석 결과
    preferences: Dict[str, Any]
    purchase_history: List[Dict[str, Any]]
    
    # 가격 비교 결과
    price_comparison_results: List[Dict[str, Any]]
    
    # 제품 추천 결과
    recommendations: List[Dict[str, Any]]
    
    # 리뷰 분석 결과
    review_analysis: Dict[str, Any]
    
    # 할인 알림
    deal_alerts: List[Dict[str, Any]]
    
    # 최종 추천
    final_recommendations: List[Dict[str, Any]]
    
    # 메타데이터
    timestamp: str
    workflow_stage: str  # "preference_analysis", "price_comparison", "product_recommendation", "review_analysis", "deal_alerts", "final"
    errors: List[str]
    warnings: List[str]

