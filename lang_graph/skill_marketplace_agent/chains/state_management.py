"""
Skill Marketplace 워크플로우 상태 관리
"""

from typing import List, Dict, Any, Optional, TypedDict


class MarketplaceState(TypedDict, total=False):
    """
    LangGraph 워크플로우를 위한 상태 정의.
    Skill Marketplace 프로세스의 각 단계를 추적합니다.
    """
    user_input: str  # 초기 사용자 요청
    learner_id: str  # 학습자 ID
    learner_profile: Dict[str, Any]  # 학습자 프로필
    skill_path: Dict[str, Any]  # 추천된 스킬 경로
    matched_instructors: List[Dict[str, Any]]  # 매칭된 강사 목록
    recommended_content: List[Dict[str, Any]]  # 추천된 컨텐츠 목록
    learning_plan: Dict[str, Any]  # 학습 계획
    marketplace_transaction: Dict[str, Any]  # Marketplace 거래 정보
    final_report: Dict[str, Any]  # 최종 리포트
    errors: List[str]  # 워크플로우 중 발생한 오류
    current_step: str  # 현재 워크플로우 단계
    retry_count: int  # 현재 단계 재시도 횟수

