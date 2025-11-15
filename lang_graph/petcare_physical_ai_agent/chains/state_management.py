"""
PetCare 워크플로우 상태 관리
"""

from typing import List, Dict, Any, Optional, TypedDict


class PetCareState(TypedDict, total=False):
    """
    LangGraph 워크플로우를 위한 상태 정의.
    반려동물 케어 프로세스의 각 단계를 추적합니다.
    """
    user_input: str  # 초기 사용자 요청
    pet_id: str  # 반려동물 ID
    pet_profile: Dict[str, Any]  # 반려동물 프로필
    health_status: Dict[str, Any]  # 건강 상태
    behavior_analysis: Dict[str, Any]  # 행동 분석 결과
    care_plan: Dict[str, Any]  # 맞춤형 케어 계획
    device_control_results: List[Dict[str, Any]]  # Physical AI 기기 제어 결과
    final_report: Dict[str, Any]  # 최종 리포트
    errors: List[str]  # 워크플로우 중 발생한 오류
    current_step: str  # 현재 워크플로우 단계
    retry_count: int  # 현재 단계 재시도 횟수

