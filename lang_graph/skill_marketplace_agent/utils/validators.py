"""
입력 유효성 검사 유틸리티
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """입력 유효성 검사 실패 시 발생하는 사용자 정의 예외"""
    pass


def validate_learner_profile(learner_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    학습자 프로필 정보의 유효성을 검사합니다.
    
    Args:
        learner_profile: 검사할 학습자 프로필 딕셔너리
    
    Returns:
        유효성이 검사된 학습자 프로필 정보
    
    Raises:
        InputValidationError: 필수 필드가 누락되었거나 형식이 잘못된 경우
    """
    required_fields = ["learner_id"]
    
    for field in required_fields:
        if field not in learner_profile or not learner_profile[field]:
            raise InputValidationError(f"필수 필드 '{field}'가 누락되었거나 비어 있습니다.")
    
    # 학습 스타일 검증
    if "learning_style" in learner_profile and learner_profile["learning_style"]:
        valid_styles = ["visual", "auditory", "kinesthetic", "reading"]
        if learner_profile["learning_style"].lower() not in valid_styles:
            logger.warning(f"Unknown learning style: {learner_profile['learning_style']}, but continuing...")
    
    # 예산 범위 검증
    if "budget_range" in learner_profile and learner_profile["budget_range"]:
        valid_budgets = ["low", "medium", "high"]
        if learner_profile["budget_range"].lower() not in valid_budgets:
            logger.warning(f"Unknown budget range: {learner_profile['budget_range']}, but continuing...")
    
    logger.info(f"학습자 프로필 유효성 검사 완료: {learner_profile['learner_id']}")
    return learner_profile

