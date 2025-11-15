"""
입력 유효성 검사 유틸리티
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """입력 유효성 검사 실패 시 발생하는 사용자 정의 예외"""
    pass


def validate_pet_profile(pet_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    반려동물 프로필 정보의 유효성을 검사합니다.
    
    Args:
        pet_profile: 검사할 반려동물 프로필 딕셔너리
    
    Returns:
        유효성이 검사된 반려동물 프로필 정보
    
    Raises:
        InputValidationError: 필수 필드가 누락되었거나 형식이 잘못된 경우
    """
    required_fields = ["pet_id"]
    
    for field in required_fields:
        if field not in pet_profile or not pet_profile[field]:
            raise InputValidationError(f"필수 필드 '{field}'가 누락되었거나 비어 있습니다.")
    
    # 종류 검증
    if "species" in pet_profile and pet_profile["species"]:
        valid_species = ["dog", "cat", "bird", "rabbit", "hamster", "other"]
        if pet_profile["species"].lower() not in valid_species:
            logger.warning(f"Unknown species: {pet_profile['species']}, but continuing...")
    
    # 나이 검증
    if "age" in pet_profile and pet_profile["age"]:
        if not isinstance(pet_profile["age"], int) or pet_profile["age"] < 0:
            raise InputValidationError("필드 'age'는 0 이상의 정수여야 합니다.")
    
    logger.info(f"반려동물 프로필 유효성 검사 완료: {pet_profile['pet_id']}")
    return pet_profile

