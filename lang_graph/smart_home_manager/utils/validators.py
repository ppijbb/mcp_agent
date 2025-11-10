"""
검증 유틸리티 함수
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def validate_device_info(device_info: Dict[str, Any]) -> bool:
    """
    기기 정보 검증
    
    Args:
        device_info: 기기 정보 딕셔너리
    
    Returns:
        검증 성공 여부
    """
    if not isinstance(device_info, dict):
        logger.error("device_info must be a dictionary")
        return False
    
    if "device_id" not in device_info:
        logger.error("device_info must contain 'device_id' field")
        return False
    
    return True


def validate_home_id(home_id: str) -> bool:
    """
    홈 ID 검증
    
    Args:
        home_id: 홈 ID 문자열
    
    Returns:
        검증 성공 여부
    """
    if not isinstance(home_id, str):
        logger.error("home_id must be a string")
        return False
    
    if not home_id.strip():
        logger.error("home_id cannot be empty")
        return False
    
    return True

