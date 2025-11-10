"""
검증 유틸리티 함수
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_device_info(device_info: Dict[str, Any]) -> bool:
    """
    의료기기 정보 검증
    
    Args:
        device_info: 의료기기 정보 딕셔너리
    
    Returns:
        검증 성공 여부
    """
    required_fields = ["name", "type", "classification"]
    
    if not isinstance(device_info, dict):
        logger.error("device_info must be a dictionary")
        return False
    
    for field in required_fields:
        if field not in device_info:
            logger.error(f"Missing required field: {field}")
            return False
    
    return True


def validate_regulatory_framework(framework: str) -> bool:
    """
    규제 프레임워크 검증
    
    Args:
        framework: 규제 프레임워크 이름
    
    Returns:
        검증 성공 여부
    """
    valid_frameworks = ["FDA 510(k)", "CE Marking", "ISO 13485", "MDR", "IVDR"]
    
    if framework not in valid_frameworks:
        logger.warning(f"Unknown regulatory framework: {framework}")
        return False
    
    return True


def validate_test_case(test_case: Dict[str, Any]) -> bool:
    """
    테스트 케이스 검증
    
    Args:
        test_case: 테스트 케이스 딕셔너리
    
    Returns:
        검증 성공 여부
    """
    required_fields = ["id", "name", "description", "requirements"]
    
    if not isinstance(test_case, dict):
        logger.error("test_case must be a dictionary")
        return False
    
    for field in required_fields:
        if field not in test_case:
            logger.error(f"Missing required field in test case: {field}")
            return False
    
    return True


def validate_test_result(test_result: Dict[str, Any]) -> bool:
    """
    테스트 결과 검증
    
    Args:
        test_result: 테스트 결과 딕셔너리
    
    Returns:
        검증 성공 여부
    """
    required_fields = ["test_case_id", "status", "timestamp"]
    
    if not isinstance(test_result, dict):
        logger.error("test_result must be a dictionary")
        return False
    
    for field in required_fields:
        if field not in test_result:
            logger.error(f"Missing required field in test result: {field}")
            return False
    
    valid_statuses = ["PASS", "FAIL", "PARTIAL", "SKIP"]
    if test_result["status"] not in valid_statuses:
        logger.error(f"Invalid test status: {test_result['status']}")
        return False
    
    return True

