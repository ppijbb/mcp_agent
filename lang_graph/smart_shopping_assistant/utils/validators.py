"""
검증 유틸리티 함수
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_user_query(query: str) -> bool:
    """
    사용자 쿼리 검증
    
    Args:
        query: 사용자 쿼리 문자열
    
    Returns:
        검증 성공 여부
    """
    if not isinstance(query, str):
        logger.error("query must be a string")
        return False
    
    if not query.strip():
        logger.error("query cannot be empty")
        return False
    
    return True


def validate_product_info(product_info: Dict[str, Any]) -> bool:
    """
    제품 정보 검증
    
    Args:
        product_info: 제품 정보 딕셔너리
    
    Returns:
        검증 성공 여부
    """
    if not isinstance(product_info, dict):
        logger.error("product_info must be a dictionary")
        return False
    
    if "name" not in product_info:
        logger.error("product_info must contain 'name' field")
        return False
    
    return True


def validate_purchase_history(purchase_history: List[Dict[str, Any]]) -> bool:
    """
    구매 이력 검증
    
    Args:
        purchase_history: 구매 이력 리스트
    
    Returns:
        검증 성공 여부
    """
    if not isinstance(purchase_history, list):
        logger.error("purchase_history must be a list")
        return False
    
    for item in purchase_history:
        if not isinstance(item, dict):
            logger.error("Each purchase history item must be a dictionary")
            return False
    
    return True

