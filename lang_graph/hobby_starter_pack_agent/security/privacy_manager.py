"""
프라이버시 관리자
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PrivacyManager:
    """프라이버시 관리자 (GDPR 준수)"""
    
    def __init__(self):
        """PrivacyManager 초기화"""
        self.consent_logs: list = []
        self.data_retention_days = 365  # 데이터 보관 기간
    
    def record_consent(
        self,
        user_id: str,
        consent_type: str,
        granted: bool,
        purpose: str
    ):
        """
        동의 기록
        
        Args:
            user_id: 사용자 ID
            consent_type: 동의 타입
            granted: 동의 여부
            purpose: 목적
        """
        log_entry = {
            "user_id": user_id,
            "consent_type": consent_type,
            "granted": granted,
            "purpose": purpose,
            "timestamp": datetime.now().isoformat(),
        }
        self.consent_logs.append(log_entry)
        logger.info(f"Consent recorded: {user_id} - {consent_type}: {granted}")
    
    def check_consent(self, user_id: str, consent_type: str) -> bool:
        """
        동의 확인
        
        Args:
            user_id: 사용자 ID
            consent_type: 동의 타입
        
        Returns:
            동의 여부
        """
        # 최근 동의 기록 확인
        for log in reversed(self.consent_logs):
            if log["user_id"] == user_id and log["consent_type"] == consent_type:
                return log["granted"]
        return False
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 익명화
        
        Args:
            data: 익명화할 데이터
        
        Returns:
            익명화된 데이터
        """
        anonymized = data.copy()
        # 개인 식별 정보 제거
        anonymized.pop("email", None)
        anonymized.pop("phone", None)
        anonymized.pop("address", None)
        if "user_id" in anonymized:
            anonymized["user_id"] = f"user_{hash(str(anonymized['user_id']))}"
        return anonymized
    
    def should_delete_data(self, created_at: datetime) -> bool:
        """
        데이터 삭제 여부 확인 (보관 기간 초과)
        
        Args:
            created_at: 생성 일시
        
        Returns:
            삭제 여부
        """
        retention_date = datetime.now() - timedelta(days=self.data_retention_days)
        return created_at < retention_date

