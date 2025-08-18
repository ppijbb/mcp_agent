"""
보안 및 프라이버시 보호 모듈
에이전트 간 통신과 외부 도구 연동 시 보안을 강화합니다.
"""

import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecurityContext:
    """보안 컨텍스트"""
    agent_id: str
    session_id: str
    permissions: List[str]
    encryption_key: bytes
    signature_key: bytes
    created_at: datetime
    expires_at: datetime
    
    def is_valid(self) -> bool:
        """보안 컨텍스트 유효성 검사"""
        return datetime.now() < self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """권한 확인"""
        return permission in self.permissions

class SecurityManager:
    """보안 관리자"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode('utf-8')
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.blacklist: set = set()
    
    def create_session(self, agent_id: str, permissions: List[str], 
                      session_duration: int = 3600) -> SecurityContext:
        """보안 세션 생성"""
        session_id = secrets.token_urlsafe(32)
        encryption_key = secrets.token_bytes(32)
        signature_key = secrets.token_bytes(32)
        
        context = SecurityContext(
            agent_id=agent_id,
            session_id=session_id,
            permissions=permissions,
            encryption_key=encryption_key,
            signature_key=signature_key,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=session_duration)
        )
        
        self.active_sessions[session_id] = context
        logger.info(f"Security session created for agent {agent_id}: {session_id}")
        return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """세션 유효성 검사"""
        if session_id in self.blacklist:
            return None
        
        context = self.active_sessions.get(session_id)
        if context and context.is_valid():
            return context
        
        # 만료된 세션 정리
        if context:
            del self.active_sessions[session_id]
        
        return None
    
    def revoke_session(self, session_id: str) -> bool:
        """세션 취소"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.blacklist.add(session_id)
            logger.info(f"Security session revoked: {session_id}")
            return True
        return False
    
    def encrypt_data(self, data: str, encryption_key: bytes) -> str:
        """데이터 암호화"""
        try:
            # 간단한 XOR 암호화 (실제 프로덕션에서는 AES 등 사용)
            data_bytes = data.encode('utf-8')
            encrypted = bytes(a ^ b for a, b in zip(data_bytes, encryption_key * (len(data_bytes) // len(encryption_key) + 1)))
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            return data
    
    def decrypt_data(self, encrypted_data: str, encryption_key: bytes) -> str:
        """데이터 복호화"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, encryption_key * (len(encrypted_bytes) // len(encryption_key) + 1)))
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return encrypted_data
    
    def sign_data(self, data: str, signature_key: bytes) -> str:
        """데이터 서명"""
        try:
            signature = hmac.new(signature_key, data.encode('utf-8'), hashlib.sha256).digest()
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.error(f"Signing failed: {str(e)}")
            return ""
    
    def verify_signature(self, data: str, signature: str, signature_key: bytes) -> bool:
        """서명 검증"""
        try:
            expected_signature = hmac.new(signature_key, data.encode('utf-8'), hashlib.sha256).digest()
            expected_signature_b64 = base64.b64encode(expected_signature).decode('utf-8')
            return hmac.compare_digest(signature, expected_signature_b64)
        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False
    
    def check_rate_limit(self, agent_id: str, operation: str, limit: int = 100, window: int = 3600) -> bool:
        """속도 제한 확인"""
        key = f"{agent_id}:{operation}"
        now = datetime.now()
        
        if key not in self.rate_limiters:
            self.rate_limiters[key] = {
                'count': 0,
                'window_start': now,
                'limit': limit,
                'window': window
            }
        
        limiter = self.rate_limiters[key]
        
        # 윈도우가 지났으면 리셋
        if (now - limiter['window_start']).total_seconds() > limiter['window']:
            limiter['count'] = 0
            limiter['window_start'] = now
        
        # 제한 확인
        if limiter['count'] >= limiter['limit']:
            logger.warning(f"Rate limit exceeded for {agent_id}:{operation}")
            return False
        
        limiter['count'] += 1
        return True
    
    def generate_jwt_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """JWT 토큰 생성"""
        try:
            payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
            payload['iat'] = datetime.utcnow()
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return token
        except Exception as e:
            logger.error(f"JWT token generation failed: {str(e)}")
            return ""
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"JWT token verification failed: {str(e)}")
            return None

class PrivacyManager:
    """프라이버시 관리자"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b',  # 신용카드 번호
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 이메일
            r'\b\d{10,11}\b',  # 전화번호
        ]
        self.data_retention_policies: Dict[str, int] = {}
        self.encryption_enabled = True
    
    def sanitize_data(self, data: str) -> str:
        """민감한 데이터 제거/마스킹"""
        import re
        
        sanitized = data
        for pattern in self.sensitive_patterns:
            if 'email' in pattern:
                sanitized = re.sub(pattern, '[EMAIL]', sanitized)
            elif 'credit' in pattern or 'SSN' in pattern:
                sanitized = re.sub(pattern, '[SENSITIVE]', sanitized)
            else:
                sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized
    
    def set_data_retention_policy(self, data_type: str, retention_days: int):
        """데이터 보존 정책 설정"""
        self.data_retention_policies[data_type] = retention_days
        logger.info(f"Data retention policy set for {data_type}: {retention_days} days")
    
    def should_retain_data(self, data_type: str, created_at: datetime) -> bool:
        """데이터 보존 여부 확인"""
        if data_type not in self.data_retention_policies:
            return True  # 정책이 없으면 보존
        
        retention_days = self.data_retention_policies[data_type]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        return created_at > cutoff_date
    
    def enable_encryption(self, enabled: bool = True):
        """암호화 활성화/비활성화"""
        self.encryption_enabled = enabled
        logger.info(f"Encryption {'enabled' if enabled else 'disabled'}")

class AuditLogger:
    """감사 로그 관리자"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_access(self, agent_id: str, resource: str, action: str, 
                   success: bool, details: Dict[str, Any] = None):
        """접근 로그 기록"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'resource': resource,
            'action': action,
            'success': success,
            'details': details or {}
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"Access: {log_entry}")
    
    def log_security_event(self, event_type: str, severity: str, 
                          agent_id: str = None, details: Dict[str, Any] = None):
        """보안 이벤트 로그 기록"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'agent_id': agent_id,
            'details': details or {}
        }
        
        level_map = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = level_map.get(severity.lower(), logging.INFO)
        self.logger.log(level, f"Security Event: {log_entry}")
    
    def log_data_access(self, agent_id: str, data_type: str, 
                       operation: str, data_size: int = 0):
        """데이터 접근 로그 기록"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'data_type': data_type,
            'operation': operation,
            'data_size': data_size
        }
        
        self.logger.info(f"Data Access: {log_entry}")

# 전역 보안 관리자 인스턴스들
security_manager = SecurityManager(secrets.token_urlsafe(32))
privacy_manager = PrivacyManager()
audit_logger = AuditLogger()

# 기본 프라이버시 정책 설정
privacy_manager.set_data_retention_policy("search_results", 30)
privacy_manager.set_data_retention_policy("user_queries", 90)
privacy_manager.set_data_retention_policy("agent_communications", 180)
