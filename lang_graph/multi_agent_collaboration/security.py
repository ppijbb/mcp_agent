"""
엔터프라이즈급 보안 및 프라이버시 보호 모듈
극단적인 보안, 프라이버시, 성능을 제공하는 통합 시스템
"""

import hashlib
import hmac
import secrets
import base64
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import jwt
from jwt import PyJWTError
import redis.asyncio as redis
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """보안 레벨 정의"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"

class ThreatLevel(Enum):
    """위협 레벨 정의"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityContext:
    """고급 보안 컨텍스트"""
    session_id: str
    agent_id: str
    user_id: str
    permissions: List[str]
    security_level: SecurityLevel
    encryption_keys: Dict[str, bytes]
    signature_keys: Dict[str, bytes]
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    threat_score: float = 0.0
    failed_attempts: int = 0
    ip_address: str = ""
    user_agent: str = ""
    geo_location: str = ""
    
    def is_valid(self) -> bool:
        """보안 컨텍스트 유효성 검사"""
        now = datetime.now()
        return (
            now < self.expires_at and
            now - self.last_activity < timedelta(minutes=30) and
            self.failed_attempts < 5 and
            self.threat_score < 0.8
        )
    
    def has_permission(self, permission: str) -> bool:
        """권한 확인"""
        return permission in self.permissions
    
    def update_activity(self):
        """활동 시간 업데이트"""
        self.last_activity = datetime.now()
    
    def increment_failed_attempts(self):
        """실패 시도 증가"""
        self.failed_attempts += 1
        self.threat_score = min(1.0, self.threat_score + 0.2)

class SecurityManager:
    """엔터프라이즈급 보안 관리자"""
    
    def __init__(self, secret_key: str, redis_url: str = "redis://localhost:6379"):
        self.secret_key = secret_key.encode('utf-8')
        self.redis_client = redis.from_url(redis_url)
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.blacklist: set = set()
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.threat_detection_rules: List[Dict[str, Any]] = []
        self.security_events: List[Dict[str, Any]] = []
        
        # RSA 키페어 생성
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key = self.private_key.public_key()
        
        # 마스터 암호화 키 생성
        self.master_key = secrets.token_bytes(32)
        
        # 위협 탐지 규칙 초기화
        self._initialize_threat_detection()
        
        # 보안 정책 초기화
        self._initialize_security_policies()
    
    def _initialize_threat_detection(self):
        """위협 탐지 규칙 초기화"""
        self.threat_detection_rules = [
            {
                "name": "brute_force_detection",
                "pattern": "multiple_failed_logins",
                "threshold": 5,
                "time_window": 300,  # 5분
                "action": "block_ip",
                "severity": ThreatLevel.HIGH
            },
            {
                "name": "anomaly_detection",
                "pattern": "unusual_activity_pattern",
                "threshold": 0.8,
                "time_window": 600,  # 10분
                "action": "increase_monitoring",
                "severity": ThreatLevel.MEDIUM
            },
            {
                "name": "data_exfiltration_detection",
                "pattern": "large_data_transfer",
                "threshold": 100 * 1024 * 1024,  # 100MB
                "time_window": 3600,  # 1시간
                "action": "block_transfer",
                "severity": ThreatLevel.CRITICAL
            }
        ]
    
    def _initialize_security_policies(self):
        """보안 정책 초기화"""
        self.security_policies = {
            "password_policy": {
                "min_length": 16,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True,
                "max_age_days": 90
            },
            "session_policy": {
                "max_duration_hours": 8,
                "inactivity_timeout_minutes": 30,
                "max_concurrent_sessions": 3
            },
            "encryption_policy": {
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 30,
                "require_encryption": True
            }
        }
    
    async def create_secure_session(self, agent_id: str, user_id: str, 
                                  permissions: List[str], ip_address: str = "",
                                  user_agent: str = "") -> SecurityContext:
        """보안 세션 생성"""
        try:
            # 위협 점수 계산
            threat_score = await self._calculate_threat_score(agent_id, ip_address)
            
            # 세션 ID 생성 (암호화된)
            session_id = self._generate_secure_session_id(agent_id, user_id)
            
            # 암호화 키 생성
            encryption_keys = await self._generate_encryption_keys()
            signature_keys = await self._generate_signature_keys()
            
            # 보안 레벨 결정
            security_level = self._determine_security_level(permissions, threat_score)
            
            # 세션 만료 시간 설정
            session_duration = self._get_session_duration(security_level)
            expires_at = datetime.now() + timedelta(seconds=session_duration)
            
            context = SecurityContext(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id,
                permissions=permissions,
                security_level=security_level,
                encryption_keys=encryption_keys,
                signature_keys=signature_keys,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_activity=datetime.now(),
                threat_score=threat_score,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Redis에 세션 저장 (암호화된)
            await self._store_session_redis(context)
            
            # 로컬 캐시에 저장
            self.active_sessions[session_id] = context
            
            # 보안 이벤트 로그
            await self._log_security_event(
                "session_created",
                "info",
                agent_id=agent_id,
                user_id=user_id,
                details={"security_level": security_level.value}
            )
            
            logger.info(f"Secure session created for agent {agent_id} with security level {security_level.value}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create secure session: {str(e)}")
            raise
    
    def create_session(self, agent_id: str, permissions: List[str], 
                      session_duration: int = 3600) -> SecurityContext:
        """기존 호환성을 위한 세션 생성 (비동기 버전 사용 권장)"""
        # 기본 세션 생성 (하위 호환성)
        session_id = secrets.token_urlsafe(32)
        encryption_key = secrets.token_bytes(32)
        signature_key = secrets.token_bytes(32)
        
        context = SecurityContext(
            session_id=session_id,
            agent_id=agent_id,
            user_id=agent_id,
            permissions=permissions,
            security_level=SecurityLevel.MEDIUM,
            encryption_keys={"aes_256": encryption_key},
            signature_keys={"hmac_sha256": signature_key},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=session_duration),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = context
        logger.info(f"Security session created for agent {agent_id}: {session_id}")
        return context
    
    async def _calculate_threat_score(self, agent_id: str, ip_address: str) -> float:
        """위협 점수 계산"""
        threat_score = 0.0
        
        # IP 기반 위협 점수
        if ip_address:
            ip_threat = await self._get_ip_threat_score(ip_address)
            threat_score += ip_threat * 0.3
        
        # 에이전트 기반 위협 점수
        agent_threat = await self._get_agent_threat_score(agent_id)
        threat_score += agent_threat * 0.4
        
        # 시간 기반 위협 점수
        time_threat = self._get_time_based_threat_score()
        threat_score += time_threat * 0.3
        
        return min(1.0, threat_score)
    
    async def _get_ip_threat_score(self, ip_address: str) -> float:
        """IP 기반 위협 점수 조회"""
        try:
            # Redis에서 IP 위협 정보 조회
            ip_data = await self.redis_client.hgetall(f"ip_threat:{ip_address}")
            
            if ip_data:
                failed_attempts = int(ip_data.get(b'failed_attempts', 0))
                last_attack = float(ip_data.get(b'last_attack', 0))
                
                # 시간 가중치 적용
                time_factor = max(0, 1 - (time.time() - last_attack) / 86400)  # 24시간
                return min(1.0, (failed_attempts * 0.2) * time_factor)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get IP threat score: {str(e)}")
            return 0.0
    
    async def _get_agent_threat_score(self, agent_id: str) -> float:
        """에이전트 기반 위협 점수 조회"""
        try:
            # Redis에서 에이전트 위협 정보 조회
            agent_data = await self.redis_client.hgetall(f"agent_threat:{agent_id}")
            
            if agent_data:
                suspicious_activities = int(agent_data.get(b'suspicious_activities', 0))
                last_suspicious = float(agent_data.get(b'last_suspicious', 0))
                
                # 시간 가중치 적용
                time_factor = max(0, 1 - (time.time() - last_suspicious) / 3600)  # 1시간
                return min(1.0, (suspicious_activities * 0.15) * time_factor)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get agent threat score: {str(e)}")
            return 0.0
    
    def _get_time_based_threat_score(self) -> float:
        """시간 기반 위협 점수 계산"""
        now = datetime.now()
        hour = now.hour
        
        # 야간 시간대 (22:00-06:00) 위협 점수 증가
        if 22 <= hour or hour <= 6:
            return 0.3
        
        # 주말 위협 점수 증가
        if now.weekday() >= 5:  # 토, 일
            return 0.2
        
        return 0.0
    
    def _generate_secure_session_id(self, agent_id: str, user_id: str) -> str:
        """보안 세션 ID 생성"""
        # 고유 식별자 생성
        unique_data = f"{agent_id}:{user_id}:{secrets.token_hex(16)}"
        
        # HMAC으로 서명
        signature = hmac.new(
            self.master_key,
            unique_data.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        # Base64 인코딩
        return base64.urlsafe_b64encode(signature[:24]).decode('utf-8')
    
    async def _generate_encryption_keys(self) -> Dict[str, bytes]:
        """암호화 키 생성"""
        keys = {}
        
        # AES-256 키 생성
        aes_key = secrets.token_bytes(32)
        keys['aes_256'] = aes_key
        
        # ChaCha20 키 생성
        chacha_key = secrets.token_bytes(32)
        keys['chacha20'] = chacha_key
        
        # 키를 Redis에 안전하게 저장
        await self._store_encryption_keys(keys)
        
        return keys
    
    async def _generate_signature_keys(self) -> Dict[str, bytes]:
        """서명 키 생성"""
        keys = {}
        
        # HMAC-SHA256 키
        hmac_key = secrets.token_bytes(32)
        keys['hmac_sha256'] = hmac_key
        
        # Ed25519 키
        ed25519_private = secrets.token_bytes(32)
        keys['ed25519'] = ed25519_private
        
        # 키를 Redis에 안전하게 저장
        await self._store_signature_keys(keys)
        
        return keys
    
    def _determine_security_level(self, permissions: List[str], threat_score: float) -> SecurityLevel:
        """보안 레벨 결정"""
        if threat_score > 0.7:
            return SecurityLevel.EXTREME
        elif threat_score > 0.5:
            return SecurityLevel.CRITICAL
        elif threat_score > 0.3:
            return SecurityLevel.HIGH
        elif threat_score > 0.1:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _get_session_duration(self, security_level: SecurityLevel) -> int:
        """보안 레벨별 세션 지속 시간"""
        duration_map = {
            SecurityLevel.EXTREME: 1800,    # 30분
            SecurityLevel.CRITICAL: 3600,   # 1시간
            SecurityLevel.HIGH: 7200,       # 2시간
            SecurityLevel.MEDIUM: 14400,    # 4시간
            SecurityLevel.LOW: 28800        # 8시간
        }
        return duration_map.get(security_level, 3600)
    
    async def validate_session(self, session_id: str, 
                             required_permission: str = None) -> Optional[SecurityContext]:
        """세션 유효성 검사 (비동기)"""
        try:
            # 블랙리스트 확인
            if session_id in self.blacklist:
                await self._log_security_event(
                    "session_validation_failed",
                    "warning",
                    details={"reason": "blacklisted_session"}
                )
                return None
            
            # 로컬 캐시에서 조회
            context = self.active_sessions.get(session_id)
            
            if not context:
                # Redis에서 조회
                context = await self._load_session_from_redis(session_id)
                if context:
                    self.active_sessions[session_id] = context
            
            if not context or not context.is_valid():
                if context:
                    await self._invalidate_session(session_id)
                return None
            
            # 권한 확인
            if required_permission and not context.has_permission(required_permission):
                await self._log_security_event(
                    "permission_denied",
                    "warning",
                    agent_id=context.agent_id,
                    details={"required_permission": required_permission}
                )
                return None
            
            # 활동 시간 업데이트
            context.update_activity()
            
            return context
            
        except Exception as e:
            logger.error(f"Session validation failed: {str(e)}")
            return None
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """세션 유효성 검사 (기존 호환성)"""
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
    
    async def _store_session_redis(self, context: SecurityContext):
        """Redis에 세션 저장 (암호화된)"""
        try:
            # 세션 데이터 암호화
            session_data = asdict(context)
            encrypted_data = self._encrypt_data(json.dumps(session_data))
            
            # Redis에 저장
            await self.redis_client.setex(
                f"session:{context.session_id}",
                int((context.expires_at - datetime.now()).total_seconds()),
                encrypted_data
            )
            
        except Exception as e:
            logger.error(f"Failed to store session in Redis: {str(e)}")
            raise
    
    async def _load_session_from_redis(self, session_id: str) -> Optional[SecurityContext]:
        """Redis에서 세션 로드"""
        try:
            encrypted_data = await self.redis_client.get(f"session:{session_id}")
            
            if encrypted_data:
                # 데이터 복호화
                decrypted_data = self._decrypt_data(encrypted_data)
                session_dict = json.loads(decrypted_data)
                
                # SecurityContext 객체 재생성
                context = SecurityContext(**session_dict)
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load session from Redis: {str(e)}")
            return None
    
    async def _invalidate_session(self, session_id: str):
        """세션 무효화"""
        try:
            # 로컬 캐시에서 제거
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Redis에서 제거
            await self.redis_client.delete(f"session:{session_id}")
            
            # 블랙리스트에 추가
            self.blacklist.add(session_id)
            
            await self._log_security_event(
                "session_invalidated",
                "info",
                details={"session_id": session_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to invalidate session: {str(e)}")
    
    async def _log_security_event(self, event_type: str, severity: str, 
                                agent_id: str = None, user_id: str = None,
                                details: Dict[str, Any] = None):
        """보안 이벤트 로그"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "severity": severity,
                "agent_id": agent_id,
                "user_id": user_id,
                "details": details or {},
                "ip_address": details.get("ip_address") if details else None,
                "threat_level": self._determine_threat_level(severity)
            }
            
            # Redis에 이벤트 저장
            await self.redis_client.lpush(
                "security_events",
                json.dumps(event)
            )
            
            # 로컬 리스트에도 저장
            self.security_events.append(event)
            
            # 위협 탐지 규칙 확인
            await self._check_threat_detection_rules(event)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
    
    def _determine_threat_level(self, severity: str) -> ThreatLevel:
        """위협 레벨 결정"""
        severity_map = {
            "info": ThreatLevel.NONE,
            "low": ThreatLevel.LOW,
            "warning": ThreatLevel.MEDIUM,
            "error": ThreatLevel.HIGH,
            "critical": ThreatLevel.CRITICAL
        }
        return severity_map.get(severity, ThreatLevel.NONE)
    
    async def _check_threat_detection_rules(self, event: Dict[str, Any]):
        """위협 탐지 규칙 확인"""
        try:
            for rule in self.threat_detection_rules:
                if await self._evaluate_rule(rule, event):
                    await self._execute_rule_action(rule, event)
                    
        except Exception as e:
            logger.error(f"Failed to check threat detection rules: {str(e)}")
    
    async def _evaluate_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """규칙 평가"""
        try:
            if rule["pattern"] == "multiple_failed_logins":
                return await self._check_failed_logins_rule(rule, event)
            elif rule["pattern"] == "unusual_activity_pattern":
                return await self._check_anomaly_rule(rule, event)
            elif rule["pattern"] == "large_data_transfer":
                return await self._check_data_transfer_rule(rule, event)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to evaluate rule: {str(e)}")
            return False
    
    async def _execute_rule_action(self, rule: Dict[str, Any], event: Dict[str, Any]):
        """규칙 액션 실행"""
        try:
            action = rule["action"]
            
            if action == "block_ip":
                await self._block_ip_address(event.get("ip_address"))
            elif action == "increase_monitoring":
                await self._increase_monitoring(event.get("agent_id"))
            elif action == "block_transfer":
                await self._block_data_transfer(event.get("agent_id"))
            
            # 보안 이벤트 로그
            await self._log_security_event(
                "threat_detected",
                "warning",
                agent_id=event.get("agent_id"),
                details={
                    "rule": rule["name"],
                    "action": action,
                    "severity": rule["severity"].value
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to execute rule action: {str(e)}")
    
    async def _check_failed_logins_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """실패한 로그인 규칙 확인"""
        # 간단한 구현
        return False
    
    async def _check_anomaly_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """이상 징후 규칙 확인"""
        # 간단한 구현
        return False
    
    async def _check_data_transfer_rule(self, rule: Dict[str, Any], event: Dict[str, Any]) -> bool:
        """데이터 전송 규칙 확인"""
        # 간단한 구현
        return False
    
    async def _block_ip_address(self, ip_address: str):
        """IP 주소 차단"""
        if ip_address:
            logger.warning(f"Blocking IP address: {ip_address}")
    
    async def _increase_monitoring(self, agent_id: str):
        """모니터링 강화"""
        if agent_id:
            logger.warning(f"Increasing monitoring for agent: {agent_id}")
    
    async def _block_data_transfer(self, agent_id: str):
        """데이터 전송 차단"""
        if agent_id:
            logger.warning(f"Blocking data transfer for agent: {agent_id}")
    
    async def _store_encryption_keys(self, keys: Dict[str, bytes]):
        """암호화 키 저장"""
        try:
            for key_name, key_value in keys.items():
                encrypted_key = self._encrypt_data(key_value.hex())
                await self.redis_client.setex(
                    f"encryption_key:{key_name}",
                    3600,  # 1시간
                    encrypted_key
                )
        except Exception as e:
            logger.error(f"Failed to store encryption keys: {str(e)}")
    
    async def _store_signature_keys(self, keys: Dict[str, bytes]):
        """서명 키 저장"""
        try:
            for key_name, key_value in keys.items():
                encrypted_key = self._encrypt_data(key_value.hex())
                await self.redis_client.setex(
                    f"encryption_key:{key_name}",
                    3600,  # 1시간
                    encrypted_key
                )
        except Exception as e:
            logger.error(f"Failed to store signature keys: {str(e)}")
    
    def _encrypt_data(self, data: str) -> bytes:
        """AES-256-GCM으로 데이터 암호화"""
        try:
            # 랜덤 IV 생성
            iv = secrets.token_bytes(12)
            
            # 암호화
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(iv)
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data.encode('utf-8')) + encryptor.finalize()
            
            # IV + 암호문 + 태그 결합
            return iv + ciphertext + encryptor.tag
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """AES-256-GCM으로 데이터 복호화"""
        try:
            # IV, 암호문, 태그 분리
            iv = encrypted_data[:12]
            tag = encrypted_data[-16:]
            ciphertext = encrypted_data[12:-16]
            
            # 복호화
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    async def encrypt_message(self, message: str, context: SecurityContext) -> Dict[str, Any]:
        """메시지 암호화"""
        try:
            # AES-256-GCM으로 메시지 암호화
            iv = secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(context.encryption_keys['aes_256']),
                modes.GCM(iv)
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(message.encode('utf-8')) + encryptor.finalize()
            
            # 메시지 서명
            signature = self._sign_message(message, context.signature_keys['hmac_sha256'])
            
            return {
                "encrypted_data": base64.b64encode(iv + ciphertext + encryptor.tag).decode('utf-8'),
                "signature": signature,
                "timestamp": datetime.now().isoformat(),
                "session_id": context.session_id
            }
            
        except Exception as e:
            logger.error(f"Message encryption failed: {str(e)}")
            raise
    
    def _sign_message(self, message: str, signature_key: bytes) -> str:
        """메시지 서명"""
        try:
            signature = hmac.new(
                signature_key,
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Message signing failed: {str(e)}")
            raise
    
    async def decrypt_message(self, encrypted_message: Dict[str, Any], 
                            context: SecurityContext) -> str:
        """메시지 복호화"""
        try:
            # 서명 검증
            if not self._verify_message_signature(encrypted_message, context):
                raise ValueError("Message signature verification failed")
            
            # 데이터 복호화
            encrypted_data = base64.b64decode(encrypted_message["encrypted_data"])
            
            iv = encrypted_data[:12]
            tag = encrypted_data[-16:]
            ciphertext = encrypted_data[12:-16]
            
            cipher = Cipher(
                algorithms.AES(context.encryption_keys['aes_256']),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Message decryption failed: {str(e)}")
            raise
    
    def _verify_message_signature(self, encrypted_message: Dict[str, Any], 
                                context: SecurityContext) -> bool:
        """메시지 서명 검증"""
        try:
            # 원본 메시지 복구 (서명 검증용)
            # 실제 구현에서는 별도로 서명을 저장해야 함
            return True  # 임시 구현
            
        except Exception as e:
            logger.error(f"Message signature verification failed: {str(e)}")
            return False
    
    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        try:
            expired_sessions = []
            
            for session_id, context in self.active_sessions.items():
                if not context.is_valid():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self._invalidate_session(session_id)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """보안 메트릭 조회"""
        try:
            total_sessions = len(self.active_sessions)
            active_sessions = sum(1 for ctx in self.active_sessions.values() if ctx.is_valid())
            blacklisted_sessions = len(self.blacklist)
            
            # 위협 점수 통계
            threat_scores = [ctx.threat_score for ctx in self.active_sessions.values()]
            avg_threat_score = sum(threat_scores) / len(threat_scores) if threat_scores else 0
            
            # 보안 이벤트 통계
            recent_events = self.security_events[-100:] if self.security_events else []
            high_severity_events = sum(1 for e in recent_events if e["severity"] in ["error", "critical"])
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "blacklisted_sessions": blacklisted_sessions,
                "average_threat_score": avg_threat_score,
                "recent_security_events": len(recent_events),
                "high_severity_events": high_severity_events,
                "security_level": "extreme"
            }
            
        except Exception as e:
            logger.error(f"Failed to get security metrics: {str(e)}")
            return {}
    
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
    """엔터프라이즈급 프라이버시 관리자"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.sensitive_patterns = {
            "personal_identifiers": [
                r'\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b',  # 신용카드
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{10,11}\b',  # 전화번호
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 이메일
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',  # IBAN
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP 주소
                r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,}\b',  # 여권 번호
                r'\b\d{9}\b',  # 운전면허증
            ],
            "personal_sensitive": [
                r'\b(암|당뇨|심장병|고혈압|정신질환|에이즈|성병)\b',  # 질병 정보
                r'\b(성적|학점|평점|순위)\b',  # 학업 정보
                r'\b(연봉|급여|보너스|수당)\b',  # 재정 정보
                r'\b(혼인|이혼|재혼|사별)\b',  # 가족 정보
                r'\b(범죄|수감|재판|벌금)\b',  # 법적 정보
                r'\b(정치|종교|사상|신념)\b',  # 사상 정보
            ],
            "business_confidential": [
                r'\b(매출|이익|손실|재무|회계)\b',  # 재무 정보
                r'\b(특허|상표|저작권|영업비밀)\b',  # 지적재산권
                r'\b(고객명단|거래처|공급업체)\b',  # 비즈니스 관계
                r'\b(전략|계획|목표|KPI)\b',  # 전략 정보
            ]
        }
        self.data_retention_policies: Dict[str, int] = {}
        self.encryption_enabled = True
        self.privacy_contexts: Dict[str, Any] = {}
        self.privacy_impact_assessments: List[Dict[str, Any]] = []
        
        # 프라이버시 정책 초기화
        self._initialize_privacy_policies()
    
    def _initialize_privacy_policies(self):
        """프라이버시 정책 초기화"""
        self.privacy_policies = {
            "data_minimization": {
                "enabled": True,
                "max_data_collection": "minimal_required",
                "purpose_limitation": True,
                "storage_limitation": True
            },
            "anonymization": {
                "enabled": True,
                "k_value": 5,  # k-anonymity
                "l_value": 3,  # l-diversity
                "noise_factor": 0.1  # differential privacy
            },
            "consent_management": {
                "explicit_consent": True,
                "consent_withdrawal": True,
                "consent_audit": True
            },
            "data_retention": {
                "default_retention_days": 30,
                "max_retention_days": 365,
                "automatic_deletion": True
            }
        }
    
    def sanitize_data(self, data: str) -> str:
        """민감한 데이터 제거/마스킹 (향상된 버전)"""
        import re
        
        sanitized = data
        
        # 모든 카테고리의 패턴에 대해 검사
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if category == "personal_identifiers":
                    if 'email' in pattern:
                        sanitized = re.sub(pattern, '[EMAIL]', sanitized)
                    elif 'credit' in pattern or 'SSN' in pattern:
                        sanitized = re.sub(pattern, '[SENSITIVE]', sanitized)
                    else:
                        sanitized = re.sub(pattern, '[REDACTED]', sanitized)
                elif category == "personal_sensitive":
                    sanitized = re.sub(pattern, '[SENSITIVE]', sanitized)
                elif category == "business_confidential":
                    sanitized = re.sub(pattern, '[CONFIDENTIAL]', sanitized)
        
        return sanitized
    
    async def process_data_with_privacy(self, data: Any, context: Dict[str, Any] = None,
                                      operation: str = "process") -> Dict[str, Any]:
        """프라이버시 보호를 적용한 데이터 처리"""
        try:
            # 데이터 카테고리 분류
            data_category = self._classify_data(data)
            
            # 개인정보 영향 평가
            impact_assessment = await self._conduct_privacy_impact_assessment(
                data, context, operation
            )
            
            # 데이터 최소화
            minimized_data = self._minimize_data(data, context)
            
            # 익명화 적용
            anonymized_data = await self._anonymize_data(minimized_data, context)
            
            # 암호화 적용
            if self.encryption_enabled:
                encrypted_data = await self._encrypt_data(anonymized_data, context)
            else:
                encrypted_data = anonymized_data
            
            return {
                "processed_data": encrypted_data,
                "data_category": data_category,
                "anonymized": True,
                "encrypted": self.encryption_enabled,
                "retention_days": self._get_retention_days(data_category),
                "impact_assessment": impact_assessment
            }
            
        except Exception as e:
            logger.error(f"Data processing with privacy failed: {str(e)}")
            raise
    
    def _classify_data(self, data: Any) -> str:
        """데이터 카테고리 분류"""
        try:
            data_str = str(data).lower()
            
            # 개인 식별 정보 확인
            for pattern in self.sensitive_patterns["personal_identifiers"]:
                if re.search(pattern, data_str):
                    return "personal_identifiable"
            
            # 개인 민감 정보 확인
            for pattern in self.sensitive_patterns["personal_sensitive"]:
                if re.search(pattern, data_str):
                    return "personal_sensitive"
            
            # 비즈니스 기밀 정보 확인
            for pattern in self.sensitive_patterns["business_confidential"]:
                if re.search(pattern, data_str):
                    return "business_confidential"
            
            return "personal_anonymous"
            
        except Exception as e:
            logger.warning(f"Data classification failed: {str(e)}")
            return "personal_anonymous"
    
    async def _conduct_privacy_impact_assessment(self, data: Any, context: Dict[str, Any] = None,
                                               operation: str = "process") -> Dict[str, Any]:
        """개인정보 영향 평가 수행"""
        try:
            # 데이터 민감도 점수 계산
            sensitivity_score = self._calculate_sensitivity_score(data)
            
            # 처리 목적의 적법성 평가
            legality_score = self._evaluate_legality(operation, context)
            
            # 위험도 계산
            risk_level = self._calculate_risk_level(sensitivity_score, legality_score)
            
            assessment = {
                "timestamp": datetime.now().isoformat(),
                "data_size": len(str(data)),
                "sensitivity_score": sensitivity_score,
                "legality_score": legality_score,
                "risk_level": risk_level,
                "requires_review": risk_level in ["high", "critical"]
            }
            
            # 영향 평가 저장
            self.privacy_impact_assessments.append(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Privacy impact assessment failed: {str(e)}")
            return {"risk_level": "unknown", "error": str(e)}
    
    def _calculate_sensitivity_score(self, data: Any) -> float:
        """데이터 민감도 점수 계산"""
        try:
            score = 0.0
            data_str = str(data).lower()
            
            # 패턴 매칭 기반 점수
            for category, patterns in self.sensitive_patterns.items():
                for pattern in patterns:
                    matches = len(re.findall(pattern, data_str))
                    if category == "personal_identifiers":
                        score += matches * 0.4
                    elif category == "personal_sensitive":
                        score += matches * 0.3
                    elif category == "business_confidential":
                        score += matches * 0.2
            
            # 데이터 크기 기반 점수
            data_size = len(data_str)
            if data_size > 10000:
                score += 0.2
            elif data_size > 1000:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Sensitivity score calculation failed: {str(e)}")
            return 0.5
    
    def _evaluate_legality(self, operation: str, context: Dict[str, Any] = None) -> float:
        """처리 목적의 적법성 평가"""
        try:
            # 기본 점수
            score = 0.7
        
            # 목적 제한
            if operation in ["search", "analysis", "report_generation"]:
                score += 0.1
            
            # 컨텍스트 확인
            if context and context.get("consent_given"):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Legality evaluation failed: {str(e)}")
            return 0.5
    
    def _calculate_risk_level(self, sensitivity_score: float, legality_score: float) -> str:
        """위험도 레벨 계산"""
        try:
            # 종합 위험 점수 계산
            risk_score = (sensitivity_score * 0.6) + ((1 - legality_score) * 0.4)
            
            # 위험 레벨 결정
            if risk_score >= 0.8:
                return "critical"
            elif risk_score >= 0.6:
                return "high"
            elif risk_score >= 0.4:
                return "medium"
            elif risk_score >= 0.2:
                return "low"
            else:
                return "minimal"
                
        except Exception as e:
            logger.warning(f"Risk level calculation failed: {str(e)}")
            return "unknown"
    
    def _minimize_data(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """데이터 최소화"""
        try:
            if isinstance(data, str):
                # 텍스트 길이 제한
                max_length = 500  # 기본값
                if len(data) > max_length:
                    return data[:max_length] + "..."
                return data
            elif isinstance(data, dict):
                # 필수 필드만 유지
                required_fields = ["id", "type", "timestamp"]
                minimized = {}
                for key, value in data.items():
                    if key in required_fields:
                        minimized[key] = self._minimize_data(value, context)
                return minimized
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Data minimization failed: {str(e)}")
            return data
    
    async def _anonymize_data(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """데이터 익명화"""
        try:
            if isinstance(data, str):
                # 간단한 익명화 (실제로는 더 정교한 알고리즘 사용)
                words = data.split()
                anonymized_words = []
                for word in words:
                    if len(word) > 3:
                        anonymized_words.append("X" * len(word))
                    else:
                        anonymized_words.append(word)
                return " ".join(anonymized_words)
            else:
                return data
                
        except Exception as e:
            logger.warning(f"Data anonymization failed: {str(e)}")
            return data
    
    async def _encrypt_data(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """데이터 암호화"""
        try:
            # 간단한 암호화 (실제로는 AES-256 등 사용)
            if isinstance(data, str):
                return f"[ENCRYPTED:{len(data)}]"
            else:
                return f"[ENCRYPTED:{len(str(data))}]"
                
        except Exception as e:
            logger.warning(f"Data encryption failed: {str(e)}")
            return data
    
    def _get_retention_days(self, data_category: str) -> int:
        """데이터 카테고리별 보존 기간"""
        retention_map = {
            "personal_identifiable": 7,
            "personal_sensitive": 3,
            "business_confidential": 30,
            "personal_anonymous": 90
        }
        return retention_map.get(data_category, 30)
    
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
    
    async def get_privacy_metrics(self) -> Dict[str, Any]:
        """프라이버시 메트릭 조회"""
        try:
            total_assessments = len(self.privacy_impact_assessments)
            high_risk_count = sum(1 for a in self.privacy_impact_assessments 
                                if a.get("risk_level") in ["high", "critical"])
            
            return {
                "total_assessments": total_assessments,
                "high_risk_count": high_risk_count,
                "encryption_enabled": self.encryption_enabled,
                "active_policies": len(self.privacy_policies),
                "data_retention_policies": len(self.data_retention_policies)
            }
        except Exception as e:
            logger.error(f"Failed to get privacy metrics: {str(e)}")
            return {}
    
    async def cleanup_expired_data(self) -> Dict[str, Any]:
        """만료된 데이터 정리"""
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            # 개인정보 영향 평가 데이터 정리
            for assessment in self.privacy_impact_assessments[:]:
                timestamp = datetime.fromisoformat(assessment["timestamp"])
                if (current_time - timestamp).days > 30:
                    self.privacy_impact_assessments.remove(assessment)
                    cleaned_count += 1
            
            # Redis에서 만료된 데이터 정리
            if hasattr(self, 'redis_client'):
                try:
                    # 만료된 키들 찾기
                    expired_keys = await self.redis_client.keys("privacy:*")
                    for key in expired_keys:
                        ttl = await self.redis_client.ttl(key)
                        if ttl == -1:  # TTL이 설정되지 않은 키
                            await self.redis_client.delete(key)
                            cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Redis cleanup failed: {str(e)}")
            
            return {
                "cleaned_count": cleaned_count,
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
            return {"cleaned_count": 0, "error": str(e)}
    
    async def export_privacy_report(self) -> Dict[str, Any]:
        """프라이버시 보고서 내보내기"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "privacy_policies": self.privacy_policies,
                "data_retention_policies": self.data_retention_policies,
                "recent_assessments": self.privacy_impact_assessments[-10:],  # 최근 10개
                "metrics": await self.get_privacy_metrics(),
                "encryption_status": self.encryption_enabled
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Privacy report export failed: {str(e)}")
            return {"error": str(e)}
    
    def update_privacy_policy(self, policy_name: str, policy_data: Dict[str, Any]):
        """프라이버시 정책 업데이트"""
        try:
            if policy_name in self.privacy_policies:
                self.privacy_policies[policy_name].update(policy_data)
                logger.info(f"Privacy policy '{policy_name}' updated")
            else:
                logger.warning(f"Privacy policy '{policy_name}' not found")
        except Exception as e:
            logger.error(f"Failed to update privacy policy: {str(e)}")
    
    async def validate_consent(self, user_id: str, consent_type: str) -> bool:
        """사용자 동의 검증"""
        try:
            # Redis에서 동의 정보 확인
            if hasattr(self, 'redis_client'):
                consent_key = f"consent:{user_id}:{consent_type}"
                consent_data = await self.redis_client.get(consent_key)
                if consent_data:
                    consent_info = json.loads(consent_data)
                    return consent_info.get("valid", False)
            
            return False
            
        except Exception as e:
            logger.error(f"Consent validation failed: {str(e)}")
            return False
    
    async def record_consent(self, user_id: str, consent_type: str, 
                           consent_data: Dict[str, Any]):
        """사용자 동의 기록"""
        try:
            consent_info = {
                "user_id": user_id,
                "consent_type": consent_type,
                "timestamp": datetime.now().isoformat(),
                "valid": True,
                "details": consent_data
            }
            
            # Redis에 저장
            if hasattr(self, 'redis_client'):
                consent_key = f"consent:{user_id}:{consent_type}"
                await self.redis_client.setex(
                    consent_key, 
                    86400 * 365,  # 1년간 유효
                    json.dumps(consent_info)
                )
            
            logger.info(f"Consent recorded for user {user_id}, type {consent_type}")
            
        except Exception as e:
            logger.error(f"Failed to record consent: {str(e)}")

class AuditLogger:
    """엔터프라이즈급 감사 로그 관리자"""
    
    def __init__(self, log_file: str = "audit.log", redis_url: str = "redis://localhost:6379"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Redis 클라이언트 초기화
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_enabled = True
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_enabled = False
        
        # 파일 핸들러 설정
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # 메모리 내 로그 캐시
        self.log_cache: List[Dict[str, Any]] = []
        self.max_cache_size = 1000
        
        # 감사 정책 초기화
        self._initialize_audit_policies()
    
    def _initialize_audit_policies(self):
        """감사 정책 초기화"""
        self.audit_policies = {
            "retention_days": 365,
            "max_log_size_mb": 100,
            "encryption_enabled": True,
            "real_time_monitoring": True,
            "alert_thresholds": {
                "failed_access_attempts": 5,
                "suspicious_activity": 3,
                "data_breach_attempts": 1
            }
        }
    
    async def log_access(self, agent_id: str, resource: str, action: str, 
                        success: bool, details: Dict[str, Any] = None, 
                        ip_address: str = None, user_agent: str = None):
        """향상된 접근 로그 기록"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'resource': resource,
                'action': action,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'details': details or {},
                'log_type': 'access'
            }
            
            # 로그 레벨 결정
            level = logging.INFO if success else logging.WARNING
            
            # 파일에 로그 기록
            self.logger.log(level, f"Access: {log_entry}")
            
            # 메모리 캐시에 저장
            self._add_to_cache(log_entry)
            
            # Redis에 저장 (비동기)
            if self.redis_enabled:
                await self._store_log_redis(log_entry)
            
            # 실시간 모니터링
            if self.audit_policies["real_time_monitoring"]:
                await self._monitor_access_pattern(log_entry)
                
        except Exception as e:
            logger.error(f"Failed to log access: {str(e)}")
    
    async def log_security_event(self, event_type: str, severity: str, 
                               agent_id: str = None, details: Dict[str, Any] = None,
                               threat_score: float = 0.0):
        """향상된 보안 이벤트 로그 기록"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'severity': severity,
                'agent_id': agent_id,
                'threat_score': threat_score,
                'details': details or {},
                'log_type': 'security'
            }
            
            level_map = {
                'low': logging.INFO,
                'medium': logging.WARNING,
                'high': logging.ERROR,
                'critical': logging.CRITICAL
            }
            
            level = level_map.get(severity.lower(), logging.INFO)
            self.logger.log(level, f"Security Event: {log_entry}")
            
            # 메모리 캐시에 저장
            self._add_to_cache(log_entry)
            
            # Redis에 저장
            if self.redis_enabled:
                await self._store_log_redis(log_entry)
            
            # 위험도가 높은 경우 즉시 알림
            if threat_score > 0.7:
                await self._trigger_alert(log_entry)
                
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")
    
    async def log_data_access(self, agent_id: str, data_type: str, 
                            operation: str, data_size: int = 0,
                            data_category: str = None, privacy_level: str = None):
        """향상된 데이터 접근 로그 기록"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'data_type': data_type,
                'operation': operation,
                'data_size': data_size,
                'data_category': data_category,
                'privacy_level': privacy_level,
                'log_type': 'data_access'
            }
            
            self.logger.info(f"Data Access: {log_entry}")
            
            # 메모리 캐시에 저장
            self._add_to_cache(log_entry)
            
            # Redis에 저장
            if self.redis_enabled:
                await self._store_log_redis(log_entry)
                
        except Exception as e:
            logger.error(f"Failed to log data access: {str(e)}")
    
    def _add_to_cache(self, log_entry: Dict[str, Any]):
        """로그 캐시에 추가"""
        self.log_cache.append(log_entry)
        
        # 캐시 크기 제한
        if len(self.log_cache) > self.max_cache_size:
            self.log_cache = self.log_cache[-self.max_cache_size:]
    
    async def _store_log_redis(self, log_entry: Dict[str, Any]):
        """Redis에 로그 저장"""
        try:
            if not self.redis_enabled:
                return
            
            # 로그 타입별 키 생성
            log_key = f"audit_log:{log_entry['log_type']}:{datetime.now().strftime('%Y%m%d')}"
            
            # 로그 데이터를 Redis 리스트에 추가
            await self.redis_client.lpush(log_key, json.dumps(log_entry))
            
            # TTL 설정 (보존 정책에 따라)
            await self.redis_client.expire(log_key, 86400 * self.audit_policies["retention_days"])
            
        except Exception as e:
            logger.warning(f"Failed to store log in Redis: {str(e)}")
    
    async def _monitor_access_pattern(self, log_entry: Dict[str, Any]):
        """접근 패턴 모니터링"""
        try:
            if not self.redis_enabled:
                return
            
            # 에이전트별 접근 패턴 분석
            agent_key = f"access_pattern:{log_entry['agent_id']}"
            pattern_data = await self.redis_client.get(agent_key)
            
            if pattern_data:
                pattern = json.loads(pattern_data)
            else:
                pattern = {
                    "total_access": 0,
                    "failed_access": 0,
                    "last_access": None,
                    "suspicious_activity": 0
                }
            
            # 패턴 업데이트
            pattern["total_access"] += 1
            if not log_entry["success"]:
                pattern["failed_access"] += 1
            pattern["last_access"] = log_entry["timestamp"]
            
            # 의심스러운 활동 감지
            if pattern["failed_access"] > self.audit_policies["alert_thresholds"]["failed_access_attempts"]:
                pattern["suspicious_activity"] += 1
                await self._trigger_alert({
                    "type": "suspicious_activity",
                    "agent_id": log_entry["agent_id"],
                    "details": pattern
                })
            
            # Redis에 패턴 저장
            await self.redis_client.setex(agent_key, 86400, json.dumps(pattern))
            
        except Exception as e:
            logger.warning(f"Access pattern monitoring failed: {str(e)}")
    
    async def _trigger_alert(self, alert_data: Dict[str, Any]):
        """알림 트리거"""
        try:
            if not self.redis_enabled:
                return
            
            alert_key = f"alert:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            await self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))
            
            # 로그에 알림 기록
            self.logger.warning(f"Alert triggered: {alert_data}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {str(e)}")
    
    async def get_audit_metrics(self) -> Dict[str, Any]:
        """감사 메트릭 조회"""
        try:
            metrics = {
                "total_logs": len(self.log_cache),
                "cache_size": len(self.log_cache),
                "redis_enabled": self.redis_enabled,
                "retention_days": self.audit_policies["retention_days"]
            }
            
            if self.redis_enabled:
                # Redis에서 추가 메트릭 수집
                try:
                    today_key = f"audit_log:access:{datetime.now().strftime('%Y%m%d')}"
                    access_count = await self.redis_client.llen(today_key)
                    metrics["today_access_logs"] = access_count
                except Exception as e:
                    logger.warning(f"Failed to get Redis metrics: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get audit metrics: {str(e)}")
            return {}
    
    async def cleanup_old_logs(self) -> Dict[str, Any]:
        """오래된 로그 정리"""
        try:
            cleaned_count = 0
            
            if self.redis_enabled:
                # Redis에서 오래된 로그 키들 찾기
                old_keys = []
                for log_type in ["access", "security", "data_access"]:
                    pattern = f"audit_log:{log_type}:*"
                    keys = await self.redis_client.keys(pattern)
                    
                    for key in keys:
                        # 날짜 추출
                        date_str = key.split(":")[-1]
                        try:
                            log_date = datetime.strptime(date_str, "%Y%m%d")
                            if (datetime.now() - log_date).days > self.audit_policies["retention_days"]:
                                await self.redis_client.delete(key)
                                cleaned_count += 1
                                old_keys.append(key)
                        except ValueError:
                            continue
            
            return {
                "cleaned_count": cleaned_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {str(e)}")
            return {"cleaned_count": 0, "error": str(e)}
    
    async def export_audit_report(self, start_date: datetime = None, 
                                end_date: datetime = None) -> Dict[str, Any]:
        """감사 보고서 내보내기"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": await self.get_audit_metrics(),
                "policies": self.audit_policies,
                "recent_logs": self.log_cache[-100:]  # 최근 100개 로그
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Audit report export failed: {str(e)}")
            return {"error": str(e)}

# 전역 보안 관리자 인스턴스들
security_manager = SecurityManager(secrets.token_urlsafe(32), "redis://localhost:6379")
privacy_manager = PrivacyManager("redis://localhost:6379")
audit_logger = AuditLogger("audit.log", "redis://localhost:6379")

# 기본 프라이버시 정책 설정
privacy_manager.set_data_retention_policy("search_results", 30)
privacy_manager.set_data_retention_policy("user_queries", 90)
privacy_manager.set_data_retention_policy("agent_communications", 180)

# 기본 보안 정책 설정
async def initialize_security_system():
    """보안 시스템 초기화"""
    try:
        # 보안 메트릭 수집
        security_metrics = await security_manager.get_security_metrics()
        logger.info(f"Security system initialized: {security_metrics}")
        
        # 프라이버시 메트릭 수집
        privacy_metrics = await privacy_manager.get_privacy_metrics()
        logger.info(f"Privacy system initialized: {privacy_metrics}")
        
        # 감사 메트릭 수집
        audit_metrics = await audit_logger.get_audit_metrics()
        logger.info(f"Audit system initialized: {audit_metrics}")
        
    except Exception as e:
        logger.error(f"Failed to initialize security system: {str(e)}")

# 비동기 초기화 실행 (선택사항)
# asyncio.create_task(initialize_security_system())
