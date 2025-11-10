"""
JWT 토큰 처리 핸들러
"""

import os
import jwt
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

logger = logging.getLogger(__name__)


class JWTHandler:
    """JWT 토큰 생성 및 검증 핸들러"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        JWTHandler 초기화
        
        Args:
            secret_key: JWT 서명에 사용할 비밀 키 (없으면 환경변수에서 가져옴)
            algorithm: JWT 알고리즘 (기본값: HS256)
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = algorithm
        
        if self.secret_key == "your-secret-key-change-in-production":
            logger.warning("Using default JWT secret key. Change JWT_SECRET_KEY in production!")
    
    def generate_token(
        self,
        user_id: str,
        email: str,
        expires_in: int = 3600,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        JWT 토큰 생성
        
        Args:
            user_id: 사용자 ID
            email: 사용자 이메일
            expires_in: 토큰 만료 시간 (초, 기본값: 1시간)
            additional_claims: 추가 클레임
        
        Returns:
            생성된 JWT 토큰
        """
        try:
            now = datetime.utcnow()
            payload = {
                "user_id": user_id,
                "email": email,
                "iat": now,
                "exp": now + timedelta(seconds=expires_in),
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"JWT token generated for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to generate JWT token: {e}")
            raise
    
    def generate_refresh_token(self, user_id: str) -> str:
        """
        리프레시 토큰 생성 (7일 유효)
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            생성된 리프레시 토큰
        """
        return self.generate_token(
            user_id=user_id,
            email="",  # 리프레시 토큰에는 이메일 불필요
            expires_in=7 * 24 * 3600,  # 7일
            additional_claims={"type": "refresh"}
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        JWT 토큰 검증
        
        Args:
            token: 검증할 JWT 토큰
        
        Returns:
            토큰 페이로드 (검증 성공 시) 또는 None (검증 실패 시)
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            logger.debug(f"JWT token verified for user {payload.get('user_id')}")
            return payload
            
        except ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
            
        except InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
            
        except Exception as e:
            logger.error(f"JWT token verification failed: {e}")
            return None
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        JWT 토큰 디코딩 (검증 없이)
        
        Args:
            token: 디코딩할 JWT 토큰
        
        Returns:
            토큰 페이로드 또는 None
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception as e:
            logger.error(f"Failed to decode JWT token: {e}")
            return None

