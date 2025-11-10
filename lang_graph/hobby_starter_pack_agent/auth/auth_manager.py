"""
통합 인증 관리자
"""

import os
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .jwt_handler import JWTHandler
from .oauth2_handler import OAuth2Handler, OAuthProvider
from .permissions import PermissionManager, SubscriptionPlan, Permission

logger = logging.getLogger(__name__)


class AuthManager:
    """통합 인증 관리자"""
    
    def __init__(self):
        """AuthManager 초기화"""
        self.jwt_handler = JWTHandler()
        self.oauth2_handler = OAuth2Handler()
        self.permission_manager = PermissionManager()
    
    def hash_password(self, password: str) -> str:
        """
        비밀번호 해시 생성
        
        Args:
            password: 평문 비밀번호
        
        Returns:
            해시된 비밀번호
        """
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # 반복 횟수
        )
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        비밀번호 검증
        
        Args:
            password: 평문 비밀번호
            hashed_password: 해시된 비밀번호
        
        Returns:
            검증 성공 여부
        """
        try:
            salt, stored_hash = hashed_password.split(":")
            password_hash = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                100000,
            )
            return password_hash.hex() == stored_hash
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_tokens(
        self,
        user_id: str,
        email: str,
        plan: SubscriptionPlan = SubscriptionPlan.FREE,
        is_admin: bool = False
    ) -> Dict[str, Any]:
        """
        액세스 토큰 및 리프레시 토큰 생성
        
        Args:
            user_id: 사용자 ID
            email: 사용자 이메일
            plan: 구독 플랜
            is_admin: 관리자 여부
        
        Returns:
            토큰 정보
        """
        permissions = self.permission_manager.get_permissions(plan, is_admin)
        permission_list = [p.value for p in permissions]
        
        access_token = self.jwt_handler.generate_token(
            user_id=user_id,
            email=email,
            expires_in=3600,  # 1시간
            additional_claims={
                "plan": plan.value,
                "is_admin": is_admin,
                "permissions": permission_list,
            }
        )
        
        refresh_token = self.jwt_handler.generate_refresh_token(user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        액세스 토큰 검증
        
        Args:
            token: 검증할 토큰
        
        Returns:
            토큰 페이로드 또는 None
        """
        return self.jwt_handler.verify_token(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        리프레시 토큰으로 새 액세스 토큰 생성
        
        Args:
            refresh_token: 리프레시 토큰
        
        Returns:
            새 토큰 정보 또는 None
        """
        payload = self.jwt_handler.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("user_id")
        if not user_id:
            return None
        
        # 사용자 정보는 데이터베이스에서 가져와야 함
        # 여기서는 기본값 사용
        return self.generate_tokens(
            user_id=user_id,
            email=payload.get("email", ""),
            plan=SubscriptionPlan.FREE,
            is_admin=False,
        )
    
    def get_oauth_authorization_url(
        self,
        provider: str,
        state: Optional[str] = None
    ) -> str:
        """
        OAuth2 인증 URL 가져오기
        
        Args:
            provider: OAuth 제공자 (google, apple, facebook)
            state: CSRF 방지 상태 값
        
        Returns:
            인증 URL
        """
        try:
            oauth_provider = OAuthProvider(provider.lower())
            return self.oauth2_handler.get_authorization_url(oauth_provider, state)
        except ValueError as e:
            logger.error(f"Invalid OAuth provider: {provider}")
            raise
    
    async def handle_oauth_callback(
        self,
        provider: str,
        code: str
    ) -> Dict[str, Any]:
        """
        OAuth2 콜백 처리
        
        Args:
            provider: OAuth 제공자
            code: 인증 코드
        
        Returns:
            사용자 정보 및 토큰
        """
        try:
            oauth_provider = OAuthProvider(provider.lower())
            token_data = await self.oauth2_handler.exchange_code_for_token(
                oauth_provider,
                code
            )
            
            user_info = token_data.get("user_info", {})
            
            # 여기서는 사용자 정보만 반환
            # 실제로는 데이터베이스에 사용자 저장/업데이트 필요
            return {
                "user_info": user_info,
                "provider": provider,
            }
        except Exception as e:
            logger.error(f"OAuth callback handling failed: {e}")
            raise

