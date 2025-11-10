"""
OAuth2 소셜 로그인 핸들러
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class OAuthProvider(Enum):
    """OAuth2 제공자"""
    GOOGLE = "google"
    APPLE = "apple"
    FACEBOOK = "facebook"


class OAuth2Handler:
    """OAuth2 소셜 로그인 처리"""
    
    def __init__(self):
        """OAuth2Handler 초기화"""
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.apple_client_id = os.getenv("APPLE_CLIENT_ID")
        self.apple_client_secret = os.getenv("APPLE_CLIENT_SECRET")
        self.facebook_app_id = os.getenv("FACEBOOK_APP_ID")
        self.facebook_app_secret = os.getenv("FACEBOOK_APP_SECRET")
        
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")
        self.apple_redirect_uri = os.getenv("APPLE_REDIRECT_URI", "http://localhost:8000/api/auth/apple/callback")
        self.facebook_redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI", "http://localhost:8000/api/auth/facebook/callback")
    
    def get_authorization_url(self, provider: OAuthProvider, state: Optional[str] = None) -> str:
        """
        OAuth2 인증 URL 생성
        
        Args:
            provider: OAuth 제공자
            state: CSRF 방지를 위한 상태 값
        
        Returns:
            인증 URL
        """
        if provider == OAuthProvider.GOOGLE:
            return self._get_google_auth_url(state)
        elif provider == OAuthProvider.APPLE:
            return self._get_apple_auth_url(state)
        elif provider == OAuthProvider.FACEBOOK:
            return self._get_facebook_auth_url(state)
        else:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
    
    def _get_google_auth_url(self, state: Optional[str]) -> str:
        """Google OAuth2 인증 URL 생성"""
        if not self.google_client_id:
            raise ValueError("GOOGLE_CLIENT_ID not configured")
        
        params = {
            "client_id": self.google_client_id,
            "redirect_uri": self.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
        }
        
        if state:
            params["state"] = state
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"https://accounts.google.com/o/oauth2/v2/auth?{query_string}"
    
    def _get_apple_auth_url(self, state: Optional[str]) -> str:
        """Apple OAuth2 인증 URL 생성"""
        if not self.apple_client_id:
            raise ValueError("APPLE_CLIENT_ID not configured")
        
        params = {
            "client_id": self.apple_client_id,
            "redirect_uri": self.apple_redirect_uri,
            "response_type": "code",
            "scope": "email name",
            "response_mode": "form_post",
        }
        
        if state:
            params["state"] = state
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"https://appleid.apple.com/auth/authorize?{query_string}"
    
    def _get_facebook_auth_url(self, state: Optional[str]) -> str:
        """Facebook OAuth2 인증 URL 생성"""
        if not self.facebook_app_id:
            raise ValueError("FACEBOOK_APP_ID not configured")
        
        params = {
            "client_id": self.facebook_app_id,
            "redirect_uri": self.facebook_redirect_uri,
            "response_type": "code",
            "scope": "email public_profile",
        }
        
        if state:
            params["state"] = state
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"https://www.facebook.com/v18.0/dialog/oauth?{query_string}"
    
    async def exchange_code_for_token(
        self,
        provider: OAuthProvider,
        code: str
    ) -> Dict[str, Any]:
        """
        인증 코드를 액세스 토큰으로 교환
        
        Args:
            provider: OAuth 제공자
            code: 인증 코드
        
        Returns:
            토큰 정보 및 사용자 정보
        """
        if provider == OAuthProvider.GOOGLE:
            return await self._exchange_google_code(code)
        elif provider == OAuthProvider.APPLE:
            return await self._exchange_apple_code(code)
        elif provider == OAuthProvider.FACEBOOK:
            return await self._exchange_facebook_code(code)
        else:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
    
    async def _exchange_google_code(self, code: str) -> Dict[str, Any]:
        """Google 인증 코드 교환"""
        if not self.google_client_id or not self.google_client_secret:
            raise ValueError("Google OAuth not configured")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": self.google_client_id,
                    "client_secret": self.google_client_secret,
                    "redirect_uri": self.google_redirect_uri,
                    "grant_type": "authorization_code",
                }
            )
            response.raise_for_status()
            token_data = response.json()
            
            # 사용자 정보 가져오기
            user_info_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()
            
            return {
                "access_token": token_data.get("access_token"),
                "refresh_token": token_data.get("refresh_token"),
                "user_info": {
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "picture": user_info.get("picture"),
                    "provider_id": user_info.get("id"),
                }
            }
    
    async def _exchange_apple_code(self, code: str) -> Dict[str, Any]:
        """Apple 인증 코드 교환"""
        if not self.apple_client_id or not self.apple_client_secret:
            raise ValueError("Apple OAuth not configured")
        
        # Apple OAuth2 구현 (실제로는 더 복잡한 JWT 서명 필요)
        logger.warning("Apple OAuth2 exchange not fully implemented")
        return {
            "access_token": "",
            "user_info": {
                "email": "",
                "name": "",
                "provider_id": "",
            }
        }
    
    async def _exchange_facebook_code(self, code: str) -> Dict[str, Any]:
        """Facebook 인증 코드 교환"""
        if not self.facebook_app_id or not self.facebook_app_secret:
            raise ValueError("Facebook OAuth not configured")
        
        async with httpx.AsyncClient() as client:
            # 액세스 토큰 교환
            token_response = await client.get(
                "https://graph.facebook.com/v18.0/oauth/access_token",
                params={
                    "client_id": self.facebook_app_id,
                    "client_secret": self.facebook_app_secret,
                    "redirect_uri": self.facebook_redirect_uri,
                    "code": code,
                }
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            
            # 사용자 정보 가져오기
            user_info_response = await client.get(
                "https://graph.facebook.com/v18.0/me",
                params={
                    "fields": "id,name,email,picture",
                    "access_token": token_data["access_token"],
                }
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()
            
            return {
                "access_token": token_data.get("access_token"),
                "user_info": {
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "picture": user_info.get("picture", {}).get("data", {}).get("url"),
                    "provider_id": user_info.get("id"),
                }
            }

