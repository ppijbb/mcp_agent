"""
인증 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

from ..auth.auth_manager import AuthManager
from ..auth.permissions import SubscriptionPlan

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class RegisterRequest(BaseModel):
    """회원가입 요청"""
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    """로그인 요청"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """토큰 응답"""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """리프레시 토큰 요청"""
    refresh_token: str


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    사용자 회원가입
    
    실제로는 데이터베이스에 사용자 저장 필요
    """
    try:
        # 비밀번호 해시
        password_hash = auth_manager.hash_password(request.password)
        
        # 여기서는 사용자 생성 로직 생략 (데이터베이스 연동 필요)
        # user = create_user(email=request.email, username=request.username, password_hash=password_hash)
        
        return {
            "message": "User registered successfully",
            "email": request.email,
        }
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    사용자 로그인
    
    실제로는 데이터베이스에서 사용자 확인 필요
    """
    try:
        # 여기서는 사용자 확인 로직 생략 (데이터베이스 연동 필요)
        # user = get_user_by_email(form_data.username)
        # if not user or not auth_manager.verify_password(form_data.password, user.password_hash):
        #     raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # 임시 사용자 정보
        user_id = "temp_user_id"
        email = form_data.username
        
        tokens = auth_manager.generate_tokens(
            user_id=user_id,
            email=email,
            plan=SubscriptionPlan.FREE,
            is_admin=False
        )
        
        return TokenResponse(**tokens)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """리프레시 토큰으로 새 액세스 토큰 발급"""
    try:
        tokens = auth_manager.refresh_access_token(request.refresh_token)
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        return TokenResponse(**tokens)
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """사용자 로그아웃"""
    # 실제로는 토큰 블랙리스트에 추가 필요
    return {"message": "Logged out successfully"}


@router.get("/profile")
async def get_profile(token: str = Depends(oauth2_scheme)):
    """현재 사용자 프로필 조회"""
    try:
        payload = auth_manager.verify_access_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # 실제로는 데이터베이스에서 사용자 정보 조회
        return {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "plan": payload.get("plan"),
            "is_admin": payload.get("is_admin"),
        }
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.get("/oauth/{provider}/authorize")
async def oauth_authorize(provider: str, state: Optional[str] = None):
    """OAuth2 인증 URL 생성"""
    try:
        auth_url = auth_manager.get_oauth_authorization_url(provider, state)
        return {"authorization_url": auth_url}
    except Exception as e:
        logger.error(f"OAuth authorization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth provider {provider} not configured"
        )


@router.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, code: str):
    """OAuth2 콜백 처리"""
    try:
        result = await auth_manager.handle_oauth_callback(provider, code)
        return result
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth callback failed"
        )

