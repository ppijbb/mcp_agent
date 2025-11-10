"""
사용자 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from ..auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["users"])

auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class UserProfileUpdate(BaseModel):
    """사용자 프로필 업데이트"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    interests: Optional[list] = None


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """현재 사용자 정보 가져오기"""
    payload = auth_manager.verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return payload


@router.get("/me")
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """현재 사용자 프로필 조회"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 프로필 조회
        return {
            "user_id": user_id,
            "email": current_user.get("email"),
            "plan": current_user.get("plan"),
            "profile": {
                "display_name": "User",
                "interests": [],
            }
        }
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )


@router.put("/me")
async def update_current_user_profile(
    profile_update: UserProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """현재 사용자 프로필 업데이트"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 프로필 업데이트
        return {
            "message": "Profile updated successfully",
            "user_id": user_id,
            "updated_fields": profile_update.dict(exclude_unset=True),
        }
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.get("/{user_id}/profile")
async def get_user_profile(user_id: str):
    """특정 사용자 프로필 조회"""
    try:
        # 실제로는 데이터베이스에서 프로필 조회
        return {
            "user_id": user_id,
            "profile": {
                "display_name": "User",
                "interests": [],
            }
        }
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.get("/me/dashboard")
async def get_user_dashboard(current_user: Dict[str, Any] = Depends(get_current_user)):
    """사용자 대시보드 데이터 조회"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 대시보드 데이터 조회
        return {
            "user_id": user_id,
            "active_hobbies": [],
            "recommended_hobbies": [],
            "community_activities": [],
            "recent_activities": [],
            "achievements": [],
        }
    except Exception as e:
        logger.error(f"Dashboard retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard"
        )

