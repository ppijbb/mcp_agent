"""
알림 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ..auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class NotificationPreferences(BaseModel):
    """알림 설정"""
    email_enabled: bool = True
    push_enabled: bool = True
    hobby_recommendations: bool = True
    community_events: bool = True
    progress_updates: bool = True


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """현재 사용자 정보 가져오기"""
    payload = auth_manager.verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return payload


@router.get("")
async def get_notifications(
    limit: int = Query(default=20, ge=1, le=100),
    unread_only: bool = Query(default=False),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """알림 목록 조회"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 알림 조회
        return {
            "user_id": user_id,
            "notifications": [],
            "unread_count": 0,
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Notifications retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notifications"
        )


@router.put("/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """알림 읽음 처리"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 알림 업데이트
        return {
            "message": "Notification marked as read",
            "notification_id": notification_id,
        }
    except Exception as e:
        logger.error(f"Notification update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update notification"
        )


@router.post("/preferences")
async def update_notification_preferences(
    preferences: NotificationPreferences,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """알림 설정 업데이트"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에 설정 저장
        return {
            "message": "Notification preferences updated",
            "user_id": user_id,
            "preferences": preferences.dict(),
        }
    except Exception as e:
        logger.error(f"Preferences update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

