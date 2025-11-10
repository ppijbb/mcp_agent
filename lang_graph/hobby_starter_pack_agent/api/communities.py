"""
커뮤니티 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ..auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/communities", tags=["communities"])

auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class CommunityCreateRequest(BaseModel):
    """커뮤니티 생성 요청"""
    name: str
    description: Optional[str] = None
    hobby_id: Optional[str] = None
    location: Optional[str] = None
    is_online: bool = False
    is_public: bool = True
    max_members: Optional[int] = None


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
async def get_communities(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    hobby_id: Optional[str] = None,
    location: Optional[str] = None,
):
    """커뮤니티 목록 조회"""
    try:
        # 실제로는 데이터베이스에서 커뮤니티 조회
        return {
            "communities": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Communities retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve communities"
        )


@router.post("")
async def create_community(
    community: CommunityCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """커뮤니티 생성"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에 커뮤니티 생성
        return {
            "message": "Community created successfully",
            "community_id": "community_1",
            "name": community.name,
        }
    except Exception as e:
        logger.error(f"Community creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create community"
        )


@router.get("/{community_id}")
async def get_community(community_id: str):
    """커뮤니티 상세 조회"""
    try:
        # 실제로는 데이터베이스에서 커뮤니티 조회
        return {
            "community_id": community_id,
            "name": "Community Name",
            "description": "Community Description",
            "member_count": 0,
        }
    except Exception as e:
        logger.error(f"Community retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Community not found"
        )


@router.post("/{community_id}/join")
async def join_community(
    community_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """커뮤니티 가입"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에 멤버 추가
        return {
            "message": "Joined community successfully",
            "user_id": user_id,
            "community_id": community_id,
        }
    except Exception as e:
        logger.error(f"Community join failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join community"
        )

