"""
취미 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from ..auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hobbies", tags=["hobbies"])

auth_manager = AuthManager()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class HobbyReviewRequest(BaseModel):
    """취미 리뷰 요청"""
    rating: int
    title: Optional[str] = None
    content: Optional[str] = None
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """현재 사용자 정보 가져오기"""
    payload = auth_manager.verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return payload


@router.get("/recommendations")
async def get_hobby_recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    category: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """취미 추천 조회"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 추천 엔진을 통해 추천 생성
        return {
            "user_id": user_id,
            "recommendations": [
                {
                    "hobby_id": "hobby_1",
                    "name": "Photography",
                    "category": "arts",
                    "score": 0.95,
                }
            ],
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Recommendation retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendations"
        )


@router.post("/{hobby_id}/start")
async def start_hobby(
    hobby_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """취미 시작"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에 사용자-취미 관계 생성
        return {
            "message": "Hobby started successfully",
            "user_id": user_id,
            "hobby_id": hobby_id,
        }
    except Exception as e:
        logger.error(f"Hobby start failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start hobby"
        )


@router.get("/me")
async def get_my_hobbies(current_user: Dict[str, Any] = Depends(get_current_user)):
    """내 취미 목록 조회"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에서 사용자 취미 조회
        return {
            "user_id": user_id,
            "hobbies": [],
        }
    except Exception as e:
        logger.error(f"Hobbies retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve hobbies"
        )


@router.post("/{hobby_id}/review")
async def create_hobby_review(
    hobby_id: str,
    review: HobbyReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """취미 리뷰 작성"""
    try:
        user_id = current_user.get("user_id")
        # 실제로는 데이터베이스에 리뷰 저장
        return {
            "message": "Review created successfully",
            "user_id": user_id,
            "hobby_id": hobby_id,
            "review_id": "review_1",
        }
    except Exception as e:
        logger.error(f"Review creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create review"
        )

