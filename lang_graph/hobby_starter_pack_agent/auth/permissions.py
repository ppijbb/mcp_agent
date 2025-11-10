"""
권한 관리 시스템
"""

import logging
from typing import List, Set, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class Permission(Enum):
    """권한 정의"""
    # 사용자 권한
    READ_PROFILE = "read_profile"
    UPDATE_PROFILE = "update_profile"
    DELETE_PROFILE = "delete_profile"
    
    # 취미 권한
    READ_HOBBIES = "read_hobbies"
    CREATE_HOBBY = "create_hobby"
    UPDATE_HOBBY = "update_hobby"
    DELETE_HOBBY = "delete_hobby"
    
    # 커뮤니티 권한
    READ_COMMUNITIES = "read_communities"
    CREATE_COMMUNITY = "create_community"
    JOIN_COMMUNITY = "join_community"
    MANAGE_COMMUNITY = "manage_community"
    
    # 관리자 권한
    ADMIN_ACCESS = "admin_access"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"


class SubscriptionPlan(Enum):
    """구독 플랜"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"


class PermissionManager:
    """권한 관리자"""
    
    def __init__(self):
        """PermissionManager 초기화"""
        self.plan_permissions: Dict[SubscriptionPlan, Set[Permission]] = {
            SubscriptionPlan.FREE: {
                Permission.READ_PROFILE,
                Permission.UPDATE_PROFILE,
                Permission.READ_HOBBIES,
                Permission.READ_COMMUNITIES,
                Permission.JOIN_COMMUNITY,
            },
            SubscriptionPlan.BASIC: {
                Permission.READ_PROFILE,
                Permission.UPDATE_PROFILE,
                Permission.DELETE_PROFILE,
                Permission.READ_HOBBIES,
                Permission.CREATE_HOBBY,
                Permission.UPDATE_HOBBY,
                Permission.READ_COMMUNITIES,
                Permission.CREATE_COMMUNITY,
                Permission.JOIN_COMMUNITY,
            },
            SubscriptionPlan.PREMIUM: {
                Permission.READ_PROFILE,
                Permission.UPDATE_PROFILE,
                Permission.DELETE_PROFILE,
                Permission.READ_HOBBIES,
                Permission.CREATE_HOBBY,
                Permission.UPDATE_HOBBY,
                Permission.DELETE_HOBBY,
                Permission.READ_COMMUNITIES,
                Permission.CREATE_COMMUNITY,
                Permission.JOIN_COMMUNITY,
                Permission.MANAGE_COMMUNITY,
                Permission.VIEW_ANALYTICS,
            },
        }
    
    def get_permissions(self, plan: SubscriptionPlan, is_admin: bool = False) -> Set[Permission]:
        """
        구독 플랜에 따른 권한 반환
        
        Args:
            plan: 구독 플랜
            is_admin: 관리자 여부
        
        Returns:
            권한 집합
        """
        permissions = self.plan_permissions.get(plan, set()).copy()
        
        if is_admin:
            permissions.add(Permission.ADMIN_ACCESS)
            permissions.add(Permission.MANAGE_USERS)
        
        return permissions
    
    def has_permission(self, user_permissions: Set[Permission], required_permission: Permission) -> bool:
        """
        사용자가 특정 권한을 가지고 있는지 확인
        
        Args:
            user_permissions: 사용자의 권한 집합
            required_permission: 필요한 권한
        
        Returns:
            권한 보유 여부
        """
        return required_permission in user_permissions
    
    def check_permission(
        self,
        user_permissions: Set[Permission],
        required_permission: Permission
    ) -> bool:
        """
        권한 확인 (에러 발생)
        
        Args:
            user_permissions: 사용자의 권한 집합
            required_permission: 필요한 권한
        
        Returns:
            권한 보유 여부
        
        Raises:
            PermissionError: 권한이 없는 경우
        """
        if not self.has_permission(user_permissions, required_permission):
            raise PermissionError(f"Required permission: {required_permission.value}")
        return True
    
    def get_plan_from_string(self, plan_str: str) -> SubscriptionPlan:
        """
        문자열에서 구독 플랜 반환
        
        Args:
            plan_str: 플랜 문자열
        
        Returns:
            구독 플랜
        """
        try:
            return SubscriptionPlan(plan_str.lower())
        except ValueError:
            logger.warning(f"Invalid subscription plan: {plan_str}, defaulting to FREE")
            return SubscriptionPlan.FREE

