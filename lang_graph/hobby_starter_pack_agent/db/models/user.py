"""
사용자 관련 데이터베이스 모델
"""

import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from ..base import Base
from ...auth.permissions import SubscriptionPlan


class User(Base):
    """사용자 기본 정보"""
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # OAuth 사용자는 None
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # 관계
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    preferences = relationship("UserPreference", back_populates="user")
    hobbies = relationship("UserHobby", back_populates="user")
    activities = relationship("UserActivity", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")


class UserProfile(Base):
    """사용자 프로필 상세 정보"""
    __tablename__ = "user_profiles"
    
    profile_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), unique=True, nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    display_name = Column(String(100), nullable=True)
    bio = Column(String(500), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    location = Column(String(100), nullable=True)
    birth_date = Column(DateTime, nullable=True)
    gender = Column(String(20), nullable=True)
    interests = Column(JSON, default=list, nullable=False)
    skill_levels = Column(JSON, default=dict, nullable=False)  # {hobby_id: level}
    time_availability = Column(JSON, default=dict, nullable=False)  # {day: hours}
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user = relationship("User", back_populates="profile")


class Subscription(Base):
    """구독 정보"""
    __tablename__ = "subscriptions"
    
    subscription_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), unique=True, nullable=False)
    plan = Column(SQLEnum(SubscriptionPlan), default=SubscriptionPlan.FREE, nullable=False)
    status = Column(String(20), default="active", nullable=False)  # active, cancelled, expired
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user = relationship("User", back_populates="subscription")


class UserPreference(Base):
    """사용자 선호도"""
    __tablename__ = "user_preferences"
    
    preference_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    category = Column(String(50), nullable=False)  # notification, privacy, display 등
    key = Column(String(100), nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        {"comment": "사용자 선호도 설정"},
    )

