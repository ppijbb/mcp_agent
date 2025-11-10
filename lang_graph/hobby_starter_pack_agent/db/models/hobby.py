"""
취미 관련 데이터베이스 모델
"""

import uuid
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from enum import Enum
from ..base import Base


class HobbyCategory(Enum):
    """취미 카테고리"""
    SPORTS = "sports"
    ARTS = "arts"
    MUSIC = "music"
    READING = "reading"
    COOKING = "cooking"
    GARDENING = "gardening"
    PHOTOGRAPHY = "photography"
    TRAVEL = "travel"
    GAMING = "gaming"
    CRAFTS = "crafts"
    OTHER = "other"


class Hobby(Base):
    """취미 기본 정보"""
    __tablename__ = "hobbies"
    
    hobby_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(SQLEnum(HobbyCategory), nullable=False, index=True)
    difficulty_level = Column(String(20), default="beginner", nullable=False)  # beginner, intermediate, advanced
    estimated_cost = Column(JSON, nullable=True)  # {min: 0, max: 100000, currency: "KRW"}
    time_required = Column(JSON, nullable=True)  # {min_hours: 1, max_hours: 5, frequency: "weekly"}
    required_skills = Column(JSON, default=list, nullable=False)
    required_equipment = Column(JSON, default=list, nullable=False)
    tags = Column(JSON, default=list, nullable=False)
    image_url = Column(String(500), nullable=True)
    popularity_score = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user_hobbies = relationship("UserHobby", back_populates="hobby")
    progress_records = relationship("HobbyProgress", back_populates="hobby")
    reviews = relationship("HobbyReview", back_populates="hobby")


class UserHobby(Base):
    """사용자-취미 관계"""
    __tablename__ = "user_hobbies"
    
    user_hobby_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=False, index=True)
    status = Column(String(20), default="interested", nullable=False)  # interested, active, paused, completed
    skill_level = Column(String(20), default="beginner", nullable=False)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_activity_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user = relationship("User", back_populates="hobbies")
    hobby = relationship("Hobby", back_populates="user_hobbies")
    progress = relationship("HobbyProgress", back_populates="user_hobby")


class HobbyProgress(Base):
    """취미 진행 상황"""
    __tablename__ = "hobby_progress"
    
    progress_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_hobby_id = Column(UUID(as_uuid=True), ForeignKey("user_hobbies.user_hobby_id"), nullable=False, index=True)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=False, index=True)
    milestone = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    hours_spent = Column(Float, default=0.0, nullable=False)
    achievements = Column(JSON, default=list, nullable=False)
    photos = Column(JSON, default=list, nullable=False)  # URL 리스트
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user_hobby = relationship("UserHobby", back_populates="progress")
    hobby = relationship("Hobby", back_populates="progress_records")


class HobbyReview(Base):
    """취미 리뷰"""
    __tablename__ = "hobby_reviews"
    
    review_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1-5
    title = Column(String(200), nullable=True)
    content = Column(Text, nullable=True)
    pros = Column(JSON, default=list, nullable=False)
    cons = Column(JSON, default=list, nullable=False)
    helpful_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    hobby = relationship("Hobby", back_populates="reviews")

