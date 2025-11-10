"""
분석 및 메트릭 데이터베이스 모델
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
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..base import Base


class UserActivity(Base):
    """사용자 활동 로그"""
    __tablename__ = "user_activities"
    
    activity_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    activity_type = Column(String(50), nullable=False, index=True)  # login, hobby_view, recommendation_click 등
    entity_type = Column(String(50), nullable=True)  # hobby, community, event 등
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    metadata = Column(JSON, default=dict, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    
    # 관계
    user = relationship("User", back_populates="activities")


class Recommendation(Base):
    """추천 이력"""
    __tablename__ = "recommendations"
    
    recommendation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=True, index=True)
    recommendation_type = Column(String(50), nullable=False)  # hobby, community, event 등
    recommendation_engine = Column(String(50), nullable=False)  # collaborative, content-based, hybrid
    score = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)
    was_clicked = Column(Boolean, default=False, nullable=False)
    was_accepted = Column(Boolean, default=False, nullable=False)
    context = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    
    # 관계
    user = relationship("User")


class Feedback(Base):
    """사용자 피드백"""
    __tablename__ = "feedback"
    
    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    feedback_type = Column(String(50), nullable=False)  # recommendation, feature, bug, general
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5
    comment = Column(Text, nullable=True)
    metadata = Column(JSON, default=dict, nullable=False)
    status = Column(String(20), default="new", nullable=False)  # new, reviewed, resolved
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    user = relationship("User", back_populates="feedback")


class Metric(Base):
    """시스템 메트릭"""
    __tablename__ = "metrics"
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict, nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        {"comment": "시스템 메트릭 저장"},
    )

