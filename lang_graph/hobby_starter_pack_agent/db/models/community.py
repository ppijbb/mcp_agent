"""
커뮤니티 관련 데이터베이스 모델
"""

import uuid
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..base import Base


class Community(Base):
    """커뮤니티 정보"""
    __tablename__ = "communities"
    
    community_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=True, index=True)
    location = Column(String(200), nullable=True)
    is_online = Column(Boolean, default=False, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    member_count = Column(Integer, default=0, nullable=False)
    max_members = Column(Integer, nullable=True)
    tags = Column(JSON, default=list, nullable=False)
    rules = Column(JSON, default=list, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    members = relationship("CommunityMember", back_populates="community")
    events = relationship("Event", back_populates="community")


class CommunityMember(Base):
    """커뮤니티 멤버"""
    __tablename__ = "community_members"
    
    member_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id = Column(UUID(as_uuid=True), ForeignKey("communities.community_id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    role = Column(String(20), default="member", nullable=False)  # member, moderator, admin
    status = Column(String(20), default="active", nullable=False)  # active, pending, banned
    joined_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_active_at = Column(DateTime, nullable=True)
    
    # 관계
    community = relationship("Community", back_populates="members")


class Event(Base):
    """이벤트 정보"""
    __tablename__ = "events"
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    community_id = Column(UUID(as_uuid=True), ForeignKey("communities.community_id"), nullable=True, index=True)
    hobby_id = Column(UUID(as_uuid=True), ForeignKey("hobbies.hobby_id"), nullable=True, index=True)
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    event_type = Column(String(50), nullable=False)  # online, offline, hybrid
    location = Column(String(200), nullable=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    max_participants = Column(Integer, nullable=True)
    current_participants = Column(Integer, default=0, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 관계
    community = relationship("Community", back_populates="events")
    participants = relationship("EventParticipant", back_populates="event")


class EventParticipant(Base):
    """이벤트 참가자"""
    __tablename__ = "event_participants"
    
    participant_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey("events.event_id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    status = Column(String(20), default="registered", nullable=False)  # registered, attended, cancelled
    registered_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # 관계
    event = relationship("Event", back_populates="participants")

