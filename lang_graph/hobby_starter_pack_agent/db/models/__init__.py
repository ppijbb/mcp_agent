"""
데이터베이스 모델 패키지
"""

from .user import User, UserProfile, Subscription, UserPreference
from .hobby import Hobby, UserHobby, HobbyProgress, HobbyReview, HobbyCategory
from .community import Community, CommunityMember, Event, EventParticipant
from .analytics import UserActivity, Recommendation, Feedback, Metric

# 기존 모델도 포함
from ..models import AgentSession, AgentConsensusLog, MCPInteractionLog

__all__ = [
    # User models
    "User",
    "UserProfile",
    "Subscription",
    "UserPreference",
    # Hobby models
    "Hobby",
    "UserHobby",
    "HobbyProgress",
    "HobbyReview",
    "HobbyCategory",
    # Community models
    "Community",
    "CommunityMember",
    "Event",
    "EventParticipant",
    # Analytics models
    "UserActivity",
    "Recommendation",
    "Feedback",
    "Metric",
    # Legacy models
    "AgentSession",
    "AgentConsensusLog",
    "MCPInteractionLog",
]

