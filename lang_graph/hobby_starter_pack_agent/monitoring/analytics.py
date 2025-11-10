"""
사용자 및 비즈니스 분석
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class UserAnalytics:
    """사용자 분석"""
    
    def __init__(self):
        """UserAnalytics 초기화"""
        self.activity_logs: List[Dict[str, Any]] = []
    
    def log_activity(
        self,
        user_id: str,
        activity_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        사용자 활동 로그 기록
        
        Args:
            user_id: 사용자 ID
            activity_type: 활동 타입
            entity_type: 엔티티 타입
            entity_id: 엔티티 ID
            metadata: 추가 메타데이터
        """
        log_entry = {
            "user_id": user_id,
            "activity_type": activity_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.activity_logs.append(log_entry)
    
    def get_user_activity_summary(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        사용자 활동 요약
        
        Args:
            user_id: 사용자 ID
            days: 분석 기간 (일)
        
        Returns:
            활동 요약
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_activities = [
            log for log in self.activity_logs
            if log["user_id"] == user_id
            and datetime.fromisoformat(log["timestamp"]) >= cutoff_date
        ]
        
        activity_counts = defaultdict(int)
        entity_counts = defaultdict(int)
        
        for activity in user_activities:
            activity_counts[activity["activity_type"]] += 1
            if activity.get("entity_type"):
                entity_counts[activity["entity_type"]] += 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_activities": len(user_activities),
            "activity_breakdown": dict(activity_counts),
            "entity_breakdown": dict(entity_counts),
            "most_active_day": self._get_most_active_day(user_activities),
        }
    
    def _get_most_active_day(self, activities: List[Dict[str, Any]]) -> Optional[str]:
        """가장 활발한 요일 반환"""
        if not activities:
            return None
        
        day_counts = defaultdict(int)
        for activity in activities:
            date = datetime.fromisoformat(activity["timestamp"]).date()
            day_counts[date] += 1
        
        if day_counts:
            most_active = max(day_counts.items(), key=lambda x: x[1])
            return most_active[0].isoformat()
        return None
    
    def get_popular_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        인기 활동 반환
        
        Args:
            limit: 반환할 개수
        
        Returns:
            인기 활동 리스트
        """
        activity_counts = defaultdict(int)
        for log in self.activity_logs:
            activity_counts[log["activity_type"]] += 1
        
        sorted_activities = sorted(
            activity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {"activity_type": activity, "count": count}
            for activity, count in sorted_activities
        ]


class BusinessMetrics:
    """비즈니스 메트릭"""
    
    def __init__(self):
        """BusinessMetrics 초기화"""
        self.user_analytics = UserAnalytics()
        self.recommendation_logs: List[Dict[str, Any]] = []
        self.feedback_logs: List[Dict[str, Any]] = []
    
    def log_recommendation(
        self,
        user_id: str,
        hobby_id: str,
        recommendation_engine: str,
        score: Optional[float] = None,
        was_clicked: bool = False,
        was_accepted: bool = False
    ):
        """
        추천 로그 기록
        
        Args:
            user_id: 사용자 ID
            hobby_id: 취미 ID
            recommendation_engine: 추천 엔진
            score: 추천 점수
            was_clicked: 클릭 여부
            was_accepted: 수락 여부
        """
        log_entry = {
            "user_id": user_id,
            "hobby_id": hobby_id,
            "recommendation_engine": recommendation_engine,
            "score": score,
            "was_clicked": was_clicked,
            "was_accepted": was_accepted,
            "timestamp": datetime.now().isoformat(),
        }
        self.recommendation_logs.append(log_entry)
    
    def log_feedback(
        self,
        user_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None
    ):
        """
        피드백 로그 기록
        
        Args:
            user_id: 사용자 ID
            feedback_type: 피드백 타입
            rating: 평점 (1-5)
            comment: 코멘트
        """
        log_entry = {
            "user_id": user_id,
            "feedback_type": feedback_type,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }
        self.feedback_logs.append(log_entry)
    
    def get_recommendation_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        추천 메트릭 반환
        
        Args:
            days: 분석 기간 (일)
        
        Returns:
            추천 메트릭
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_recommendations = [
            log for log in self.recommendation_logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_date
        ]
        
        if not recent_recommendations:
            return {
                "total_recommendations": 0,
                "click_rate": 0.0,
                "acceptance_rate": 0.0,
            }
        
        total = len(recent_recommendations)
        clicked = sum(1 for log in recent_recommendations if log.get("was_clicked", False))
        accepted = sum(1 for log in recent_recommendations if log.get("was_accepted", False))
        
        return {
            "total_recommendations": total,
            "click_rate": (clicked / total) * 100 if total > 0 else 0.0,
            "acceptance_rate": (accepted / total) * 100 if total > 0 else 0.0,
            "period_days": days,
        }
    
    def get_feedback_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        피드백 메트릭 반환
        
        Args:
            days: 분석 기간 (일)
        
        Returns:
            피드백 메트릭
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_feedback = [
            log for log in self.feedback_logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_date
        ]
        
        if not recent_feedback:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
            }
        
        ratings = [log["rating"] for log in recent_feedback if log.get("rating")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            "total_feedback": len(recent_feedback),
            "average_rating": avg_rating,
            "period_days": days,
        }
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """전체 비즈니스 메트릭 반환"""
        return {
            "recommendations": self.get_recommendation_metrics(),
            "feedback": self.get_feedback_metrics(),
            "popular_activities": self.user_analytics.get_popular_activities(),
        }

