"""
MCP Event Search Server
Provides event and discount search capabilities for hobby activities.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class EventSearchMCPServer:
    """
    MCP Server for event and discount search functionality.
    
    Capabilities:
    - Search for upcoming hobby events
    - Find discount events and promotions
    - Get class/workshop schedules
    - Search for community activities
    """
    
    def __init__(self):
        self.name = "event_search"
        self.description = "Event and discount search server for hobby activities"
        self.capabilities = [
            "search_events",
            "search_discounts",
            "search_classes",
            "search_workshops",
            "search_community_activities"
        ]
    
    async def search_events(
        self, 
        hobby: str, 
        days: int = 30, 
        location: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for upcoming hobby events.
        
        Args:
            hobby: Hobby name
            days: Lookahead period in days
            location: Event location
            limit: Maximum results
            
        Returns:
            Dict containing upcoming events
        """
        try:
            events = [
                {
                    "name": f"{hobby} 입문 클래스",
                    "type": "class",
                    "date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    "time": "14:00",
                    "location": "서울 강남구",
                    "venue": "Hobby Center",
                    "price": 35000,
                    "discounted_price": 28000,
                    "capacity": 20,
                    "available": 12,
                    "url": "https://example.com/event1",
                    "organizer": "HobbyList"
                },
                {
                    "name": f"{hobby} 워크숍",
                    "type": "workshop", 
                    "date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "time": "10:00",
                    "location": "서울 마포구",
                    "venue": "Creative Space",
                    "price": 50000,
                    "discounted_price": 45000,
                    "capacity": 15,
                    "available": 8,
                    "url": "https://example.com/event2",
                    "organizer": "Creative Studio"
                },
                {
                    "name": f"{hobby} 모임",
                    "type": "community",
                    "date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                    "time": "15:00",
                    "location": "서울 용산구",
                    "venue": "Hangout Cafe",
                    "price": 0,
                    "discounted_price": 0,
                    "capacity": 30,
                    "available": 25,
                    "url": "https://example.com/event3",
                    "organizer": "Hobby Community"
                }
            ]
            
            if location:
                events = [e for e in events if location in e.get("location", "")]
            
            return {
                "success": True,
                "source": "event_search",
                "data": {
                    "events": events[:limit],
                    "searched_hobby": hobby,
                    "period_days": days,
                    "location": location,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Event search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_discounts(
        self, 
        hobby: str, 
        discount_min: int = 20, 
        expires_within_days: int = 7,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for discount events and promotions.
        
        Args:
            hobby: Hobby name
            discount_min: Minimum discount percentage
            expires_within_days: Discount expires within days
            limit: Maximum results
            
        Returns:
            Dict containing active discounts
        """
        try:
            discounts = [
                {
                    "name": f"{hobby} 신규회원 할인",
                    "type": "promotion",
                    "discount_percent": 30,
                    "original_price": 50000,
                    "sale_price": 35000,
                    "expires_at": (datetime.now() + timedelta(days=3)).isoformat(),
                    "platform": "HobbyList",
                    "coupon_code": "HobbyNEW30",
                    "url": "https://example.com/discount1",
                    "conditions": "신규 회원만 적용"
                },
                {
                    "name": f"{hobby} 시즌 오프",
                    "type": "clearance",
                    "discount_percent": 40,
                    "original_price": 100000,
                    "sale_price": 60000,
                    "expires_at": (datetime.now() + timedelta(days=5)).isoformat(),
                    "platform": "Partner Store",
                    "coupon_code": "SEASON40",
                    "url": "https://example.com/discount2",
                    "conditions": "한정 수량"
                },
                {
                    "name": f"{hobby} 패키지 할인",
                    "type": "bundle",
                    "discount_percent": 25,
                    "original_price": 80000,
                    "sale_price": 60000,
                    "expires_at": (datetime.now() + timedelta(days=10)).isoformat(),
                    "platform": "HobbyList",
                    "coupon_code": "PACK25",
                    "url": "https://example.com/discount3",
                    "conditions": "2개 이상 구매 시"
                }
            ]
            
            filtered = [d for d in discounts if d["discount_percent"] >= discount_min]
            
            return {
                "success": True,
                "source": "event_search",
                "data": {
                    "discounts": filtered[:limit],
                    "searched_hobby": hobby,
                    "min_discount": discount_min,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Discount search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_classes(
        self, 
        hobby: str, 
        level: str = "beginner",
        location: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for hobby classes and lessons.
        
        Args:
            hobby: Hobby name
            level: Skill level (beginner, intermediate, advanced)
            location: Class location
            limit: Maximum results
            
        Returns:
            Dict containing class listings
        """
        try:
            classes = [
                {
                    "name": f"[초급] {hobby} 4주 과정",
                    "level": "beginner",
                    "duration": "4주 (주 2회)",
                    "total_sessions": 8,
                    "price": 200000,
                    "discounted_price": 160000,
                    "schedule": "화, 목 19:00-21:00",
                    "location": "서울 강남구",
                    "venue": "Hobby Academy",
                    "instructor": "김강사",
                    "url": "https://example.com/class1"
                },
                {
                    "name": f"[중급] {hobby} 심화반",
                    "level": "intermediate",
                    "duration": "6주 (주 2회)",
                    "total_sessions": 12,
                    "price": 350000,
                    "discounted_price": 300000,
                    "schedule": "수, 금 20:00-22:00",
                    "location": "서울 마포구",
                    "venue": "Creative Studio",
                    "instructor": "이강사",
                    "url": "https://example.com/class2"
                },
                {
                    "name": f"{hobby} 원데이 클래스",
                    "level": "all",
                    "duration": "1일 (4시간)",
                    "total_sessions": 1,
                    "price": 50000,
                    "discounted_price": 40000,
                    "schedule": "토요일 14:00-18:00",
                    "location": "서울 용산구",
                    "venue": "Art Center",
                    "instructor": "박강사",
                    "url": "https://example.com/class3"
                }
            ]
            
            if level != "all":
                classes = [c for c in classes if c["level"] == level or c["level"] == "all"]
            
            if location:
                classes = [c for c in classes if location in c.get("location", "")]
            
            return {
                "success": True,
                "source": "event_search",
                "data": {
                    "classes": classes[:limit],
                    "searched_hobby": hobby,
                    "level": level,
                    "location": location,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Class search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_workshops(
        self, 
        hobby: str, 
        duration_hours_max: Optional[float] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for hobby workshops.
        
        Args:
            hobby: Hobby name
            duration_hours_max: Maximum workshop duration
            limit: Maximum results
            
        Returns:
            Dict containing workshop listings
        """
        try:
            workshops = [
                {
                    "name": f"{hobby} 주말 워크숍",
                    "duration_hours": 4,
                    "date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "time": "10:00-14:00",
                    "location": "서울 종로구",
                    "venue": "Art Factory",
                    "price": 75000,
                    "discounted_price": 60000,
                    "materials_included": True,
                    "takeaway": True,
                    "url": "https://example.com/workshop1"
                },
                {
                    "name": f"{hobby} 저녁 워크숍",
                    "duration_hours": 3,
                    "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                    "time": "19:00-22:00",
                    "location": "서울 성동구",
                    "venue": "Make Space",
                    "price": 55000,
                    "discounted_price": 50000,
                    "materials_included": True,
                    "takeaway": False,
                    "url": "https://example.com/workshop2"
                }
            ]
            
            if duration_hours_max:
                workshops = [w for w in workshops if w["duration_hours"] <= duration_hours_max]
            
            return {
                "success": True,
                "source": "event_search",
                "data": {
                    "workshops": workshops[:limit],
                    "searched_hobby": hobby,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Workshop search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_community_activities(
        self, 
        hobby: str, 
        activity_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for community activities and meetups.
        
        Args:
            hobby: Hobby name
            activity_type: Type (meeting, contest, exhibition, etc.)
            limit: Maximum results
            
        Returns:
            Dict containing community activities
        """
        try:
            activities = [
                {
                    "name": f"{hobby} 정기 모임",
                    "type": "meeting",
                    "frequency": "매주 토요일",
                    "location": "서울 강남구",
                    "venue": "Cafe Meeting",
                    "fee": 0,
                    "participants": 45,
                    "organizer": f"{hobby} Lovers Club",
                    "url": "https://example.com/activity1"
                },
                {
                    "name": f"{hobby} 대회",
                    "type": "contest",
                    "date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    "location": "서울 송파구",
                    "venue": "Sports Center",
                    "fee": 10000,
                    "participants": 100,
                    "registration_deadline": (datetime.now() + timedelta(days=20)).isoformat(),
                    "organizer": "Hobby Sports Association",
                    "url": "https://example.com/activity2"
                },
                {
                    "name": f"{hobby} 전시회",
                    "type": "exhibition",
                    "start_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                    "end_date": (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"),
                    "location": "서울 용산구",
                    "venue": "Art Gallery",
                    "fee": 5000,
                    "organizer": "Art Community",
                    "url": "https://example.com/activity3"
                }
            ]
            
            if activity_type:
                activities = [a for a in activities if a["type"] == activity_type]
            
            return {
                "success": True,
                "source": "event_search",
                "data": {
                    "activities": activities[:limit],
                    "searched_hobby": hobby,
                    "activity_type": activity_type,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Community activity search error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities
        }


# Singleton instance
_event_search_mcp = None

def get_event_search_mcp() -> EventSearchMCPServer:
    """Get singleton EventSearchMCPServer instance."""
    global _event_search_mcp
    if _event_search_mcp is None:
        _event_search_mcp = EventSearchMCPServer()
    return _event_search_mcp
