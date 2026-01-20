"""
MCP Location Search Server
Provides location-based search capabilities for hobby activities.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class LocationSearchMCPServer:
    """
    MCP Server for location-based search functionality.
    
    Capabilities:
    - Search for nearby hobby shops
    - Find hobby venues and studios
    - Locate community centers
    - Get regional hobby information
    """
    
    def __init__(self):
        self.name = "location_search"
        self.description = "Location-based search server for hobby shops and venues"
        self.capabilities = [
            "search_nearby_shops",
            "search_venues",
            "search_community_centers",
            "search_regional_hobby_info"
        ]
    
    async def search_nearby_shops(
        self, 
        hobby: str, 
        location: str,
        radius_km: float = 5.0,
        shop_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for nearby hobby shops.
        
        Args:
            hobby: Hobby name
            location: User location (e.g., "서울 강남구")
            radius_km: Search radius in km
            shop_type: Type (retail, online, rental)
            limit: Maximum results
            
        Returns:
            Dict containing nearby shops
        """
        try:
            shops = [
                {
                    "name": f"{hobby} 전문점",
                    "type": "retail",
                    "address": f"{location} 테헤란로 123",
                    "distance_km": 0.8,
                    "phone": "02-1234-5678",
                    "hours": "10:00-21:00",
                    "products": [f"{hobby} 장비", f"{hobby} 소모품"],
                    "price_range": "중간",
                    "rating": 4.5,
                    "url": "https://example.com/shop1"
                },
                {
                    "name": f"{hobby} 대용",
                    "type": "rental",
                    "address": f"{location} 삼성로 45",
                    "distance_km": 1.2,
                    "phone": "02-2345-6789",
                    "hours": "09:00-20:00",
                    "products": [f"{hobby} 장비 대여"],
                    "price_range": "저렴",
                    "rating": 4.2,
                    "url": "https://example.com/shop2"
                },
                {
                    "name": f"{hobby} 온라인 스토어",
                    "type": "online",
                    "address": "온라인몰",
                    "distance_km": 0,
                    "phone": "1588-0000",
                    "hours": "24시간",
                    "products": ["전체 {hobby} 제품"],
                    "price_range": "다양함",
                    "rating": 4.3,
                    "url": "https://example.com/shop3"
                }
            ]
            
            if shop_type:
                shops = [s for s in shops if s["type"] == shop_type]
            
            return {
                "success": True,
                "source": "location_search",
                "data": {
                    "shops": shops[:limit],
                    "searched_hobby": hobby,
                    "location": location,
                    "radius_km": radius_km,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Shop search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_venues(
        self, 
        hobby: str, 
        location: str,
        venue_type: Optional[str] = None,
        price_max: Optional[int] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for hobby venues and studios.
        
        Args:
            hobby: Hobby name
            location: Location to search
            venue_type: Type (studio, gym, outdoor, cafe)
            price_max: Maximum hourly price
            limit: Maximum results
            
        Returns:
            Dict containing venue listings
        """
        try:
            venues = [
                {
                    "name": f"{hobby} 스튜디오",
                    "type": "studio",
                    "address": f"{location} 예술로 77",
                    "price_per_hour": 30000,
                    "capacity": 20,
                    "amenities": ["에어컨", "주차장", "화장실"],
                    "rating": 4.7,
                    "booking_url": "https://example.com/venue1"
                },
                {
                    "name": f"{hobby} 센터",
                    "type": "gym",
                    "address": f"{location} 건강로 88",
                    "price_per_hour": 20000,
                    "capacity": 30,
                    "amenities": ["탈의실", "샤워실", "주차장"],
                    "rating": 4.4,
                    "booking_url": "https://example.com/venue2"
                },
                {
                    "name": f"{hobby} 야외 공간",
                    "type": "outdoor",
                    "address": f"{location} 공원로 99",
                    "price_per_hour": 0,
                    "capacity": 50,
                    "amenities": ["화장실", "자율 parking"],
                    "rating": 4.1,
                    "booking_url": "https://example.com/venue3"
                },
                {
                    "name": f"{hobby} 카페",
                    "type": "cafe",
                    "address": f"{location} 카페거리 11",
                    "price_per_hour": 15000,
                    "capacity": 15,
                    "amenities": ["와이파이", "주차", "간단 다과"],
                    "rating": 4.3,
                    "booking_url": "https://example.com/venue4"
                }
            ]
            
            if venue_type:
                venues = [v for v in venues if v["type"] == venue_type]
            
            if price_max:
                venues = [v for v in venues if v["price_per_hour"] <= price_max]
            
            return {
                "success": True,
                "source": "location_search",
                "data": {
                    "venues": venues[:limit],
                    "searched_hobby": hobby,
                    "location": location,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Venue search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_community_centers(
        self, 
        hobby: str, 
        location: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for community centers offering hobby programs.
        
        Args:
            hobby: Hobby name
            location: Location to search
            limit: Maximum results
            
        Returns:
            Dict containing community centers
        """
        try:
            centers = [
                {
                    "name": f"{location} 주민센터",
                    "address": f"{location} 주민센터로 1",
                    "programs": [f"{hobby} 입문 강좌", f"{hobby} 동아리"],
                    "program_fee": "무료 또는 저가",
                    "registration": "필요",
                    "contact": "02-1111-1111",
                    "url": "https://example.com/center1"
                },
                {
                    "name": f"{location} 문화센터",
                    "address": f"{location} 문화로 22",
                    "programs": [f"{hobby} 정기 강좌", f"{hobby} 워크숍"],
                    "program_fee": "저렴",
                    "registration": "필요",
                    "contact": "02-2222-2222",
                    "url": "https://example.com/center2"
                },
                {
                    "name": f"{location} 체육센터",
                    "address": f"{location} 체육로 33",
                    "programs": [f"{hobby} 교실"],
                    "program_fee": "중간",
                    "registration": "필요",
                    "contact": "02-3333-3333",
                    "url": "https://example.com/center3"
                }
            ]
            
            return {
                "success": True,
                "source": "location_search",
                "data": {
                    "centers": centers[:limit],
                    "searched_hobby": hobby,
                    "location": location,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Community center search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_regional_hobby_info(
        self, 
        hobby: str, 
        region: str,
        info_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for regional hobby information.
        
        Args:
            hobby: Hobby name
            region: Region to search
            info_type: Type (clubs, events, facilities)
            limit: Maximum results
            
        Returns:
            Dict containing regional hobby information
        """
        try:
            info = {
                "clubs": [
                    {
                        "name": f"{region} {hobby} 동호회",
                        "members": 150,
                        "activities": "주간 모임, 월간 행사",
                        "contact": "카카오톡 ID: club123"
                    }
                ],
                "events": [
                    {
                        "name": f"{region} {hobby} 페스티벌",
                        "date": "2024-05",
                        "venue": f"{region} 축제장",
                        "expected_visitors": 5000
                    }
                ],
                "facilities": [
                    {
                        "name": f"{region} {hobby} 체험관",
                        "address": f"{region} 체험로 5",
                        "hours": "10:00-18:00",
                        "fee": "성인 5,000원"
                    }
                ]
            }
            
            if info_type:
                info = {info_type: info.get(info_type, [])}
            
            return {
                "success": True,
                "source": "location_search",
                "data": {
                    "regional_info": info,
                    "searched_hobby": hobby,
                    "region": region,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Regional info search error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities
        }


# Singleton instance
_location_search_mcp = None

def get_location_search_mcp() -> LocationSearchMCPServer:
    """Get singleton LocationSearchMCPServer instance."""
    global _location_search_mcp
    if _location_search_mcp is None:
        _location_search_mcp = LocationSearchMCPServer()
    return _location_search_mcp
