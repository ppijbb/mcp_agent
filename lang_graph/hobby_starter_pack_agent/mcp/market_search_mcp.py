"""
MCP Market Search Server
Provides market search capabilities for hobby equipment and products.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MarketSearchMCPServer:
    """
    MCP Server for market search functionality.
    
    Capabilities:
    - Search for hobby equipment prices
    - Find discounts and deals
    - Compare products across platforms
    - Get product recommendations
    """
    
    def __init__(self):
        self.name = "market_search"
        self.description = "Market search server for hobby equipment and products"
        self.capabilities = [
            "search_equipment",
            "compare_prices",
            "search_discounts",
            "find_discounts",
            "get_product_recommendations",
            "search_reviews"
        ]
    
    async def search_equipment(
        self, 
        hobby: str, 
        category: Optional[str] = None,
        budget_min: Optional[int] = None,
        budget_max: Optional[int] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for hobby equipment/products.
        
        Args:
            hobby: Hobby name
            category: Product category (shoes, clothes, tools, etc.)
            budget_min: Minimum budget
            budget_max: Maximum budget
            limit: Maximum results
            
        Returns:
            Dict containing product search results
        """
        try:
            # Simulated product database
            products = [
                {
                    "name": f"{hobby} 입문용 세트",
                    "category": "세트",
                    "brand": "HobbyStart",
                    "price": 89000,
                    "original_price": 120000,
                    "discount": 26,
                    "platform": "쿠팡",
                    "rating": 4.5,
                    "reviews": 234,
                    "url": "https://example.com/product1",
                    "description": f"{hobby} 입문에 필요한 모든 것"
                },
                {
                    "name": f"{hobby} 고급 장비",
                    "category": "전문",
                    "brand": "ProGear",
                    "price": 250000,
                    "original_price": 300000,
                    "discount": 17,
                    "platform": "네이버 스마트스토어",
                    "rating": 4.8,
                    "reviews": 89,
                    "url": "https://example.com/product2",
                    "description": f"전문가를 위한 고품질 {hobby} 장비"
                },
                {
                    "name": f"{hobby} 기본 장비",
                    "category": "기본",
                    "brand": "BasicFit",
                    "price": 45000,
                    "original_price": 50000,
                    "discount": 10,
                    "platform": "G마켓",
                    "rating": 4.2,
                    "reviews": 567,
                    "url": "https://example.com/product3",
                    "description": f"{hobby} 시작하는 분들을 위한 기본 장비"
                }
            ]
            
            # Filter by budget
            if budget_min or budget_max:
                filtered = []
                for p in products:
                    if budget_min and p["price"] < budget_min:
                        continue
                    if budget_max and p["price"] > budget_max:
                        continue
                    filtered.append(p)
                products = filtered
            
            return {
                "success": True,
                "source": "market_search",
                "data": {
                    "products": products[:limit],
                    "searched_hobby": hobby,
                    "filters": {
                        "category": category,
                        "budget_min": budget_min,
                        "budget_max": budget_max
                    },
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Market search error: {e}")
            return {"success": False, "error": str(e)}
    
    async def compare_prices(self, product_name: str, platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare prices across platforms.
        
        Args:
            product_name: Product to compare
            platforms: List of platforms to compare
            
        Returns:
            Dict containing price comparison
        """
        try:
            # Simulated price comparison
            comparison = [
                {
                    "platform": "쿠팡",
                    "price": 89000,
                    "delivery": "무료",
                    "url": "https://example.com/coupang",
                    "in_stock": True
                },
                {
                    "platform": "네이버 스마트스토어",
                    "price": 92000,
                    "delivery": "₩3,000",
                    "url": "https://example.com/naver",
                    "in_stock": True
                },
                {
                    "platform": "G마켓",
                    "price": 85000,
                    "delivery": "₩2,500",
                    "url": "https://example.com/gmarket",
                    "in_stock": True
                },
                {
                    "platform": "11번가",
                    "price": 88000,
                    "delivery": "무료",
                    "url": "https://example.com/11st",
                    "in_stock": True
                }
            ]
            
            if platforms:
                comparison = [p for p in comparison if p["platform"] in platforms]
            
            # Sort by price
            comparison.sort(key=lambda x: x["price"])
            
            best = comparison[0] if comparison else None
            
            return {
                "success": True,
                "source": "market_search",
                "data": {
                    "product": product_name,
                    "comparison": comparison,
                    "best_price": best,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Price comparison error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_discounts(self, hobby: str, discount_min: int = 20, limit: int = 10) -> Dict[str, Any]:
        """
        Search for discounted products for a hobby.
        
        Args:
            hobby: Hobby name
            discount_min: Minimum discount percentage
            limit: Maximum results
            
        Returns:
            Dict containing discounted products
        """
        return await self.find_discounts(hobby, discount_min, limit)
    
    async def find_discounts(self, hobby: str, discount_min: int = 20, limit: int = 10) -> Dict[str, Any]:
        """
        Find discounted products for a hobby.
        
        Args:
            hobby: Hobby name
            discount_min: Minimum discount percentage
            limit: Maximum results
            
        Returns:
            Dict containing discounted products
        """
        try:
            discounts = [
                {
                    "name": f"{hobby} 세트 상품",
                    "discount_percent": 35,
                    "current_price": 65000,
                    "original_price": 100000,
                    "platform": "쿠팡",
                    "ends_at": (datetime.now()).isoformat(),
                    "url": "https://example.com/discount1"
                },
                {
                    "name": f"{hobby} 시즌 오프",
                    "discount_percent": 30,
                    "current_price": 42000,
                    "original_price": 60000,
                    "platform": "G마켓",
                    "ends_at": (datetime.now()).isoformat(),
                    "url": "https://example.com/discount2"
                },
                {
                    "name": f"{hobby} 특별 할인",
                    "discount_percent": 25,
                    "current_price": 75000,
                    "original_price": 100000,
                    "platform": "옥션",
                    "ends_at": (datetime.now()).isoformat(),
                    "url": "https://example.com/discount3"
                }
            ]
            
            filtered = [d for d in discounts if d["discount_percent"] >= discount_min]
            
            return {
                "success": True,
                "source": "market_search",
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
    
    async def get_product_recommendations(self, hobby: str, budget: int, purpose: str = "beginner") -> Dict[str, Any]:
        """
        Get product recommendations based on budget and purpose.
        
        Args:
            hobby: Hobby name
            budget: Available budget
            purpose: Purpose (beginner, intermediate, gift)
            
        Returns:
            Dict containing recommendations
        """
        try:
            recommendations = {
                "beginner": [
                    {
                        "name": f"{hobby} 입문 키트",
                        "items": [f"{hobby} 기본 도구", f"{hobby} 가이드북", f"{hobby} 소모품 세트"],
                        "total_price": budget,
                        "value_score": 9.0,
                        "url": "https://example.com/starter-kit"
                    }
                ],
                "intermediate": [
                    {
                        "name": f"{hobby} 중급 키트",
                        "items": [f"{hobby} 전문 도구", f"{hobby} 고급 재료", f"{hobby} 케이스"],
                        "total_price": budget,
                        "value_score": 8.5,
                        "url": "https://example.com/intermediate-kit"
                    }
                ],
                "gift": [
                    {
                        "name": f"{hobby} 선물 세트",
                        "items": [f"{hobby} 고급 선물", f"{hobby} 포장 세트", f"{hobby} 선물 카드"],
                        "total_price": budget,
                        "value_score": 8.8,
                        "url": "https://example.com/gift-kit"
                    }
                ]
            }
            
            result = recommendations.get(purpose, recommendations["beginner"])
            
            return {
                "success": True,
                "source": "market_search",
                "data": {
                    "recommendations": result,
                    "hobby": hobby,
                    "budget": budget,
                    "purpose": purpose,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_reviews(self, product_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for product reviews.
        
        Args:
            product_name: Product to search reviews for
            limit: Maximum results
            
        Returns:
            Dict containing reviews
        """
        try:
            reviews = [
                {
                    "author": "구매자1",
                    "rating": 5,
                    "content": "정말 만족스러워요. 다음에도 여기서 구매할게요.",
                    "date": datetime.now().isoformat(),
                    "platform": "쿠팡"
                },
                {
                    "author": "구매자2", 
                    "rating": 4,
                    "content": "좋아요. 다만 배송이 조금 걸렸습니다.",
                    "date": datetime.now().isoformat(),
                    "platform": "네이버"
                }
            ]
            
            return {
                "success": True,
                "source": "market_search",
                "data": {
                    "product": product_name,
                    "reviews": reviews[:limit],
                    "average_rating": 4.5,
                    "searched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Review search error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities
        }


# Singleton instance
_market_search_mcp = None

def get_market_search_mcp() -> MarketSearchMCPServer:
    """Get singleton MarketSearchMCPServer instance."""
    global _market_search_mcp
    if _market_search_mcp is None:
        _market_search_mcp = MarketSearchMCPServer()
    return _market_search_mcp
