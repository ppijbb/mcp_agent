"""
A2A Bridge for Sales Communication
HSP Agent와 Sales Agent 간의 A2A 프로토콜 통신
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SalesMessageType(Enum):
    """Sales 관련 A2A 메시지 유형"""
    # HSP -> Sales
    TICKET_RECOMMENDATION_REQUEST = "ticket_recommendation_request"
    STARTER_PACK_REQUEST = "starter_pack_request"
    PRODUCT_SEARCH_REQUEST = "product_search_request"
    PRICE_INQUIRY = "price_inquiry"
    
    # Sales -> HSP
    TICKET_RECOMMENDATION = "ticket_recommendation"
    STARTER_PACK_OFFER = "starter_pack_offer"
    PRODUCT_INFO = "product_info"
    PRICE_QUOTE = "price_quote"
    PURCHASE_CONFIRMATION = "purchase_confirmation"
    
    # Generic
    ERROR = "error"


class SalesA2ABridge:
    """
    HSP Agent와 Sales Agent 간 A2A 프로토콜 브리지
    
    사용자가 취미를 탐색하다가 구매意向을 보이면
    HSP Agent가 Sales Agent에게 요청을 전송합니다.
    """
    
    def __init__(self):
        self.name = "SalesA2ABridge"
        self.sales_agents = {}
        self.message_history = []
        
    async def register_sales_agent(self, agent_id: str, agent_type: str):
        """
        Sales Agent를 등록합니다.
        
        Args:
            agent_id: 에이전트 ID
            agent_type: 에이전트 유형 (ticket, starter_pack, order, bulk)
        """
        self.sales_agents[agent_id] = {
            "type": agent_type,
            "registered_at": datetime.now().isoformat()
        }
        logger.info(f"Sales Agent 등록 완료: {agent_id} ({agent_type})")
    
    async def send_to_sales(
        self,
        sender: str,
        message_type: str,
        payload: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Sales Agent에게 메시지를 전송합니다.
        
        Args:
            sender: 발신자 (HSP Agent명)
            message_type: 메시지 유형
            payload: 메시지 데이터
            session_id: 세션 ID
            
        Returns:
            Sales Agent 응답
        """
        try:
            from autogen.sales_agents import sales_manager
            
            # 메시지 유형별 처리
            request_map = {
                SalesMessageType.TICKET_RECOMMENDATION_REQUEST.value: {
                    "handler": "search_tickets",
                    "params": {
                        "hobby": payload.get("hobby"),
                        "ticket_type": payload.get("ticket_type", "all"),
                        "max_price": payload.get("max_price"),
                        "location": payload.get("location")
                    }
                },
                SalesMessageType.STARTER_PACK_REQUEST.value: {
                    "handler": "get_recommended_packs",
                    "params": {
                        "user_profile": payload.get("user_profile", {})
                    }
                },
                SalesMessageType.PRODUCT_SEARCH_REQUEST.value: {
                    "handler": "search_tickets",  # 재사용
                    "params": {
                        "hobby": payload.get("hobby"),
                        "ticket_type": payload.get("product_type", "all")
                    }
                },
                SalesMessageType.PRICE_INQUIRY.value: {
                    "handler": "get_starter_pack",
                    "params": {
                        "hobby": payload.get("hobby")
                    }
                }
            }
            
            if message_type not in request_map:
                return {
                    "success": False,
                    "error": f"Unknown message type: {message_type}"
                }
            
            request_info = request_map[message_type]
            
            # Sales Manager에게 요청
            result = await sales_manager.handle_sales_request(
                request_type=request_info["handler"],
                params=request_info["params"]
            )
            
            # 메시지 이력 저장
            self.message_history.append({
                "sender": sender,
                "message_type": message_type,
                "payload": payload,
                "response": result,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Sales 메시지 전송 완료: {message_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Sales 메시지 전송 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def request_ticket_recommendation(
        self,
        hobby: str,
        user_profile: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        티켓 추천을 요청합니다.
        
        Args:
            hobby: 취미명
            user_profile: 사용자 프로필
            session_id: 세션 ID
            
        Returns:
            티켓 추천 결과
        """
        from autogen.sales_agents import sales_manager
        
        result = await sales_manager.handle_sales_request(
            request_type="search_tickets",
            params={
                "hobby": hobby,
                "ticket_type": "all",
                "max_price": None,
                "location": user_profile.get("location") if user_profile else None
            }
        )
        
        return result
    
    async def request_starter_pack(
        self,
        hobby: str,
        user_profile: Optional[Dict[str, Any]] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        입문 키트 추천을 요청합니다.
        
        Args:
            hobby: 취미명
            user_profile: 사용자 프로필
            session_id: 세션 ID
            
        Returns:
            입문 키트 추천 결과
        """
        from autogen.sales_agents import sales_manager
        
        if user_profile:
            result = await sales_manager.handle_sales_request(
                request_type="get_recommended_packs",
                params={"user_profile": user_profile}
            )
        else:
            result = await sales_manager.handle_sales_request(
                request_type="get_starter_pack",
                params={"hobby": hobby}
            )
        
        return result
    
    async def create_purchase(
        self,
        user_id: str,
        product_type: str,  # ticket, starter_pack
        product_id: str,
        hobby: str,
        payment_info: Dict[str, Any],
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        구매를 생성합니다.
        
        Args:
            user_id: 사용자 ID
            product_type: 상품 유형
            product_id: 상품 ID
            hobby: 취미명
            payment_info: 결제 정보
            session_id: 세션 ID
            
        Returns:
            구매 결과
        """
        from autogen.sales_agents import sales_manager
        
        if product_type == "ticket":
            # 티켓 구매
            result = await sales_manager.handle_sales_request(
                request_type="purchase_ticket",
                params={
                    "user_id": user_id,
                    "ticket_id": product_id,
                    "payment_info": payment_info
                }
            )
        elif product_type == "starter_pack":
            # 입문 키트 구매
            result = await sales_manager.handle_sales_request(
                request_type="purchase_starter_pack",
                params={
                    "user_id": user_id,
                    "hobby": hobby,
                    "shipping_info": payment_info.get("shipping", {})
                }
            )
        else:
            return {
                "success": False,
                "error": f"Unknown product type: {product_type}"
            }
        
        return result
    
    async def get_price_quote(
        self,
        hobby: str,
        item_type: str,  # ticket, pack, product
        quantity: int = 1,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        가격 견적을 요청합니다.
        
        Args:
            hobby: 취미명
            item_type: 아이템 유형
            quantity: 수량
            session_id: 세션 ID
            
        Returns:
            가격 견적
        """
        from autogen.sales_agents import sales_manager
        
        if item_type == "pack":
            result = await sales_manager.handle_sales_request(
                request_type="get_starter_pack",
                params={"hobby": hobby}
            )
        else:
            result = await sales_manager.handle_sales_request(
                request_type="search_tickets",
                params={
                    "hobby": hobby,
                    "ticket_type": item_type
                }
            )
        
        return result
    
    def get_message_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        메시지 이력을 조회합니다.
        
        Args:
            session_id: 필터링할 세션 ID (선택)
            
        Returns:
            메시지 이력 목록
        """
        if session_id:
            return [m for m in self.message_history if m.get("session_id") == session_id]
        return self.message_history
    
    def clear_history(self):
        """메시지 이력을 초기화합니다."""
        self.message_history = []
        logger.info("Sales A2A 메시지 이력 초기화")


# Singleton instance
sales_a2a_bridge = SalesA2ABridge()


async def handle_hsp_to_sales_request(
    hsp_agent: str,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    HSP Agent의 Sales 요청을 처리합니다.
    
    Args:
        hsp_agent: HSP Agent명
        request: 요청 데이터
        
    Returns:
        응답 데이터
    """
    request_type = request.get("type")
    payload = request.get("payload", {})
    session_id = request.get("session_id", "default")
    
    return await sales_a2a_bridge.send_to_sales(
        sender=hsp_agent,
        message_type=request_type,
        payload=payload,
        session_id=session_id
    )
