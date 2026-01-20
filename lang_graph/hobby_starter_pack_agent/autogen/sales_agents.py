"""
Sales Agents for HobbyList
비즈니스 로직을 처리하는 Sales Agent들

Agents:
- TicketAgent: 티켓/클래스/이벤트 판매
- StarterPackAgent: 입문 키트 판매
- OrderAgent: 주문/결제 처리
- BulkAgent: 대량구매 관리
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Product:
    """상품 정보"""
    product_id: str
    name: str
    category: str
    original_price: int
    sale_price: int
    margin: int
    margin_percent: float
    stock: int
    description: str
    supplier: str
    image_url: str = ""
    rating: float = 0.0
    reviews: int = 0


@dataclass
class Order:
    """주문 정보"""
    order_id: str
    user_id: str
    products: List[Dict[str, Any]]
    total_amount: int
    status: str  # pending, paid, cancelled, completed
    payment_method: str = ""
    payment_id: str = ""
    created_at: str = ""
    completed_at: str = ""


@dataclass
class Ticket:
    """티켓 정보"""
    ticket_id: str
    order_id: str
    product_id: str
    user_id: str
    qr_code: str
    status: str  # valid, used, expired
    valid_from: str
    valid_until: str
    event_name: str


@dataclass
class BulkDeal:
    """대량구매 딜 정보"""
    deal_id: str
    product_name: str
    supplier: str
    quantity: int
    unit_price: int
    total_price: int
    our_sale_price: int
    margin_per_unit: int
    margin_percent: float
    status: str  # negotiating, ordered, received, sold_out
    created_at: str = ""


# ============================================================================
# TicketAgent
# ============================================================================

class TicketAgent:
    """
    티켓/클래스/이벤트 판매 Agent
    
    Responsibilities:
    - Search and recommend tickets/classes/events
    - Handle ticket purchases
    - Generate QR codes
    - Manage ticket validity
    """
    
    def __init__(self):
        self.name = "TicketAgent"
        self.description = "티켓 및 클래스 판매 Agent"
        
        # 목업 티켓 데이터베이스
        self.tickets_db = {}
        
    async def search_tickets(
        self, 
        hobby: str, 
        ticket_type: str = "all",
        max_price: Optional[int] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        티켓을 검색합니다.
        
        Args:
            hobby: 취미명
            ticket_type: 티켓 유형 (class, event, workshop, all)
            max_price: 최대 가격
            location: 위치
            
        Returns:
            검색된 티켓 목록
        """
        try:
            # MCP Event Server에서 검색한 것으로 가정
            tickets = [
                {
                    "ticket_id": f"TKT-{hobby[:3].upper()}-001",
                    "name": f"[초급] {hobby} 4주 과정",
                    "type": "class",
                    "date": "2024-02-15",
                    "time": "19:00",
                    "location": "서울 강남구",
                    "venue": "Hobby Center",
                    "price": 200000,
                    "sale_price": 160000,
                    "discount": 20,
                    "available": 12,
                    "duration": "4주 (주 2회)"
                },
                {
                    "ticket_id": f"TKT-{hobby[:3].upper()}-002",
                    "name": f"{hobby} 원데이 클래스",
                    "type": "workshop",
                    "date": "2024-02-20",
                    "time": "14:00",
                    "location": "서울 마포구",
                    "venue": "Creative Space",
                    "price": 50000,
                    "sale_price": 40000,
                    "discount": 20,
                    "available": 8,
                    "duration": "4시간"
                }
            ]
            
            # 필터링
            if ticket_type != "all":
                tickets = [t for t in tickets if t["type"] == ticket_type]
            
            if max_price:
                tickets = [t for t in tickets if t["sale_price"] <= max_price]
            
            if location:
                tickets = [t for t in tickets if location in t.get("location", "")]
            
            logger.info(f"티켓 검색 완료: {hobby}, {len(tickets)}개 결과")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "tickets": tickets,
                    "searched_hobby": hobby,
                    "filters": {
                        "type": ticket_type,
                        "max_price": max_price,
                        "location": location
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"티켓 검색 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def purchase_ticket(
        self, 
        user_id: str, 
        ticket_id: str,
        payment_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        티켓을 구매합니다.
        
        Args:
            user_id: 사용자 ID
            ticket_id: 티켓 ID
            payment_info: 결제 정보
            
        Returns:
            구매 결과 및 티켓 정보
        """
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
            ticket_code = f"QR-{uuid.uuid4().hex[:8].upper()}"
            
            # 티켓 생성
            ticket = {
                "ticket_id": ticket_id,
                "order_id": order_id,
                "user_id": user_id,
                "qr_code": ticket_code,
                "status": "valid",
                "purchased_at": datetime.now().isoformat(),
                "valid_until": (datetime.now() + timedelta(days=365)).isoformat()
            }
            
            # OrderAgent에게 주문 생성 요청
            order_agent = OrderAgent()
            
            order_result = await order_agent.create_order(
                user_id=user_id,
                products=[{
                    "product_id": ticket_id,
                    "name": f"티켓 {ticket_id}",
                    "price": 50000  # 티켓 가격
                }],
                payment_info=payment_info
            )
            
            logger.info(f"티켓 구매 완료: {user_id} -> {ticket_id}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "order_id": order_id,
                    "ticket": ticket,
                    "qr_code_url": f"https://hobbylist.com/ticket/{ticket_code}"
                }
            }
            
        except Exception as e:
            logger.error(f"티켓 구매 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_ticket(self, qr_code: str) -> Dict[str, Any]:
        """
        티켓을 검증합니다.
        
        Args:
            qr_code: QR 코드
            
        Returns:
            검증 결과
        """
        try:
            # 실제로는 데이터베이스에서 조회
            ticket_info = {
                "valid": True,
                "ticket_id": "TKT-CLA-001",
                "event_name": "[초급] 클라이밍 4주 과정",
                "event_date": "2024-02-15",
                "user_id": "user123"
            }
            
            return {
                "success": True,
                "agent": self.name,
                "data": ticket_info
            }
            
        except Exception as e:
            logger.error(f"티켓 검증 실패: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# StarterPackAgent
# ============================================================================

class StarterPackAgent:
    """
    입문 키트 판매 Agent
    
    Responsibilities:
    - Recommend starter packs based on hobby
    - Create custom starter packs
    - Manage inventory
    """
    
    def __init__(self):
        self.name = "StarterPackAgent"
        self.description = "입문 키트 판매 Agent"
        
        # 입문 키트 데이터베이스
        self.starter_packs = {
            "등산": {
                "name": "등산 입문 키트",
                "items": [
                    {"name": "등산화 (중급)", "price": 69000},
                    {"name": "등산배낭 30L", "price": 45000},
                    {"name": "보온병", "price": 15000},
                    {"name": "등산 가이드북", "price": 12000}
                ],
                "total_original": 141000,
                "pack_price": 99000,
                "margin": 15000,
                "margin_percent": 15.2,
                "stock": 50
            },
            "요리": {
                "name": "요리 입문 키트",
                "items": [
                    {"name": "요리 도구 세트", "price": 59000},
                    {"name": "앞치마", "price": 15000},
                    {"name": "레시피 북", "price": 10000}
                ],
                "total_original": 84000,
                "pack_price": 59000,
                "margin": 12000,
                "margin_percent": 20.3,
                "stock": 30
            },
            "사진": {
                "name": "사진 입문 키트",
                "items": [
                    {"name": "디지털 카메라 entry", "price": 350000},
                    {"name": "삼각대", "price": 80000},
                    {"name": "메모리 카드 64GB", "price": 25000},
                    {"name": "카메라 가방", "price": 35000}
                ],
                "total_original": 490000,
                "pack_price": 399000,
                "margin": 49000,
                "margin_percent": 12.3,
                "stock": 20
            }
        }
    
    async def get_starter_pack(self, hobby: str) -> Dict[str, Any]:
        """
        입문 키트를 조회합니다.
        
        Args:
            hobby: 취미명
            
        Returns:
            입문 키트 정보
        """
        try:
            pack = self.starter_packs.get(hobby)
            
            if not pack:
                return {
                    "success": False,
                    "agent": self.name,
                    "error": f"입문 키트가 없습니다: {hobby}"
                }
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "hobby": hobby,
                    "pack": pack
                }
            }
            
        except Exception as e:
            logger.error(f"입문 키트 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_recommended_packs(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 프로필 기반 추천 입문 키트 목록을 반환합니다.
        
        Args:
            user_profile: 사용자 프로필
            
        Returns:
            추천 입문 키트 목록
        """
        try:
            interests = user_profile.get("interests", [])
            
            # 관심사와 매칭되는 키트 찾기
            recommended = []
            for hobby, pack in self.starter_packs.items():
                if any(hobby.lower() in interest.lower() for interest in interests):
                    pack_copy = pack.copy()
                    pack_copy["hobby"] = hobby
                    pack_copy["match_reason"] = f"'{hobby}' 관련 관심사"
                    recommended.append(pack_copy)
            
            # 관심사와 매칭되는 것이 없으면 기본 추천
            if not recommended:
                for hobby, pack in list(self.starter_packs.items())[:3]:
                    pack_copy = pack.copy()
                    pack_copy["hobby"] = hobby
                    pack_copy["match_reason"] = "인기 입문 키트"
                    recommended.append(pack_copy)
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "recommended_packs": recommended,
                    "based_on": interests
                }
            }
            
        except Exception as e:
            logger.error(f"추천 입문 키트 조회 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def purchase_starter_pack(
        self, 
        user_id: str, 
        hobby: str,
        shipping_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        입문 키트를 구매합니다.
        
        Args:
            user_id: 사용자 ID
            hobby: 취미명
            shipping_info: 배송 정보
            
        Returns:
            구매 결과
        """
        try:
            pack = self.starter_packs.get(hobby)
            
            if not pack:
                return {
                    "success": False,
                    "agent": self.name,
                    "error": f"입문 키트가 없습니다: {hobby}"
                }
            
            if pack["stock"] <= 0:
                return {
                    "success": False,
                    "agent": self.name,
                    "error": "재고가 없습니다"
                }
            
            order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
            
            # 주문 생성
            order = {
                "order_id": order_id,
                "user_id": user_id,
                "product_type": "starter_pack",
                "hobby": hobby,
                "items": pack["items"],
                "total_price": pack["pack_price"],
                "shipping_info": shipping_info,
                "status": "paid",
                "created_at": datetime.now().isoformat()
            }
            
            # 재고 감소
            self.starter_packs[hobby]["stock"] -= 1
            
            logger.info(f"입문 키트 구매 완료: {user_id} -> {hobby}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "order_id": order_id,
                    "hobby": hobby,
                    "items": pack["items"],
                    "total_price": pack["pack_price"],
                    "shipping": shipping_info,
                    "estimated_delivery": "3-5일 내"
                }
            }
            
        except Exception as e:
            logger.error(f"입문 키트 구매 실패: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# OrderAgent
# ============================================================================

class OrderAgent:
    """
    주문/결제 처리 Agent
    
    Responsibilities:
    - Create and manage orders
    - Process payments
    - Handle refunds
    """
    
    def __init__(self):
        self.name = "OrderAgent"
        self.description = "주문 및 결제 처리 Agent"
        
        self.orders_db = {}
    
    async def create_order(
        self, 
        user_id: str, 
        products: List[Dict[str, Any]],
        payment_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        주문을 생성합니다.
        
        Args:
            user_id: 사용자 ID
            products: 상품 목록
            payment_info: 결제 정보
            
        Returns:
            생성된 주문 정보
        """
        try:
            order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
            
            # 총액 계산
            total_amount = sum(p["price"] for p in products)
            
            order = {
                "order_id": order_id,
                "user_id": user_id,
                "products": products,
                "total_amount": total_amount,
                "status": "pending",
                "payment_info": payment_info,
                "created_at": datetime.now().isoformat()
            }
            
            self.orders_db[order_id] = order
            
            logger.info(f"주문 생성 완료: {order_id}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": order
            }
            
        except Exception as e:
            logger.error(f"주문 생성 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_payment(
        self, 
        order_id: str, 
        payment_method: str,
        payment_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        결제를 처리합니다.
        
        Args:
            order_id: 주문 ID
            payment_method: 결제 수단
            payment_details: 결제 상세 정보
            
        Returns:
            결제 결과
        """
        try:
            if order_id not in self.orders_db:
                return {
                    "success": False,
                    "agent": self.name,
                    "error": "주문을 찾을 수 없습니다"
                }
            
            order = self.orders_db[order_id]
            
            # 결제 처리 (실제로는 PG API 호출)
            payment_id = f"PAY-{uuid.uuid4().hex[:12].upper()}"
            
            order["status"] = "paid"
            order["payment_method"] = payment_method
            order["payment_id"] = payment_id
            order["paid_at"] = datetime.now().isoformat()
            
            logger.info(f"결제 처리 완료: {order_id} -> {payment_id}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "order_id": order_id,
                    "payment_id": payment_id,
                    "status": "paid",
                    "amount": order["total_amount"]
                }
            }
            
        except Exception as e:
            logger.error(f"결제 처리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def refund_order(self, order_id: str, reason: str) -> Dict[str, Any]:
        """
        환불을 처리합니다.
        
        Args:
            order_id: 주문 ID
            reason: 환불 사유
            
        Returns:
            환불 결과
        """
        try:
            if order_id not in self.orders_db:
                return {
                    "success": False,
                    "agent": self.name,
                    "error": "주문을 찾을 수 없습니다"
                }
            
            order = self.orders_db[order_id]
            
            if order["status"] != "paid":
                return {
                    "success": False,
                    "agent": self.name,
                    "error": "환불 가능한 상태가 아닙니다"
                }
            
            # 환불 처리
            order["status"] = "refunded"
            order["refunded_at"] = datetime.now().isoformat()
            order["refund_reason"] = reason
            
            logger.info(f"환불 처리 완료: {order_id}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "order_id": order_id,
                    "status": "refunded",
                    "refund_amount": order["total_amount"],
                    "reason": reason
                }
            }
            
        except Exception as e:
            logger.error(f"환불 처리 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_order_history(self, user_id: str) -> Dict[str, Any]:
        """
        사용자의 주문 이력을 조회합니다.
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            주문 이력
        """
        try:
            user_orders = [
                order for order in self.orders_db.values()
                if order["user_id"] == user_id
            ]
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "user_id": user_id,
                    "orders": user_orders,
                    "total_orders": len(user_orders),
                    "total_spent": sum(o["total_amount"] for o in user_orders if o["status"] == "paid")
                }
            }
            
        except Exception as e:
            logger.error(f"주문 이력 조회 실패: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# BulkAgent
# ============================================================================

class BulkAgent:
    """
    대량구매 관리 Agent
    
    Responsibilities:
    - Find bulk purchase opportunities
    - Negotiate with suppliers
    - Track inventory and margins
    """
    
    def __init__(self):
        self.name = "BulkAgent"
        self.description = "대량구매 관리 Agent"
        
        self.deals_db = {}
        self.inventory = {}
    
    async def search_bulk_deals(
        self, 
        product_name: str, 
        quantity: int = 10
    ) -> Dict[str, Any]:
        """
        대량구매 딜을 검색합니다.
        
        Args:
            product_name: 제품명
            quantity: 수량
            
        Returns:
            대량구매 딜 목록
        """
        try:
            # 실제로는 MCP Market Server에서 검색
            deals = [
                {
                    "deal_id": f"DEAL-{uuid.uuid4().hex[:8].upper()}",
                    "supplier": "도매업체 A",
                    "product": product_name,
                    "quantity": quantity,
                    "unit_price": 45000,
                    "total_price": quantity * 45000,
                    "discount": 35,
                    "delivery": "3-5일",
                    "warranty": "1년"
                },
                {
                    "deal_id": f"DEAL-{uuid.uuid4().hex[:8].upper()}",
                    "supplier": "도매업체 B",
                    "product": product_name,
                    "quantity": quantity,
                    "unit_price": 48000,
                    "total_price": quantity * 48000,
                    "discount": 30,
                    "delivery": "2-3일",
                    "warranty": "1년"
                }
            ]
            
            logger.info(f"대량구매 딜 검색 완료: {product_name}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "product": product_name,
                    "quantity": quantity,
                    "deals": deals
                }
            }
            
        except Exception as e:
            logger.error(f"대량구매 딜 검색 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def place_bulk_order(
        self, 
        product_name: str, 
        quantity: int,
        supplier: str,
        unit_price: int
    ) -> Dict[str, Any]:
        """
        대량주문을发出합니다.
        
        Args:
            product_name: 제품명
            quantity: 수량
            supplier: 공급업체
            unit_price: 단가
            
        Returns:
            주문 결과
        """
        try:
            deal_id = f"DEAL-{uuid.uuid4().hex[:8].upper()}"
            total_price = quantity * unit_price
            
            deal = {
                "deal_id": deal_id,
                "product": product_name,
                "supplier": supplier,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": total_price,
                "status": "ordered",
                "ordered_at": datetime.now().isoformat(),
                "expected_delivery": (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            self.deals_db[deal_id] = deal
            
            # 재고에 추가
            if product_name not in self.inventory:
                self.inventory[product_name] = 0
            self.inventory[product_name] += quantity
            
            logger.info(f"대량주문 완료: {deal_id}")
            
            return {
                "success": True,
                "agent": self.name,
                "data": deal
            }
            
        except Exception as e:
            logger.error(f"대량주문 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def calculate_margin(
        self, 
        product_name: str, 
        purchase_price: int,
        sale_price: int
    ) -> Dict[str, Any]:
        """
        마진을 계산합니다.
        
        Args:
            product_name: 제품명
            purchase_price: 구매가
            sale_price: 판매가
            
        Returns:
            마진 정보
        """
        try:
            margin = sale_price - purchase_price
            margin_percent = (margin / sale_price) * 100
            
            # 목표 마진율 권장
            recommended_price = int(purchase_price / 0.75)  # 25% 마진
            min_price = int(purchase_price / 0.85)  # 15% 마진
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "product": product_name,
                    "purchase_price": purchase_price,
                    "sale_price": sale_price,
                    "margin": margin,
                    "margin_percent": round(margin_percent, 1),
                    "recommendations": {
                        "aggressive": sale_price,  # 공격적
                        "recommended": recommended_price,  # 권장
                        "minimum": min_price  # 최소
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"마진 계산 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_inventory_status(self) -> Dict[str, Any]:
        """
        재고 현황을 조회합니다.
        
        Returns:
            재고 현황
        """
        try:
            inventory_list = [
                {
                    "product": product,
                    "quantity": quantity,
                    "value": quantity * 50000  # 평균 단가 가정
                }
                for product, quantity in self.inventory.items()
            ]
            
            total_value = sum(i["value"] for i in inventory_list)
            
            return {
                "success": True,
                "agent": self.name,
                "data": {
                    "inventory": inventory_list,
                    "total_items": sum(self.inventory.values()),
                    "total_value": total_value
                }
            }
            
        except Exception as e:
            logger.error(f"재고 조회 실패: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# Sales Manager
# ============================================================================

class SalesManager:
    """
    Sales Agent들을 통합 관리하는 Manager
    
    Responsibilities:
    - Coordinate between different sales agents
    - Handle complex sales workflows
    """
    
    def __init__(self):
        self.ticket_agent = TicketAgent()
        self.starter_pack_agent = StarterPackAgent()
        self.order_agent = OrderAgent()
        self.bulk_agent = BulkAgent()
        
    async def handle_sales_request(
        self, 
        request_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sales 요청을 처리합니다.
        
        Args:
            request_type: 요청 유형
            params: 파라미터
            
        Returns:
            처리 결과
        """
        request_handlers = {
            "search_tickets": self.ticket_agent.search_tickets,
            "purchase_ticket": self.ticket_agent.purchase_ticket,
            "get_starter_pack": self.starter_pack_agent.get_starter_pack,
            "get_recommended_packs": self.starter_pack_agent.get_recommended_packs,
            "purchase_starter_pack": self.starter_pack_agent.purchase_starter_pack,
            "create_order": self.order_agent.create_order,
            "process_payment": self.order_agent.process_payment,
            "refund_order": self.order_agent.refund_order,
            "search_bulk_deals": self.bulk_agent.search_bulk_deals,
            "place_bulk_order": self.bulk_agent.place_bulk_order,
            "calculate_margin": self.bulk_agent.calculate_margin,
            "get_inventory": self.bulk_agent.get_inventory_status
        }
        
        if request_type not in request_handlers:
            return {
                "success": False,
                "error": f"알 수 없는 요청 유형: {request_type}"
            }
        
        handler = request_handlers[request_type]
        return await handler(**params)


# Singleton instances
ticket_agent = TicketAgent()
starter_pack_agent = StarterPackAgent()
order_agent = OrderAgent()
bulk_agent = BulkAgent()
sales_manager = SalesManager()
