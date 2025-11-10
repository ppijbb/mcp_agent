"""
스마트 쇼핑 어시스턴트 Agent 메인 진입점
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from .llm.model_manager import ModelManager, ModelProvider
from .llm.fallback_handler import FallbackHandler
from .chains.shopping_chain import ShoppingChain
from .config.shopping_config import ShoppingConfig
from .utils.validators import validate_user_query

logger = logging.getLogger(__name__)


class SmartShoppingAssistant:
    """
    스마트 쇼핑 어시스턴트 Agent
    
    개인화된 쇼핑 추천, 가격 비교, 할인 정보 수집, 제품 리뷰 분석을
    제공하는 Multi-Agent 시스템
    """
    
    def __init__(
        self,
        config: Optional[ShoppingConfig] = None,
        output_dir: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        SmartShoppingAssistant 초기화
        
        Args:
            config: 설정 객체 (선택, 없으면 환경 변수에서 로드)
            output_dir: 출력 디렉토리 (선택)
            data_dir: 데이터 저장 디렉토리 (선택)
        """
        self.config = config or ShoppingConfig.from_env()
        self.output_dir = output_dir or self.config.output_dir
        self.data_dir = data_dir or self.config.data_dir
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # LLM Manager 및 Fallback Handler 초기화
        self.model_manager = ModelManager(
            budget_limit=self.config.llm.budget_limit
        )
        self.fallback_handler = FallbackHandler(self.model_manager)
        
        # Shopping Chain 초기화
        self.shopping_chain = ShoppingChain(
            model_manager=self.model_manager,
            fallback_handler=self.fallback_handler,
            output_dir=self.output_dir,
            data_dir=self.data_dir
        )
        
        logger.info("Smart Shopping Assistant Agent initialized")
    
    def search(
        self,
        user_id: str,
        query: str,
        purchase_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        쇼핑 검색 및 추천 실행
        
        Args:
            user_id: 사용자 ID
            query: 검색 쿼리
            purchase_history: 구매 이력 (선택)
        
        Returns:
            검색 결과
        """
        try:
            # 입력 검증
            if not validate_user_query(query):
                return {
                    "success": False,
                    "error": "Invalid query",
                    "user_id": user_id,
                    "query": query
                }
            
            logger.info(f"Starting shopping search for user: {user_id}, query: {query}")
            
            # 워크플로우 실행
            final_state = self.shopping_chain.run(user_id, query, purchase_history)
            
            # 결과 반환
            return {
                "success": len(final_state.get("errors", [])) == 0,
                "user_id": user_id,
                "query": query,
                "preferences": final_state.get("preferences", {}),
                "price_comparison_results": final_state.get("price_comparison_results", []),
                "recommendations": final_state.get("recommendations", []),
                "review_analysis": final_state.get("review_analysis", {}),
                "deal_alerts": final_state.get("deal_alerts", []),
                "final_recommendations": final_state.get("final_recommendations", []),
                "errors": final_state.get("errors", []),
                "warnings": final_state.get("warnings", []),
                "timestamp": final_state.get("timestamp")
            }
        
        except Exception as e:
            logger.error(f"Shopping search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "query": query
            }


async def main():
    """메인 실행 함수"""
    # 예제 사용자 정보
    user_id = "user_001"
    query = "wireless headphones"
    purchase_history = [
        {"name": "Sony WH-1000XM4", "category": "Electronics", "price": 350, "brand": "Sony"},
        {"name": "Apple AirPods Pro", "category": "Electronics", "price": 249, "brand": "Apple"}
    ]
    
    # Agent 초기화 및 실행
    agent = SmartShoppingAssistant()
    result = agent.search(user_id, query, purchase_history)
    
    print(f"Shopping search completed: {result['success']}")
    if result.get("recommendations"):
        print(f"Found {len(result['recommendations'])} recommendations")
    if result.get("deal_alerts"):
        print(f"Found {len(result['deal_alerts'])} deal alerts")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

