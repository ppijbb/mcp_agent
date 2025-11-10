"""
Shopping Chain

LangGraph StateGraph 기반 스마트 쇼핑 어시스턴트 워크플로우
"""

import logging
from typing import Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state_management import ShoppingState
from ..agents.preference_analyzer import PreferenceAnalyzerAgent
from ..agents.price_comparison import PriceComparisonAgent
from ..agents.product_recommender import ProductRecommenderAgent
from ..agents.review_analyzer import ReviewAnalyzerAgent
from ..agents.deal_alert import DealAlertAgent
from ..llm.model_manager import ModelManager
from ..llm.fallback_handler import FallbackHandler

logger = logging.getLogger(__name__)


class ShoppingChain:
    """
    스마트 쇼핑 어시스턴트 워크플로우 체인
    
    LangGraph StateGraph를 사용하여 다음 단계를 순차적으로 실행:
    1. Preference Analysis
    2. Price Comparison
    3. Product Recommendation
    4. Review Analysis
    5. Deal Alerts
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        output_dir: str = "smart_shopping_reports",
        data_dir: str = "shopping_data"
    ):
        """
        ShoppingChain 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            output_dir: 출력 디렉토리
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Agents 초기화
        self.preference_analyzer = PreferenceAnalyzerAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.price_comparison = PriceComparisonAgent(
            model_manager, fallback_handler
        )
        self.product_recommender = ProductRecommenderAgent(
            model_manager, fallback_handler, data_dir=data_dir
        )
        self.review_analyzer = ReviewAnalyzerAgent(
            model_manager, fallback_handler
        )
        self.deal_alert = DealAlertAgent(
            model_manager, fallback_handler
        )
        
        # LangGraph 워크플로우 설정
        self.memory = MemorySaver()
        self._setup_workflow()
    
    def _setup_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(ShoppingState)
        
        # 노드 추가
        workflow.add_node("preference_analysis", self._preference_analysis_node)
        workflow.add_node("price_comparison", self._price_comparison_node)
        workflow.add_node("product_recommendation", self._product_recommendation_node)
        workflow.add_node("review_analysis", self._review_analysis_node)
        workflow.add_node("deal_alerts", self._deal_alerts_node)
        
        # 엣지 설정
        workflow.set_entry_point("preference_analysis")
        workflow.add_edge("preference_analysis", "price_comparison")
        workflow.add_edge("price_comparison", "product_recommendation")
        workflow.add_edge("product_recommendation", "review_analysis")
        workflow.add_edge("review_analysis", "deal_alerts")
        workflow.add_edge("deal_alerts", END)
        
        # 컴파일
        self.app = workflow.compile(checkpointer=self.memory)
        logger.info("Shopping Chain workflow initialized")
    
    def _preference_analysis_node(self, state: ShoppingState) -> ShoppingState:
        """선호도 분석 노드"""
        try:
            logger.info("Starting preference analysis")
            result = self.preference_analyzer.analyze(
                state["user_id"],
                state.get("purchase_history")
            )
            
            state["preferences"] = result.get("preferences", {})
            state["purchase_history"] = result.get("purchase_history", [])
            state["workflow_stage"] = "preference_analysis"
            
            if not result.get("success"):
                state["errors"].append(f"Preference analysis failed: {result.get('analysis', 'Unknown error')}")
            
            logger.info(f"Preference analysis completed for user: {state['user_id']}")
        
        except Exception as e:
            logger.error(f"Preference analysis error: {e}")
            state["errors"].append(f"Preference analysis error: {str(e)}")
        
        return state
    
    def _price_comparison_node(self, state: ShoppingState) -> ShoppingState:
        """가격 비교 노드"""
        try:
            logger.info("Starting price comparison")
            
            # 쿼리에서 제품 이름 추출 (간단한 추출)
            product_name = state.get("query", "").split()[0] if state.get("query") else "product"
            
            result = self.price_comparison.compare(product_name)
            
            state["price_comparison_results"] = result.get("comparison_results", [])
            state["workflow_stage"] = "price_comparison"
            
            logger.info(f"Price comparison completed: {len(state['price_comparison_results'])} results")
        
        except Exception as e:
            logger.error(f"Price comparison error: {e}")
            state["errors"].append(f"Price comparison error: {str(e)}")
        
        return state
    
    def _product_recommendation_node(self, state: ShoppingState) -> ShoppingState:
        """제품 추천 노드"""
        try:
            logger.info("Starting product recommendation")
            recommendations = self.product_recommender.recommend(
                state["user_id"],
                state["query"],
                state.get("preferences")
            )
            
            state["recommendations"] = recommendations
            state["workflow_stage"] = "product_recommendation"
            
            logger.info(f"Product recommendation completed: {len(recommendations)} recommendations")
        
        except Exception as e:
            logger.error(f"Product recommendation error: {e}")
            state["errors"].append(f"Product recommendation error: {str(e)}")
        
        return state
    
    def _review_analysis_node(self, state: ShoppingState) -> ShoppingState:
        """리뷰 분석 노드"""
        try:
            logger.info("Starting review analysis")
            
            # 추천 제품 중 첫 번째 제품의 리뷰 분석
            if state.get("recommendations"):
                product_name = state["recommendations"][0].get("name", state["query"])
            else:
                product_name = state["query"]
            
            result = self.review_analyzer.analyze(product_name)
            
            state["review_analysis"] = result.get("analysis", {})
            state["workflow_stage"] = "review_analysis"
            
            logger.info(f"Review analysis completed for: {product_name}")
        
        except Exception as e:
            logger.error(f"Review analysis error: {e}")
            state["errors"].append(f"Review analysis error: {str(e)}")
        
        return state
    
    def _deal_alerts_node(self, state: ShoppingState) -> ShoppingState:
        """할인 알림 노드"""
        try:
            logger.info("Starting deal alerts")
            
            # 추천 제품들에 대한 할인 정보 확인
            alerts = []
            for recommendation in state.get("recommendations", []):
                product_name = recommendation.get("name", "")
                if product_name:
                    deal_alerts = self.deal_alert.check_deals(product_name)
                    alerts.extend(deal_alerts)
            
            state["deal_alerts"] = alerts
            state["workflow_stage"] = "deal_alerts"
            
            # 최종 추천 생성
            state["final_recommendations"] = self._generate_final_recommendations(state)
            
            logger.info(f"Deal alerts completed: {len(alerts)} alerts")
        
        except Exception as e:
            logger.error(f"Deal alerts error: {e}")
            state["errors"].append(f"Deal alerts error: {str(e)}")
        
        return state
    
    def _generate_final_recommendations(self, state: ShoppingState) -> List[Dict[str, Any]]:
        """최종 추천 생성"""
        final_recommendations = []
        
        for recommendation in state.get("recommendations", []):
            final_rec = {
                "product": recommendation,
                "price_comparison": None,
                "review_summary": state.get("review_analysis", {}),
                "deal_alerts": []
            }
            
            # 가격 비교 결과 매칭
            for price_result in state.get("price_comparison_results", []):
                if price_result.get("product_name") == recommendation.get("name"):
                    final_rec["price_comparison"] = price_result
                    break
            
            # 할인 알림 매칭
            for alert in state.get("deal_alerts", []):
                if alert.get("product_name") == recommendation.get("name"):
                    final_rec["deal_alerts"].append(alert)
            
            final_recommendations.append(final_rec)
        
        return final_recommendations
    
    def run(
        self,
        user_id: str,
        query: str,
        purchase_history: Optional[List[Dict[str, Any]]] = None
    ) -> ShoppingState:
        """
        워크플로우 실행
        
        Args:
            user_id: 사용자 ID
            query: 검색 쿼리
            purchase_history: 구매 이력 (선택)
        
        Returns:
            최종 상태
        """
        # 초기 상태 생성
        initial_state: ShoppingState = {
            "user_id": user_id,
            "query": query,
            "preferences": {},
            "purchase_history": purchase_history or [],
            "price_comparison_results": [],
            "recommendations": [],
            "review_analysis": {},
            "deal_alerts": [],
            "final_recommendations": [],
            "timestamp": datetime.now().isoformat(),
            "workflow_stage": "initialized",
            "errors": [],
            "warnings": []
        }
        
        try:
            # 워크플로우 실행
            config = {"configurable": {"thread_id": f"shopping_workflow_{user_id}"}}
            final_state = self.app.invoke(initial_state, config)
            
            logger.info("Shopping workflow completed successfully")
            return final_state
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state["errors"].append(f"Workflow execution error: {str(e)}")
            return initial_state

