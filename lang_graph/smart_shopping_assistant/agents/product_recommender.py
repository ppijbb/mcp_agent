"""
Product Recommender Agent

개인화된 제품 추천
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.shopping_tools import ShoppingTools
from ..tools.preference_tools import PreferenceTools

logger = logging.getLogger(__name__)


class ProductRecommenderAgent:
    """
    제품 추천 Agent
    
    사용자의 선호도와 구매 이력을 기반으로 개인화된 제품을 추천합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "shopping_data"
    ):
        """
        ProductRecommenderAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
            data_dir: 데이터 저장 디렉토리
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        # 도구 초기화
        self.mcp_tools = MCPToolsWrapper()
        self.shopping_tools = ShoppingTools()
        self.preference_tools = PreferenceTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.shopping_tools.get_tools() + self.preference_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert product recommendation specialist specializing in personalized shopping recommendations.

Your task is to recommend products based on user preferences, purchase history, and current needs.

For each recommendation request, you must:
1. Analyze user preferences and purchase history
2. Search for relevant products that match preferences
3. Consider price, quality, and user budget
4. Provide personalized recommendations with detailed reasoning

Use the available tools to:
- Analyze user preferences
- Search for products
- Compare prices
- Find product information

Provide detailed, personalized product recommendations with specific reasoning."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        try:
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )
            
            logger.info("Product Recommender Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def recommend(
        self,
        user_id: str,
        query: str,
        preferences: Optional[Dict[str, Any]] = None,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        제품 추천 수행
        
        Args:
            user_id: 사용자 ID
            query: 검색 쿼리
            preferences: 사용자 선호도 (선택)
            max_recommendations: 최대 추천 수
        
        Returns:
            추천 제품 목록
        """
        try:
            input_text = f"""Recommend products for user {user_id}:

Query: {query}
User Preferences: {preferences or 'Analyze from purchase history'}
Max Recommendations: {max_recommendations}

Please:
1. Analyze user preferences and purchase history
2. Search for products matching the query and preferences
3. Consider price, quality, and user budget
4. Provide {max_recommendations} personalized recommendations with detailed reasoning

Use the available tools to analyze preferences and search for products."""
            
            try:
                result = self.agent_executor.invoke({"input": input_text})
                output = result.get("output", "")
            except Exception as e:
                logger.warning(f"Agent execution failed, trying fallback: {e}")
                messages = [HumanMessage(content=input_text)]
                response = self.fallback_handler.invoke_with_fallback(
                    messages,
                    self.preferred_provider
                )
                output = response.content if hasattr(response, 'content') else str(response)
            
            # 결과 파싱
            recommendations = self._parse_recommendations(output)
            
            return recommendations[:max_recommendations]
        
        except Exception as e:
            logger.error(f"Product recommendation failed: {e}")
            return []
    
    def _parse_recommendations(self, analysis: str) -> List[Dict[str, Any]]:
        """분석 결과에서 추천 제품 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        recommendations = []
        # LLM 출력에서 추천 제품 추출 로직
        return recommendations

