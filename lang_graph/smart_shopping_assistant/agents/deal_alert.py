"""
Deal Alert Agent

최적 구매 시점 알림 및 가격 모니터링
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.shopping_tools import ShoppingTools

logger = logging.getLogger(__name__)


class DealAlertAgent:
    """
    할인 알림 Agent
    
    제품 가격을 모니터링하고 최적 구매 시점을 알림합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        DealAlertAgent 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
            fallback_handler: FallbackHandler 인스턴스
            preferred_provider: 선호하는 Provider
        """
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        # 도구 초기화
        self.mcp_tools = MCPToolsWrapper()
        self.shopping_tools = ShoppingTools()
        self.tools = self.mcp_tools.get_tools() + self.shopping_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert deal alert specialist specializing in monitoring product prices and identifying the best time to buy.

Your task is to monitor product prices and alert users when there are good deals or price drops.

For each product, you must:
1. Check current prices and compare with historical prices
2. Identify discount opportunities
3. Monitor price trends
4. Alert users when prices drop significantly
5. Recommend the best time to buy

Use the available tools to:
- Search for current prices
- Find discount information
- Monitor price changes

Provide timely deal alerts with specific price information and recommendations."""),
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
            
            logger.info("Deal Alert Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def check_deals(
        self,
        product_name: str,
        target_price: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        할인 정보 확인
        
        Args:
            product_name: 제품 이름
            target_price: 목표 가격 (선택)
        
        Returns:
            할인 알림 목록
        """
        try:
            input_text = f"""Check for deals on the following product:

Product: {product_name}
Target Price: {target_price or 'Not specified'}

Please:
1. Check current prices and discounts
2. Compare with target price if specified
3. Identify any good deals
4. Alert if price drops significantly

Use the available tools to search for prices and discounts."""
            
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
            alerts = self._parse_deal_alerts(output)
            
            return alerts
        
        except Exception as e:
            logger.error(f"Deal check failed: {e}")
            return []
    
    def _parse_deal_alerts(self, analysis: str) -> List[Dict[str, Any]]:
        """분석 결과에서 할인 알림 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        alerts = []
        # LLM 출력에서 할인 알림 추출 로직
        return alerts

