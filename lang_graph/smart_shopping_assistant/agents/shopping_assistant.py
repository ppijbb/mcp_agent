"""
Shopping Assistant Agent

종합 쇼핑 어시스턴트 (모든 기능 통합)
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


class ShoppingAssistantAgent:
    """
    종합 쇼핑 어시스턴트 Agent
    
    모든 쇼핑 관련 기능을 통합하여 종합적인 쇼핑 어시스턴트를 제공합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "shopping_data"
    ):
        """
        ShoppingAssistantAgent 초기화
        
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
            ("system", """You are a comprehensive shopping assistant that helps users with all aspects of shopping.

Your capabilities include:
1. Analyzing user preferences and purchase history
2. Comparing prices across multiple stores
3. Recommending personalized products
4. Analyzing product reviews
5. Alerting users about deals and discounts

For each user request, you must:
- Understand the user's intent
- Use appropriate tools to gather information
- Provide comprehensive, actionable recommendations
- Consider user preferences, budget, and needs

Use all available tools to provide the best shopping assistance possible."""),
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
                max_iterations=15
            )
            
            logger.info("Shopping Assistant Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def assist(
        self,
        user_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        쇼핑 어시스턴트 실행
        
        Args:
            user_id: 사용자 ID
            query: 사용자 쿼리
        
        Returns:
            어시스턴트 응답
        """
        try:
            input_text = f"""Help user {user_id} with their shopping request:

Query: {query}

Please provide comprehensive shopping assistance including:
- Product recommendations
- Price comparisons
- Review analysis
- Deal alerts

Use all available tools to gather information and provide the best recommendations."""
            
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
            
            return {
                "user_id": user_id,
                "query": query,
                "response": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Shopping assistance failed: {e}")
            return {
                "user_id": user_id,
                "query": query,
                "response": f"Error: {str(e)}",
                "success": False
            }

