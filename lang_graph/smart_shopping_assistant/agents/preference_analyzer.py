"""
Preference Analyzer Agent

구매 이력 및 선호도 분석
"""

import logging
import json
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.preference_tools import PreferenceTools

logger = logging.getLogger(__name__)


class PreferenceAnalyzerAgent:
    """
    선호도 분석 Agent
    
    구매 이력 및 선호도를 분석하여 사용자의 쇼핑 패턴을 파악합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "shopping_data"
    ):
        """
        PreferenceAnalyzerAgent 초기화
        
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
        self.preference_tools = PreferenceTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.preference_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert shopping preference analyst specializing in understanding customer shopping patterns and preferences.

Your task is to analyze purchase history and extract meaningful preferences that can be used for personalized product recommendations.

For each user, you must:
1. Analyze purchase history to identify patterns
2. Extract preferences (categories, brands, price ranges, etc.)
3. Identify shopping behavior patterns
4. Create a comprehensive preference profile

Use the available tools to:
- Analyze purchase history from files
- Extract preferences from purchase data
- Identify purchase patterns and trends

Provide detailed, actionable preference analysis based on real purchase data."""),
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
            
            logger.info("Preference Analyzer Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def analyze(
        self,
        user_id: str,
        purchase_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        선호도 분석 수행
        
        Args:
            user_id: 사용자 ID
            purchase_history: 구매 이력 (선택, 없으면 파일에서 로드)
        
        Returns:
            분석 결과 (선호도 프로필, 구매 패턴)
        """
        try:
            input_text = f"""Analyze shopping preferences for user {user_id}:

User ID: {user_id}
Purchase History: {json.dumps(purchase_history, ensure_ascii=False) if purchase_history else 'Load from file'}

Please:
1. Analyze purchase history to identify shopping patterns
2. Extract preferences (categories, brands, price ranges, etc.)
3. Identify shopping behavior patterns
4. Create a comprehensive preference profile

Use the available tools to analyze purchase history and extract preferences."""
            
            # Fallback 메커니즘을 사용하여 실행
            messages = [HumanMessage(content=input_text)]
            
            try:
                result = self.agent_executor.invoke({"input": input_text})
                output = result.get("output", "")
            except Exception as e:
                logger.warning(f"Agent execution failed, trying fallback: {e}")
                # Fallback으로 직접 LLM 호출
                response = self.fallback_handler.invoke_with_fallback(
                    messages,
                    self.preferred_provider
                )
                output = response.content if hasattr(response, 'content') else str(response)
            
            # 결과 파싱
            preferences = self._parse_preferences(output)
            patterns = self._parse_patterns(output)
            
            return {
                "user_id": user_id,
                "preferences": preferences,
                "patterns": patterns,
                "analysis": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Preference analysis failed: {e}")
            return {
                "user_id": user_id,
                "preferences": {},
                "patterns": {},
                "analysis": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_preferences(self, analysis: str) -> Dict[str, Any]:
        """분석 결과에서 선호도 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        preferences = {
            "preferred_categories": [],
            "preferred_brands": [],
            "price_range": {"min": 0, "max": 0},
            "shopping_frequency": "unknown"
        }
        
        # LLM 출력에서 선호도 추출 로직
        # 실제로는 JSON 파싱 또는 구조화된 출력 사용
        return preferences
    
    def _parse_patterns(self, analysis: str) -> Dict[str, Any]:
        """분석 결과에서 패턴 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        patterns = {
            "purchase_frequency": "unknown",
            "seasonal_patterns": {},
            "category_trends": {}
        }
        
        return patterns

