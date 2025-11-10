"""
Review Analyzer Agent

제품 리뷰 분석 및 요약
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

logger = logging.getLogger(__name__)


class ReviewAnalyzerAgent:
    """
    리뷰 분석 Agent
    
    제품 리뷰를 분석하고 요약하여 장단점을 파악합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        ReviewAnalyzerAgent 초기화
        
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
            ("system", """You are an expert product review analyst specializing in analyzing customer reviews and extracting meaningful insights.

Your task is to analyze product reviews and provide comprehensive summaries of pros, cons, and overall sentiment.

For each product, you must:
1. Collect reviews from multiple sources
2. Analyze sentiment and key themes
3. Identify common pros and cons
4. Summarize overall customer satisfaction
5. Provide actionable insights

Use the available tools to:
- Search for product reviews
- Fetch review content from websites
- Analyze review sentiment and themes

Provide detailed review analysis with specific pros, cons, and overall rating."""),
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
            
            logger.info("Review Analyzer Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def analyze(
        self,
        product_name: str,
        max_reviews: int = 50
    ) -> Dict[str, Any]:
        """
        리뷰 분석 수행
        
        Args:
            product_name: 제품 이름
            max_reviews: 최대 리뷰 수
        
        Returns:
            리뷰 분석 결과
        """
        try:
            input_text = f"""Analyze reviews for the following product:

Product: {product_name}
Max Reviews: {max_reviews}

Please:
1. Search for product reviews from multiple sources
2. Analyze sentiment and key themes
3. Identify common pros and cons
4. Summarize overall customer satisfaction
5. Provide actionable insights

Use the available tools to search for and fetch reviews."""
            
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
            analysis = self._parse_review_analysis(output)
            
            return {
                "product_name": product_name,
                "analysis": analysis,
                "summary": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Review analysis failed: {e}")
            return {
                "product_name": product_name,
                "analysis": {},
                "summary": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_review_analysis(self, analysis: str) -> Dict[str, Any]:
        """분석 결과에서 리뷰 분석 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return {
            "pros": [],
            "cons": [],
            "overall_rating": 0.0,
            "sentiment": "neutral"
        }

