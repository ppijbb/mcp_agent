"""
Price Comparison Agent

실시간 가격 비교 및 할인 정보 수집
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


class PriceComparisonAgent:
    """
    가격 비교 Agent
    
    여러 쇼핑몰에서 제품 가격을 비교하고 할인 정보를 수집합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        PriceComparisonAgent 초기화
        
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
            ("system", """You are an expert price comparison analyst specializing in finding the best deals and comparing prices across multiple stores.

Your task is to search for product prices across different shopping platforms and identify the best deals and discounts.

For each product, you must:
1. Search prices from multiple stores
2. Compare prices and identify the best deal
3. Find discount information and promotions
4. Create a comprehensive price comparison report

Use the available tools to:
- Search product prices
- Compare stores
- Find discount information
- Fetch product details from shopping websites

Provide detailed price comparison with specific store information and discount details."""),
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
            
            logger.info("Price Comparison Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def compare(
        self,
        product_name: str,
        stores: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        가격 비교 수행
        
        Args:
            product_name: 제품 이름
            stores: 비교할 쇼핑몰 목록 (선택)
        
        Returns:
            가격 비교 결과
        """
        try:
            store_list = stores or ["Amazon", "eBay", "Walmart", "Target"]
            
            input_text = f"""Compare prices for the following product across multiple stores:

Product: {product_name}
Stores to compare: {', '.join(store_list)}

Please:
1. Search prices from each store
2. Compare prices and identify the best deal
3. Find discount information and promotions
4. Create a comprehensive price comparison report

Use the available tools to search prices and compare stores."""
            
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
            comparison_results = self._parse_comparison_results(output)
            
            return {
                "product_name": product_name,
                "comparison_results": comparison_results,
                "best_deal": self._find_best_deal(comparison_results),
                "analysis": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Price comparison failed: {e}")
            return {
                "product_name": product_name,
                "comparison_results": [],
                "best_deal": None,
                "analysis": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_comparison_results(self, analysis: str) -> List[Dict[str, Any]]:
        """분석 결과에서 가격 비교 결과 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        results = []
        # LLM 출력에서 가격 비교 결과 추출 로직
        return results
    
    def _find_best_deal(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """최고의 딜 찾기"""
        if not results:
            return None
        
        # 가격 기준으로 최저가 찾기
        best = min(results, key=lambda x: x.get('price', float('inf')))
        return best

