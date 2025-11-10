"""
Energy Optimizer Agent

에너지 사용 최적화
"""

import logging
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.energy_tools import EnergyTools

logger = logging.getLogger(__name__)


class EnergyOptimizerAgent:
    """
    에너지 최적화 Agent
    
    에너지 사용 패턴을 분석하고 최적화 방안을 제안합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "home_data"
    ):
        """
        EnergyOptimizerAgent 초기화
        
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
        self.energy_tools = EnergyTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.energy_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert energy optimization specialist specializing in analyzing energy usage and providing optimization recommendations.

Your task is to analyze energy usage patterns and suggest ways to reduce energy consumption and costs.

For each request, you must:
1. Analyze energy usage patterns
2. Identify peak usage times
3. Suggest optimization strategies
4. Provide cost-saving recommendations

Use the available tools to:
- Get energy usage data
- Analyze energy patterns
- Suggest optimizations
- Generate energy reports

Provide detailed energy optimization recommendations with specific actions and potential savings."""),
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
            
            logger.info("Energy Optimizer Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def optimize(
        self,
        period: str = "week",
        target_reduction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        에너지 최적화 수행
        
        Args:
            period: 분석 기간
            target_reduction: 목표 절감률 (%) (선택)
        
        Returns:
            최적화 결과
        """
        try:
            input_text = f"""Optimize energy usage:

Period: {period}
Target Reduction: {target_reduction or 'Not specified'}%

Please:
1. Analyze energy usage patterns
2. Identify optimization opportunities
3. Suggest specific actions to reduce energy consumption
4. Calculate potential savings

Use the available tools to analyze energy usage and suggest optimizations."""
            
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
            optimization = self._parse_optimization(output)
            
            return {
                "period": period,
                "target_reduction": target_reduction,
                "optimization": optimization,
                "output": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Energy optimization failed: {e}")
            return {
                "period": period,
                "target_reduction": target_reduction,
                "optimization": {},
                "output": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_optimization(self, output: str) -> Dict[str, Any]:
        """출력에서 최적화 정보 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return {"suggestions": []}

