"""
Framework Analyzer Agent

FDA 510(k), CE 마킹, ISO 13485 등 규제 프레임워크 분석
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.regulatory_tools import RegulatoryTools

logger = logging.getLogger(__name__)


class FrameworkAnalyzerAgent:
    """
    규제 프레임워크 분석 Agent
    
    의료기기 정보를 기반으로 적용 가능한 규제 프레임워크를 분석하고
    각 프레임워크의 요구사항을 식별합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        """
        FrameworkAnalyzerAgent 초기화
        
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
        self.regulatory_tools = RegulatoryTools()
        self.tools = self.mcp_tools.get_tools() + self.regulatory_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert regulatory framework analyst specializing in medical device compliance.

Your task is to analyze medical device information and identify applicable regulatory frameworks (FDA 510(k), CE Marking, ISO 13485, etc.).

For each regulatory framework, you must:
1. Determine if it applies to the device
2. Identify specific requirements
3. Search for relevant regulatory documents and standards
4. Create a comprehensive requirements matrix

Use the available tools to:
- Search FDA 510(k) database
- Search CE marking requirements
- Search ISO 13485 standards
- Fetch regulatory documents from official sources

Provide detailed, actionable analysis based on real regulatory requirements."""),
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
            
            logger.info("Framework Analyzer Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def analyze(
        self,
        device_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        규제 프레임워크 분석 수행
        
        Args:
            device_info: 의료기기 정보
        
        Returns:
            분석 결과 (규제 프레임워크 목록 및 요구사항)
        """
        try:
            input_text = f"""Analyze the following medical device and identify applicable regulatory frameworks:

Device Information:
{self._format_device_info(device_info)}

Please:
1. Identify all applicable regulatory frameworks (FDA 510(k), CE Marking, ISO 13485, etc.)
2. For each framework, search for specific requirements
3. Create a comprehensive requirements matrix
4. Identify any gaps or missing information

Use the available tools to search regulatory databases and fetch official documents."""
            
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
            frameworks = self._parse_frameworks(output)
            requirements = self._parse_requirements(output)
            
            return {
                "frameworks": frameworks,
                "requirements": requirements,
                "analysis": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Framework analysis failed: {e}")
            return {
                "frameworks": [],
                "requirements": {},
                "analysis": f"Error: {str(e)}",
                "success": False
            }
    
    def _format_device_info(self, device_info: Dict[str, Any]) -> str:
        """의료기기 정보 포맷팅"""
        lines = []
        for key, value in device_info.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _parse_frameworks(self, analysis: str) -> List[str]:
        """분석 결과에서 규제 프레임워크 추출"""
        frameworks = []
        framework_keywords = {
            "FDA 510(k)": ["FDA", "510(k)", "510k", "premarket"],
            "CE Marking": ["CE", "CE Marking", "MDR", "IVDR"],
            "ISO 13485": ["ISO 13485", "ISO13485", "quality management"]
        }
        
        analysis_lower = analysis.lower()
        for framework, keywords in framework_keywords.items():
            if any(keyword.lower() in analysis_lower for keyword in keywords):
                frameworks.append(framework)
        
        return frameworks
    
    def _parse_requirements(self, analysis: str) -> Dict[str, Any]:
        """분석 결과에서 요구사항 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        requirements = {}
        
        # FDA 510(k) 요구사항
        if "FDA" in analysis or "510(k)" in analysis:
            requirements["FDA 510(k)"] = {
                "applicable": True,
                "requirements": ["Premarket notification", "Device classification", "Substantial equivalence"]
            }
        
        # CE Marking 요구사항
        if "CE" in analysis or "MDR" in analysis:
            requirements["CE Marking"] = {
                "applicable": True,
                "requirements": ["Technical documentation", "Clinical evaluation", "Quality management system"]
            }
        
        # ISO 13485 요구사항
        if "ISO 13485" in analysis:
            requirements["ISO 13485"] = {
                "applicable": True,
                "requirements": ["Quality management system", "Risk management", "Design controls"]
            }
        
        return requirements

