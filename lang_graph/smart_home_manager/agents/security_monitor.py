"""
Security Monitor Agent

보안 모니터링 및 알림
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.security_tools import SecurityTools

logger = logging.getLogger(__name__)


class SecurityMonitorAgent:
    """
    보안 모니터링 Agent
    
    홈 보안 상태를 모니터링하고 위협을 감지합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "home_data"
    ):
        """
        SecurityMonitorAgent 초기화
        
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
        self.security_tools = SecurityTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.security_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert security monitoring specialist specializing in home security and threat detection.

Your task is to monitor home security status, detect threats, and create security alerts.

For each request, you must:
1. Check security status of devices and network
2. Detect potential threats
3. Create security alerts when needed
4. Provide security recommendations

Use the available tools to:
- Get security status
- Detect threats
- Create security alerts
- Generate security reports

Provide detailed security monitoring with specific threat information and recommendations."""),
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
            
            logger.info("Security Monitor Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def monitor(
        self,
        check_type: str = "all"
    ) -> Dict[str, Any]:
        """
        보안 모니터링 수행
        
        Args:
            check_type: 검사 유형 (all, network, devices, access)
        
        Returns:
            모니터링 결과
        """
        try:
            input_text = f"""Monitor home security:

Check Type: {check_type}

Please:
1. Check security status
2. Detect threats
3. Create alerts if needed
4. Provide security recommendations

Use the available tools to monitor security."""
            
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
            security_status = self._parse_security_status(output)
            
            return {
                "check_type": check_type,
                "security_status": security_status,
                "output": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Security monitoring failed: {e}")
            return {
                "check_type": check_type,
                "security_status": {},
                "output": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_security_status(self, output: str) -> Dict[str, Any]:
        """출력에서 보안 상태 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return {"status": "secure"}

