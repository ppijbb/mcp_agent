"""
Maintenance Alert Agent

유지보수 알림
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.iot_tools import IoTTools

logger = logging.getLogger(__name__)


class MaintenanceAlertAgent:
    """
    유지보수 알림 Agent
    
    가전 제품의 유지보수 일정을 관리하고 알림을 제공합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "home_data"
    ):
        """
        MaintenanceAlertAgent 초기화
        
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
        self.iot_tools = IoTTools(data_dir=data_dir)
        self.tools = self.mcp_tools.get_tools() + self.iot_tools.get_tools()
        
        # LLM 초기화
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        # Agent 초기화
        self._initialize_agent()
    
    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert maintenance scheduling specialist specializing in home appliance maintenance.

Your task is to track maintenance schedules for home appliances and provide timely alerts.

For each request, you must:
1. Check device maintenance schedules
2. Identify devices needing maintenance
3. Create maintenance alerts
4. Provide maintenance recommendations

Use the available tools to:
- Check device status
- Get device information
- Create maintenance schedules

Provide detailed maintenance alerts with specific dates and recommendations."""),
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
            
            logger.info("Maintenance Alert Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def check_maintenance(
        self,
        device_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        유지보수 확인
        
        Args:
            device_id: 기기 ID (없으면 모든 기기)
        
        Returns:
            유지보수 알림 목록
        """
        try:
            input_text = f"""Check maintenance schedules:

Device ID: {device_id or 'all devices'}

Please:
1. Check device maintenance schedules
2. Identify devices needing maintenance
3. Create maintenance alerts
4. Provide maintenance recommendations

Use the available tools to check device status and maintenance schedules."""
            
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
            alerts = self._parse_maintenance_alerts(output)
            
            return alerts
        
        except Exception as e:
            logger.error(f"Maintenance check failed: {e}")
            return []
    
    def _parse_maintenance_alerts(self, output: str) -> List[Dict[str, Any]]:
        """출력에서 유지보수 알림 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return []

