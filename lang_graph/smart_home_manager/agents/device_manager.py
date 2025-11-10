"""
Device Manager Agent

IoT 기기 통합 관리
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
from ..tools.iot_tools import IoTTools

logger = logging.getLogger(__name__)


class DeviceManagerAgent:
    """
    IoT 기기 통합 관리 Agent
    
    IoT 기기를 통합 관리하고 제어합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "home_data"
    ):
        """
        DeviceManagerAgent 초기화
        
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
            ("system", """You are an expert IoT device manager specializing in managing and controlling smart home devices.

Your task is to manage IoT devices, control them, monitor their status, and create device groups.

For each request, you must:
1. Control devices (turn on/off, set temperature, adjust brightness, etc.)
2. Check device status
3. Configure device settings
4. Manage device groups

Use the available tools to:
- Control IoT devices
- Get device status
- Configure devices
- Manage device groups

Provide detailed device management with specific control commands and status information."""),
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
            
            logger.info("Device Manager Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def manage_devices(
        self,
        action: str,
        device_id: Optional[str] = None,
        value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        기기 관리 수행
        
        Args:
            action: 액션 (control, status, config, group)
            device_id: 기기 ID (선택)
            value: 값 (선택)
        
        Returns:
            관리 결과
        """
        try:
            input_text = f"""Manage IoT devices:

Action: {action}
Device ID: {device_id or 'all devices'}
Value: {value or 'N/A'}

Please:
1. Control devices if action is 'control'
2. Check device status if action is 'status'
3. Configure devices if action is 'config'
4. Manage device groups if action is 'group'

Use the available tools to manage devices."""
            
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
            device_status = self._parse_device_status(output)
            
            return {
                "action": action,
                "device_id": device_id,
                "status": device_status,
                "output": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Device management failed: {e}")
            return {
                "action": action,
                "device_id": device_id,
                "status": {},
                "output": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_device_status(self, output: str) -> Dict[str, Any]:
        """출력에서 기기 상태 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return {"status": "managed"}

