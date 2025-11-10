"""
Automation Scenario Agent

자동화 시나리오 생성 및 관리
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


class AutomationScenarioAgent:
    """
    자동화 시나리오 Agent
    
    IoT 기기 자동화 시나리오를 생성하고 관리합니다.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None,
        data_dir: str = "home_data"
    ):
        """
        AutomationScenarioAgent 초기화
        
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
            ("system", """You are an expert home automation specialist specializing in creating and managing automation scenarios.

Your task is to create automation scenarios that control multiple IoT devices based on conditions and triggers.

For each request, you must:
1. Understand the automation requirements
2. Create automation scenarios
3. Configure device groups
4. Set up triggers and actions

Use the available tools to:
- Control devices
- Manage device groups
- Configure devices

Provide detailed automation scenarios with specific triggers, conditions, and actions."""),
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
            
            logger.info("Automation Scenario Agent initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def create_scenario(
        self,
        scenario_name: str,
        description: str,
        triggers: List[Dict[str, Any]],
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        자동화 시나리오 생성
        
        Args:
            scenario_name: 시나리오 이름
            description: 시나리오 설명
            triggers: 트리거 목록
            actions: 액션 목록
        
        Returns:
            생성된 시나리오
        """
        try:
            input_text = f"""Create automation scenario:

Name: {scenario_name}
Description: {description}
Triggers: {triggers}
Actions: {actions}

Please:
1. Create the automation scenario
2. Configure device groups if needed
3. Set up triggers and actions
4. Test the scenario

Use the available tools to create and configure the automation scenario."""
            
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
            scenario = self._parse_scenario(output)
            
            return {
                "scenario_name": scenario_name,
                "scenario": scenario,
                "output": output,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Scenario creation failed: {e}")
            return {
                "scenario_name": scenario_name,
                "scenario": {},
                "output": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_scenario(self, output: str) -> Dict[str, Any]:
        """출력에서 시나리오 추출"""
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        return {"status": "created"}

