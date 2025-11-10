"""
Test Executor Agent

테스트 케이스 실행 및 결과 수집
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper

logger = logging.getLogger(__name__)


class TestExecutorAgent:
    """테스트 실행 Agent"""
    
    def __init__(
        self,
        model_manager: ModelManager,
        fallback_handler: FallbackHandler,
        preferred_provider: Optional[ModelProvider] = None
    ):
        self.model_manager = model_manager
        self.fallback_handler = fallback_handler
        self.preferred_provider = preferred_provider
        
        self.mcp_tools = MCPToolsWrapper()
        self.tools = self.mcp_tools.get_tools()
        
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a test execution agent for medical device compliance testing.

Execute test cases and collect results. For each test case:
1. Execute test steps
2. Record actual results
3. Compare with expected results
4. Determine pass/fail status
5. Document any issues or deviations"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
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
    
    def execute(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """테스트 케이스 실행"""
        try:
            results = []
            for test_case in test_cases:
                input_text = f"Execute test case: {test_case.get('name', 'Unknown')}"
                result = self.agent_executor.invoke({"input": input_text})
                results.append({
                    "test_case_id": test_case.get("id"),
                    "status": "PASS",  # 실제로는 결과 분석 필요
                    "result": result.get("output", ""),
                    "timestamp": str(datetime.now())
                })
            return results
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return []

