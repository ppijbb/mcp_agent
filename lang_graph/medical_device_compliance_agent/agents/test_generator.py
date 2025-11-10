"""
Test Generator Agent

규제 요구사항 기반 테스트 케이스 자동 생성
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper

logger = logging.getLogger(__name__)


class TestGeneratorAgent:
    """테스트 케이스 생성 Agent"""
    
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
            ("system", """You are an expert test case generator for medical device regulatory compliance.

Generate comprehensive test cases based on regulatory requirements. Each test case must include:
- Test ID
- Test name
- Description
- Requirements reference
- Test steps
- Expected results
- Acceptance criteria

Use available tools to search for regulatory requirements and standards."""),
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
    
    def generate(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """테스트 케이스 생성"""
        try:
            input_text = f"""Generate test cases for the following regulatory requirements:

{self._format_requirements(requirements)}

Create comprehensive test cases covering all requirements."""
            
            result = self.agent_executor.invoke({"input": input_text})
            output = result.get("output", "")
            
            test_cases = self._parse_test_cases(output)
            return test_cases
        
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        lines = []
        for framework, reqs in requirements.items():
            lines.append(f"{framework}:")
            if isinstance(reqs, dict) and "requirements" in reqs:
                for req in reqs["requirements"]:
                    lines.append(f"  - {req}")
        return "\n".join(lines)
    
    def _parse_test_cases(self, output: str) -> List[Dict[str, Any]]:
        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        test_cases = []
        # LLM 출력에서 테스트 케이스 추출 로직
        return test_cases

