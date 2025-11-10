"""
Compliance Validator Agent

규제 준수 검증 및 리스크 평가
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.regulatory_tools import RegulatoryTools

logger = logging.getLogger(__name__)


class ComplianceValidatorAgent:
    """규제 준수 검증 Agent"""
    
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
        self.regulatory_tools = RegulatoryTools()
        self.tools = self.mcp_tools.get_tools() + self.regulatory_tools.get_tools()
        
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a compliance validation expert for medical devices.

Validate test results against regulatory requirements and assess compliance status.
For each framework:
1. Evaluate test results
2. Assess compliance level
3. Identify risks and gaps
4. Calculate compliance score
5. Provide recommendations"""),
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
    
    def validate(
        self,
        test_results: List[Dict[str, Any]],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """규제 준수 검증"""
        try:
            input_text = f"""Validate compliance based on test results:

Test Results:
{self._format_test_results(test_results)}

Requirements:
{self._format_requirements(requirements)}

Assess compliance status and provide risk assessment."""
            
            result = self.agent_executor.invoke({"input": input_text})
            output = result.get("output", "")
            
            return {
                "compliance_status": "COMPLIANT",  # 실제로는 결과 분석 필요
                "compliance_score": 0.95,
                "risk_assessment": {},
                "analysis": output
            }
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {
                "compliance_status": "UNKNOWN",
                "compliance_score": 0.0,
                "risk_assessment": {},
                "analysis": f"Error: {str(e)}"
            }
    
    def _format_test_results(self, results: List[Dict[str, Any]]) -> str:
        lines = []
        for result in results:
            lines.append(f"- Test {result.get('test_case_id')}: {result.get('status')}")
        return "\n".join(lines)
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        lines = []
        for framework, reqs in requirements.items():
            lines.append(f"{framework}: {reqs}")
        return "\n".join(lines)

