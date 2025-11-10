"""
Report Generator Agent

규제 기관 제출용 리포트 자동 생성
"""

import logging
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..llm.model_manager import ModelManager, ModelProvider
from ..llm.fallback_handler import FallbackHandler
from ..tools.mcp_tools import MCPToolsWrapper
from ..tools.document_tools import DocumentTools

logger = logging.getLogger(__name__)


class ReportGeneratorAgent:
    """리포트 생성 Agent"""
    
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
        self.document_tools = DocumentTools()
        self.tools = self.mcp_tools.get_tools() + self.document_tools.get_tools()
        
        self.llm = self.model_manager.get_llm(preferred_provider)
        if not self.llm:
            raise ValueError("No available LLM found")
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a regulatory report generator for medical devices.

Generate comprehensive compliance reports for regulatory submission (FDA 510(k), CE Marking, etc.).

Reports must include:
- Executive summary
- Device description
- Regulatory framework analysis
- Test results
- Compliance assessment
- Risk analysis
- Recommendations"""),
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
    
    def generate(
        self,
        report_type: str,
        device_info: Dict[str, Any],
        compliance_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """리포트 생성"""
        try:
            input_text = f"""Generate {report_type} compliance report:

Device Info: {device_info}
Compliance Data: {compliance_data}

Save report to: {output_path}"""
            
            result = self.agent_executor.invoke({"input": input_text})
            return result.get("output", "")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error: {str(e)}"

