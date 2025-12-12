"""
Agent Tool Wrapper - 에이전트를 도구로 노출하는 래퍼

기존 에이전트를 수정하지 않고, 에이전트를 LangChain Tool로 변환하여
Cross-Agent Communication을 가능하게 합니다.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AskAgentInput(BaseModel):
    """에이전트 질의 입력 스키마"""
    query: str = Field(description="The question or task to ask the agent")
    context: Optional[str] = Field(default=None, description="Additional context for the agent")


class AgentToolWrapper:
    """에이전트를 도구로 노출하는 래퍼 - 기존 에이전트 수정 없음"""
    
    def __init__(
        self,
        planner_agent=None,
        executor_agent=None,
        verifier_agent=None,
        generator_agent=None
    ):
        """
        초기화
        
        Args:
            planner_agent: PlannerAgent 인스턴스 (선택적)
            executor_agent: ExecutorAgent 인스턴스 (선택적)
            verifier_agent: VerifierAgent 인스턴스 (선택적)
            generator_agent: GeneratorAgent 인스턴스 (선택적)
        """
        # 기존 에이전트 인스턴스 그대로 사용 (수정 없음)
        self.planner = planner_agent
        self.executor = executor_agent
        self.verifier = verifier_agent
        self.generator = generator_agent
        logger.info("Agent Tool Wrapper initialized")
    
    def create_tools(self) -> List[BaseTool]:
        """
        기존 에이전트를 도구로 변환
        
        Returns:
            LangChain Tool 리스트
        """
        tools = []
        
        # PlannerAgent를 도구로 노출 (기존 execute 메서드 그대로 사용)
        if self.planner:
            async def ask_planner(query: str, context: Optional[str] = None) -> str:
                """Ask the planning agent for research planning"""
                try:
                    # AgentState 생성 (기존 구조 그대로)
                    from src.core.agent_orchestrator import AgentState
                    state: AgentState = {
                        'messages': [],
                        'user_query': query,
                        'research_plan': None,
                        'research_tasks': [],
                        'research_results': [],
                        'verified_results': [],
                        'final_report': None,
                        'current_agent': None,
                        'iteration': 0,
                        'session_id': None,
                        'research_failed': False,
                        'verification_failed': False,
                        'report_failed': False,
                        'error': None
                    }
                    
                    # 기존 execute 메서드 그대로 호출
                    result = await self.planner.execute(state)
                    return result.get('research_plan', 'No plan generated')
                except Exception as e:
                    logger.error(f"Error calling planner agent: {e}")
                    return f"Error: {str(e)}"
            
            tools.append(StructuredTool.from_function(
                func=ask_planner,
                name="ask_planner",
                description="Ask the planning agent to create a research plan for a given query"
            ))
        
        # ExecutorAgent를 도구로 노출
        if self.executor:
            async def ask_executor(query: str, context: Optional[str] = None) -> str:
                """Ask the executor agent to perform research"""
                try:
                    from src.core.agent_orchestrator import AgentState
                    state: AgentState = {
                        'messages': [],
                        'user_query': query,
                        'research_plan': context or '',
                        'research_tasks': [],
                        'research_results': [],
                        'verified_results': [],
                        'final_report': None,
                        'current_agent': None,
                        'iteration': 0,
                        'session_id': None,
                        'research_failed': False,
                        'verification_failed': False,
                        'report_failed': False,
                        'error': None
                    }
                    
                    # 기존 execute 메서드 그대로 호출
                    result = await self.executor.execute(state)
                    results = result.get('research_results', [])
                    if results:
                        return str(results[:3])  # 처음 3개 결과만 반환
                    return "No results found"
                except Exception as e:
                    logger.error(f"Error calling executor agent: {e}")
                    return f"Error: {str(e)}"
            
            tools.append(StructuredTool.from_function(
                func=ask_executor,
                name="ask_executor",
                description="Ask the executor agent to perform research on a given query"
            ))
        
        # VerifierAgent를 도구로 노출
        if self.verifier:
            async def ask_verifier(query: str, context: Optional[str] = None) -> str:
                """Ask the verifier agent to verify research results"""
                try:
                    from src.core.agent_orchestrator import AgentState
                    state: AgentState = {
                        'messages': [],
                        'user_query': query,
                        'research_plan': None,
                        'research_tasks': [],
                        'research_results': [],
                        'verified_results': [],
                        'final_report': None,
                        'current_agent': None,
                        'iteration': 0,
                        'session_id': None,
                        'research_failed': False,
                        'verification_failed': False,
                        'report_failed': False,
                        'error': None
                    }
                    
                    # 기존 execute 메서드 그대로 호출
                    result = await self.verifier.execute(state)
                    verified = result.get('verified_results', [])
                    if verified:
                        return str(verified[:3])  # 처음 3개 결과만 반환
                    return "No verified results"
                except Exception as e:
                    logger.error(f"Error calling verifier agent: {e}")
                    return f"Error: {str(e)}"
            
            tools.append(StructuredTool.from_function(
                func=ask_verifier,
                name="ask_verifier",
                description="Ask the verifier agent to verify research results"
            ))
        
        # GeneratorAgent를 도구로 노출
        if self.generator:
            async def ask_generator(query: str, context: Optional[str] = None) -> str:
                """Ask the generator agent to generate a report"""
                try:
                    from src.core.agent_orchestrator import AgentState
                    state: AgentState = {
                        'messages': [],
                        'user_query': query,
                        'research_plan': None,
                        'research_tasks': [],
                        'research_results': [],
                        'verified_results': [],
                        'final_report': None,
                        'current_agent': None,
                        'iteration': 0,
                        'session_id': None,
                        'research_failed': False,
                        'verification_failed': False,
                        'report_failed': False,
                        'error': None
                    }
                    
                    # 기존 execute 메서드 그대로 호출
                    result = await self.generator.execute(state)
                    return result.get('final_report', 'No report generated')
                except Exception as e:
                    logger.error(f"Error calling generator agent: {e}")
                    return f"Error: {str(e)}"
            
            tools.append(StructuredTool.from_function(
                func=ask_generator,
                name="ask_generator",
                description="Ask the generator agent to generate a final report"
            ))
        
        logger.info(f"Created {len(tools)} agent tools")
        return tools

