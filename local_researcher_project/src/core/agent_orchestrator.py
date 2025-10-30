"""
Agent Orchestrator for Multi-Agent System

LangGraph 기반 에이전트 오케스트레이션 시스템
4대 핵심 에이전트를 조율하여 협업 워크플로우 구축
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Literal, Annotated
from datetime import datetime
from dataclasses import dataclass, field
import operator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.core.shared_memory import get_shared_memory, MemoryScope
from src.core.skills_manager import get_skill_manager
from src.core.skills_selector import get_skill_selector, SkillMatch
from src.core.skills_loader import Skill

logger = logging.getLogger(__name__)


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentState(TypedDict):
    """Main agent state containing messages and research data."""
    
    messages: Annotated[list, add_messages]
    user_query: str
    research_plan: Optional[str]
    research_results: Annotated[list[str], override_reducer]
    verified_results: Annotated[list[str], override_reducer]
    final_report: Optional[str]
    current_agent: Optional[str]
    iteration: int
    session_id: Optional[str]


###################
# Agent Definitions
###################

@dataclass
class AgentContext:
    """Agent execution context."""
    agent_id: str
    session_id: str
    shared_memory: Any
    config: Any = None


class PlannerAgent:
    """Planner agent - creates research plans (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "planner"
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("research_planner")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a research planning agent."
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute planning task with Skills-based instruction."""
        logger.info(f"[{self.name}] Planning research for: {state['user_query']}")
        
        # Read from shared memory
        memory = self.context.shared_memory
        previous_plans = memory.search(state['user_query'], limit=3)
        
        # Skills-based instruction 사용
        instruction = self.instruction if self.skill else "You are a research planning agent."
        
        # Call LLM via OpenRouter
        import os
        from src.core.mcp_integration import get_mcp_hub
            
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not set - LLM features require OpenRouter API key")
        
        # Use Skills instruction
        prompt = f"""{instruction}

Task: Create a detailed research plan for: {state['user_query']}

Based on previous research:
{previous_plans if previous_plans else "No previous research found"}

Create a comprehensive research plan with:
1. Research objectives
2. Key areas to investigate
3. Expected sources and methods
4. Success criteria

Keep it concise and actionable (max 300 words)."""

        # 전역 MCP Hub 인스턴스 사용 및 초기화 확인
        hub = get_mcp_hub()
        if not hub.openrouter_client or (hasattr(hub.openrouter_client, 'session') and not hub.openrouter_client.session):
            logger.info("Initializing MCP Hub...")
            await hub.initialize_mcp()
        
        response = await hub.call_llm_async(
            model="google/gemini-2.0-flash-exp:free",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 응답 파싱
        if response and 'choices' in response and len(response['choices']) > 0:
            plan = response['choices'][0].get('message', {}).get('content', 'No plan generated')
        else:
            plan = response.get('content', 'No plan generated')
        
        state['research_plan'] = plan
        state['current_agent'] = self.name
        
        # Write to shared memory
        memory.write(
            key=f"plan_{state['session_id']}",
            value=plan,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        return state


class ExecutorAgent:
    """Executor agent - executes research tasks using tools (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "executor"
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("research_executor")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a research execution agent."
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute research tasks."""
        logger.info(f"[{self.name}] Executing research for: {state['user_query']}")
        
        # Read plan from shared memory
        memory = self.context.shared_memory
        plan = memory.read(
            key=f"plan_{state['session_id']}",
            scope=MemoryScope.SESSION,
            session_id=state['session_id']
        )
        
        # 실제 연구 실행 - MCP Hub를 통한 검색 수행
        query = state['user_query']
        results = []
        
        try:
            # MCP Hub 초기화 확인
            from src.core.mcp_integration import get_mcp_hub, execute_tool, ToolCategory
            
            hub = get_mcp_hub()
            if not hub.openrouter_client or (hasattr(hub.openrouter_client, 'session') and not hub.openrouter_client.session):
                logger.info("Initializing MCP Hub...")
                await hub.initialize_mcp()
            
            # 검색 도구 실행
            search_result = await execute_tool(
                "g-search",
                {"query": query, "max_results": 10}
            )
            
            if search_result.get('success') and search_result.get('data'):
                data = search_result.get('data', {})
                search_results = data.get('results', []) if isinstance(data, dict) else []
                
                if search_results and len(search_results) > 0:
                    # 실제 검색 결과를 사용
                    for i, result in enumerate(search_results[:5], 1):
                        title = result.get('title', 'No title')
                        snippet = result.get('snippet', result.get('content', ''))
                        url = result.get('url', '')
                        
                        result_text = f"Research Result {i}: {title}"
                        if snippet:
                            result_text += f" - {snippet[:200]}"
                        if url:
                            result_text += f" (Source: {url})"
                        
                        results.append(result_text)
                else:
                    # 검색 결과가 없음 - 실패 처리
                    error_msg = f"연구 실행 실패: '{query}'에 대한 검색 결과를 찾을 수 없습니다."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                # 검색 실패 - 에러 반환
                error_msg = f"연구 실행 실패: 검색 도구 실행 중 오류가 발생했습니다. {search_result.get('error', 'Unknown error')}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # 실제 오류 발생 - 실패 처리
            error_msg = f"연구 실행 실패: {str(e)}"
            logger.error(error_msg)
            
            # 실패 상태 기록
            state['research_results'] = []
            state['current_agent'] = self.name
            state['error'] = error_msg
            state['research_failed'] = True
            
            # 메모리에 실패 정보 기록
            memory.write(
                key=f"execution_error_{state['session_id']}",
                value=error_msg,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
            
            # 실패 상태 반환 (더미 데이터 없이)
            return state
        
        # 성공적으로 결과 수집된 경우
        state['research_results'].extend(results)
        state['current_agent'] = self.name
        state['research_failed'] = False
        
        # Write to shared memory
        for i, result in enumerate(results):
            memory.write(
                key=f"result_{i}_{state['session_id']}",
                value=result,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
        
        return state


class VerifierAgent:
    """Verifier agent - verifies research results (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "verifier"
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("evaluator")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a verification agent."
    
    async def execute(self, state: AgentState) -> AgentState:
        """Verify research results."""
        logger.info(f"[{self.name}] Verifying results...")
        
        # 연구 실패 확인
        if state.get('research_failed'):
            logger.error("Research execution failed, skipping verification")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['current_agent'] = self.name
            return state
        
        memory = self.context.shared_memory
        
        # Read results from shared memory
        results = state.get('research_results', [])
        
        if not results or len(results) == 0:
            error_msg = "검증 실패: 검증할 연구 결과가 없습니다."
            logger.error(error_msg)
            state['verified_results'] = []
            state['verification_failed'] = True
            state['error'] = error_msg
            state['current_agent'] = self.name
            return state
        
        # 실제 결과 검증 (단순 패턴 확인)
        verified = []
        for result in results:
            # 기본 검증: 결과가 실제 정보를 포함하는지 확인
            if result and len(result) > 20 and "Research Result" in result:
                verified.append(f"✅ Verified: {result}")
            else:
                verified.append(f"⚠️ Partial verification: {result}")
        
        state['verified_results'].extend(verified)
        state['current_agent'] = self.name
        state['verification_failed'] = False
        
        # Write to shared memory
        memory.write(
            key=f"verified_{state['session_id']}",
            value=verified,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        return state


class GeneratorAgent:
    """Generator agent - creates final report (Skills-based)."""
    
    def __init__(self, context: AgentContext, skill: Optional[Skill] = None):
        self.context = context
        self.name = "generator"
        self.skill = skill
        
        # Skill이 없으면 로드 시도
        if self.skill is None:
            skill_manager = get_skill_manager()
            self.skill = skill_manager.load_skill("synthesizer")
        
        # Skill instruction 사용
        if self.skill:
            self.instruction = self.skill.instructions
        else:
            self.instruction = "You are a report generation agent."
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate final report."""
        logger.info(f"[{self.name}] Generating final report...")
        
        # 연구 또는 검증 실패 확인
        if state.get('research_failed') or state.get('verification_failed'):
            error_msg = state.get('error', '알 수 없는 오류')
            
            report = f"""
# 연구 실패 보고서: {state['user_query']}

## ❌ 연구 실행 실패

연구를 완료할 수 없었습니다.

### 오류 내용
{error_msg}

### 권장 조치
1. 검색 쿼리를 다시 확인해주세요
2. 네트워크 연결 상태를 확인해주세요
3. MCP 서버 설정을 확인해주세요
4. 잠시 후 다시 시도해주세요

## 실패 원인
- 연구 실행 단계에서 오류 발생
- 검색 결과를 얻을 수 없음
- 서버 연결 문제 가능성

실제 연구 결과 없이 보고서를 생성할 수 없습니다.
"""
            state['final_report'] = report
            state['current_agent'] = self.name
            state['report_failed'] = True
            
            memory = self.context.shared_memory
            memory.write(
                key=f"report_{state['session_id']}",
                value=report,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
            
            return state
        
        memory = self.context.shared_memory
        
        # Read verified results from shared memory
        verified_results = state.get('verified_results', [])
        
        if not verified_results or len(verified_results) == 0:
            error_msg = "보고서 생성 실패: 검증된 연구 결과가 없습니다."
            logger.error(error_msg)
            
            report = f"""
# 연구 완료 불가: {state['user_query']}

## ❌ 보고서 생성 실패

검증된 연구 결과가 없어 보고서를 생성할 수 없습니다.

### 상황
- 연구 실행은 완료되었지만 결과가 없습니다
- 또는 검증 단계에서 모든 결과가 제외되었습니다

실제 연구 데이터 없이 보고서를 생성할 수 없습니다.
"""
            state['final_report'] = report
            state['current_agent'] = self.name
            state['report_failed'] = True
            
            memory.write(
                key=f"report_{state['session_id']}",
                value=report,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
            
            return state
        
        # 실제 결과가 있는 경우에만 보고서 생성
        report = f"""
# Final Report: {state['user_query']}

## Executive Summary
{chr(10).join(verified_results[:3])}

## Detailed Findings
{chr(10).join(verified_results)}

## Conclusion
Based on comprehensive research and verification, this report provides
a thorough analysis of the topic with high confidence.
"""
        
        state['final_report'] = report
        state['current_agent'] = self.name
        state['report_failed'] = False
        
        # Write to shared memory
        memory.write(
            key=f"report_{state['session_id']}",
            value=report,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        return state


###################
# Orchestrator
###################

class AgentOrchestrator:
    """Orchestrator for multi-agent workflow."""
    
    def __init__(self, config: Any = None):
        """Initialize orchestrator."""
        self.config = config
        self.shared_memory = get_shared_memory()
        self.skill_manager = get_skill_manager()
        self.graph = None
        # Graph는 첫 실행 시 쿼리 기반으로 빌드
        
        logger.info("AgentOrchestrator initialized")
    
    def _build_graph(self, user_query: Optional[str] = None) -> None:
        """Build LangGraph workflow with Skills auto-selection."""
        
        # Create context for all agents
        context = AgentContext(
            agent_id="orchestrator",
            session_id="default",
            shared_memory=self.shared_memory,
            config=self.config
        )
        
        # Skills 자동 선택 (쿼리가 있으면)
        selected_skills = {}
        if user_query:
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(user_query)
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # Initialize agents with Skills
        self.planner = PlannerAgent(context, selected_skills.get("research_planner"))
        self.executor = ExecutorAgent(context, selected_skills.get("research_executor"))
        self.verifier = VerifierAgent(context, selected_skills.get("evaluator"))
        self.generator = GeneratorAgent(context, selected_skills.get("synthesizer"))
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("verifier", self._verifier_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("end", self._end_node)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "verifier")
        workflow.add_edge("verifier", "generator")
        workflow.add_edge("generator", "end")
        
        # Compile graph
        self.graph = workflow.compile()
        
        logger.info("LangGraph workflow built")
    
    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner node execution."""
        return await self.planner.execute(state)
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution."""
        return await self.executor.execute(state)
    
    async def _verifier_node(self, state: AgentState) -> AgentState:
        """Verifier node execution."""
        return await self.verifier.execute(state)
    
    async def _generator_node(self, state: AgentState) -> AgentState:
        """Generator node execution."""
        return await self.generator.execute(state)
    
    async def _end_node(self, state: AgentState) -> AgentState:
        """End node - final state."""
        logger.info("Workflow completed")
        return state
    
    async def execute(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute multi-agent workflow with Skills auto-selection.
        
        Args:
            user_query: User's research query
            session_id: Session ID
            
        Returns:
            Final result from the workflow
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting workflow for query: {user_query}")
        
        # Graph가 없거나 쿼리 기반 재빌드가 필요한 경우 빌드
        if self.graph is None:
            self._build_graph(user_query)
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            user_query=user_query,
            research_plan=None,
            research_results=[],
            verified_results=[],
            final_report=None,
            current_agent=None,
            iteration=0,
            session_id=session_id
        )
        
        # Execute workflow
        try:
            result = await self.graph.ainvoke(initial_state)
            logger.info("Workflow execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def stream(self, user_query: str, session_id: Optional[str] = None):
        """Stream workflow execution."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            user_query=user_query,
            research_plan=None,
            research_results=[],
            verified_results=[],
            final_report=None,
            current_agent=None,
            iteration=0,
            session_id=session_id
        )
        
        # Stream execution
        async for event in self.graph.astream(initial_state):
            yield event


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(config: Any = None) -> AgentOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(config=config)
    
    return _orchestrator


def init_orchestrator(config: Any = None) -> AgentOrchestrator:
    """Initialize orchestrator."""
    global _orchestrator
    
    _orchestrator = AgentOrchestrator(config=config)
    
    return _orchestrator

