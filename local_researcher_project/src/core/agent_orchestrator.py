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
        """Execute planning task with Skills-based instruction and detailed logging."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting research planning")
        logger.info(f"Query: {state['user_query']}")
        logger.info(f"Session: {state['session_id']}")
        logger.info(f"=" * 80)
        
        # Read from shared memory
        memory = self.context.shared_memory
        previous_plans = memory.search(state['user_query'], limit=3)
        
        logger.info(f"[{self.name}] Previous plans found: {len(previous_plans) if previous_plans else 0}")
        
        # Skills-based instruction 사용
        instruction = self.instruction if self.skill else "You are a research planning agent."
        
        logger.info(f"[{self.name}] Using skill: {self.skill is not None}")
        
        # LLM 호출은 llm_manager를 통해 Gemini 직결 사용
        from src.core.llm_manager import execute_llm_task, TaskType
        
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

        logger.info(f"[{self.name}] Calling LLM for planning...")
        # Gemini 실행
        model_result = await execute_llm_task(
            prompt=prompt,
            task_type=TaskType.PLANNING,
            model_name=None,
            system_message=None
        )
        plan = model_result.content or 'No plan generated'
        
        logger.info(f"[{self.name}] ✅ Plan generated: {len(plan)} characters")
        logger.info(f"[{self.name}] Plan preview: {plan[:200]}...")
        
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
        
        logger.info(f"[{self.name}] Plan saved to shared memory")
        logger.info(f"=" * 80)
        
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
        """Execute research tasks with detailed logging."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting research execution")
        logger.info(f"Query: {state['user_query']}")
        logger.info(f"Session: {state['session_id']}")
        logger.info(f"=" * 80)
        
        # Read plan from shared memory
        memory = self.context.shared_memory
        plan = memory.read(
            key=f"plan_{state['session_id']}",
            scope=MemoryScope.SESSION,
            session_id=state['session_id']
        )
        
        logger.info(f"[{self.name}] Research plan loaded: {plan is not None}")
        if plan:
            logger.info(f"[{self.name}] Plan preview: {plan[:200]}...")
        
        # 실제 연구 실행 - MCP Hub를 통한 검색 수행
        query = state['user_query']
        results = []
        
        try:
            # MCP Hub 초기화 확인
            from src.core.mcp_integration import get_mcp_hub, execute_tool, ToolCategory
            
            hub = get_mcp_hub()
            logger.info(f"[{self.name}] MCP Hub status: {len(hub.mcp_sessions) if hub.mcp_sessions else 0} servers connected")
            
            if not hub.mcp_sessions:
                logger.info(f"[{self.name}] Initializing MCP Hub...")
                await hub.initialize_mcp()
                logger.info(f"[{self.name}] MCP Hub initialized: {len(hub.mcp_sessions)} servers")
            
            # 검색 도구 실행
            logger.info(f"[{self.name}] Executing search: '{query}'")
            search_result = await execute_tool(
                "g-search",
                {"query": query, "max_results": 10}
            )
            
            logger.info(f"[{self.name}] Search completed: success={search_result.get('success')}, error={search_result.get('error')}")
            
            if search_result.get('success') and search_result.get('data'):
                data = search_result.get('data', {})
                
                # 검색 결과 파싱 - 다양한 형식 지원
                search_results = []
                if isinstance(data, dict):
                    # 표준 형식: {"results": [...]}
                    search_results = data.get('results', [])
                    if not search_results:
                        # 다른 키 시도
                        search_results = data.get('items', data.get('data', []))
                elif isinstance(data, list):
                    search_results = data
                
                logger.info(f"[{self.name}] Parsed {len(search_results)} search results")
                
                if search_results and len(search_results) > 0:
                    # 실제 검색 결과를 구조화된 형식으로 저장
                    unique_results = []
                    seen_urls = set()
                    
                    for i, result in enumerate(search_results, 1):
                        # 다양한 형식 지원
                        if isinstance(result, dict):
                            title = result.get('title', result.get('name', 'No title'))
                            snippet = result.get('snippet', result.get('content', result.get('summary', '')))
                            url = result.get('url', result.get('link', result.get('href', '')))
                        elif isinstance(result, str):
                            # 문자열 형식인 경우 파싱 시도
                            logger.warning(f"[{self.name}] Result {i} is string, skipping: {result[:100]}")
                            continue
                        else:
                            logger.warning(f"[{self.name}] Unknown result format: {type(result)}")
                            continue
                        
                        # URL 중복 제거
                        if url and url in seen_urls:
                            logger.debug(f"[{self.name}] Duplicate URL skipped: {url}")
                            continue
                        if url:
                            seen_urls.add(url)
                        
                        # 구조화된 결과 저장
                        result_dict = {
                            "index": len(unique_results) + 1,
                            "title": title,
                            "snippet": snippet[:500] if snippet else "",
                            "url": url,
                            "source": "search"
                        }
                        unique_results.append(result_dict)
                        
                        logger.info(f"[{self.name}] Result {i}: {title[:50]}... (URL: {url[:50] if url else 'N/A'}...)")
                    
                    # 결과를 구조화된 형식으로 저장
                    if unique_results:
                        results = unique_results
                        logger.info(f"[{self.name}] ✅ Collected {len(results)} unique results")
                    else:
                        error_msg = f"연구 실행 실패: 검색 결과를 파싱할 수 없습니다."
                        logger.error(f"[{self.name}] ❌ {error_msg}")
                        raise RuntimeError(error_msg)
                else:
                    # 검색 결과가 없음 - 실패 처리
                    error_msg = f"연구 실행 실패: '{query}'에 대한 검색 결과를 찾을 수 없습니다."
                    logger.error(f"[{self.name}] ❌ {error_msg}")
                    raise RuntimeError(error_msg)
            else:
                # 검색 실패 - 에러 반환
                error_msg = f"연구 실행 실패: 검색 도구 실행 중 오류가 발생했습니다. {search_result.get('error', 'Unknown error')}"
                logger.error(f"[{self.name}] ❌ {error_msg}")
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
        state['research_results'] = results  # 리스트로 저장 (덮어쓰기)
        state['current_agent'] = self.name
        state['research_failed'] = False
        
        logger.info(f"[{self.name}] ✅ Research execution completed: {len(results)} results")
        
        # Write to shared memory (구조화된 형식)
        memory.write(
            key=f"research_results_{state['session_id']}",
            value=results,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Results saved to shared memory")
        logger.info(f"=" * 80)
        
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
        """Verify research results with LLM-based verification."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting verification")
        logger.info(f"=" * 80)
        
        # 연구 실패 확인
        if state.get('research_failed'):
            logger.error(f"[{self.name}] ❌ Research execution failed, skipping verification")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['current_agent'] = self.name
            return state
        
        memory = self.context.shared_memory
        
        # Read results from state or shared memory
        results = state.get('research_results', [])
        if not results:
            results = memory.read(
                key=f"research_results_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        logger.info(f"[{self.name}] Found {len(results)} results to verify")
        
        if not results or len(results) == 0:
            error_msg = "검증 실패: 검증할 연구 결과가 없습니다."
            logger.error(f"[{self.name}] ❌ {error_msg}")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['error'] = error_msg
            state['current_agent'] = self.name
            return state
        
        # LLM을 사용한 실제 검증
        from src.core.llm_manager import execute_llm_task, TaskType
        
        verified = []
        for i, result in enumerate(results, 1):
            if isinstance(result, dict):
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                url = result.get('url', '')
                
                # LLM으로 검증
                verification_prompt = f"""다음 검색 결과를 검증하세요:

제목: {title}
내용: {snippet[:300]}
URL: {url}

원래 쿼리: {state['user_query']}

이 결과가 쿼리와 관련이 있고 신뢰할 수 있는지 검증하세요.
응답 형식: "VERIFIED" 또는 "REJECTED"와 간단한 이유를 한 줄로 작성하세요."""
                
                try:
                    verification_result = await execute_llm_task(
                        prompt=verification_prompt,
                        task_type=TaskType.VERIFICATION,
                        model_name=None,
                        system_message="You are a verification agent. Verify if search results are relevant and reliable."
                    )
                    
                    verification_text = verification_result.content or "UNKNOWN"
                    is_verified = "VERIFIED" in verification_text.upper() or "REJECT" not in verification_text.upper()
                    
                    if is_verified:
                        verified.append({
                            "index": i,
                            "title": title,
                            "snippet": snippet,
                            "url": url,
                            "status": "verified",
                            "verification_note": verification_text[:200]
                        })
                        logger.info(f"[{self.name}] ✅ Result {i} verified: {title[:50]}...")
                    else:
                        logger.info(f"[{self.name}] ⚠️ Result {i} rejected: {title[:50]}...")
                        continue
                except Exception as e:
                    logger.warning(f"[{self.name}] Verification failed for result {i}: {e}, including anyway")
                    verified.append({
                        "index": i,
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "status": "partial",
                        "verification_note": "Verification failed, but included"
                    })
            else:
                logger.warning(f"[{self.name}] Unknown result format: {type(result)}")
                continue
        
        logger.info(f"[{self.name}] ✅ Verification completed: {len(verified)}/{len(results)} results verified")
        
        state['verified_results'] = verified
        state['current_agent'] = self.name
        state['verification_failed'] = False if verified else True
        
        # Write to shared memory
        memory.write(
            key=f"verified_{state['session_id']}",
            value=verified,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Verified results saved to shared memory")
        logger.info(f"=" * 80)
        
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
        
        # Read verified results from state or shared memory
        verified_results = state.get('verified_results', [])
        if not verified_results:
            verified_results = memory.read(
                key=f"verified_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        logger.info(f"[{self.name}] Found {len(verified_results)} verified results for report generation")
        
        if not verified_results or len(verified_results) == 0:
            error_msg = "보고서 생성 실패: 검증된 연구 결과가 없습니다."
            logger.error(error_msg)
            
            report = f"""
# 연구 실패 보고서: {state['user_query']}

## ❌ 연구 실행 실패

검증된 연구 결과가 없어 보고서를 생성할 수 없습니다.

### 오류 내용
{error_msg}

### 상황 분석
- 연구 실행은 완료되었지만 결과가 없습니다
- 또는 검증 단계에서 모든 결과가 제외되었습니다
- 연구 쿼리: {state['user_query']}

### 권장 조치
1. 검색 쿼리를 다시 확인해주세요
2. 네트워크 연결 상태를 확인해주세요
3. MCP 서버 설정을 확인해주세요
4. 잠시 후 다시 시도해주세요

실제 연구 데이터 없이 보고서를 생성할 수 없습니다.
"""
            state['final_report'] = report
            state['current_agent'] = self.name
            state['report_failed'] = True
            state['error'] = error_msg  # 에러 메시지 명시
            
            memory.write(
                key=f"report_{state['session_id']}",
                value=report,
                scope=MemoryScope.SESSION,
                session_id=state['session_id'],
                agent_id=self.name
            )
            
            return state
        
        # 실제 결과가 있는 경우 LLM으로 보고서 생성
        logger.info(f"[{self.name}] Generating report with LLM from {len(verified_results)} verified results...")
        
        # 검증된 결과를 텍스트로 변환
        verified_text = ""
        for result in verified_results:
            if isinstance(result, dict):
                verified_text += f"\n- {result.get('title', '')}: {result.get('snippet', '')[:200]}... (Source: {result.get('url', '')})\n"
            else:
                verified_text += f"\n- {str(result)}\n"
        
        # LLM으로 보고서 생성
        from src.core.llm_manager import execute_llm_task, TaskType
        
        generation_prompt = f"""다음 검증된 연구 결과를 바탕으로 상세한 보고서를 작성하세요.

원래 질문: {state['user_query']}

검증된 결과:
{verified_text}

보고서 구조:
1. Executive Summary (요약)
2. 주요 발견사항 (Main Findings)
3. 관련 부품 및 소비재 (Related Components & Consumables)
4. 결론 (Conclusion)

각 섹션을 상세히 작성하고, 구체적인 정보와 출처를 포함하세요."""

        try:
            report_result = await execute_llm_task(
                prompt=generation_prompt,
                task_type=TaskType.GENERATION,
                model_name=None,
                system_message="You are an expert technical writer. Create comprehensive, detailed reports based on verified research results."
            )
            
            report = report_result.content or f"# Report: {state['user_query']}\n\nNo report generated."
            logger.info(f"[{self.name}] ✅ Report generated: {len(report)} characters")
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Report generation failed: {e}")
            # Fallback: 기본 보고서
            report = f"""# Final Report: {state['user_query']}

## Executive Summary
Based on {len(verified_results)} verified research results.

## Detailed Findings
{verified_text}

## Conclusion
Report generation completed with {len(verified_results)} verified results.
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
        
        logger.info(f"[{self.name}] ✅ Report saved to shared memory")
        logger.info(f"=" * 80)
        
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
        """Planner node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🔵 [WORKFLOW] → Planner Node")
        logger.info("=" * 80)
        result = await self.planner.execute(state)
        logger.info(f"🔵 [WORKFLOW] ✓ Planner completed: {result.get('current_agent')}")
        return result
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🟢 [WORKFLOW] → Executor Node")
        logger.info("=" * 80)
        result = await self.executor.execute(state)
        logger.info(f"🟢 [WORKFLOW] ✓ Executor completed: {len(result.get('research_results', []))} results")
        return result
    
    async def _verifier_node(self, state: AgentState) -> AgentState:
        """Verifier node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🟡 [WORKFLOW] → Verifier Node")
        logger.info("=" * 80)
        result = await self.verifier.execute(state)
        logger.info(f"🟡 [WORKFLOW] ✓ Verifier completed: {len(result.get('verified_results', []))} verified")
        return result
    
    async def _generator_node(self, state: AgentState) -> AgentState:
        """Generator node execution with tracking."""
        logger.info("=" * 80)
        logger.info("🟣 [WORKFLOW] → Generator Node")
        logger.info("=" * 80)
        result = await self.generator.execute(state)
        logger.info(f"🟣 [WORKFLOW] ✓ Generator completed: report_length={len(result.get('final_report', ''))}")
        return result
    
    async def _end_node(self, state: AgentState) -> AgentState:
        """End node - final state with summary."""
        logger.info("=" * 80)
        logger.info("✅ [WORKFLOW] → End Node - Workflow Completed")
        logger.info("=" * 80)
        logger.info(f"Session: {state.get('session_id')}")
        logger.info(f"Final Agent: {state.get('current_agent')}")
        logger.info(f"Research Results: {len(state.get('research_results', []))}")
        logger.info(f"Verified Results: {len(state.get('verified_results', []))}")
        logger.info(f"Report Generated: {bool(state.get('final_report'))}")
        logger.info(f"Failed: {state.get('research_failed') or state.get('verification_failed') or state.get('report_failed')}")
        logger.info("=" * 80)
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

