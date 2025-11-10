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
from src.core.agent_result_sharing import SharedResultsManager, AgentDiscussionManager
from src.core.researcher_config import get_agent_config

logger = logging.getLogger(__name__)

# Logger가 handler가 없으면 root logger의 handler 사용
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # Root logger의 handler 사용 (main.py에서 설정된 handler)
    parent_logger = logging.getLogger()
    if parent_logger.handlers:
        logger.handlers = parent_logger.handlers
        logger.propagate = True
    else:
        # Fallback: 기본 handler 설정
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


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
    research_tasks: Annotated[list, override_reducer]  # List of research tasks for parallel execution
    research_results: Annotated[list, override_reducer]  # Changed: supports both dict and str
    verified_results: Annotated[list, override_reducer]  # Changed: supports both dict and str
    final_report: Optional[str]
    current_agent: Optional[str]
    iteration: int
    session_id: Optional[str]
    research_failed: bool
    verification_failed: bool
    report_failed: bool
    error: Optional[str]


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
    shared_results_manager: Optional[SharedResultsManager] = None
    discussion_manager: Optional[AgentDiscussionManager] = None


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
        
        # 작업 분할: 연구 계획을 여러 독립적인 작업으로 분할
        logger.info(f"[{self.name}] Splitting research plan into parallel tasks...")
        
        task_split_prompt = f"""연구 계획:
{plan}

원래 질문: {state['user_query']}

위 연구 계획을 분석하여 여러 독립적으로 실행 가능한 연구 작업으로 분할하세요.
각 작업은 별도의 연구자(ExecutorAgent)가 동시에 처리할 수 있어야 합니다.

응답 형식 (JSON):
{{
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "작업 설명",
      "search_queries": ["검색 쿼리 1", "검색 쿼리 2"],
      "priority": 1,
      "estimated_time": "medium",
      "dependencies": []
    }},
    ...
  ]
}}

각 작업은:
- 독립적으로 실행 가능해야 함
- 명확한 검색 쿼리를 포함해야 함
- 우선순위와 예상 시간을 포함해야 함
- 의존성이 없어야 함 (병렬 실행을 위해)

작업 수: 3-5개 권장"""

        try:
            task_split_result = await execute_llm_task(
                prompt=task_split_prompt,
                task_type=TaskType.PLANNING,
                model_name=None,
                system_message="You are a task decomposition agent. Split research plans into independent parallel tasks."
            )
            
            task_split_text = task_split_result.content or ""
            
            # JSON 파싱 시도
            import json
            import re
            
            # JSON 블록 추출
            json_match = re.search(r'\{[\s\S]*\}', task_split_text)
            if json_match:
                task_split_json = json.loads(json_match.group())
                tasks = task_split_json.get('tasks', [])
            else:
                # JSON이 없으면 텍스트에서 작업 추출 시도
                tasks = []
                lines = task_split_text.split('\n')
                current_task = None
                for line in lines:
                    line = line.strip()
                    if 'task_id' in line.lower() or 'task' in line.lower() and ':' in line:
                        if current_task:
                            tasks.append(current_task)
                        task_id_match = re.search(r'task[_\s]*(\d+)', line, re.IGNORECASE)
                        task_id = f"task_{task_id_match.group(1) if task_id_match else len(tasks) + 1}"
                        current_task = {
                            "task_id": task_id,
                            "description": "",
                            "search_queries": [],
                            "priority": len(tasks) + 1,
                            "estimated_time": "medium",
                            "dependencies": []
                        }
                    elif current_task:
                        if 'description' in line.lower() or '설명' in line:
                            desc_match = re.search(r':\s*(.+)', line)
                            if desc_match:
                                current_task["description"] = desc_match.group(1).strip()
                        elif 'query' in line.lower() or '쿼리' in line:
                            query_match = re.search(r':\s*(.+)', line)
                            if query_match:
                                current_task["search_queries"].append(query_match.group(1).strip())
                
                if current_task:
                    tasks.append(current_task)
            
            # 작업이 없으면 기본 작업 생성
            if not tasks:
                logger.warning(f"[{self.name}] Failed to parse tasks, creating default task")
                tasks = [{
                    "task_id": "task_1",
                    "description": state['user_query'],
                    "search_queries": [state['user_query']],
                    "priority": 1,
                    "estimated_time": "medium",
                    "dependencies": []
                }]
            
            # 각 작업에 메타데이터 추가
            for i, task in enumerate(tasks):
                if 'task_id' not in task:
                    task['task_id'] = f"task_{i + 1}"
                if 'description' not in task:
                    task['description'] = state['user_query']
                if 'search_queries' not in task or not task['search_queries']:
                    task['search_queries'] = [state['user_query']]
                if 'priority' not in task:
                    task['priority'] = i + 1
                if 'estimated_time' not in task:
                    task['estimated_time'] = "medium"
                if 'dependencies' not in task:
                    task['dependencies'] = []
            
            state['research_tasks'] = tasks
            logger.info(f"[{self.name}] ✅ Split research plan into {len(tasks)} parallel tasks")
            for task in tasks:
                logger.info(f"[{self.name}]   - {task.get('task_id')}: {task.get('description', '')[:50]}... ({len(task.get('search_queries', []))} queries)")
                
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Failed to split tasks: {e}")
            # 실패 시 기본 작업 생성
            state['research_tasks'] = [{
                "task_id": "task_1",
                "description": state['user_query'],
                "search_queries": [state['user_query']],
                "priority": 1,
                "estimated_time": "medium",
                "dependencies": []
            }]
            logger.warning(f"[{self.name}] Using default single task")
        
        state['current_agent'] = self.name
        
        # Write to shared memory
        memory.write(
            key=f"plan_{state['session_id']}",
            value=plan,
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        memory.write(
            key=f"tasks_{state['session_id']}",
            value=state['research_tasks'],
            scope=MemoryScope.SESSION,
            session_id=state['session_id'],
            agent_id=self.name
        )
        
        logger.info(f"[{self.name}] Plan and tasks saved to shared memory")
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
    
    async def execute(self, state: AgentState, assigned_task: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute research tasks with detailed logging."""
        logger.info(f"=" * 80)
        logger.info(f"[{self.name.upper()}] Starting research execution")
        logger.info(f"Agent ID: {self.context.agent_id}")
        logger.info(f"Query: {state['user_query']}")
        logger.info(f"Session: {state['session_id']}")
        logger.info(f"=" * 80)
        
        # 작업 할당: assigned_task가 있으면 사용, 없으면 state에서 찾기
        if assigned_task is None:
            # state['research_tasks']에서 이 에이전트에게 할당된 작업 찾기
            tasks = state.get('research_tasks', [])
            if tasks:
                # agent_id를 기반으로 작업 할당 (라운드로빈)
                agent_id = self.context.agent_id
                if agent_id.startswith("executor_"):
                    try:
                        agent_index = int(agent_id.split("_")[1])
                        if agent_index < len(tasks):
                            assigned_task = tasks[agent_index]
                            logger.info(f"[{self.name}] Assigned task {assigned_task.get('task_id', 'unknown')} to {agent_id}")
                        else:
                            # 인덱스가 범위를 벗어나면 첫 번째 작업 할당
                            assigned_task = tasks[0]
                            logger.info(f"[{self.name}] Agent index out of range, using first task")
                    except (ValueError, IndexError):
                        assigned_task = tasks[0] if tasks else None
                        logger.info(f"[{self.name}] Using first task (fallback)")
                else:
                    # agent_id가 executor_ 형식이 아니면 첫 번째 작업 사용
                    assigned_task = tasks[0] if tasks else None
            else:
                # 작업이 없으면 메모리에서 읽기
                memory = self.context.shared_memory
                tasks = memory.read(
                    key=f"tasks_{state['session_id']}",
                    scope=MemoryScope.SESSION,
                    session_id=state['session_id']
                ) or []
                if tasks:
                    assigned_task = tasks[0] if tasks else None
        
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
        
        # 실제 연구 실행 - MCP Hub를 통한 병렬 검색 수행
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
            
            # 작업 할당이 있으면 해당 작업의 검색 쿼리 사용
            search_queries = []
            if assigned_task:
                search_queries = assigned_task.get('search_queries', [])
                logger.info(f"[{self.name}] Using assigned task queries: {len(search_queries)} queries from task {assigned_task.get('task_id', 'unknown')}")
            
            # 작업 할당이 없거나 쿼리가 없으면 기존 로직 사용
            if not search_queries:
                search_queries = [query]  # 기본 쿼리
                if plan:
                    # LLM으로 연구 계획에서 검색 쿼리 추출
                    query_generation_prompt = f"""연구 계획:
{plan}

원래 질문: {query}

위 연구 계획을 바탕으로 검색에 사용할 구체적인 검색 쿼리 3-5개를 생성하세요.
각 쿼리는 서로 다른 관점이나 측면을 다루어야 합니다.
응답 형식: 각 줄에 하나의 검색 쿼리만 작성하세요. 번호나 기호 없이 쿼리만 작성하세요."""
                
                try:
                    query_result = await execute_llm_task(
                        prompt=query_generation_prompt,
                        task_type=TaskType.PLANNING,
                        model_name=None,
                        system_message="You are a research query generator. Generate specific search queries based on research plans."
                    )
                    
                    generated_queries = query_result.content or ""
                    # 각 줄을 쿼리로 파싱
                    for line in generated_queries.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 5:
                            search_queries.append(line)
                    
                    # 중복 제거
                    search_queries = list(dict.fromkeys(search_queries))[:5]  # 최대 5개
                    logger.info(f"[{self.name}] Generated {len(search_queries)} search queries from plan")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to generate search queries from plan: {e}, using original query only")
            
            # 병렬 검색 실행
            logger.info(f"[{self.name}] Executing {len(search_queries)} searches in parallel...")
            
            async def execute_single_search(search_query: str, query_index: int) -> Dict[str, Any]:
                """단일 검색 실행."""
                try:
                    logger.info(f"[{self.name}] Search {query_index + 1}/{len(search_queries)}: '{search_query}'")
                    search_result = await execute_tool(
                        "g-search",
                        {"query": search_query, "max_results": 10}
                    )
                    return {
                        "query": search_query,
                        "index": query_index,
                        "result": search_result,
                        "success": search_result.get('success', False)
                    }
                except Exception as e:
                    logger.error(f"[{self.name}] Search {query_index + 1} failed: {e}")
                    return {
                        "query": search_query,
                        "index": query_index,
                        "result": {"success": False, "error": str(e)},
                        "success": False
                    }
            
            # 모든 검색을 병렬로 실행
            search_tasks = [execute_single_search(q, i) for i, q in enumerate(search_queries)]
            search_results_list = await asyncio.gather(*search_tasks)
            
            logger.info(f"[{self.name}] ✅ Completed {len(search_results_list)} parallel searches")
            
            # 모든 성공한 검색 결과 통합
            successful_results = [sr for sr in search_results_list if sr.get('success') and sr.get('result', {}).get('data')]
            
            if not successful_results:
                error_msg = f"연구 실행 실패: 모든 검색 쿼리 실행이 실패했습니다."
                logger.error(f"[{self.name}] ❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # 모든 검색 결과를 통합 (하드코딩 제거, 동적 통합)
            all_search_data = []
            for sr in successful_results:
                result_data = sr['result'].get('data', {})
                if isinstance(result_data, dict):
                    items = result_data.get('results', result_data.get('items', []))
                    if isinstance(items, list):
                        all_search_data.extend(items)
                elif isinstance(result_data, list):
                    all_search_data.extend(result_data)
            
            # 통합된 결과를 하나의 검색 결과 형식으로 구성
            search_result = {
                'success': True,
                'data': {
                    'results': all_search_data,
                    'total_results': len(all_search_data),
                    'source': 'parallel_search'
                }
            }
            
            logger.info(f"[{self.name}] ✅ Integrated {len(all_search_data)} results from {len(successful_results)} successful searches")
            
            # 모든 검색 결과를 SharedResultsManager에 공유
            if self.context.shared_results_manager:
                for sr in search_results_list:
                    if sr.get('success'):
                        task_id = f"search_{sr['index']}"
                        await self.context.shared_results_manager.share_result(
                            task_id=task_id,
                            agent_id=self.context.agent_id,  # 고유한 agent_id 사용
                            result=sr['result'],
                            metadata={"query": sr['query'], "index": sr['index']},
                            confidence=1.0 if sr.get('success') else 0.0
                        )
                        logger.info(f"[{self.name}] Shared search result for query: '{sr['query']}' (agent_id: {self.context.agent_id})")
            
            logger.info(f"[{self.name}] Search completed: success={search_result.get('success')}, total_results={search_result.get('data', {}).get('total_results', 0)}")
            logger.info(f"[{self.name}] Search result type: {type(search_result)}, keys: {list(search_result.keys()) if isinstance(search_result, dict) else 'N/A'}")
            
            if search_result.get('success') and search_result.get('data'):
                data = search_result.get('data', {})
                logger.info(f"[{self.name}] Data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                
                # 검색 결과 파싱 - 다양한 형식 지원
                search_results = []
                if isinstance(data, dict):
                    # 표준 형식: {"query": "...", "results": [...], "total_results": N, "source": "..."}
                    search_results = data.get('results', [])
                    logger.info(f"[{self.name}] Found 'results' key: {len(search_results)} items")
                    
                    if not search_results:
                        # 다른 키 시도
                        search_results = data.get('items', data.get('data', []))
                        logger.info(f"[{self.name}] Tried 'items' or 'data' keys: {len(search_results)} items")
                    
                    # data 자체가 리스트인 경우 (중첩된 경우)
                    if not search_results and isinstance(data, dict):
                        # data의 값 중 리스트 찾기
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 0:
                                # 첫 번째 항목이 dict인지 확인
                                if value and isinstance(value[0], dict):
                                    search_results = value
                                    logger.info(f"[{self.name}] Found list in key '{key}': {len(search_results)} items")
                                    break
                elif isinstance(data, list):
                    search_results = data
                    logger.info(f"[{self.name}] Data is directly a list: {len(search_results)} items")
                
                logger.info(f"[{self.name}] ✅ Parsed {len(search_results)} search results")
                
                # 디버깅: 첫 번째 결과 샘플 출력
                if search_results and len(search_results) > 0:
                    first_result = search_results[0]
                    logger.info(f"[{self.name}] First result type: {type(first_result)}, sample: {str(first_result)[:200]}")
                
                if search_results and len(search_results) > 0:
                    # 실제 검색 결과를 구조화된 형식으로 저장
                    unique_results = []
                    seen_urls = set()
                    
                    logger.info(f"[{self.name}] Processing {len(search_results)} results...")
                    
                    for i, result in enumerate(search_results, 1):
                        # 다양한 형식 지원
                        if isinstance(result, dict):
                            title = result.get('title', result.get('name', result.get('Title', 'No title')))
                            snippet = result.get('snippet', result.get('content', result.get('summary', result.get('description', result.get('abstract', '')))))
                            url = result.get('url', result.get('link', result.get('href', result.get('URL', ''))))
                            
                            # snippet에 마크다운 형식의 여러 결과가 들어있는 경우 파싱
                            if snippet and ("Found" in snippet or "search results" in snippet.lower() or "\n1." in snippet):
                                logger.info(f"[{self.name}] Detected markdown format in snippet, parsing...")
                                import re
                                parsed_results = []
                                lines = snippet.split('\n')
                                current_result = None
                                
                                for line in lines:
                                    original_line = line
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # 패턴 1: 마크다운 링크 "1. [Title](URL)"
                                    link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', line)
                                    # 패턴 2: 번호와 제목만 "1. [Title]" 또는 "1. Title"
                                    title_match = re.match(r'^\d+\.\s*(?:\[([^\]]+)\]|(.+?))(?:\s*$|:)', line)
                                    # 패턴 3: URL 줄 "   URL: https://..."
                                    url_match = re.search(r'URL:\s*(https?://[^\s]+)', line, re.IGNORECASE)
                                    # 패턴 4: Summary 줄 "   Summary: ..."
                                    summary_match = re.search(r'Summary:\s*(.+)$', line, re.IGNORECASE)
                                    
                                    if link_match:
                                        # 이전 결과 저장
                                        if current_result and current_result.get('title'):
                                            parsed_results.append(current_result)
                                        
                                        title_parsed = link_match.group(1)
                                        url_parsed = link_match.group(2)
                                        current_result = {
                                            "title": title_parsed,
                                            "url": url_parsed,
                                            "snippet": ""
                                        }
                                    elif title_match and not current_result:
                                        # 번호와 제목만 있는 경우 (다음 줄에 URL이 올 것으로 예상)
                                        title_parsed = title_match.group(1) or title_match.group(2)
                                        if title_parsed:
                                            current_result = {
                                                "title": title_parsed.strip(),
                                                "url": "",
                                                "snippet": ""
                                            }
                                    elif url_match:
                                        # URL이 별도 줄에 있는 경우
                                        if current_result:
                                            current_result["url"] = url_match.group(1)
                                        else:
                                            # URL만 있고 제목이 없는 경우 (이전 결과에 추가)
                                            if parsed_results:
                                                parsed_results[-1]["url"] = url_match.group(1)
                                    elif summary_match and current_result:
                                        # Summary 줄
                                        current_result["snippet"] = summary_match.group(1).strip()
                                    elif current_result and line and not any([
                                        line.startswith('URL:'), 
                                        line.startswith('Summary:'),
                                        line.startswith('Found'),
                                        'search results' in line.lower()
                                    ]):
                                        # 설명 텍스트 (들여쓰기된 경우)
                                        if original_line.startswith('   ') or original_line.startswith('\t'):
                                            if current_result["snippet"]:
                                                current_result["snippet"] += " " + line
                                            else:
                                                current_result["snippet"] = line
                                
                                # 마지막 결과 추가
                                if current_result and current_result.get('title'):
                                    parsed_results.append(current_result)
                                
                                if parsed_results:
                                    logger.info(f"[{self.name}] Parsed {len(parsed_results)} results from markdown snippet")
                                    # 파싱된 결과들을 unique_results에 추가
                                    for parsed_result in parsed_results:
                                        parsed_url = parsed_result.get('url', '')
                                        if parsed_url and parsed_url in seen_urls:
                                            continue
                                        if parsed_url:
                                            seen_urls.add(parsed_url)
                                        
                                        unique_results.append({
                                            "index": len(unique_results) + 1,
                                            "title": parsed_result.get('title', ''),
                                            "snippet": parsed_result.get('snippet', '')[:500],
                                            "url": parsed_url,
                                            "source": "search"
                                        })
                                        logger.info(f"[{self.name}] Parsed result: {parsed_result.get('title', '')[:50]}... (URL: {parsed_url[:50] if parsed_url else 'N/A'}...)")
                                    
                                    # 원본 결과는 건너뛰기
                                    continue
                            
                            logger.debug(f"[{self.name}] Result {i}: title={title[:50] if title else 'N/A'}, url={url[:50] if url else 'N/A'}")
                        elif isinstance(result, str):
                            # 문자열 형식인 경우 파싱 시도 (마크다운 링크 형식)
                            import re
                            link_match = re.match(r'^\d+\.\s*\[([^\]]+)\]\(([^\)]+)\)', result.strip())
                            if link_match:
                                title = link_match.group(1)
                                url = link_match.group(2)
                                snippet = ""
                                logger.info(f"[{self.name}] Parsed string result {i} as markdown: {title[:50]}")
                            else:
                                logger.warning(f"[{self.name}] Result {i} is string but not markdown format, skipping: {result[:100]}")
                                continue
                        else:
                            logger.warning(f"[{self.name}] Unknown result format for result {i}: {type(result)}, value: {str(result)[:100]}")
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
        
        # SharedResultsManager에서 다른 Executor의 결과도 가져오기
        if self.context.shared_results_manager:
            shared_results = await self.context.shared_results_manager.get_shared_results(
                exclude_agent_id=self.name
            )
            logger.info(f"[{self.name}] Found {len(shared_results)} shared results from other agents")
            
            # 공유된 결과를 results에 추가
            for shared_result in shared_results:
                if isinstance(shared_result.result, dict) and shared_result.result.get('data'):
                    # 검색 결과에서 구조화된 데이터 추출
                    data = shared_result.result.get('data', {})
                    if isinstance(data, dict):
                        shared_search_results = data.get('results', data.get('items', []))
                        if isinstance(shared_search_results, list):
                            results.extend(shared_search_results)
                    elif isinstance(data, list):
                        results.extend(data)
        
        logger.info(f"[{self.name}] Found {len(results)} results to verify (including shared results)")
        
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
        
        # 검증 결과를 SharedResultsManager에 공유
        if self.context.shared_results_manager:
            for verified_result in verified:
                task_id = f"verification_{verified_result.get('index', 0)}"
                await self.context.shared_results_manager.share_result(
                    task_id=task_id,
                    agent_id=self.context.agent_id,  # 고유한 agent_id 사용
                    result=verified_result,
                    metadata={"status": verified_result.get('status', 'unknown')},
                    confidence=1.0 if verified_result.get('status') == 'verified' else 0.5
                )
            
            # 다른 에이전트의 검증 결과와 토론 (검증 결과가 다른 경우)
            if self.context.discussion_manager and len(verified) > 0:
                other_verified = await self.context.shared_results_manager.get_shared_results(
                    agent_id=None,  # 모든 에이전트
                    exclude_agent_id=self.context.agent_id  # 고유한 agent_id 사용
                )
                
                # 검증된 결과만 필터링
                other_verified_results = [r for r in other_verified if isinstance(r.result, dict) and r.result.get('status') == 'verified']
                
                if other_verified_results:
                    # 첫 번째 검증 결과에 대해 토론
                    first_verified = verified[0]
                    result_id = f"verification_{first_verified.get('index', 0)}"
                    discussion = await self.context.discussion_manager.agent_discuss_result(
                        result_id=result_id,
                        agent_id=self.context.agent_id,  # 고유한 agent_id 사용
                        other_agent_results=other_verified_results[:3]  # 최대 3개
                    )
                    if discussion:
                        logger.info(f"[{self.name}] Discussion completed: {discussion[:100]}... (agent_id: {self.context.agent_id})")
        
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
        
        # 연구 또는 검증 실패 확인 - Fallback 제거, 명확한 에러만 반환
        if state.get('research_failed') or state.get('verification_failed'):
            error_msg = state.get('error', '알 수 없는 오류')
            logger.error(f"[{self.name}] ❌ Research or verification failed: {error_msg}")
            state['final_report'] = None
            state['current_agent'] = self.name
            state['report_failed'] = True
            state['error'] = error_msg
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
        
        # SharedResultsManager에서 모든 공유된 검증 결과 가져오기
        if self.context.shared_results_manager:
            all_shared_results = await self.context.shared_results_manager.get_shared_results()
            logger.info(f"[{self.name}] Found {len(all_shared_results)} shared results from all agents")
            
            # 검증된 결과만 필터링하여 추가
            for shared_result in all_shared_results:
                if isinstance(shared_result.result, dict):
                    # 검증된 결과인 경우
                    if shared_result.result.get('status') == 'verified':
                        # 중복 제거 (URL 기준)
                        existing_urls = {r.get('url', '') for r in verified_results if isinstance(r, dict)}
                        result_url = shared_result.result.get('url', '')
                        if result_url and result_url not in existing_urls:
                            verified_results.append(shared_result.result)
                            logger.info(f"[{self.name}] Added shared verified result: {shared_result.result.get('title', '')[:50]}...")
        
        logger.info(f"[{self.name}] Found {len(verified_results)} verified results for report generation (including shared results)")
        
        if not verified_results or len(verified_results) == 0:
            # Fallback 제거 - 명확한 에러만 반환
            error_msg = "보고서 생성 실패: 검증된 연구 결과가 없습니다."
            logger.error(f"[{self.name}] ❌ {error_msg}")
            state['final_report'] = None
            state['current_agent'] = self.name
            state['report_failed'] = True
            state['error'] = error_msg
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
        
        # LLM으로 사용자 요청에 맞는 형식으로 생성
        from src.core.llm_manager import execute_llm_task, TaskType
        
        # 사용자 요청을 그대로 전달 - LLM이 형식을 결정하도록
        generation_prompt = f"""사용자 요청: {state['user_query']}

검증된 연구 결과:
{verified_text}

사용자의 요청을 정확히 이해하고, 요청한 형식에 맞게 결과를 생성하세요.
- 보고서를 요청했다면 보고서 형식으로
- 코드를 요청했다면 실행 가능한 코드로
- 문서를 요청했다면 문서 형식으로

요청된 형식에 맞게 완전하고 실행 가능한 결과를 생성하세요."""

        try:
            report_result = await execute_llm_task(
                prompt=generation_prompt,
                task_type=TaskType.GENERATION,
                model_name=None,
                system_message="You are an expert assistant. Generate results in the exact format requested by the user. If they ask for a report, create a report. If they ask for code, create executable code. Follow the user's request precisely without adding unnecessary templates or structures."
            )
            
            report = report_result.content or f"# Report: {state['user_query']}\n\nNo report generated."
            
            # Safety filter 차단 확인 - Fallback 제거, 명확한 오류 반환
            if "blocked by safety" in report.lower() or "content blocked" in report.lower() or len(report) < 100:
                error_msg = "보고서 생성 실패: Safety filter에 의해 차단되었습니다. 프롬프트를 수정하거나 다른 모델을 사용해주세요."
                logger.error(f"[{self.name}] ❌ {error_msg}")
                state['final_report'] = None
                state['report_failed'] = True
                state['error'] = error_msg
                state['current_agent'] = self.name
                return state
            else:
                logger.info(f"[{self.name}] ✅ Report generated: {len(report)} characters")
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Report generation failed: {e}")
            # Fallback 제거 - 명확한 오류 반환
            error_msg = f"보고서 생성 실패: {str(e)}"
            state['final_report'] = None
            state['report_failed'] = True
            state['error'] = error_msg
            state['current_agent'] = self.name
            return state
        
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
        self.agent_config = get_agent_config()
        self.graph = None
        # Graph는 첫 실행 시 쿼리 기반으로 빌드
        
        # SharedResultsManager와 AgentDiscussionManager는 execute 시점에 초기화
        # (objective_id가 필요하므로)
        self.shared_results_manager: Optional[SharedResultsManager] = None
        self.discussion_manager: Optional[AgentDiscussionManager] = None
        
        logger.info("AgentOrchestrator initialized")
    
    def _build_graph(self, user_query: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """Build LangGraph workflow with Skills auto-selection."""
        
        # Create context for all agents
        context = AgentContext(
            agent_id="orchestrator",
            session_id=session_id or "default",
            shared_memory=self.shared_memory,
            config=self.config,
            shared_results_manager=self.shared_results_manager,
            discussion_manager=self.discussion_manager
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
        workflow.add_node("executor", self._executor_node)  # Legacy
        workflow.add_node("parallel_executor", self._parallel_executor_node)  # New parallel executor
        workflow.add_node("verifier", self._verifier_node)  # Legacy
        workflow.add_node("parallel_verifier", self._parallel_verifier_node)  # New parallel verifier
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("end", self._end_node)
        
        # Define edges - 병렬 실행 노드 사용
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "parallel_executor")  # 병렬 실행 사용
        workflow.add_edge("parallel_executor", "parallel_verifier")  # 병렬 검증 사용
        workflow.add_edge("parallel_verifier", "generator")
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
        """Executor node execution with tracking (legacy - for backward compatibility)."""
        logger.info("=" * 80)
        logger.info("🟢 [WORKFLOW] → Executor Node (legacy)")
        logger.info("=" * 80)
        result = await self.executor.execute(state)
        logger.info(f"🟢 [WORKFLOW] ✓ Executor completed: {len(result.get('research_results', []))} results")
        return result
    
    async def _parallel_executor_node(self, state: AgentState) -> AgentState:
        """Parallel executor node - runs multiple ExecutorAgent instances simultaneously."""
        logger.info("=" * 80)
        logger.info("🟢 [WORKFLOW] → Parallel Executor Node")
        logger.info("=" * 80)
        
        # 작업 목록 가져오기
        tasks = state.get('research_tasks', [])
        if not tasks:
            # 메모리에서 읽기
            memory = self.shared_memory
            tasks = memory.read(
                key=f"tasks_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        if not tasks:
            logger.warning("[WORKFLOW] No tasks found, falling back to single executor")
            return await self._executor_node(state)
        
        logger.info(f"[WORKFLOW] Executing {len(tasks)} tasks in parallel with {len(tasks)} ExecutorAgent instances")
        
        # 동적 동시성 관리 통합
        from src.core.concurrency_manager import get_concurrency_manager
        concurrency_manager = get_concurrency_manager()
        max_concurrent = concurrency_manager.get_current_concurrency() or self.agent_config.max_concurrent_research_units
        max_concurrent = min(max_concurrent, len(tasks))  # 작업 수를 초과하지 않도록
        
        logger.info(f"[WORKFLOW] Using concurrency limit: {max_concurrent} (from concurrency_manager)")
        
        # Skills 자동 선택
        selected_skills = {}
        if state.get('user_query'):
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(state['user_query'])
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # 여러 ExecutorAgent 인스턴스 생성 및 병렬 실행
        async def execute_single_task(task: Dict[str, Any], task_index: int) -> AgentState:
            """단일 작업을 실행하는 ExecutorAgent."""
            agent_id = f"executor_{task_index}"
            context = AgentContext(
                agent_id=agent_id,
                session_id=state['session_id'],
                shared_memory=self.shared_memory,
                config=self.config,
                shared_results_manager=self.shared_results_manager,
                discussion_manager=self.discussion_manager
            )
            
            executor_agent = ExecutorAgent(context, selected_skills.get("research_executor"))
            
            try:
                logger.info(f"[WORKFLOW] ExecutorAgent {agent_id} starting task {task.get('task_id', 'unknown')}")
                result_state = await executor_agent.execute(state, assigned_task=task)
                logger.info(f"[WORKFLOW] ExecutorAgent {agent_id} completed: {len(result_state.get('research_results', []))} results")
                return result_state
            except Exception as e:
                logger.error(f"[WORKFLOW] ExecutorAgent {agent_id} failed: {e}")
                # 실패한 에이전트의 상태 반환
                failed_state = state.copy()
                failed_state['research_results'] = []
                failed_state['research_failed'] = True
                failed_state['error'] = f"Task {task.get('task_id', 'unknown')} failed: {str(e)}"
                failed_state['current_agent'] = agent_id
                return failed_state
        
        # 모든 작업을 병렬로 실행 (동적 동시성 제한 적용)
        if max_concurrent < len(tasks):
            # Semaphore를 사용하여 동시 실행 수 제한
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_with_limit(task: Dict[str, Any], task_index: int) -> AgentState:
                async with semaphore:
                    return await execute_single_task(task, task_index)
            
            executor_tasks = [execute_with_limit(task, i) for i, task in enumerate(tasks)]
        else:
            # 동시성 제한이 작업 수보다 크면 모든 작업을 동시에 실행
            executor_tasks = [execute_single_task(task, i) for i, task in enumerate(tasks)]
        
        # 병렬 실행
        executor_results = await asyncio.gather(*executor_tasks, return_exceptions=True)
        
        # 결과 통합
        all_results = []
        all_failed = False
        errors = []
        
        for i, result in enumerate(executor_results):
            if isinstance(result, Exception):
                logger.error(f"[WORKFLOW] ExecutorAgent {i} raised exception: {result}")
                all_failed = True
                errors.append(f"Task {tasks[i].get('task_id', 'unknown')}: {str(result)}")
            elif isinstance(result, dict):
                # 결과 수집
                task_results = result.get('research_results', [])
                if task_results:
                    all_results.extend(task_results)
                    logger.info(f"[WORKFLOW] ExecutorAgent {i} contributed {len(task_results)} results")
                
                # 실패 상태 확인
                if result.get('research_failed'):
                    all_failed = True
                    if result.get('error'):
                        errors.append(result['error'])
        
        # 통합된 상태 생성
        final_state = state.copy()
        final_state['research_results'] = all_results
        final_state['research_failed'] = all_failed
        final_state['current_agent'] = "parallel_executor"
        
        if errors:
            final_state['error'] = "; ".join(errors)
        
        logger.info(f"[WORKFLOW] ✅ Parallel execution completed: {len(all_results)} total results from {len(tasks)} tasks")
        logger.info(f"[WORKFLOW] Failed: {all_failed}")
        
        return final_state
    
    async def _verifier_node(self, state: AgentState) -> AgentState:
        """Verifier node execution with tracking (legacy - for backward compatibility)."""
        logger.info("=" * 80)
        logger.info("🟡 [WORKFLOW] → Verifier Node (legacy)")
        logger.info("=" * 80)
        result = await self.verifier.execute(state)
        logger.info(f"🟡 [WORKFLOW] ✓ Verifier completed: {len(result.get('verified_results', []))} verified")
        return result
    
    async def _parallel_verifier_node(self, state: AgentState) -> AgentState:
        """Parallel verifier node - runs multiple VerifierAgent instances simultaneously."""
        logger.info("=" * 80)
        logger.info("🟡 [WORKFLOW] → Parallel Verifier Node")
        logger.info("=" * 80)
        
        # 연구 실패 확인
        if state.get('research_failed'):
            logger.error("[WORKFLOW] Research execution failed, skipping verification")
            state['verified_results'] = []
            state['verification_failed'] = True
            state['current_agent'] = "parallel_verifier"
            return state
        
        # 검증할 결과 가져오기
        results = state.get('research_results', [])
        if not results:
            memory = self.shared_memory
            results = memory.read(
                key=f"research_results_{state['session_id']}",
                scope=MemoryScope.SESSION,
                session_id=state['session_id']
            ) or []
        
        if not results:
            logger.warning("[WORKFLOW] No results to verify, falling back to single verifier")
            return await self._verifier_node(state)
        
        # 결과를 여러 청크로 분할하여 여러 VerifierAgent에 할당
        num_verifiers = min(len(results), self.agent_config.max_concurrent_research_units or 3)
        chunk_size = max(1, len(results) // num_verifiers)
        result_chunks = [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]
        
        logger.info(f"[WORKFLOW] Verifying {len(results)} results with {len(result_chunks)} VerifierAgent instances")
        
        # 동적 동시성 관리 통합
        from src.core.concurrency_manager import get_concurrency_manager
        concurrency_manager = get_concurrency_manager()
        max_concurrent = concurrency_manager.get_current_concurrency() or self.agent_config.max_concurrent_research_units
        max_concurrent = min(max_concurrent, len(result_chunks))
        
        logger.info(f"[WORKFLOW] Using concurrency limit: {max_concurrent} (from concurrency_manager)")
        
        # Skills 자동 선택
        selected_skills = {}
        if state.get('user_query'):
            skill_selector = get_skill_selector()
            matches = skill_selector.select_skills_for_task(state['user_query'])
            for match in matches:
                skill = self.skill_manager.load_skill(match.skill_id)
                if skill:
                    selected_skills[match.skill_id] = skill
        
        # 여러 VerifierAgent 인스턴스 생성 및 병렬 실행
        async def verify_single_chunk(chunk: List[Dict[str, Any]], chunk_index: int) -> List[Dict[str, Any]]:
            """단일 청크를 검증하는 VerifierAgent."""
            agent_id = f"verifier_{chunk_index}"
            context = AgentContext(
                agent_id=agent_id,
                session_id=state['session_id'],
                shared_memory=self.shared_memory,
                config=self.config,
                shared_results_manager=self.shared_results_manager,
                discussion_manager=self.discussion_manager
            )
            
            verifier_agent = VerifierAgent(context, selected_skills.get("evaluator"))
            
            # 청크만 포함하는 임시 state 생성
            chunk_state = state.copy()
            chunk_state['research_results'] = chunk
            
            try:
                logger.info(f"[WORKFLOW] VerifierAgent {agent_id} starting verification of {len(chunk)} results")
                result_state = await verifier_agent.execute(chunk_state)
                verified_chunk = result_state.get('verified_results', [])
                logger.info(f"[WORKFLOW] VerifierAgent {agent_id} completed: {len(verified_chunk)} verified")
                return verified_chunk
            except Exception as e:
                logger.error(f"[WORKFLOW] VerifierAgent {agent_id} failed: {e}")
                return []  # 실패 시 빈 리스트 반환
        
        # 모든 청크를 병렬로 검증 (동적 동시성 제한 적용)
        if max_concurrent < len(result_chunks):
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def verify_with_limit(chunk: List[Dict[str, Any]], chunk_index: int) -> List[Dict[str, Any]]:
                async with semaphore:
                    return await verify_single_chunk(chunk, chunk_index)
            
            verifier_tasks = [verify_with_limit(chunk, i) for i, chunk in enumerate(result_chunks)]
        else:
            verifier_tasks = [verify_single_chunk(chunk, i) for i, chunk in enumerate(result_chunks)]
        
        # 병렬 실행
        verifier_results = await asyncio.gather(*verifier_tasks, return_exceptions=True)
        
        # 결과 통합
        all_verified = []
        for i, result in enumerate(verifier_results):
            if isinstance(result, Exception):
                logger.error(f"[WORKFLOW] VerifierAgent {i} raised exception: {result}")
            elif isinstance(result, list):
                all_verified.extend(result)
                logger.info(f"[WORKFLOW] VerifierAgent {i} contributed {len(result)} verified results")
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_verified = []
        for verified_result in all_verified:
            if isinstance(verified_result, dict):
                url = verified_result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_verified.append(verified_result)
                elif not url:
                    unique_verified.append(verified_result)
        
        # 여러 VerifierAgent 간 토론 (검증 결과가 다른 경우)
        if self.discussion_manager and len(unique_verified) > 0:
            # 다른 VerifierAgent의 검증 결과 가져오기
            if self.shared_results_manager:
                other_verified = await self.shared_results_manager.get_shared_results()
                other_verified_results = [r for r in other_verified if isinstance(r.result, dict) and r.result.get('status') == 'verified']
                
                if other_verified_results:
                    # 첫 번째 검증 결과에 대해 토론
                    first_verified = unique_verified[0]
                    result_id = f"verification_{first_verified.get('index', 0)}"
                    discussion = await self.discussion_manager.agent_discuss_result(
                        result_id=result_id,
                        agent_id="parallel_verifier",
                        other_agent_results=other_verified_results[:3]
                    )
                    if discussion:
                        logger.info(f"[WORKFLOW] Discussion completed: {discussion[:100]}...")
        
        # 통합된 상태 생성
        final_state = state.copy()
        final_state['verified_results'] = unique_verified
        final_state['verification_failed'] = False if unique_verified else True
        final_state['current_agent'] = "parallel_verifier"
        
        logger.info(f"[WORKFLOW] ✅ Parallel verification completed: {len(unique_verified)} total verified results from {len(result_chunks)} verifiers")
        
        return final_state
    
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
        
        # Objective ID 생성 (병렬 실행 및 결과 공유용)
        objective_id = f"objective_{session_id}"
        
        # SharedResultsManager와 AgentDiscussionManager 초기화 (병렬 실행 활성화 시)
        if self.agent_config.enable_agent_communication:
            self.shared_results_manager = SharedResultsManager(objective_id=objective_id)
            self.discussion_manager = AgentDiscussionManager(
                objective_id=objective_id,
                shared_results_manager=self.shared_results_manager
            )
            logger.info("✅ Agent result sharing and discussion enabled")
        else:
            self.shared_results_manager = None
            self.discussion_manager = None
            logger.info("Agent communication disabled")
        
        # Graph가 없거나 쿼리 기반 재빌드가 필요한 경우 빌드
        if self.graph is None:
            self._build_graph(user_query, session_id)
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            user_query=user_query,
            research_plan=None,
            research_tasks=[],
            research_results=[],
            verified_results=[],
            final_report=None,
            current_agent=None,
            iteration=0,
            session_id=session_id,
            research_failed=False,
            verification_failed=False,
            report_failed=False,
            error=None
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
            research_tasks=[],
            research_results=[],
            verified_results=[],
            final_report=None,
            current_agent=None,
            iteration=0,
            session_id=session_id,
            research_failed=False,
            verification_failed=False,
            report_failed=False,
            error=None
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

