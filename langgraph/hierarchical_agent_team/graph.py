from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from .agents import (
    supervisor_agent, search_agent, analyst_agent, 
    outline_agent, writer_agent, editor_agent
)
from .utils import search_tool
from langchain_core.runnables import Runnable
from langchain_core.agents import AgentFinish
from langchain_core.tools import tool

# 상태 정의: 그래프의 각 노드가 공유하는 데이터 구조
class AgentState(TypedDict):
    query: str
    supervisor_feedback: Optional[str] = None
    search_queries: Optional[List[str]] = None
    search_results: Optional[str] = None
    critique: Optional[str] = None
    outline: Optional[str] = None
    draft: Optional[str] = None
    editor_feedback: Optional[str] = None
    final_report: Optional[str] = None

# 헬퍼 함수: 에이전트를 실행하고 결과를 파싱
def run_agent(agent: Runnable, state: AgentState, name: str):
    print(f"--- 실행중인 노드: {name} ---")
    result = agent.invoke({"input": state['query'], "agent_scratchpad": []})
    if isinstance(result, AgentFinish):
        return {"messages": [result.return_values['output']]}
    return {"messages": [result]}

# 노드 실행 함수 정의
def supervisor_node(state: AgentState):
    print("---SUPERVISOR---")
    # supervisor_feedback을 바탕으로 검색 쿼리를 생성하거나 재생성
    prompt = f"Original query: {state['query']}"
    if state['supervisor_feedback']:
        prompt += f"\nFeedback for improvement: {state['supervisor_feedback']}"
    
    queries = supervisor_agent.invoke({"input": prompt})
    state['search_queries'] = queries.split('\n')
    return state

def search_node(state: AgentState):
    print("---SEARCH---")
    results = ""
    for q in state['search_queries']:
        # Tavily search tool 직접 호출
        res = search_tool.invoke({"query": q})
        results += f"Query: '{q}'\nResult: {res}\n\n"
    state['search_results'] = results
    return state

def analyst_node(state: AgentState):
    print("---ANALYST---")
    prompt = f"Original query: {state['query']}\n\nSearch results:\n{state['search_results']}"
    critique = analyst_agent.invoke({"input": prompt})
    state['critique'] = critique
    return state

def outline_node(state: AgentState):
    print("---OUTLINER---")
    prompt = f"Research results:\n{state['search_results']}"
    outline = outline_agent.invoke({"input": prompt})
    state['outline'] = outline
    return state

def writer_node(state: AgentState):
    print("---WRITER---")
    prompt = f"Outline:\n{state['outline']}\n\nResearch material:\n{state['search_results']}"
    if state['editor_feedback']:
        prompt += f"\n\nEditor's feedback: {state['editor_feedback']}"
    draft = writer_agent.invoke({"input": prompt})
    state['draft'] = draft
    return state
    
def editor_node(state: AgentState):
    print("---EDITOR---")
    feedback = editor_agent.invoke({"input": state['draft']})
    state['editor_feedback'] = feedback
    return state

# 조건부 엣지 (라우팅) 로직
def route_after_analysis(state: AgentState):
    print("---ROUTING (ANALYSIS)---")
    if "CONTINUE" in state['critique']:
        print("결과 양호, 개요 작성으로 이동")
        return "outliner"
    else:
        print("결과 미흡, 감독관에게 피드백 전달")
        state['supervisor_feedback'] = state['critique']
        return "supervisor"

def route_after_edit(state: AgentState):
    print("---ROUTING (EDIT)---")
    if "PERFECT" in state['editor_feedback']:
        print("보고서 완벽, 최종본으로 채택")
        state['final_report'] = state['draft']
        return END
    else:
        print("수정 필요, 작성가에게 피드백 전달")
        return "writer"

# 그래프 빌드
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("search", search_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("outliner", outline_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("supervisor")
workflow.add_edge("supervisor", "search")
workflow.add_edge("search", "analyst")
workflow.add_conditional_edge("analyst", route_after_analysis, {
    "outliner": "outliner",
    "supervisor": "supervisor"
})
workflow.add_edge("outliner", "writer")
workflow.add_edge("writer", "editor")
workflow.add_conditional_edge("editor", route_after_edit, {
    "writer": "writer",
    END: END
})

# 그래프 컴파일
app = workflow.compile()
print("계층적 에이전트 팀 그래프가 성공적으로 컴파일되었습니다.") 