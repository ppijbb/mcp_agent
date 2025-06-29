import os
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 설정 확인
if 'OPENAI_API_KEY' not in os.environ or 'TAVILY_API_KEY' not in os.environ:
    print("'.env' 파일에 OPENAI_API_KEY와 TAVILY_API_KEY를 설정해주세요.")
    exit()

# 1. 도구 정의
# 리서처 에이전트가 사용할 웹 검색 도구입니다.
tools = [TavilySearchResults(max_results=3, name="web_search")]
tool_executor = AgentExecutor(
    tools=tools,
    agent=create_tool_calling_agent(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        tools=tools,
        prompt=ChatPromptTemplate.from_messages([("system", "You are a helpful assistant.")])
    ),
)

# 2. 에이전트 상태 정의
# 에이전트들이 작업하는 동안 상태를 저장하는 객체입니다.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# 3. 에이전트(노드) 정의
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def researcher_node(state: AgentState):
    """리서처 에이전트: 웹 검색을 수행하여 정보를 수집합니다."""
    print("---리서처 노드 실행---")
    response = tool_executor.invoke({"messages": state["messages"]})
    return {"messages": [response["output"]]}

def writer_node(state: AgentState):
    """작성자 에이전트: 수집된 정보를 바탕으로 보고서를 작성합니다."""
    print("---작성자 노드 실행---")
    # 보고서 작성 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional report writer. Your task is to write a concise and clear report based on the provided research information."),
        ("user", "{research_summary}")
    ])
    writer_chain = prompt | llm
    
    # 마지막 메시지(리서치 결과)를 기반으로 보고서 작성
    research_summary = state["messages"][-1]
    report = writer_chain.invoke({"research_summary": research_summary})
    return {"messages": [report]}

# 4. 감독관(조건부 엣지) 정의
def supervisor_router(state: AgentState) -> str:
    """
    감독관: 대화 내용을 분석하여 다음에 어떤 에이전트를 호출할지 결정합니다.
    - 리서치가 필요하면 'researcher'
    - 보고서 작성이 필요하면 'writer'
    - 모든 작업이 끝났으면 'END'
    """
    print("---감독관 노드 실행---")
    last_message = state["messages"][-1]
    
    # ToolMessage는 리서처의 결과물이므로, 다음은 작성자 차례입니다.
    if isinstance(last_message, ToolMessage) or "web_search" in str(last_message.content):
        print("라우팅: 작성자에게 전달")
        return "writer"
    
    # 사용자의 초기 입력이거나, 다른 종류의 메시지일 경우 리서처를 먼저 실행합니다.
    # 간단한 시스템을 위해, 초기 입력 후에는 항상 리서처 -> 작성자 순으로 진행됩니다.
    # 만약 보고서가 이미 생성되었다면(AIMessage 이면서 길이가 길다면) 종료합니다.
    if isinstance(last_message, HumanMessage):
        print("라우팅: 리서처에게 전달")
        return "researcher"
    else:
        print("라우팅: 작업 종료")
        return "END"

# 5. 그래프 생성 및 연결
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# 엣지 추가
workflow.add_conditional_edges(
    "researcher",
    supervisor_router,
    {"writer": "writer", "END": END}
)
workflow.add_conditional_edges(
    "writer",
    supervisor_router,
    {"researcher": "researcher", "END": END}
)

# 시작점 설정
workflow.set_entry_point("researcher")

# 그래프 컴파일
app = workflow.compile()

# 6. 실행 코드
if __name__ == "__main__":
    query = "What is LangGraph and how does it work?"
    print(f"사용자 질문: {query}\n")

    # 스트리밍 출력을 위해 stream 사용
    inputs = {"messages": [HumanMessage(content=query)]}
    for event in app.stream(inputs, stream_mode="values"):
        # state["messages"][-1]는 현재 단계에서 추가된 최신 메시지를 의미합니다.
        latest_message = event["messages"][-1]
        print("---스트림 이벤트 발생---")
        if isinstance(latest_message, HumanMessage):
            print(f"사용자 입력: {latest_message.content}")
        elif isinstance(latest_message, ToolMessage):
            print(f"리서처 결과: {latest_message.content[:300]}...") # 너무 길어서 일부만 출력
        else: # AIMessage
            print(f"최종 보고서:\n{latest_message.content}") 