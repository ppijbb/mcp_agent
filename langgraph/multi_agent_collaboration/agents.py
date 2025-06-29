from langchain_core.prompts import ChatPromptTemplate
from langchain_core.agents import create_tool_calling_agent
from langchain_core.runnables import Runnable
from .utils import search_tool, model

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> Runnable:
    """주어진 LLM, 도구, 시스템 프롬프트를 바탕으로 에이전트를 생성하는 헬퍼 함수"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

# 1. 리서치 감독관 (Research Supervisor)
# 사용자의 요청을 바탕으로 구체적인 검색 쿼리들을 생성합니다.
supervisor_agent = create_agent(
    model,
    [],
    "You are a research supervisor. Your role is to understand the user's request and generate a list of 3-5 specific, targeted search queries to find the best information on the topic. Provide the queries as a numbered list."
)

# 2. 검색 에이전트 (Search Agent)
# supervisor가 생성한 쿼리를 받아 웹 검색을 수행합니다.
search_agent = create_agent(
    model,
    [search_tool],
    "You are a research agent. You must use the 'web_search' tool to find information for the given search queries."
)

# 3. 정보 분석가 (Info Analyst)
# 검색 결과를 평가하고, 정보가 충분한지 판단합니다.
analyst_agent = create_agent(
    model,
    [],
    "You are an information analyst. Your role is to evaluate the provided search results. Determine if the information is sufficient and relevant to the user's original query. Respond with 'CONTINUE' if the information is good, or provide feedback on what's missing and suggest new search queries if it's not."
)

# 4. 개요 작성가 (Outliner)
# 충분한 정보가 모이면, 보고서의 구조적인 개요를 작성합니다.
outline_agent = create_agent(
    model,
    [],
    "You are an expert report outliner. Based on the provided research, create a detailed, structured outline for a comprehensive report. The outline should include a title, introduction, main sections with bullet points for key topics, and a conclusion."
)

# 5. 보고서 작성가 (Writer)
# 개요와 정보를 바탕으로 보고서 초안을 작성합니다.
writer_agent = create_agent(
    model,
    [],
    "You are a professional report writer. Using the provided outline and research information, write a clear, concise, and well-structured report. The report should be easy to understand and cover all points from the outline."
)

# 6. 교정/편집가 (Editor)
# 초안을 검토하고, 최종본으로 만듭니다. 필요시 수정 방향을 제시합니다.
editor_agent = create_agent(
    model,
    [],
    """You are an expert editor. Review the draft report for any grammatical errors, stylistic issues, and logical inconsistencies. 
    If the report is perfect, respond with 'PERFECT'.
    If there are issues, provide specific feedback on how to improve it."""
) 