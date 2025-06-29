from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# 웹 검색 도구 정의
# 이 도구는 '검색 에이전트'가 사용합니다.
search_tool = TavilySearchResults(max_results=5, name="web_search")

# LLM 모델 정의
# 모든 에이전트들이 공통으로 사용하는 언어 모델입니다.
# gpt-4o 모델을 사용하여 강력한 추론 능력을 활용합니다.
model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True) 