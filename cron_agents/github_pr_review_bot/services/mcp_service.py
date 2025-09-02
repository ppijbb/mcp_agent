"""
MCP Service

Model Context Protocol 서버들과의 상호작용을 담당하는 서비스입니다.
실제 존재하는 MCP 서버들을 통해 코드 분석, 검색, 파일 시스템 접근 등의 기능을 제공합니다.
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END

from ..core.config import config

logger = logging.getLogger(__name__)

class MCPService:
    """MCP 서비스"""
    
    def __init__(self):
        """MCP 서비스 초기화"""
        self.mcp_client = None
        self.agent = None
        self.langgraph_app = None
        self.tools = []
        
        self._initialize_llm()
        self._initialize_mcp_client()
        self._initialize_agent()
        self._initialize_langgraph()
        
        logger.info("MCP 서비스 초기화 완료")
    
    def _initialize_llm(self) -> None:
        """LLM 초기화"""
        if not config.llm.openai_api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            if config.llm.fail_on_llm_error:
                sys.exit(1)
            raise ValueError("OpenAI API 키가 필요합니다.")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=config.llm.openai_api_key
        )
        logger.info("LLM 초기화 완료")
    
    def _initialize_mcp_client(self) -> None:
        """MCP 클라이언트 초기화"""
        server_configs = {
            "github": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-github"],
                "transport": "stdio",
            },
            "filesystem": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio",
            },
            "brave_search": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-brave-search"],
                "transport": "stdio",
            },
            "fetch": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-fetch"],
                "transport": "stdio",
            }
        }
        
        try:
            self.mcp_client = MultiServerMCPClient(server_configs)
            self.tools = self.mcp_client.get_tools()
            logger.info(f"MCP 클라이언트 초기화 완료: {len(self.tools)}개 도구 로드")
        except Exception as e:
            logger.error(f"MCP 클라이언트 초기화 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"MCP 클라이언트 초기화 실패: {e}")
    
    def _initialize_agent(self) -> None:
        """LangChain 에이전트 초기화"""
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            logger.info("LangChain 에이전트 초기화 완료")
        except Exception as e:
            logger.error(f"LangChain 에이전트 초기화 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"LangChain 에이전트 초기화 실패: {e}")
    
    def _initialize_langgraph(self) -> None:
        """LangGraph 워크플로우 초기화"""
        try:
            from typing import TypedDict
            
            class ReviewState(TypedDict):
                code: str
                language: str
                context: Dict[str, Any]
                analysis_results: Dict[str, Any]
                final_review: str
            
            workflow = StateGraph(ReviewState)
            workflow.add_node("analyze_code", self._analyze_code_node)
            workflow.add_node("generate_review", self._generate_review_node)
            workflow.set_entry_point("analyze_code")
            workflow.add_edge("analyze_code", "generate_review")
            workflow.add_edge("generate_review", END)
            
            self.langgraph_app = workflow.compile()
            logger.info("LangGraph 워크플로우 초기화 완료")
        except Exception as e:
            logger.error(f"LangGraph 초기화 실패: {e}")
            if config.github.fail_fast_on_error:
                sys.exit(1)
            raise ValueError(f"LangGraph 초기화 실패: {e}")
    
    def _analyze_code_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """코드 분석 노드"""
        try:
            # GitHub 서버로 PR 정보 분석
            github_tool = next((t for t in self.tools if "github" in t.name.lower()), None)
            if github_tool:
                result = github_tool.invoke({
                    "action": "analyze_pr",
                    "code": state["code"],
                    "language": state["language"]
                })
                state["analysis_results"]["github_analysis"] = result
            
            # 파일 시스템으로 코드베이스 분석
            filesystem_tool = next((t for t in self.tools if "filesystem" in t.name.lower()), None)
            if filesystem_tool:
                result = filesystem_tool.invoke({
                    "action": "analyze_codebase",
                    "path": "."
                })
                state["analysis_results"]["filesystem_analysis"] = result
            
            return state
        except Exception as e:
            raise ValueError(f"코드 분석 실패: {e}")
    
    def _generate_review_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """리뷰 생성 노드"""
        try:
            # 웹 검색으로 모범 사례 찾기
            search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
            if search_tool:
                search_result = search_tool.invoke({
                    "query": f"{state['language']} code review best practices"
                })
                state["analysis_results"]["best_practices"] = search_result
            
            # LangChain 에이전트로 종합 리뷰 생성
            prompt = f"""
            다음 코드를 분석하고 리뷰를 생성해주세요:
            
            언어: {state['language']}
            코드: {state['code']}
            분석 결과: {state['analysis_results']}
            
            GitHub PR 리뷰 관점에서 코드 품질, 보안, 성능, 스타일을 종합적으로 검토하고 리뷰를 작성해주세요.
            """
            
            result = self.agent.run(prompt)
            state["final_review"] = result
            
            return state
        except Exception as e:
            raise ValueError(f"리뷰 생성 실패: {e}")
    
    def analyze_code(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """코드 분석"""
        initial_state = {
            "code": code,
            "language": language,
            "context": context or {},
            "analysis_results": {},
            "final_review": ""
        }
        
        try:
            final_state = self.langgraph_app.invoke(initial_state)
            return {
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'analysis_results': final_state.get('analysis_results', {}),
                'final_review': final_state.get('final_review', '')
            }
        except Exception as e:
            raise ValueError(f"코드 분석 실패: {e}")
    
    def search_best_practices(self, language: str, topic: str = "code review") -> Dict[str, Any]:
        """모범 사례 검색"""
        try:
            search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
            if not search_tool:
                raise ValueError("검색 도구를 찾을 수 없습니다.")
            
            query = f"{language} {topic} best practices"
            result = search_tool.invoke({"query": query})
            return {"result": result, "query": query}
        except Exception as e:
            raise ValueError(f"모범 사례 검색 실패: {e}")
    
    def fetch_external_data(self, url: str) -> Dict[str, Any]:
        """외부 데이터 가져오기"""
        try:
            fetch_tool = next((t for t in self.tools if "fetch" in t.name.lower()), None)
            if not fetch_tool:
                raise ValueError("Fetch 도구를 찾을 수 없습니다.")
            
            result = fetch_tool.invoke({"url": url})
            return {"result": result, "url": url}
        except Exception as e:
            raise ValueError(f"외부 데이터 가져오기 실패: {e}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 조회"""
        connected_servers = []
        for server_name in ["github", "filesystem", "brave_search", "fetch"]:
            if any(server_name in tool.name.lower() for tool in self.tools):
                connected_servers.append(server_name)
        
        return {
            'total_servers': 4,
            'connected_servers': len(connected_servers),
            'available_servers': ["github", "filesystem", "brave_search", "fetch"],
            'active_connections': connected_servers,
            'total_tools': len(self.tools)
        }
