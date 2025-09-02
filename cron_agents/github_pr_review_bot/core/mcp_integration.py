"""
MCP Integration - 실제 MCP 서버와 LangChain/LangGraph 통합

이 모듈은 실제 구현된 MCP 서버들과 LangChain/LangGraph를 통합합니다.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

# LangChain과 LangGraph MCP 통합
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI
    from langchain.agents import initialize_agent, AgentType
    from langgraph.graph import StateGraph, END
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
except ImportError:
    logging.error("LangChain MCP 어댑터가 설치되지 않았습니다. 'pip install langchain-mcp-adapters'를 실행하세요.")
    raise

from .config import config

logger = logging.getLogger(__name__)

class MCPIntegrationManager:
    """실제 MCP 서버와 LangChain/LangGraph 통합 관리자"""
    
    def __init__(self):
        # LangChain MCP 클라이언트 초기화
        self.mcp_client = None
        self.agent = None
        self.langgraph_app = None
        self.tools = []
        
        # 실제 존재하는 MCP 서버 설정 (LangChain 방식)
        self.server_configs = {
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
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=config.llm.openai_api_key
        )
        
        self._initialize_mcp_client()
        self._initialize_agent()
        self._initialize_langgraph()
        
        logger.info(f"MCP Integration Manager initialized with real MCP servers")
    
    def _initialize_mcp_client(self):
        """MCP 클라이언트 초기화 (LangChain 방식)"""
        try:
            self.mcp_client = MultiServerMCPClient(self.server_configs)
            self.tools = self.mcp_client.get_tools()
            logger.info(f"MCP 클라이언트 초기화 완료: {len(self.tools)}개 도구 로드")
        except Exception as e:
            logger.error(f"MCP 클라이언트 초기화 실패: {e}")
            raise ValueError(f"MCP 클라이언트 초기화 실패: {e}")
    
    def _initialize_agent(self):
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
            raise ValueError(f"LangChain 에이전트 초기화 실패: {e}")
    
    def _initialize_langgraph(self):
        """LangGraph 워크플로우 초기화"""
        try:
            # LangGraph 상태 정의
            from typing import TypedDict
            
            class ReviewState(TypedDict):
                code: str
                language: str
                context: Dict[str, Any]
                analysis_results: Dict[str, Any]
                final_review: str
            
            # 그래프 구성
            workflow = StateGraph(ReviewState)
            
            # 노드 추가
            workflow.add_node("analyze_code", self._analyze_code_node)
            workflow.add_node("generate_review", self._generate_review_node)
            
            # 엣지 추가
            workflow.set_entry_point("analyze_code")
            workflow.add_edge("analyze_code", "generate_review")
            workflow.add_edge("generate_review", END)
            
            self.langgraph_app = workflow.compile()
            logger.info("LangGraph 워크플로우 초기화 완료")
        except Exception as e:
            logger.error(f"LangGraph 초기화 실패: {e}")
            raise ValueError(f"LangGraph 초기화 실패: {e}")
    
    def _analyze_code_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """코드 분석 노드 (실제 MCP 서버 사용)"""
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
        """리뷰 생성 노드 (실제 MCP 서버 사용)"""
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
    
    async def connect_to_server(self, server_name: str) -> bool:
        """MCP 서버에 연결 (LangChain 방식)"""
        if server_name not in self.server_configs:
            raise ValueError(f"Unknown MCP server: {server_name}")
        
        # LangChain MCP 클라이언트는 이미 초기화 시 모든 서버에 연결됨
        server_tools = [tool for tool in self.tools if server_name in tool.name.lower()]
        
        if server_tools:
            logger.info(f"MCP 서버 {server_name}에 연결됨: {len(server_tools)}개 도구")
            return True
        else:
            raise ValueError(f"MCP 서버 {server_name}에 연결할 수 없습니다.")
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """MCP 서버의 도구 호출 (LangChain 방식)"""
        # LangChain 도구에서 해당 도구 찾기
        target_tool = None
        for tool in self.tools:
            if server_name in tool.name.lower() and tool_name in tool.name.lower():
                target_tool = tool
                break
        
        if not target_tool:
            raise ValueError(f"도구를 찾을 수 없습니다: {server_name}.{tool_name}")
        
        # LangChain 도구 호출
        result = target_tool.invoke(kwargs)
        if not result:
            raise ValueError(f"No result from MCP tool {tool_name} on {server_name}")
        
        return {"result": result}
    
    async def get_comprehensive_review(self, code: str, language: str, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """종합적인 코드 리뷰 (LangGraph 워크플로우 활용)"""
        # LangGraph 워크플로우 실행
        initial_state = {
            "code": code,
            "language": language,
            "context": context or {},
            "analysis_results": {},
            "final_review": ""
        }
        
        try:
            # LangGraph 앱 실행
            final_state = self.langgraph_app.invoke(initial_state)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'mcp_analyses': final_state.get('analysis_results', {}),
                'summary': {
                    'total_analyses': len(final_state.get('analysis_results', {})),
                    'final_review': final_state.get('final_review', ''),
                    'recommendations': [final_state.get('final_review', '')]
                }
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"LangGraph 워크플로우 실행 실패: {e}")
    
    def _generate_comprehensive_summary(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """종합 분석 요약 생성 (LangChain 방식)"""
        summary = {
            'total_analyses': len(analyses),
            'critical_issues': 0,
            'high_priority_issues': 0,
            'medium_priority_issues': 0,
            'recommendations': [],
            'expert_insights': [],
            'security_findings': [],
            'performance_insights': []
        }
        
        for analysis_name, analysis_result in analyses.items():
            if 'error' in analysis_result:
                continue
            
            # 보안 분석 결과 처리
            if 'security_analysis' in analysis_name:
                security_data = analysis_result.get('vulnerabilities', [])
                summary['security_findings'] = security_data
                summary['critical_issues'] += len([v for v in security_data if v.get('severity') == 'critical'])
                summary['high_priority_issues'] += len([v for v in security_data if v.get('severity') == 'high'])
            
            # 성능 분석 결과 처리
            elif 'performance_analysis' in analysis_name:
                performance_data = analysis_result.get('issues', [])
                summary['performance_insights'] = performance_data
                summary['critical_issues'] += len([i for i in performance_data if i.get('impact') == 'critical'])
                summary['high_priority_issues'] += len([i for i in performance_data if i.get('impact') == 'high'])
            
            # 전문가 리뷰 결과 처리
            elif 'expert_review' in analysis_name:
                expert_data = analysis_result.get('recommendations', [])
                summary['expert_insights'] = expert_data
                summary['recommendations'].extend(expert_data)
            
            # 일반 권장사항 수집
            if 'recommendations' in analysis_result:
                summary['recommendations'].extend(analysis_result['recommendations'])
        
        # 중복 제거
        summary['recommendations'] = list(set(summary['recommendations']))
        summary['expert_insights'] = list(set(summary['expert_insights']))
        
        return summary
    
    async def get_specialized_analysis(self, analysis_type: str, code: str, 
                                     language: str, **kwargs) -> Dict[str, Any]:
        """특화된 분석 수행 (실제 MCP 서버 사용)"""
        try:
            if analysis_type == "github":
                # GitHub 서버로 PR 분석
                github_tool = next((t for t in self.tools if "github" in t.name.lower()), None)
                if github_tool:
                    result = github_tool.invoke({
                        "action": "analyze_pr",
                        "code": code,
                        "language": language,
                        **kwargs
                    })
                    return {"result": result, "analysis_type": "github"}
            
            elif analysis_type == "filesystem":
                # 파일 시스템 서버로 코드베이스 분석
                filesystem_tool = next((t for t in self.tools if "filesystem" in t.name.lower()), None)
                if filesystem_tool:
                    result = filesystem_tool.invoke({
                        "action": "analyze_codebase",
                        "path": kwargs.get("path", "."),
                        "code": code
                    })
                    return {"result": result, "analysis_type": "filesystem"}
            
            elif analysis_type == "search":
                # Brave Search 서버로 웹 검색
                search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
                if search_tool:
                    query = kwargs.get("query", f"{language} code review best practices")
                    result = search_tool.invoke({"query": query})
                    return {"result": result, "analysis_type": "search"}
            
            elif analysis_type == "fetch":
                # Fetch 서버로 API 호출
                fetch_tool = next((t for t in self.tools if "fetch" in t.name.lower()), None)
                if fetch_tool:
                    url = kwargs.get("url", "https://api.github.com/repos/microsoft/vscode")
                    result = fetch_tool.invoke({"url": url})
                    return {"result": result, "analysis_type": "fetch"}
            
            else:
                # LangChain 에이전트로 일반 분석
                prompt = f"""
                다음 {analysis_type} 분석을 수행해주세요:
                
                분석 유형: {analysis_type}
                언어: {language}
                코드: {code}
                추가 매개변수: {kwargs}
                
                GitHub PR 리뷰 관점에서 분석을 수행하고 결과를 반환해주세요.
                """
                
                result = self.agent.run(prompt)
                return {"result": result, "analysis_type": analysis_type}
                
        except Exception as e:
            raise ValueError(f"특화 분석 실패 ({analysis_type}): {e}")
    
    def get_available_servers(self) -> List[Dict[str, Any]]:
        """사용 가능한 MCP 서버 목록 반환 (LangChain 방식)"""
        return [
            {
                'name': name,
                'info': config,
                'connected': any(name in tool.name.lower() for tool in self.tools)
            }
            for name, config in self.server_configs.items()
        ]
    
    def get_server_status(self) -> Dict[str, Any]:
        """서버 상태 정보 반환 (LangChain 방식)"""
        connected_servers = []
        for server_name in self.server_configs.keys():
            if any(server_name in tool.name.lower() for tool in self.tools):
                connected_servers.append(server_name)
        
        return {
            'total_servers': len(self.server_configs),
            'connected_servers': len(connected_servers),
            'available_servers': list(self.server_configs.keys()),
            'active_connections': connected_servers,
            'total_tools': len(self.tools)
        }
    
    async def health_check_all_servers(self) -> Dict[str, Any]:
        """모든 MCP 서버 상태 확인 (LangChain 방식)"""
        health_results = {}
        
        for server_name in self.server_configs.keys():
            try:
                is_connected = await self.connect_to_server(server_name)
                health_results[server_name] = {
                    'status': 'healthy' if is_connected else 'unhealthy',
                    'connected': is_connected
                }
            except Exception as e:
                health_results[server_name] = {
                    'status': 'error',
                    'error': str(e),
                    'connected': False
                }
        
        return health_results

# 전역 인스턴스
mcp_integration_manager = MCPIntegrationManager() 