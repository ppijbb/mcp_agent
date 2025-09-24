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
        self.audit_log = []
        
        # 보안이 강화된 MCP 서버 설정 (읽기 전용, 샌드박스 환경)
        self.server_configs = {
            "github_secure": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-github", "--read-only", "--sandbox"],
                "transport": "stdio",
                "permissions": ["read"],
                "sandbox": True
            },
            "filesystem_secure": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem", "/tmp/mcp_sandbox", "--read-only"],
                "transport": "stdio",
                "permissions": ["read"],
                "sandbox": True,
                "restricted_paths": ["/tmp/mcp_sandbox"]
            },
            "memory_secure": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-memory", "--secure-mode"],
                "transport": "stdio",
                "permissions": ["read"],
                "sandbox": True
            },
            "sequential_thinking": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-sequential-thinking", "--sandbox"],
                "transport": "stdio",
                "permissions": ["read"],
                "sandbox": True
            }
        }
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=config.llm.openai_api_key
        )
        
        # 샌드박스 환경 생성
        self._create_sandbox_environment()
        
        self._initialize_mcp_client()
        self._initialize_agent()
        self._initialize_langgraph()
        
        logger.info(f"MCP Integration Manager initialized with secure MCP servers (sandbox mode)")
    
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
        """MCP 서버의 도구 호출 (보안 검증 포함)"""
        # 보안 권한 검증
        if not self._validate_mcp_permissions(server_name, tool_name):
            raise ValueError(f"보안 위반: {server_name}에서 {tool_name} 실행 권한 없음")
        
        # 활동 로깅
        self._log_mcp_activity(server_name, tool_name, kwargs)
        
        # LangChain 도구에서 해당 도구 찾기
        target_tool = None
        for tool in self.tools:
            if server_name in tool.name.lower() and tool_name in tool.name.lower():
                target_tool = tool
                break
        
        if not target_tool:
            raise ValueError(f"도구를 찾을 수 없습니다: {server_name}.{tool_name}")
        
        try:
            # LangChain 도구 호출
            result = target_tool.invoke(kwargs)
            if not result:
                raise ValueError(f"No result from MCP tool {tool_name} on {server_name}")
            
            # 성공 로깅
            self._log_mcp_activity(server_name, f"{tool_name}_success", {"result_length": len(str(result))})
            
            return {"result": result}
            
        except Exception as e:
            # 오류 로깅
            self._log_mcp_activity(server_name, f"{tool_name}_error", {"error": str(e)})
            raise
    
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
            
            # 상세 변경사항 정보 추가
            detailed_changes = context.get('detailed_changes', {}) if context else {}
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'mcp_analyses': final_state.get('analysis_results', {}),
                'change_analysis': self._analyze_changes_for_review(detailed_changes),
                'summary': {
                    'total_analyses': len(final_state.get('analysis_results', {})),
                    'final_review': final_state.get('final_review', ''),
                    'recommendations': [final_state.get('final_review', '')]
                }
            }
            
            return results
            
        except Exception as e:
            raise ValueError(f"LangGraph 워크플로우 실행 실패: {e}")
    
    def _analyze_changes_for_review(self, detailed_changes: Dict[str, Any]) -> Dict[str, Any]:
        """리뷰를 위한 변경사항 분석"""
        if not detailed_changes:
            return {}
        
        analysis = {
            'change_summary': detailed_changes.get('summary', {}),
            'critical_issues': [],
            'recommendations': [],
            'focus_areas': []
        }
        
        # 중요 파일 변경사항 분석
        categories = detailed_changes.get('change_categories', {})
        if categories.get('critical_files'):
            analysis['critical_issues'].extend([
                f"중요 파일 변경: {file['filename']} ({file['change_type']})"
                for file in categories['critical_files']
            ])
        
        # API 변경사항 분석
        impact_analysis = detailed_changes.get('impact_analysis', {})
        if impact_analysis.get('api_changes'):
            analysis['critical_issues'].extend([
                f"API 변경 감지: {change['file']}"
                for change in impact_analysis['api_changes']
            ])
            analysis['recommendations'].append("API 변경사항에 대한 테스트를 추가하세요")
        
        # Breaking changes 분석
        if impact_analysis.get('breaking_changes'):
            analysis['critical_issues'].extend([
                f"잠재적 Breaking Change: {change['file']}"
                for change in impact_analysis['breaking_changes']
            ])
            analysis['recommendations'].append("Breaking Change 가능성을 검토하고 문서화하세요")
        
        # 의존성 변경사항 분석
        if impact_analysis.get('dependency_changes'):
            analysis['focus_areas'].extend([
                f"의존성 변경: {change['file']}"
                for change in impact_analysis['dependency_changes']
            ])
            analysis['recommendations'].append("의존성 변경사항의 호환성을 확인하세요")
        
        # 의미적 변경사항 분석
        semantic_changes = detailed_changes.get('semantic_changes', {})
        if semantic_changes.get('security_updates'):
            analysis['focus_areas'].append("보안 업데이트 감지")
            analysis['recommendations'].append("보안 변경사항에 대한 추가 검토가 필요합니다")
        
        if semantic_changes.get('performance_improvements'):
            analysis['focus_areas'].append("성능 개선 감지")
            analysis['recommendations'].append("성능 개선 효과를 측정하고 문서화하세요")
        
        return analysis
    
    async def _gather_external_codebase_context(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """외부 코드베이스에서 관련 컨텍스트 수집"""
        external_context = {
            'best_practices': [],
            'security_patterns': [],
            'performance_insights': [],
            'common_issues': [],
            'framework_guidelines': [],
            'library_documentation': []
        }
        
        try:
            # 코드에서 주요 키워드 추출
            keywords = self._extract_code_keywords(code, language)
            
            # 각 키워드에 대해 외부 검색 수행
            for keyword in keywords[:5]:  # 최대 5개 키워드만 검색
                try:
                    # 보안 패턴 검색
                    security_info = await self._search_security_patterns(keyword, language)
                    if security_info:
                        external_context['security_patterns'].extend(security_info)
                    
                    # 모범 사례 검색
                    best_practices = await self._search_best_practices(keyword, language)
                    if best_practices:
                        external_context['best_practices'].extend(best_practices)
                    
                    # 성능 인사이트 검색
                    performance_info = await self._search_performance_insights(keyword, language)
                    if performance_info:
                        external_context['performance_insights'].extend(performance_info)
                    
                    # 일반적인 이슈 검색
                    common_issues = await self._search_common_issues(keyword, language)
                    if common_issues:
                        external_context['common_issues'].extend(common_issues)
                        
                except Exception as e:
                    logger.warning(f"키워드 '{keyword}' 검색 실패: {e}")
                    continue
            
            # 중복 제거 및 정리
            for key in external_context:
                external_context[key] = list(set(external_context[key]))[:10]  # 최대 10개씩만 유지
            
            logger.info(f"외부 컨텍스트 수집 완료: {sum(len(v) for v in external_context.values())}개 항목")
            return external_context
            
        except Exception as e:
            logger.error(f"외부 컨텍스트 수집 실패: {e}")
            return external_context
    
    def _extract_code_keywords(self, code: str, language: str) -> List[str]:
        """코드에서 검색할 키워드 추출"""
        keywords = []
        
        # 언어별 주요 패턴 추출
        if language.lower() == 'python':
            # Python 함수, 클래스, import 추출
            import re
            functions = re.findall(r'def\s+(\w+)', code)
            classes = re.findall(r'class\s+(\w+)', code)
            imports = re.findall(r'import\s+(\w+)', code)
            keywords.extend(functions + classes + imports)
        
        elif language.lower() == 'javascript':
            # JavaScript 함수, 변수, import 추출
            import re
            functions = re.findall(r'function\s+(\w+)', code)
            consts = re.findall(r'const\s+(\w+)', code)
            imports = re.findall(r'import.*?from\s+[\'"]([^\'"]+)[\'"]', code)
            keywords.extend(functions + consts + imports)
        
        elif language.lower() == 'java':
            # Java 클래스, 메서드 추출
            import re
            classes = re.findall(r'class\s+(\w+)', code)
            methods = re.findall(r'public\s+\w+\s+(\w+)\s*\(', code)
            keywords.extend(classes + methods)
        
        # 일반적인 프로그래밍 키워드
        common_keywords = ['api', 'database', 'security', 'auth', 'config', 'error', 'exception', 'test']
        keywords.extend(common_keywords)
        
        # 중복 제거 및 길이 제한
        keywords = list(set(keywords))
        keywords = [k for k in keywords if len(k) > 2 and len(k) < 20]
        
        return keywords[:10]  # 최대 10개 키워드만 반환
    
    async def _search_security_patterns(self, keyword: str, language: str) -> List[str]:
        """보안 패턴 검색 (보안 강화된 GitHub + 웹 검색)"""
        results = []
        
        try:
            # 1. 보안 강화된 GitHub 서버에서 검색
            github_tool = next((t for t in self.tools if "github_secure" in t.name.lower()), None)
            if github_tool:
                try:
                    # 보안 검증 후 실행
                    if self._validate_mcp_permissions("github_secure", "search_code"):
                        self._log_mcp_activity("github_secure", "search_security_patterns", {"keyword": keyword})
                        
                        github_result = github_tool.invoke({
                            "action": "search_code",
                            "query": f"{keyword} security {language}",
                            "language": language,
                            "read_only": True
                        })
                        if github_result and isinstance(github_result, str):
                            results.append(f"GitHub 보안 패턴: {github_result[:150]}")
                except Exception as e:
                    logger.warning(f"GitHub 보안 검색 실패: {e}")
                    self._log_mcp_activity("github_secure", "search_error", {"error": str(e)})
            
            # 2. 메모리 서버를 통한 안전한 검색
            memory_tool = next((t for t in self.tools if "memory_secure" in t.name.lower()), None)
            if memory_tool:
                try:
                    if self._validate_mcp_permissions("memory_secure", "search"):
                        self._log_mcp_activity("memory_secure", "search_security", {"keyword": keyword})
                        
                        memory_result = memory_tool.invoke({
                            "action": "search",
                            "query": f"{keyword} security patterns {language}",
                            "secure_mode": True
                        })
                        if memory_result and isinstance(memory_result, str):
                            results.append(f"메모리 보안 정보: {memory_result[:150]}")
                except Exception as e:
                    logger.warning(f"메모리 보안 검색 실패: {e}")
                    
        except Exception as e:
            logger.warning(f"보안 패턴 검색 실패: {e}")
            self._log_mcp_activity("security_search", "general_error", {"error": str(e)})
        
        return results[:2]  # 최대 2개 결과만 반환
    
    async def _search_best_practices(self, keyword: str, language: str) -> List[str]:
        """모범 사례 검색 (GitHub + 웹 검색)"""
        results = []
        
        try:
            # 1. GitHub에서 모범 사례 코드 검색
            github_tool = next((t for t in self.tools if "github" in t.name.lower()), None)
            if github_tool:
                try:
                    github_result = github_tool.invoke({
                        "action": "search_code",
                        "query": f"{keyword} best practice {language}",
                        "language": language
                    })
                    if github_result and isinstance(github_result, str):
                        results.append(f"GitHub 모범 사례: {github_result[:150]}")
                except Exception as e:
                    logger.warning(f"GitHub 모범 사례 검색 실패: {e}")
            
            # 2. 웹 검색
            search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
            if search_tool:
                query = f"{language} {keyword} best practices coding standards"
                result = search_tool.invoke({"query": query})
                if result and isinstance(result, str):
                    results.append(f"웹 모범 사례: {result[:150]}")
                    
        except Exception as e:
            logger.warning(f"모범 사례 검색 실패: {e}")
        
        return results[:2]
    
    async def _search_performance_insights(self, keyword: str, language: str) -> List[str]:
        """성능 인사이트 검색 (GitHub + 웹 검색)"""
        results = []
        
        try:
            # 1. GitHub에서 성능 관련 코드 검색
            github_tool = next((t for t in self.tools if "github" in t.name.lower()), None)
            if github_tool:
                try:
                    github_result = github_tool.invoke({
                        "action": "search_code",
                        "query": f"{keyword} performance optimization {language}",
                        "language": language
                    })
                    if github_result and isinstance(github_result, str):
                        results.append(f"GitHub 성능 패턴: {github_result[:150]}")
                except Exception as e:
                    logger.warning(f"GitHub 성능 검색 실패: {e}")
            
            # 2. 웹 검색
            search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
            if search_tool:
                query = f"{language} {keyword} performance optimization tips"
                result = search_tool.invoke({"query": query})
                if result and isinstance(result, str):
                    results.append(f"웹 성능 정보: {result[:150]}")
                    
        except Exception as e:
            logger.warning(f"성능 인사이트 검색 실패: {e}")
        
        return results[:2]
    
    async def _search_common_issues(self, keyword: str, language: str) -> List[str]:
        """일반적인 이슈 검색 (GitHub + 웹 검색)"""
        results = []
        
        try:
            # 1. GitHub에서 이슈 검색
            github_tool = next((t for t in self.tools if "github" in t.name.lower()), None)
            if github_tool:
                try:
                    github_result = github_tool.invoke({
                        "action": "search_issues",
                        "query": f"{keyword} {language} common problems",
                        "language": language
                    })
                    if github_result and isinstance(github_result, str):
                        results.append(f"GitHub 이슈: {github_result[:150]}")
                except Exception as e:
                    logger.warning(f"GitHub 이슈 검색 실패: {e}")
            
            # 2. 웹 검색
            search_tool = next((t for t in self.tools if "search" in t.name.lower()), None)
            if search_tool:
                query = f"{language} {keyword} common problems issues troubleshooting"
                result = search_tool.invoke({"query": query})
                if result and isinstance(result, str):
                    results.append(f"웹 이슈 정보: {result[:150]}")
                    
        except Exception as e:
            logger.warning(f"일반적인 이슈 검색 실패: {e}")
        
        return results[:2]
    
    def _log_mcp_activity(self, server_name: str, action: str, details: Dict[str, Any] = None):
        """MCP 서버 활동 감사 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "server": server_name,
            "action": action,
            "details": details or {},
            "security_level": "high" if "write" in action.lower() else "medium"
        }
        self.audit_log.append(log_entry)
        
        # 보안 위험 활동 감지
        if log_entry["security_level"] == "high":
            logger.warning(f"보안 위험 활동 감지: {server_name} - {action}")
        
        # 로그 크기 제한 (최대 1000개 항목)
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def _validate_mcp_permissions(self, server_name: str, action: str) -> bool:
        """MCP 서버 권한 검증"""
        server_config = self.server_configs.get(server_name, {})
        permissions = server_config.get("permissions", [])
        
        # 읽기 전용 모드에서 쓰기 작업 차단
        if "write" in action.lower() and "write" not in permissions:
            logger.error(f"권한 없음: {server_name}에서 {action} 실행 시도")
            return False
        
        # 샌드박스 모드 검증
        if not server_config.get("sandbox", False):
            logger.warning(f"샌드박스 모드 비활성화: {server_name}")
            return False
        
        return True
    
    def _create_sandbox_environment(self):
        """MCP 샌드박스 환경 생성"""
        import os
        import tempfile
        
        try:
            # 안전한 임시 디렉토리 생성
            sandbox_dir = "/tmp/mcp_sandbox"
            os.makedirs(sandbox_dir, exist_ok=True)
            
            # 권한 제한 (읽기 전용)
            os.chmod(sandbox_dir, 0o555)
            
            # 샌드박스 환경 변수 설정
            os.environ["MCP_SANDBOX_MODE"] = "true"
            os.environ["MCP_RESTRICTED_PATHS"] = sandbox_dir
            
            logger.info(f"샌드박스 환경 생성 완료: {sandbox_dir}")
            return True
            
        except Exception as e:
            logger.error(f"샌드박스 환경 생성 실패: {e}")
            return False
    
    def get_security_audit_log(self) -> List[Dict[str, Any]]:
        """보안 감사 로그 조회"""
        return self.audit_log.copy()
    
    def get_security_status(self) -> Dict[str, Any]:
        """보안 상태 조회"""
        return {
            "sandbox_enabled": all(config.get("sandbox", False) for config in self.server_configs.values()),
            "read_only_mode": all("read" in config.get("permissions", []) for config in self.server_configs.values()),
            "total_servers": len(self.server_configs),
            "secure_servers": len([s for s in self.server_configs.values() if s.get("sandbox", False)]),
            "audit_log_entries": len(self.audit_log),
            "security_incidents": len([log for log in self.audit_log if log.get("security_level") == "high"])
        }
    
    async def analyze_code(self, code: str, language: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """향상된 코드 분석 (외부 코드베이스 조회 포함)"""
        try:
            # 외부 코드베이스 조회
            external_context = await self._gather_external_codebase_context(code, language, context)
            
            # 기본 MCP 분석
            comprehensive_review = await self.get_comprehensive_review(code, language, context)
            
            # 변경사항 분석 추가
            detailed_changes = context.get('detailed_changes', {}) if context else {}
            change_analysis = self._analyze_changes_for_review(detailed_changes)
            
            # 외부 컨텍스트와 통합된 리뷰 생성
            review_content = self._generate_enhanced_review_content(
                comprehensive_review, 
                change_analysis, 
                detailed_changes,
                external_context
            )
            
            return {
                'analysis_type': 'mcp_enhanced_gemini_with_external_context',
                'result': {
                    'review': review_content,
                    'change_analysis': change_analysis,
                    'comprehensive_analysis': comprehensive_review,
                    'external_context': external_context
                },
                'github_metadata': {'status': 'success'},
                'comments_analysis': {'status': 'success'}
            }
            
        except Exception as e:
            logger.error(f"코드 분석 실패: {e}")
            return {
                'analysis_type': 'error',
                'result': {
                    'review': f"코드 분석 중 오류가 발생했습니다: {e}",
                    'error': str(e)
                },
                'github_metadata': {'status': 'error'},
                'comments_analysis': {'status': 'error'}
            }
    
    def _generate_enhanced_review_content(self, comprehensive_review: Dict[str, Any], 
                                        change_analysis: Dict[str, Any], 
                                        detailed_changes: Dict[str, Any],
                                        external_context: Dict[str, Any] = None) -> str:
        """향상된 리뷰 내용 생성"""
        review_parts = []
        
        # 기본 AI 분석 결과
        if comprehensive_review.get('summary', {}).get('final_review'):
            review_parts.append("### 🤖 AI 코드 분석")
            review_parts.append(comprehensive_review['summary']['final_review'])
            review_parts.append("")
        
        # 변경사항 기반 분석
        if change_analysis.get('critical_issues'):
            review_parts.append("### ⚠️ 중요 이슈")
            for issue in change_analysis['critical_issues']:
                review_parts.append(f"- {issue}")
            review_parts.append("")
        
        if change_analysis.get('focus_areas'):
            review_parts.append("### 🎯 집중 검토 영역")
            for area in change_analysis['focus_areas']:
                review_parts.append(f"- {area}")
            review_parts.append("")
        
        if change_analysis.get('recommendations'):
            review_parts.append("### 💡 권장사항")
            for rec in change_analysis['recommendations']:
                review_parts.append(f"- {rec}")
            review_parts.append("")
        
        # 상세 변경사항 요약
        if detailed_changes.get('summary'):
            summary = detailed_changes['summary']
            review_parts.append("### 📊 변경사항 요약")
            review_parts.append(f"- **총 파일 수**: {summary.get('total_files', 0)}개")
            review_parts.append(f"- **추가된 라인**: {summary.get('total_additions', 0)}줄")
            review_parts.append(f"- **삭제된 라인**: {summary.get('total_deletions', 0)}줄")
            review_parts.append(f"- **커밋 수**: {summary.get('commits_count', 0)}개")
            review_parts.append("")
        
        # 외부 코드베이스 컨텍스트 (새로운 기능)
        if external_context:
            review_parts.append("### 🌐 외부 코드베이스 인사이트")
            
            if external_context.get('security_patterns'):
                review_parts.append("#### 🔒 보안 관련 정보")
                for pattern in external_context['security_patterns'][:3]:
                    review_parts.append(f"- {pattern}")
                review_parts.append("")
            
            if external_context.get('best_practices'):
                review_parts.append("#### ✅ 모범 사례")
                for practice in external_context['best_practices'][:3]:
                    review_parts.append(f"- {practice}")
                review_parts.append("")
            
            if external_context.get('performance_insights'):
                review_parts.append("#### ⚡ 성능 최적화 팁")
                for insight in external_context['performance_insights'][:3]:
                    review_parts.append(f"- {insight}")
                review_parts.append("")
            
            if external_context.get('common_issues'):
                review_parts.append("#### ⚠️ 주의사항")
                for issue in external_context['common_issues'][:3]:
                    review_parts.append(f"- {issue}")
                review_parts.append("")
        
        return "\n".join(review_parts) if review_parts else "변경사항 분석을 완료했습니다."
    
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