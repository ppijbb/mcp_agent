"""
향상된 에이전트 모듈
MCP와 A2A 프로토콜을 지원하는 에이전트들을 정의합니다.
"""

import uuid
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.agents import create_tool_calling_agent
from langchain_core.runnables import Runnable

# 로깅 설정
logger = logging.getLogger(__name__)

from .mcp_integration import mcp_registry, mcp_executor
from .a2a_protocol import (
    A2AAgent, AgentCard, AgentCapability, A2AMessage, 
    MessageType, MessagePriority, a2a_message_broker
)
from .security import security_manager, privacy_manager, audit_logger
from .utils import search_tool, model, performance_manager, enhanced_search_tool

class EnhancedAgent(A2AAgent):
    """MCP와 A2A를 지원하는 향상된 에이전트 기본 클래스"""
    
    def __init__(self, agent_card: AgentCard, llm: ChatOpenAI, 
                 tools: List = None, system_prompt: str = ""):
        super().__init__(agent_card, a2a_message_broker)
        
        self.llm = llm
        self.tools = tools or []
        self.system_prompt = system_prompt
        
        # MCP 도구 등록
        self._register_mcp_tools()
        
        # 보안 세션 생성 (향상된 버전)
        self.security_context = None
        asyncio.create_task(self._initialize_security_context(agent_card))
        
        # 성능 모니터링 초기화
        self.performance_metrics = {}
        
        # 에이전트 시작
        asyncio.create_task(self.start())
    
    async def _initialize_security_context(self, agent_card: AgentCard):
        """보안 컨텍스트 초기화"""
        try:
            permissions = agent_card.capabilities[0].supported_operations if agent_card.capabilities else []
            
            # 향상된 보안 세션 생성
            self.security_context = await security_manager.create_secure_session(
                agent_id=agent_card.agent_id,
                user_id=f"agent_{agent_card.agent_id}",
                permissions=permissions,
                ip_address="127.0.0.1",  # 내부 에이전트
                user_agent=f"EnhancedAgent/{agent_card.version}"
            )
            
            logger.info(f"Security context initialized for agent {agent_card.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize security context: {str(e)}")
            # 기본 세션 생성
            self.security_context = security_manager.create_session(
                agent_id=agent_card.agent_id,
                permissions=permissions
            )
    
    def _register_mcp_tools(self):
        """MCP 도구들을 에이전트에 등록"""
        available_tools = mcp_registry.list_tools(self.agent_card.agent_id)
        for tool in available_tools:
            if tool.name not in [t.name for t in self.tools]:
                # MCP 도구를 LangChain 도구로 래핑
                from langchain_core.tools import tool
                
                @tool(name=tool.name, description=tool.description)
                async def mcp_tool_wrapper(**kwargs):
                    result = await mcp_executor.execute_tool(
                        tool_name=tool.name,
                        parameters=kwargs,
                        agent_id=self.agent_card.agent_id,
                        context={"agent_name": self.agent_card.name}
                    )
                    return result
                
                self.tools.append(mcp_tool_wrapper)
    
    async def _handle_request(self, message: A2AMessage):
        """요청 메시지 처리 - 하위 클래스에서 오버라이드"""
        try:
            # 감사 로그 기록
            audit_logger.log_access(
                agent_id=message.sender_id,
                resource=self.agent_card.name,
                action="request",
                success=True,
                details={"message_type": message.message_type.value}
            )
            
            # 요청 처리 로직은 하위 클래스에서 구현
            await super()._handle_request(message)
            
        except Exception as e:
            audit_logger.log_access(
                agent_id=message.sender_id,
                resource=self.agent_card.name,
                action="request",
                success=False,
                details={"error": str(e)}
            )
            raise
    
    @performance_manager.performance_monitor("agent_task_execution")
    async def execute_task(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """향상된 작업 실행"""
        try:
            # 보안 컨텍스트 검증
            if not self.security_context or not self.security_context.is_valid():
                await self._initialize_security_context(self.agent_card)
            
            # 속도 제한 확인
            if not security_manager.check_rate_limit(
                self.agent_card.agent_id, "task_execution", limit=50
            ):
                raise Exception("Rate limit exceeded for task execution")
            
            # 프라이버시 검사 및 데이터 처리
            privacy_result = await privacy_manager.process_data_with_privacy(
                task_description, 
                context={"agent_id": self.agent_card.agent_id, "operation": "task_execution"},
                operation="task_processing"
            )
            
            # 작업 실행 로직은 하위 클래스에서 구현
            result = await self._execute_task_internal(privacy_result["processed_data"], **kwargs)
            
            # 결과 암호화 (필요한 경우)
            if self.security_context and self.security_context.encryption_keys:
                encrypted_result = await security_manager.encrypt_message(
                    result, self.security_context
                )
                result = {"encrypted": True, "data": encrypted_result}
            
            # 향상된 감사 로그 기록
            await audit_logger.log_data_access(
                agent_id=self.agent_card.agent_id,
                data_type="task_result",
                operation="execute",
                data_size=len(str(result)),
                data_category=privacy_result["data_category"],
                privacy_level="high"
            )
            
            # 성능 메트릭 업데이트
            self.performance_metrics["last_task_duration"] = time.time()
            self.performance_metrics["total_tasks_executed"] = self.performance_metrics.get("total_tasks_executed", 0) + 1
            
            return result
            
        except Exception as e:
            # 향상된 보안 이벤트 로깅
            await audit_logger.log_security_event(
                event_type="task_execution_failed",
                severity="medium",
                agent_id=self.agent_card.agent_id,
                details={"error": str(e), "task": task_description},
                threat_score=0.3
            )
            raise
    
    async def _execute_task_internal(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """내부 작업 실행 - 하위 클래스에서 구현"""
        raise NotImplementedError("Subclasses must implement _execute_task_internal")

class EnhancedSupervisorAgent(EnhancedAgent):
    """향상된 감독관 에이전트"""
    
    def __init__(self):
        # 에이전트 카드 생성
        agent_card = AgentCard(
            agent_id=f"supervisor_{uuid.uuid4().hex[:8]}",
            name="Research Supervisor",
            description="리서치 프로젝트를 감독하고 검색 쿼리를 생성하는 에이전트",
            version="2.0.0",
            capabilities=[
                AgentCapability(
                    name="query_generation",
                    description="사용자 요청을 바탕으로 검색 쿼리 생성",
                    version="1.0",
                    supported_operations=["query_generation", "project_management"],
                    input_schema={"query": "string", "context": "string"},
                    output_schema={"queries": "array", "strategy": "string"}
                )
            ],
            contact_info={"protocol": "a2a", "endpoint": "internal"},
            security_requirements=["authentication", "authorization"],
            api_endpoints=["/api/supervisor/query", "/api/supervisor/strategy"],
            authentication_methods=["jwt", "session"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # 시스템 프롬프트
        system_prompt = """You are an enhanced research supervisor with MCP and A2A capabilities. 
        Your role is to understand user requests and generate targeted search queries.
        You can collaborate with other agents through the A2A protocol and use MCP tools for enhanced functionality."""
        
        super().__init__(agent_card, model, [], system_prompt)
    
    async def _execute_task_internal(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """감독관 작업 실행"""
        # 다른 에이전트들과 협업하여 검색 쿼리 생성
        queries = await self._generate_search_queries(task_description)
        
        # A2A 메시지로 결과 공유
        await self._notify_agents("search_queries_ready", {
            "queries": queries,
            "task": task_description
        })
        
        return {
            "queries": queries,
            "strategy": "collaborative_research",
            "collaboration_partners": await self._get_available_agents()
        }
    
    async def _generate_search_queries(self, task_description: str) -> List[str]:
        """검색 쿼리 생성"""
        # LLM을 사용한 쿼리 생성
        prompt = f"Generate 3-5 specific search queries for: {task_description}"
        response = self.llm.invoke(prompt)
        
        # 응답 파싱
        queries = [line.strip() for line in response.content.split('\n') if line.strip()]
        return queries[:5]  # 최대 5개
    
    async def _notify_agents(self, event_type: str, data: Dict[str, Any]):
        """다른 에이전트들에게 알림 전송"""
        notification = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_card.agent_id,
            receiver_id="*",  # 브로드캐스트
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.NORMAL,
            content={"event_type": event_type, "data": data},
            timestamp=datetime.now()
        )
        await self.send_message(notification)
    
    async def _get_available_agents(self) -> List[str]:
        """사용 가능한 에이전트 목록 조회"""
        agents = a2a_message_broker.list_agents()
        return [agent.name for agent in agents if agent.agent_id != self.agent_card.agent_id]

class EnhancedSearchAgent(EnhancedAgent):
    """향상된 검색 에이전트"""
    
    def __init__(self):
        agent_card = AgentCard(
            agent_id=f"search_{uuid.uuid4().hex[:8]}",
            name="Enhanced Search Agent",
            description="MCP 도구를 활용하여 고급 검색 기능을 제공하는 에이전트",
            version="2.0.0",
            capabilities=[
                AgentCapability(
                    name="web_search",
                    description="다양한 검색 엔진과 MCP 도구를 활용한 검색",
                    version="1.0",
                    supported_operations=["web_search", "data_collection"],
                    input_schema={"query": "string", "filters": "object"},
                    output_schema={"results": "array", "metadata": "object"}
                )
            ],
            contact_info={"protocol": "a2a", "endpoint": "internal"},
            security_requirements=["authentication", "rate_limiting"],
            api_endpoints=["/api/search/query", "/api/search/filters"],
            authentication_methods=["jwt", "session"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        system_prompt = """You are an enhanced search agent with MCP integration.
        You can use various search tools and collaborate with other agents through A2A protocol."""
        
        super().__init__(agent_card, model, [search_tool], system_prompt)
    
    @performance_manager.performance_monitor("search_task_execution")
    async def _execute_task_internal(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """향상된 검색 작업 실행"""
        queries = kwargs.get("queries", [task_description])
        
        # 성능 최적화된 병렬 검색 실행
        search_tasks = []
        for query in queries:
            # 백그라운드 작업으로 검색 추가
            await performance_manager.add_background_task(
                self._execute_single_search, query
            )
            task = asyncio.create_task(self._execute_single_search(query))
            search_tasks.append(task)
        
        # 모든 검색 완료 대기
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 결과 통합 및 정리
        consolidated_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                # 에러 처리
                consolidated_results.append({
                    "query": queries[i],
                    "error": str(result),
                    "status": "failed"
                })
            else:
                consolidated_results.append({
                    "query": queries[i],
                    "results": result,
                    "status": "success"
                })
        
        # 성능 메트릭 수집
        performance_data = await performance_manager.get_performance_metrics()
        
        return {
            "search_results": consolidated_results,
            "total_queries": len(queries),
            "successful_searches": len([r for r in consolidated_results if r["status"] == "success"]),
            "execution_time": datetime.now().isoformat(),
            "performance_metrics": performance_data
        }
    
    @performance_manager.performance_monitor("single_search_execution")
    async def _execute_single_search(self, query: str) -> List[Dict[str, Any]]:
        """향상된 단일 검색 실행"""
        try:
            # 캐시된 검색 결과 확인
            cache_key = f"search:{query}"
            cached_result = await performance_manager.async_cache_get(cache_key)
            if cached_result:
                logger.info(f"Using cached search result for query: {query}")
                return cached_result
            
            # 향상된 검색 도구 사용
            results = await enhanced_search_tool(query)
            
            # 결과 품질 평가
            quality_score = await self._evaluate_search_quality(results, query)
            
            search_result = {
                "query": query,
                "results": results,
                "quality_score": quality_score,
                "timestamp": datetime.now().isoformat()
            }
            
            # 결과 캐싱
            await performance_manager.async_cache_set(cache_key, search_result, 1800)  # 30분
            
            return search_result
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            raise
    
    async def _evaluate_search_quality(self, results: List[Dict[str, Any]], query: str) -> float:
        """검색 결과 품질 평가"""
        try:
            if not results:
                return 0.0
            
            # 간단한 품질 점수 계산
            score = 0.0
            
            # 결과 개수 기반 점수
            if len(results) >= 5:
                score += 0.3
            elif len(results) >= 3:
                score += 0.2
            elif len(results) >= 1:
                score += 0.1
            
            # 결과 내용 기반 점수 (간단한 키워드 매칭)
            query_words = set(query.lower().split())
            for result in results:
                if "content" in result:
                    content_words = set(result["content"].lower().split())
                    if query_words.intersection(content_words):
                        score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {str(e)}")
            return 0.5

class EnhancedAnalystAgent(EnhancedAgent):
    """향상된 분석 에이전트"""
    
    def __init__(self):
        agent_card = AgentCard(
            agent_id=f"analyst_{uuid.uuid4().hex[:8]}",
            name="Enhanced Information Analyst",
            description="검색 결과를 분석하고 품질을 평가하는 에이전트",
            version="2.0.0",
            capabilities=[
                AgentCapability(
                    name="data_analysis",
                    description="검색 결과 분석 및 품질 평가",
                    version="1.0",
                    supported_operations=["data_analysis", "quality_assessment"],
                    input_schema={"search_results": "object", "criteria": "object"},
                    output_schema={"analysis": "object", "recommendations": "array"}
                )
            ],
            contact_info={"protocol": "a2a", "endpoint": "internal"},
            security_requirements=["authentication", "data_privacy"],
            api_endpoints=["/api/analyst/analyze", "/api/analyst/quality"],
            authentication_methods=["jwt", "session"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        system_prompt = """You are an enhanced information analyst with A2A collaboration capabilities.
        Analyze search results and provide quality assessments with recommendations."""
        
        super().__init__(agent_card, model, [], system_prompt)
    
    async def _execute_task_internal(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """분석 작업 실행"""
        search_results = kwargs.get("search_results", {})
        
        # 검색 결과 분석
        analysis = await self._analyze_search_results(search_results)
        
        # 다른 에이전트들과 협업하여 개선 방안 제시
        recommendations = await self._get_collaborative_recommendations(analysis)
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "quality_score": self._calculate_quality_score(analysis)
        }
    
    async def _analyze_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """검색 결과 분석"""
        # LLM을 사용한 분석
        prompt = f"Analyze these search results for quality and relevance: {search_results}"
        response = self.llm.invoke(prompt)
        
        return {
            "content_quality": "high" if "CONTINUE" in response.content else "needs_improvement",
            "relevance_score": 0.8,
            "coverage_assessment": "comprehensive",
            "analysis": response.content
        }
    
    async def _get_collaborative_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """협업을 통한 개선 방안 제시"""
        # 다른 에이전트들에게 분석 결과 공유
        await self.send_request(
            receiver_id="*",
            content={
                "request_type": "improvement_suggestions",
                "analysis": analysis
            },
            priority=MessagePriority.HIGH
        )
        
        # 기본 권장사항
        return [
            "Expand search scope if quality is low",
            "Refine search queries for better precision",
            "Consider alternative information sources"
        ]
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """품질 점수 계산"""
        score = 0.0
        if analysis["content_quality"] == "high":
            score += 0.6
        if analysis["relevance_score"] > 0.7:
            score += 0.4
        return min(score, 1.0)

# 에이전트 팩토리 함수
def create_enhanced_agent(agent_type: str, **kwargs) -> EnhancedAgent:
    """향상된 에이전트 생성"""
    if agent_type == "supervisor":
        return EnhancedSupervisorAgent()
    elif agent_type == "search":
        return EnhancedSearchAgent()
    elif agent_type == "analyst":
        return EnhancedAnalystAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# 전역 에이전트 인스턴스들
enhanced_supervisor = EnhancedSupervisorAgent()
enhanced_search = EnhancedSearchAgent()
enhanced_analyst = EnhancedAnalystAgent()
