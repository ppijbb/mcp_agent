import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Awaitable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from datetime import datetime
from enum import Enum

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorCategory(Enum):
    """에러 카테고리"""
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    BUSINESS_LOGIC_ERROR = "business_logic_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler:
    """에러 처리 핸들러"""
    
    def __init__(self):
        """ErrorHandler 초기화"""
        self.error_log: list = []
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        에러 처리 및 로깅
        
        Args:
            error: 발생한 예외
            context: 에러 발생 컨텍스트
            severity: 에러 심각도
            category: 에러 카테고리
            metadata: 추가 메타데이터
        
        Returns:
            에러 정보 딕셔너리
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity.value,
            "category": category.value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # 에러 로깅
        log_level = self._get_log_level(severity)
        logger.log(log_level, f"Error in {context}: {error}", exc_info=True)
        
        # 에러 카운트 업데이트
        error_key = f"{category.value}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 에러 로그에 추가
        self.error_log.append(error_info)
        
        # 심각한 에러는 알림
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_alert(error_info)
        
        return error_info
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """심각도에 따른 로그 레벨 반환"""
        severity_map = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.WARNING)
    
    def _send_alert(self, error_info: Dict[str, Any]):
        """심각한 에러 알림 전송"""
        logger.critical(f"ALERT: {error_info}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        return {
            "total_errors": len(self.error_log),
            "error_counts": self.error_counts,
            "recent_errors": self.error_log[-10:] if self.error_log else [],
        }


class RetryHandler:
    """재시도 핸들러"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0
    ):
        """
        RetryHandler 초기화
        
        Args:
            max_retries: 최대 재시도 횟수
            initial_delay: 초기 지연 시간 (초)
            backoff_factor: 지연 시간 배수
            max_delay: 최대 지연 시간 (초)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    async def retry_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        retry_on: type = Exception,
        **kwargs
    ) -> T:
        """
        재시도 로직이 포함된 비동기 함수 호출
        
        Args:
            func: 호출할 비동기 함수
            *args: 함수 인자
            retry_on: 재시도할 예외 타입
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 반환값
        
        Raises:
            Exception: 최대 재시도 횟수 초과 시
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except retry_on as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded")
        
        raise last_exception

# Import necessary classes from other modules
from .state import HSPAgentState
from .vector_store import HSPVectorStore
from ..autogen.agents import HSPAutoGenAgents
from ..mcp.manager import MCPServerManager

class HSPLangGraphWorkflow:
    """LangGraph 기반 메인 워크플로우"""
    
    def __init__(self, autogen_agents: Optional[HSPAutoGenAgents] = None, 
                 mcp_manager: Optional[MCPServerManager] = None,
                 vector_store: Optional[HSPVectorStore] = None):
        self.autogen_agents = autogen_agents or HSPAutoGenAgents()
        self.mcp_manager = mcp_manager or MCPServerManager()
        self.vector_store = vector_store or HSPVectorStore()
        self.a2a_bridge = None  # Will be set during workflow run
        self.error_handler = ErrorHandler()
        self.retry_handler = RetryHandler(max_retries=3, initial_delay=1.0)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(HSPAgentState)
        
        # 노드 정의
        workflow.add_node("initialize_session", self._initialize_session)
        workflow.add_node("collect_user_preferences", self._collect_user_preferences)
        workflow.add_node("determine_question_completeness", self._determine_question_completeness)
        workflow.add_node("analyze_user_profile", self._analyze_user_profile)
        workflow.add_node("discover_hobbies", self._discover_hobbies)
        workflow.add_node("integrate_schedule", self._integrate_schedule)
        workflow.add_node("match_communities", self._match_communities)
        workflow.add_node("track_progress", self._track_progress)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("autogen_consensus", self._autogen_consensus)
        
        # 엣지 정의 (조건부 라우팅)
        workflow.set_entry_point("initialize_session")
        
        # 대화형 상호작용 플로우
        workflow.add_edge("initialize_session", "collect_user_preferences")
        workflow.add_edge("collect_user_preferences", "determine_question_completeness")
        
        # 정보 수집 완성도에 따른 라우팅
        workflow.add_conditional_edges(
            "determine_question_completeness",
            self._route_conversation_completeness,
            {
                "continue_conversation": "collect_user_preferences",
                "proceed_to_analysis": "analyze_user_profile"
            }
        )
        
        workflow.add_edge("analyze_user_profile", "autogen_consensus")
        
        workflow.add_conditional_edges(
            "autogen_consensus",
            self._route_next_step,
            {
                "discover_hobbies": "discover_hobbies",
                "integrate_schedule": "integrate_schedule",
                "match_communities": "match_communities",
                "track_progress": "track_progress",
                "generate_insights": "generate_insights",
                "END": END
            }
        )
        
        # 모든 노드에서 다시 합의로 돌아갈 수 있음
        workflow.add_edge("discover_hobbies", "autogen_consensus")
        workflow.add_edge("integrate_schedule", "autogen_consensus")
        workflow.add_edge("match_communities", "autogen_consensus")
        workflow.add_edge("track_progress", "autogen_consensus")
        workflow.add_edge("generate_insights", END)
        
        return workflow.compile(checkpointer=InMemorySaver())
    
    async def _initialize_session(self, state: HSPAgentState) -> HSPAgentState:
        """세션 초기화"""
        print("---Initializing Session---")
        
        # A2A 브리지에 LangGraph 워크플로우 등록
        if self.a2a_bridge:
            await self.a2a_bridge.register_agent(
                agent_id="LangGraphWorkflow",
                agent_type="workflow_orchestrator",
                framework="langgraph"
            )
        
        # 모든 초기화는 빈 값으로 시작
        state["agent_session"] = {"session_id": f"hsp_session_{datetime.now().isoformat()}"}
        state["workflow_context"] = {"started_at": datetime.now().isoformat()}
        state["current_decision_point"] = "profile_analysis"
        state["user_profile"] = {}
        state["hobby_recommendations"] = []
        state["schedule_analysis"] = {}
        state["community_matches"] = []
        state["progress_metrics"] = {}
        state["weekly_journal"] = ""
        state["error_log"] = []
        state["agent_consensus"] = {}
        state["mcp_responses"] = {}
        state["a2a_messages"] = []
        
        # 대화 관련 필드 초기화
        if "conversation_history" not in state:
            state["conversation_history"] = []
        if "collected_preferences" not in state:
            state["collected_preferences"] = {}
        if "question_completeness_score" not in state:
            state["question_completeness_score"] = 0.0
        if "current_question" not in state:
            state["current_question"] = None
        if "waiting_for_user_response" not in state:
            state["waiting_for_user_response"] = False
        
        return state
    
    async def _collect_user_preferences(self, state: HSPAgentState) -> HSPAgentState:
        """사용자와의 대화형 질문-답변 진행"""
        print("---Collecting User Preferences---")
        
        try:
            # 사용자 입력이 있는 경우 (답변)
            user_input = state.get("user_input", "")
            conversation_history = state.get("conversation_history", [])
            collected_preferences = state.get("collected_preferences", {})
            
            # 적응형 질문 생성
            question_result = await self.autogen_agents.generate_adaptive_question(
                conversation_history=conversation_history,
                collected_preferences=collected_preferences,
                user_input=user_input
            )
            
            # 사용자 답변이 있는 경우 히스토리에 추가
            if user_input and conversation_history:
                conversation_history[-1]["answer"] = user_input
                collected_preferences = question_result["collected_preferences"]
            
            # 새로운 질문을 히스토리에 추가
            new_question_entry = {
                "question": question_result["next_question"],
                "category": question_result["category"],
                "timestamp": datetime.now().isoformat()
            }
            conversation_history.append(new_question_entry)
            
            # 상태 업데이트
            state["conversation_history"] = conversation_history
            state["collected_preferences"] = collected_preferences
            state["current_question"] = question_result["next_question"]
            state["question_completeness_score"] = question_result["completeness_score"]
            state["waiting_for_user_response"] = True
            
            # 수집된 정보를 user_profile에 반영
            if collected_preferences:
                state["user_profile"].update(collected_preferences)
            
            logger.info(f"Question generated for session {state.get('session_id', 'unknown')}: "
                       f"completeness={question_result['completeness_score']:.2f}")
            
            return state
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, "collect_user_preferences",
                ErrorSeverity.MEDIUM, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"session_id": state.get("session_id", "unknown")}
            )
            state["error_log"].append(f"Preference collection failed: {str(e)}")
            state["error_log"].append(f"Error details: {error_info.get('error_message', str(e))}")
            # 기본 질문으로 fallback
            state["current_question"] = "안녕하세요! 취미를 찾는 데 도움을 드리겠습니다. 나이를 알려주세요."
            state["waiting_for_user_response"] = True
            return state
    
    async def _determine_question_completeness(self, state: HSPAgentState) -> HSPAgentState:
        """수집된 정보가 충분한지 판단"""
        print("---Determining Question Completeness---")
        
        try:
            completeness_score = state.get("question_completeness_score", 0.0)
            collected_preferences = state.get("collected_preferences", {})
            
            # 최소 완성도 임계값 (0.7 = 70%)
            min_completeness = 0.7
            
            # 필수 정보 확인
            required_fields = ["age", "occupation", "available_days"]
            has_required = all(field in collected_preferences for field in required_fields)
            
            # 완성도 판단
            is_complete = completeness_score >= min_completeness and has_required
            
            state["question_completeness_score"] = completeness_score
            
            if is_complete:
                logger.info(f"Information collection complete: score={completeness_score:.2f}")
                state["waiting_for_user_response"] = False
            else:
                logger.info(f"Information collection incomplete: score={completeness_score:.2f}, "
                          f"required_fields={has_required}")
            
            return state
            
        except Exception as e:
            logger.error(f"완성도 판단 실패: {e}")
            state["error_log"].append(f"Completeness determination failed: {str(e)}")
            return state
    
    def _route_conversation_completeness(self, state: HSPAgentState) -> str:
        """대화 완성도에 따른 라우팅"""
        completeness_score = state.get("question_completeness_score", 0.0)
        min_completeness = 0.7
        collected_preferences = state.get("collected_preferences", {})
        required_fields = ["age", "occupation", "available_days"]
        has_required = all(field in collected_preferences for field in required_fields)
        
        if completeness_score >= min_completeness and has_required:
            return "proceed_to_analysis"
        else:
            return "continue_conversation"

    async def _analyze_user_profile(self, state: HSPAgentState) -> HSPAgentState:
        print("---Analyzing User Profile---")
        
        try:
            # MCP 서버를 통해 사용자 데이터 수집
            mcp_results = {}
            
            # 구글 캘린더에서 스케줄 정보 가져오기 (재시도 로직 포함)
            try:
                calendar_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "google_calendar", 
                    "list_events", 
                    {"timeframe": "next_month"}
                )
                mcp_results["calendar"] = calendar_data
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "analyze_user_profile.calendar",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                mcp_results["calendar"] = {"error": error_info}
            
            # 소셜 미디어에서 관심사 분석 (재시도 로직 포함)
            try:
                social_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "social_search",
                    "search_groups", 
                    {"user_interests": state.get("user_profile", {}).get("interests", [])}
                )
                mcp_results["social"] = social_data
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "analyze_user_profile.social",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                mcp_results["social"] = {"error": error_info}
            
            state["mcp_responses"].update(mcp_results)
            state["current_decision_point"] = "hobby_discovery"
            
            # A2A 메시지로 프로필 분석 완료 알림
            if self.a2a_bridge:
                await self._send_a2a_update(state, "profile_analysis_complete", mcp_results)
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, "analyze_user_profile",
                ErrorSeverity.HIGH, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"state": state.get("session_id", "unknown")}
            )
            state["error_log"].append(f"Profile analysis failed: {str(e)}")
            state["mcp_responses"]["profile_analysis_error"] = error_info
        
        return state

    async def _discover_hobbies(self, state: HSPAgentState) -> HSPAgentState:
        print("---Discovering Hobbies---")
        
        user_profile = state.get("user_profile", {})
        hobby_recommendations = []
        
        # 벡터 스토어에서 유사 취미 검색
        try:
            query = f"사용자 관심사: {', '.join(user_profile.get('interests', []))}"
            vector_results = self.vector_store.search_similar_hobbies(
                query, user_profile, top_k=10
            )
            hobby_recommendations.extend(vector_results)
        except Exception as e:
            logger.error(f"벡터 스토어 검색 실패: {e}")
        
        # 교육 플랫폼에서 취미 관련 강의 검색
        education_data = await self.mcp_manager.call_mcp_server(
            "education",
            "search_courses",
            {"user_profile": user_profile}
        )
        
        # 전자상거래에서 관련 용품 검색
        ecommerce_data = await self.mcp_manager.call_mcp_server(
            "ecommerce",
            "search_products", 
            {"hobby_categories": [h.get("hobby_name", "") for h in hobby_recommendations[:5]]}
        )
        
        # 취미 추천 결과 업데이트
        state["hobby_recommendations"] = hobby_recommendations
        state["mcp_responses"].update({
            "education": education_data,
            "ecommerce": ecommerce_data,
            "vector_search": vector_results if 'vector_results' in locals() else []
        })
        state["current_decision_point"] = "integrate_schedule"
        
        if self.a2a_bridge:
            await self._send_a2a_update(state, "hobby_discovery_complete", {
                "education": education_data,
                "ecommerce": ecommerce_data,
                "vector_results": vector_results if 'vector_results' in locals() else []
            })
        
        return state

    async def _integrate_schedule(self, state: HSPAgentState) -> HSPAgentState:
        print("---Integrating Schedule---")
        
        try:
            # 날씨 정보로 야외 활동 계획 (재시도 로직 포함)
            try:
                weather_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "weather_api",
                    "forecast",
                    {"location": state.get("user_profile", {}).get("location", "Seoul")}
                )
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "integrate_schedule.weather",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                weather_data = {"error": error_info}
            
            # 구글 맵스로 근처 취미 장소 검색 (재시도 로직 포함)
            try:
                maps_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "google_maps",
                    "search_places",
                    {"hobby_types": [h.get("hobby_name", "") if isinstance(h, dict) else str(h) for h in state.get("hobby_recommendations", [])[:5]]}
                )
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "integrate_schedule.maps",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                maps_data = {"error": error_info}
            
            state["mcp_responses"].update({
                "weather": weather_data,
                "maps": maps_data
            })
            state["current_decision_point"] = "match_communities"
            
            if self.a2a_bridge:
                await self._send_a2a_update(state, "schedule_integration_complete", {
                    "weather": weather_data,
                    "maps": maps_data
                })
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, "integrate_schedule",
                ErrorSeverity.HIGH, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"state": state.get("session_id", "unknown")}
            )
            state["error_log"].append(f"Schedule integration failed: {str(e)}")
        
        return state

    async def _match_communities(self, state: HSPAgentState) -> HSPAgentState:
        print("---Matching Communities---")
        
        try:
            user_profile = state.get("user_profile", {})
            hobby_recommendations = state.get("hobby_recommendations", [])
            community_matches = []
            vector_communities = []
            
            # 벡터 스토어에서 커뮤니티 검색
            try:
                hobby_categories = [h.get("category", "") for h in hobby_recommendations if isinstance(h, dict)]
                query = f"취미 카테고리: {', '.join(hobby_categories)}"
                vector_communities = self.vector_store.search_communities(
                    query, user_profile, top_k=5
                )
                community_matches.extend(vector_communities)
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "match_communities.vector_search",
                    ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_ERROR
                )
                logger.error(f"벡터 스토어 커뮤니티 검색 실패: {e}")
            
            # 소셜 미디어에서 관련 그룹 찾기 (재시도 로직 포함)
            try:
                community_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "social_search",
                    "find_communities",
                    {"hobbies": [h.get("hobby_name", "") if isinstance(h, dict) else str(h) for h in hobby_recommendations[:5]]}
                )
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "match_communities.social_search",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                community_data = {"error": error_info}
            
            # 커뮤니티 매칭 결과 업데이트
            state["community_matches"] = community_matches
            state["mcp_responses"]["communities"] = community_data
            state["mcp_responses"]["vector_communities"] = vector_communities
            state["current_decision_point"] = "track_progress"
            
            if self.a2a_bridge:
                await self._send_a2a_update(state, "community_matching_complete", {
                    "mcp_communities": community_data,
                    "vector_communities": vector_communities
                })
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, "match_communities",
                ErrorSeverity.HIGH, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"state": state.get("session_id", "unknown")}
            )
            state["error_log"].append(f"Community matching failed: {str(e)}")
        
        return state

    async def _track_progress(self, state: HSPAgentState) -> HSPAgentState:
        print("---Tracking Progress---")
        
        # 피트니스 트래커로 운동 관련 취미 추적
        fitness_data = await self.mcp_manager.call_mcp_server(
            "fitness_tracker",
            "get_stats",
            {"user_id": state.get("user_profile", {}).get("user_id", "")}
        )
        
        state["mcp_responses"]["fitness"] = fitness_data
        state["current_decision_point"] = "generate_insights"
        
        if self.a2a_bridge:
            await self._send_a2a_update(state, "progress_tracking_complete", fitness_data)
        
        return state

    async def _generate_insights(self, state: HSPAgentState) -> HSPAgentState:
        print("---Generating Insights---")
        
        try:
            # 모든 MCP 응답을 종합하여 최종 인사이트 생성
            all_mcp_data = state.get("mcp_responses", {})
            
            # 음악/독서 플랫폼에서 추가 추천 (재시도 로직 포함)
            try:
                music_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "music_platform",
                    "get_recommendations",
                    {"user_preferences": state.get("user_profile", {})}
                )
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "generate_insights.music",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                music_data = {"error": error_info}
            
            try:
                reading_data = await self.retry_handler.retry_async(
                    self.mcp_manager.call_mcp_server,
                    "reading_platform", 
                    "search_books",
                    {"interests": state.get("hobby_recommendations", [])}
                )
            except Exception as e:
                error_info = self.error_handler.handle_error(
                    e, "generate_insights.reading",
                    ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_SERVICE_ERROR
                )
                reading_data = {"error": error_info}
            
            state["mcp_responses"].update({
                "music": music_data,
                "reading": reading_data
            })
            
            # 모든 MCP 데이터를 로깅 (디버깅용)
            logger.info(f"Generated insights with {len(all_mcp_data)} MCP data sources")
            
            if self.a2a_bridge:
                await self._send_a2a_update(state, "insights_generated", {
                    "music": music_data,
                    "reading": reading_data,
                    "summary": "Final hobby recommendations generated",
                    "total_mcp_sources": len(all_mcp_data)
                })
            
        except Exception as e:
            error_info = self.error_handler.handle_error(
                e, "generate_insights",
                ErrorSeverity.HIGH, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"state": state.get("session_id", "unknown")}
            )
            state["error_log"].append(f"Insights generation failed: {str(e)}")
        
        return state

    def _route_next_step(self, state: HSPAgentState) -> str:
        """다음 단계 라우팅 - 에이전트 합의에 기반"""
        consensus = state.get("agent_consensus", {})
        next_step = consensus.get("next_step", "END")
        print(f"---Routing to {next_step}---")
        return next_step
    
    async def _autogen_consensus(self, state: HSPAgentState) -> HSPAgentState:
        """AutoGen 에이전트들의 합의 과정"""
        current_decision = state.get("current_decision_point", "")
        print(f"---AutoGen Consensus for {current_decision}---")
        
        # 의사결정 포인트에 따라 관련 에이전트 선택
        relevant_agents = self._select_relevant_agents(current_decision)
        
        if not relevant_agents:
            state["agent_consensus"] = {"next_step": "END"}
            return state

        # A2A 브리지를 통한 AutoGen 합의 실행
        session_id = state.get("agent_session", {}).get("session_id", "")
        
        try:
            # AutoGen 에이전트들과 합의 진행
            consensus_result = await self.autogen_agents.run_consensus(
                agents=relevant_agents,
                topic=current_decision,
                context={
                    "current_state": current_decision,
                    "mcp_responses": state.get("mcp_responses", {}),
                    "user_profile": state.get("user_profile", {}),
                    "workflow_context": state.get("workflow_context", {})
                },
                user_profile=state.get("user_profile", {}),
                bridge=self.a2a_bridge,
                session_id=session_id
            )
            
            # 합의 결과를 상태에 저장
            state["agent_consensus"] = consensus_result
            
            # 다음 단계 결정 (에이전트 합의 기반)
            if consensus_result.get("consensus_reached", False):
                # 실제 합의 결과에서 다음 단계 추출
                next_step = self._extract_next_step_from_consensus(consensus_result, current_decision)
            else:
                # 기본 순서대로 진행
                next_step_mapping = {
                    "profile_analysis": "discover_hobbies",
                    "hobby_discovery": "integrate_schedule", 
                    "integrate_schedule": "match_communities",
                    "match_communities": "track_progress",
                    "track_progress": "generate_insights",
                    "generate_insights": "END",
                }
                next_step = next_step_mapping.get(current_decision, "END")
            
            state["agent_consensus"]["next_step"] = next_step
            
        except Exception as e:
            # 에러 처리: 빈 값 fallback
            error_info = self.error_handler.handle_error(
                e, f"autogen_consensus.{current_decision}",
                ErrorSeverity.HIGH, ErrorCategory.BUSINESS_LOGIC_ERROR,
                {"session_id": session_id, "current_decision": current_decision}
            )
            state["error_log"].append(f"Consensus failed for {current_decision}: {str(e)}")
            
            # 기본 다음 단계로 진행
            next_step_mapping = {
                "profile_analysis": "discover_hobbies",
                "hobby_discovery": "integrate_schedule",
                "integrate_schedule": "match_communities", 
                "match_communities": "track_progress",
                "track_progress": "generate_insights",
                "generate_insights": "END",
            }
            
            state["agent_consensus"] = {
                "next_step": next_step_mapping.get(current_decision, "END"),
                "error": "Consensus failed, using default routing",
                "error_info": error_info
            }
        
        return state
    
    def _select_relevant_agents(self, decision_point: str) -> List[str]:
        """의사결정 포인트에 따른 관련 에이전트 선택"""
        agent_mapping = {
            "profile_analysis": ["profile_analyst", "decision_moderator"],
            "hobby_discovery": ["hobby_discoverer", "profile_analyst", "decision_moderator"],
            "schedule_integration": ["schedule_integrator", "decision_moderator"],
            "community_matching": ["community_matcher", "decision_moderator"],
            "progress_tracking": ["progress_tracker", "decision_moderator"],
            "final_insights": ["progress_tracker", "decision_moderator"]
        }
        return agent_mapping.get(decision_point, [])
    
    async def _send_a2a_update(self, state: HSPAgentState, update_type: str, data: Any):
        """A2A 브리지를 통해 업데이트 메시지 전송"""
        if not self.a2a_bridge:
            return
        
        from ..bridge.a2a_bridge import A2AMessage
        
        session_id = state.get("agent_session", {}).get("session_id", "")
        
        message = A2AMessage(
            sender_agent="LangGraphWorkflow",
            receiver_agent="AutoGenConsensus",
            message_type=update_type,
            payload=data,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        state["a2a_messages"].append(message.__dict__)
        await self.a2a_bridge.send_message(message)
    
    def _extract_next_step_from_consensus(self, consensus_result: Dict[str, Any], current_decision: str) -> str:
        """합의 결과에서 다음 단계 추출"""
        try:
            # DecisionModerator의 최종 결론에서 다음 단계 찾기
            final_consensus = consensus_result.get("final_consensus", "")
            
            # 키워드 매칭으로 다음 단계 결정
            if "취미 발견" in final_consensus or "hobby" in final_consensus.lower():
                return "discover_hobbies"
            elif "스케줄" in final_consensus or "schedule" in final_consensus.lower():
                return "integrate_schedule"
            elif "커뮤니티" in final_consensus or "community" in final_consensus.lower():
                return "match_communities"
            elif "진행" in final_consensus or "progress" in final_consensus.lower():
                return "track_progress"
            elif "인사이트" in final_consensus or "insight" in final_consensus.lower():
                return "generate_insights"
            elif "완료" in final_consensus or "end" in final_consensus.lower():
                return "END"
            else:
                # 기본 순서 사용
                next_step_mapping = {
                    "profile_analysis": "discover_hobbies",
                    "hobby_discovery": "integrate_schedule",
                    "integrate_schedule": "match_communities",
                    "match_communities": "track_progress", 
                    "track_progress": "generate_insights",
                    "generate_insights": "END",
                }
                return next_step_mapping.get(current_decision, "END")
                
        except Exception:
            # 에러 시 기본 순서 사용
            next_step_mapping = {
                "profile_analysis": "discover_hobbies",
                "hobby_discovery": "integrate_schedule", 
                "integrate_schedule": "match_communities",
                "match_communities": "track_progress",
                "track_progress": "generate_insights",
                "generate_insights": "END",
            }
            return next_step_mapping.get(current_decision, "END")
    
    async def run_workflow(self, user_input: str, user_profile: Optional[Dict[str, Any]] = None,
                          preferences: Optional[Dict[str, Any]] = None, 
                          a2a_bridge=None, mcp_manager=None) -> Dict[str, Any]:
        """
        워크플로우 실행 (API에서 호출)
        
        Args:
            user_input: 사용자 입력
            user_profile: 사용자 프로필 (선택)
            preferences: 사용자 선호도 (선택)
            a2a_bridge: A2A 브리지 (선택)
            mcp_manager: MCP 매니저 (선택)
        
        Returns:
            워크플로우 실행 결과 딕셔너리
        """
        start_time = datetime.now()
        
        try:
            # 외부에서 전달받은 컴포넌트들 설정
            if a2a_bridge:
                self.a2a_bridge = a2a_bridge
            if mcp_manager:
                self.mcp_manager = mcp_manager
            
            # 초기 상태 구성
            initial_state = {
                "user_input": user_input,
                "user_profile": user_profile or {},
                "preferences": preferences or {},
                "agent_session": {},
                "workflow_context": {"started_at": start_time.isoformat()},
                "current_decision_point": "profile_analysis",
                "hobby_recommendations": [],
                "schedule_analysis": {},
                "community_matches": [],
                "progress_metrics": {},
                "weekly_journal": "",
                "error_log": [],
                "agent_consensus": {},
                "mcp_responses": {},
                "a2a_messages": [],
                "conversation_history": [],
                "collected_preferences": {},
                "question_completeness_score": 0.0,
                "current_question": None,
                "waiting_for_user_response": False
            }
            
            # 워크플로우 실행
            result = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"hsp_{start_time.isoformat()}"}}
            )
            
            # 실행 시간 계산
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 에러 통계 수집
            error_stats = self.error_handler.get_error_statistics()
            
            return {
                "success": True,
                "final_state": result,
                "recommendations": result.get("hobby_recommendations", []),
                "mcp_data": result.get("mcp_responses", {}),
                "consensus_history": result.get("agent_consensus", {}),
                "error_log": result.get("error_log", []),
                "performance_metrics": {
                    "execution_time_seconds": execution_time,
                    "error_count": error_stats.get("total_errors", 0),
                    "started_at": start_time.isoformat(),
                    "completed_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_info = self.error_handler.handle_error(
                e, "run_workflow",
                ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_ERROR,
                {"user_input": user_input[:100] if user_input else ""}
            )
            
            return {
                "success": False,
                "error": str(e),
                "error_info": error_info,
                "recommendations": [],
                "mcp_data": {},
                "consensus_history": {},
                "error_log": [f"Workflow execution failed: {str(e)}"],
                "performance_metrics": {
                    "execution_time_seconds": execution_time,
                    "failed_at": datetime.now().isoformat()
                }
            } 