import asyncio
import logging
from typing import List, Dict, Any, Optional
from langgraph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from datetime import datetime

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary classes from other modules
from .state import HSPAgentState, StateManager
from ..autogen.agents import HSPAutoGenAgents
from ..autogen.decision_engine import AutoGenDecisionEngine
from ..mcp.manager import MCPServerManager, MCPManager

class OptimizedHSPWorkflow:
    """최적화된 워크플로우 - A2A 제거하고 직접 연동"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.autogen_engine = AutoGenDecisionEngine()
        self.mcp_manager = MCPManager()
        
    def create_workflow(self) -> StateGraph:
        """최적화된 워크플로우 생성"""
        workflow = StateGraph(HSPAgentState)
        
        # 단순화된 노드들
        workflow.add_node("profile_analysis", self.analyze_profile)
        workflow.add_node("hobby_discovery", self.discover_hobbies)
        workflow.add_node("community_matching", self.match_communities)
        workflow.add_node("final_recommendation", self.generate_recommendations)
        
        # 효율적인 라우팅
        workflow.add_edge("profile_analysis", "hobby_discovery")
        workflow.add_edge("hobby_discovery", "community_matching")
        workflow.add_edge("community_matching", "final_recommendation")
        workflow.add_edge("final_recommendation", END)
        
        workflow.set_entry_point("profile_analysis")
        return workflow.compile()
    
    async def analyze_profile(self, state: HSPAgentState) -> HSPAgentState:
        """사용자 프로필 분석 - 직접 AutoGen 호출"""
        try:
            # AutoGen에 직접 요청 (A2A 우회)
            profile_prompt = f"사용자 입력을 분석하여 프로필을 생성해주세요: {state['user_input']}"
            profile_result = await self.autogen_engine.analyze_user_profile(profile_prompt)
            
            # 상태 업데이트 (캐시 활용)
            state["current_step"] = "hobby_discovery"
            state["user_profile"] = {
                "interests": profile_result.get("interests", []),
                "skill_level": profile_result.get("skill_level", "beginner"),
                "time_availability": profile_result.get("time_availability", "weekend")
            }
            
            # 현재 단계 결과 저장
            state["current_step_result"] = profile_result
            
            # 상태 압축 저장
            self.state_manager.save_state(state)
            
            logger.info(f"Profile analysis completed for session {state['session_id']}")
            return state
            
        except Exception as e:
            state["error_context"] = f"프로필 분석 실패: {str(e)}"
            logger.error(f"Profile analysis failed: {e}")
            return state
    
    async def discover_hobbies(self, state: HSPAgentState) -> HSPAgentState:
        """취미 발견 - 병렬 MCP 호출 최적화"""
        try:
            # 병렬로 여러 MCP 서버 호출
            hobby_tasks = [
                self.mcp_manager.get_hobby_suggestions(state["user_profile"]),
                self.mcp_manager.get_trending_hobbies(),
                self.mcp_manager.get_local_activities(state["user_profile"].get("location"))
            ]
            
            results = await asyncio.gather(*hobby_tasks, return_exceptions=True)
            
            # 결과 병합
            hobby_suggestions = []
            for result in results:
                if isinstance(result, list):
                    hobby_suggestions.extend(result)
            
            # AutoGen 필터링
            filtered_hobbies = await self.autogen_engine.filter_hobbies(
                hobby_suggestions, state["user_profile"]
            )
            
            state["current_step"] = "community_matching"
            state["current_step_result"] = {"discovered_hobbies": filtered_hobbies}
            
            self.state_manager.save_state(state)
            return state
            
        except Exception as e:
            state["error_context"] = f"취미 발견 실패: {str(e)}"
            return state
    
    async def match_communities(self, state: HSPAgentState) -> HSPAgentState:
        """커뮤니티 매칭"""
        try:
            # 이전 단계 결과를 캐시에서 로드
            hobbies = state["current_step_result"].get("discovered_hobbies", [])
            
            # MCP로 커뮤니티 검색
            communities = await self.mcp_manager.find_communities(hobbies)
            
            state["current_step"] = "final_recommendation"
            state["current_step_result"] = {"matched_communities": communities}
            
            self.state_manager.save_state(state)
            return state
            
        except Exception as e:
            state["error_context"] = f"커뮤니티 매칭 실패: {str(e)}"
            return state
    
    async def generate_recommendations(self, state: HSPAgentState) -> HSPAgentState:
        """최종 추천 생성"""
        try:
            # 이전 결과들 통합
            hobbies = self.state_manager.get_cached_data(
                state["session_id"], 
                f"step_result:{state['session_id']}:hobby_discovery"
            )
            
            communities = state["current_step_result"].get("matched_communities", [])
            
            # AutoGen으로 최종 추천 생성
            final_recommendations = await self.autogen_engine.generate_final_recommendations(
                user_profile=state["user_profile"],
                hobbies=hobbies,
                communities=communities
            )
            
            state["hobby_recommendations"] = final_recommendations
            state["current_step"] = "completed"
            
            self.state_manager.save_state(state)
            return state
            
        except Exception as e:
            state["error_context"] = f"최종 추천 생성 실패: {str(e)}"
            return state

# 워크플로우 팩토리
def create_optimized_workflow() -> StateGraph:
    """최적화된 워크플로우 인스턴스 생성"""
    hsp_workflow = OptimizedHSPWorkflow()
    return hsp_workflow.create_workflow()

class HSPLangGraphWorkflow:
    """LangGraph 기반 메인 워크플로우"""
    
    def __init__(self, autogen_agents: Optional[HSPAutoGenAgents] = None, 
                 mcp_manager: Optional[MCPServerManager] = None):
        self.autogen_agents = autogen_agents or HSPAutoGenAgents()
        self.mcp_manager = mcp_manager or MCPServerManager()
        self.a2a_bridge = None  # Will be set during workflow run
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(HSPAgentState)
        
        # 노드 정의
        workflow.add_node("initialize_session", self._initialize_session)
        workflow.add_node("analyze_user_profile", self._analyze_user_profile)
        workflow.add_node("discover_hobbies", self._discover_hobbies)
        workflow.add_node("integrate_schedule", self._integrate_schedule)
        workflow.add_node("match_communities", self._match_communities)
        workflow.add_node("track_progress", self._track_progress)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("autogen_consensus", self._autogen_consensus)
        
        # 엣지 정의 (조건부 라우팅)
        workflow.set_entry_point("initialize_session")
        
        workflow.add_edge("initialize_session", "analyze_user_profile")
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
        return state

    async def _analyze_user_profile(self, state: HSPAgentState) -> HSPAgentState:
        print("---Analyzing User Profile---")
        
        # MCP 서버를 통해 사용자 데이터 수집
        mcp_results = {}
        
        # 구글 캘린더에서 스케줄 정보 가져오기
        calendar_data = await self.mcp_manager.call_mcp_server(
            "google_calendar", 
            "list_events", 
            {"timeframe": "next_month"}
        )
        mcp_results["calendar"] = calendar_data
        
        # 소셜 미디어에서 관심사 분석
        social_data = await self.mcp_manager.call_mcp_server(
            "social_search",
            "search_groups", 
            {"user_interests": state.get("user_profile", {}).get("interests", [])}
        )
        mcp_results["social"] = social_data
        
        state["mcp_responses"].update(mcp_results)
        state["current_decision_point"] = "hobby_discovery"
        
        # A2A 메시지로 프로필 분석 완료 알림
        if self.a2a_bridge:
            await self._send_a2a_update(state, "profile_analysis_complete", mcp_results)
        
        return state

    async def _discover_hobbies(self, state: HSPAgentState) -> HSPAgentState:
        print("---Discovering Hobbies---")
        
        # 교육 플랫폼에서 취미 관련 강의 검색
        education_data = await self.mcp_manager.call_mcp_server(
            "education",
            "search_courses",
            {"user_profile": state.get("user_profile", {})}
        )
        
        # 전자상거래에서 관련 용품 검색
        ecommerce_data = await self.mcp_manager.call_mcp_server(
            "ecommerce",
            "search_products", 
            {"hobby_categories": state.get("hobby_recommendations", [])}
        )
        
        state["mcp_responses"].update({
            "education": education_data,
            "ecommerce": ecommerce_data
        })
        state["current_decision_point"] = "integrate_schedule"
        
        if self.a2a_bridge:
            await self._send_a2a_update(state, "hobby_discovery_complete", {
                "education": education_data,
                "ecommerce": ecommerce_data
            })
        
        return state

    async def _integrate_schedule(self, state: HSPAgentState) -> HSPAgentState:
        print("---Integrating Schedule---")
        
        # 날씨 정보로 야외 활동 계획
        weather_data = await self.mcp_manager.call_mcp_server(
            "weather_api",
            "forecast",
            {"location": state.get("user_profile", {}).get("location", "Seoul")}
        )
        
        # 구글 맵스로 근처 취미 장소 검색
        maps_data = await self.mcp_manager.call_mcp_server(
            "google_maps",
            "search_places",
            {"hobby_types": state.get("hobby_recommendations", [])}
        )
        
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
        
        return state

    async def _match_communities(self, state: HSPAgentState) -> HSPAgentState:
        print("---Matching Communities---")
        
        # 소셜 미디어에서 관련 그룹 찾기
        community_data = await self.mcp_manager.call_mcp_server(
            "social_search",
            "find_communities",
            {"hobbies": state.get("hobby_recommendations", [])}
        )
        
        state["mcp_responses"]["communities"] = community_data
        state["current_decision_point"] = "track_progress"
        
        if self.a2a_bridge:
            await self._send_a2a_update(state, "community_matching_complete", community_data)
        
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
        
        # 모든 MCP 응답을 종합하여 최종 인사이트 생성
        all_mcp_data = state.get("mcp_responses", {})
        
        # 음악/독서 플랫폼에서 추가 추천
        music_data = await self.mcp_manager.call_mcp_server(
            "music_platform",
            "get_recommendations",
            {"user_preferences": state.get("user_profile", {})}
        )
        
        reading_data = await self.mcp_manager.call_mcp_server(
            "reading_platform", 
            "search_books",
            {"interests": state.get("hobby_recommendations", [])}
        )
        
        state["mcp_responses"].update({
            "music": music_data,
            "reading": reading_data
        })
        
        if self.a2a_bridge:
            await self._send_a2a_update(state, "insights_generated", {
                "music": music_data,
                "reading": reading_data,
                "summary": "Final hobby recommendations generated"
            })
        
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
            print(f"Consensus error: {e}")
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
                "error": "Consensus failed, using default routing"
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
        """워크플로우 실행 (API에서 호출)"""
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
                "workflow_context": {},
                "current_decision_point": "profile_analysis",
                "hobby_recommendations": [],
                "schedule_analysis": {},
                "community_matches": [],
                "progress_metrics": {},
                "weekly_journal": "",
                "error_log": [],
                "agent_consensus": {},
                "mcp_responses": {},
                "a2a_messages": []
            }
            
            # 워크플로우 실행
            result = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"hsp_{datetime.now().isoformat()}"}}
            )
            
            return {
                "success": True,
                "final_state": result,
                "recommendations": result.get("hobby_recommendations", []),
                "mcp_data": result.get("mcp_responses", {}),
                "consensus_history": result.get("agent_consensus", {}),
                "error_log": result.get("error_log", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "recommendations": [],
                "mcp_data": {},
                "consensus_history": {},
                "error_log": [f"Workflow execution failed: {str(e)}"]
            } 