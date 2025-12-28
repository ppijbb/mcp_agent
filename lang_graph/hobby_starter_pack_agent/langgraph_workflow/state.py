import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional
import json

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HSPAgentState(TypedDict, total=False):
    """HSP Agent 상태 정의 - 모든 필수 및 선택적 필드 포함"""
    # 세션 관리
    session_id: Annotated[str, "세션 고유 ID"]
    current_step: Annotated[str, "현재 처리 단계"]
    agent_session: Annotated[Dict[str, Any], "에이전트 세션 정보"]
    workflow_context: Annotated[Dict[str, Any], "워크플로우 컨텍스트"]
    current_decision_point: Annotated[str, "현재 의사결정 포인트"]
    
    # 사용자 데이터
    user_profile: Annotated[Dict[str, Any], "사용자 프로필"]
    user_input: Annotated[str, "사용자 입력"]
    preferences: Annotated[Dict[str, Any], "사용자 선호도"]
    
    # 현재 단계 결과
    current_step_result: Annotated[Dict[str, Any], "현재 단계 처리 결과"]
    
    # 최종 출력
    hobby_recommendations: Annotated[List[Dict], "취미 추천 결과"]
    schedule_analysis: Annotated[Dict[str, Any], "스케줄 분석 결과"]
    community_matches: Annotated[List[Dict], "커뮤니티 매칭 결과"]
    progress_metrics: Annotated[Dict[str, Any], "진행 상황 메트릭"]
    weekly_journal: Annotated[str, "주간 저널"]
    
    # 에러 처리
    error_context: Annotated[Optional[str], "에러 발생시에만 저장"]
    error_log: Annotated[List[str], "에러 로그"]
    
    # 에이전트 합의
    agent_consensus: Annotated[Dict[str, Any], "에이전트 합의 결과"]
    
    # MCP 및 A2A
    mcp_responses: Annotated[Dict[str, Any], "MCP 서버 응답"]
    a2a_messages: Annotated[List[Dict[str, Any]], "A2A 메시지"]
    
    # 캐시 참조
    cached_data_keys: Annotated[List[str], "캐시된 데이터 키 목록"]
    
    # 대화형 상호작용 관련 필드
    conversation_history: Annotated[List[Dict[str, Any]], "질문-답변 히스토리"]
    collected_preferences: Annotated[Dict[str, Any], "수집된 선호도 정보"]
    question_completeness_score: Annotated[float, "정보 수집 완성도 점수 (0.0-1.0)"]
    current_question: Annotated[Optional[str], "현재 질문"]
    waiting_for_user_response: Annotated[bool, "사용자 응답 대기 상태"]

class StateManager:
    """상태 압축 및 캐시 관리"""
    
    def __init__(self):
        import redis
        import json
        self.redis_client = redis.Redis(decode_responses=True)
        self.cache_ttl = 3600  # 1시간
        logger.info("StateManager 초기화 완료")
    
    def create_initial_state(self, session_id: str, user_input: str) -> HSPAgentState:
        """최적화된 초기 상태 생성"""
        return {
            "session_id": session_id,
            "current_step": "collect_user_preferences",
            "user_profile": {},
            "user_input": user_input,
            "current_step_result": {},
            "hobby_recommendations": [],
            "error_context": None,
            "cached_data_keys": [],
            "conversation_history": [],
            "collected_preferences": {},
            "question_completeness_score": 0.0,
            "current_question": None,
            "waiting_for_user_response": False
        }
    
    def save_state(self, state: HSPAgentState) -> None:
        """압축된 상태를 Redis에 저장"""
        session_id = state["session_id"]
        
        # 큰 객체들은 별도 캐시에 저장
        large_objects = {}
        if state.get("current_step_result"):
            cache_key = f"step_result:{session_id}:{state['current_step']}"
            large_objects[cache_key] = state["current_step_result"]
            state["cached_data_keys"].append(cache_key)
            state["current_step_result"] = {}  # 참조만 남기고 제거
        
        # 큰 객체들 별도 저장
        for key, value in large_objects.items():
            self.redis_client.setex(key, self.cache_ttl, json.dumps(value))
        
        # 압축된 상태 저장
        state_json = json.dumps(state, separators=(',', ':'))
        self.redis_client.setex(f"hsp_state:{session_id}", self.cache_ttl, state_json)
    
    def load_state(self, session_id: str) -> Optional[HSPAgentState]:
        """Redis에서 상태 로드 및 복원"""
        state_json = self.redis_client.get(f"hsp_state:{session_id}")
        if not state_json:
            return None
        
        state = json.loads(state_json)
        
        # 캐시된 데이터 복원
        for cache_key in state.get("cached_data_keys", []):
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                # 필요시에만 복원 (lazy loading)
                pass
        
        return state
    
    def get_cached_data(self, session_id: str, cache_key: str) -> Optional[Dict]:
        """특정 캐시 데이터 조회"""
        cached_json = self.redis_client.get(cache_key)
        return json.loads(cached_json) if cached_json else None 