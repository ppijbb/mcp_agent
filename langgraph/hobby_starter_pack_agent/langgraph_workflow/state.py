import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional
import json

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HSPAgentState(TypedDict):
    """최적화된 상태 - 핵심 정보만 유지"""
    # 세션 관리
    session_id: Annotated[str, "세션 고유 ID"]
    current_step: Annotated[str, "현재 처리 단계"]
    
    # 사용자 데이터 (압축된 핵심 정보만)
    user_profile: Annotated[Dict[str, Any], "압축된 사용자 프로필"]
    user_input: Annotated[str, "사용자 입력"]
    
    # 현재 단계 결과만 보관 (이전 단계는 캐시로 이동)
    current_step_result: Annotated[Dict[str, Any], "현재 단계 처리 결과"]
    
    # 최종 출력
    hobby_recommendations: Annotated[List[Dict], "취미 추천 결과"]
    
    # 에러 처리
    error_context: Annotated[Optional[str], "에러 발생시에만 저장"]
    
    # 캐시 참조 (실제 데이터는 외부 저장)
    cached_data_keys: Annotated[List[str], "캐시된 데이터 키 목록"]

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
            "current_step": "profile_analysis",
            "user_profile": {},
            "user_input": user_input,
            "current_step_result": {},
            "hobby_recommendations": [],
            "error_context": None,
            "cached_data_keys": []
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