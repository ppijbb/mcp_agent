import logging
import hashlib
from typing import Dict, Any, List, TypedDict, Annotated, Optional
from datetime import datetime, timedelta
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

class CacheManager:
    """캐시 관리자"""
    
    def __init__(self, default_ttl: int = 3600):
        """
        CacheManager 초기화
        
        Args:
            default_ttl: 기본 TTL (초)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 가져오기
        
        Args:
            key: 캐시 키
        
        Returns:
            캐시된 값 또는 None
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # TTL 확인
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초, None이면 기본값 사용)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now(),
        }
    
    def delete(self, key: str):
        """캐시에서 값 삭제"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache.clear()
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        캐시 키 생성
        
        Args:
            prefix: 키 접두사
            *args: 위치 인자
            **kwargs: 키워드 인자
        
        Returns:
            생성된 캐시 키
        """
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"


class StateManager:
    """상태 압축 및 캐시 관리"""
    
    def __init__(self, use_redis: bool = False):
        """
        StateManager 초기화
        
        Args:
            use_redis: Redis 사용 여부 (False면 인메모리 캐시 사용)
        """
        self.use_redis = use_redis
        self.cache_manager = CacheManager(default_ttl=3600)
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(decode_responses=True)
                logger.info("StateManager Redis 초기화 완료")
            except ImportError:
                logger.warning("Redis가 설치되지 않아 인메모리 캐시를 사용합니다.")
                self.use_redis = False
                self.redis_client = None
        else:
            self.redis_client = None
            logger.info("StateManager 인메모리 캐시 초기화 완료")
    
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
        """압축된 상태를 저장 (Redis 또는 인메모리 캐시)"""
        session_id = state.get("session_id", "unknown")
        
        # 큰 객체들은 별도 캐시에 저장
        large_objects = {}
        if state.get("current_step_result"):
            cache_key = f"step_result:{session_id}:{state.get('current_step', 'unknown')}"
            large_objects[cache_key] = state["current_step_result"]
            if "cached_data_keys" not in state:
                state["cached_data_keys"] = []
            if cache_key not in state["cached_data_keys"]:
                state["cached_data_keys"].append(cache_key)
            state["current_step_result"] = {}  # 참조만 남기고 제거
        
        # 큰 객체들 별도 저장
        for key, value in large_objects.items():
            if self.use_redis and self.redis_client:
                self.redis_client.setex(key, self.cache_manager.default_ttl, json.dumps(value))
            else:
                self.cache_manager.set(key, value, ttl=self.cache_manager.default_ttl)
        
        # 압축된 상태 저장
        state_json = json.dumps(state, separators=(',', ':'))
        state_key = f"hsp_state:{session_id}"
        
        if self.use_redis and self.redis_client:
            self.redis_client.setex(state_key, self.cache_manager.default_ttl, state_json)
        else:
            self.cache_manager.set(state_key, json.loads(state_json), ttl=self.cache_manager.default_ttl)
    
    def load_state(self, session_id: str) -> Optional[HSPAgentState]:
        """상태 로드 및 복원 (Redis 또는 인메모리 캐시)"""
        state_key = f"hsp_state:{session_id}"
        
        if self.use_redis and self.redis_client:
            state_json = self.redis_client.get(state_key)
            if not state_json:
                return None
            state = json.loads(state_json)
        else:
            cached_state = self.cache_manager.get(state_key)
            if not cached_state:
                return None
            state = cached_state
        
        # 캐시된 데이터는 lazy loading으로 필요시에만 로드
        return state
    
    def get_cached_data(self, session_id: str, cache_key: str) -> Optional[Dict]:
        """특정 캐시 데이터 조회"""
        if self.use_redis and self.redis_client:
            cached_json = self.redis_client.get(cache_key)
            return json.loads(cached_json) if cached_json else None
        else:
            return self.cache_manager.get(cache_key)
    
    def cache_data(self, key: str, value: Any, ttl: Optional[int] = None):
        """데이터를 캐시에 저장"""
        if self.use_redis and self.redis_client:
            ttl = ttl or self.cache_manager.default_ttl
            self.redis_client.setex(key, ttl, json.dumps(value))
        else:
            self.cache_manager.set(key, value, ttl=ttl)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        if self.use_redis and self.redis_client:
            return {
                "type": "redis",
                "cache_size": "unknown"  # Redis는 크기 측정이 복잡함
            }
        else:
            return {
                "type": "in_memory",
                "cache_size": len(self.cache_manager.cache),
                "entries": list(self.cache_manager.cache.keys())
            } 