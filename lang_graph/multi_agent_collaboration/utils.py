import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis.asyncio as redis
from dataclasses import dataclass
from datetime import datetime, timedelta

# 성능 최적화를 위한 uvloop 사용 (Linux/macOS에서만)
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop for enhanced performance")
except ImportError:
    print("uvloop not available, using standard asyncio")

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# 로깅 설정
logger = logging.getLogger(__name__)

# 웹 검색 도구 정의
# 이 도구는 '검색 에이전트'가 사용합니다.
search_tool = TavilySearchResults(max_results=5, name="web_search")

# LLM 모델 정의
# 모든 에이전트들이 공통으로 사용하는 언어 모델입니다.
# gemini-2.5-flash-lite-preview-06-07 모델을 사용하여 강력한 추론 능력을 활용합니다.
model = ChatOpenAI(model="gemini-2.5-flash-lite-preview-06-07", temperature=0, streaming=True)

# 성능 모니터링을 위한 메트릭 수집기
@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PerformanceManager:
    """엔터프라이즈급 성능 관리자"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.metrics: List[PerformanceMetrics] = []
        self.max_metrics = 10000
        
        # 스레드 풀과 프로세스 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        # 캐시 설정
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1시간
        
        # 로드 밸런싱 설정
        self.load_balancers: Dict[str, Any] = {}
        
        # 백그라운드 작업 큐
        self.background_tasks: List[Dict[str, Any]] = []
        
        # 성능 정책 초기화
        self._initialize_performance_policies()
    
    def _initialize_performance_policies(self):
        """성능 정책 초기화"""
        self.performance_policies = {
            "caching": {
                "enabled": True,
                "max_cache_size": 10000,
                "ttl_seconds": 3600,
                "eviction_policy": "lru"
            },
            "load_balancing": {
                "enabled": True,
                "algorithm": "weighted_round_robin",
                "health_check_interval": 30,
                "failover_enabled": True
            },
            "background_processing": {
                "enabled": True,
                "max_concurrent_tasks": 5,
                "task_timeout": 300
            },
            "monitoring": {
                "enabled": True,
                "metrics_collection_interval": 60,
                "alert_thresholds": {
                    "response_time_ms": 1000,
                    "error_rate": 0.05,
                    "memory_usage_mb": 1024
                }
            }
        }
    
    def performance_monitor(self, operation_name: str = None):
        """성능 모니터링 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error = None
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    metric = PerformanceMetrics(
                        operation=operation_name or func.__name__,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        error=error
                    )
                    
                    self._record_metric(metric)
                    
                    # 성능 임계값 확인
                    await self._check_performance_thresholds(metric)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error = None
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    metric = PerformanceMetrics(
                        operation=operation_name or func.__name__,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        success=success,
                        error=error
                    )
                    
                    self._record_metric(metric)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_metric(self, metric: PerformanceMetrics):
        """메트릭 기록"""
        self.metrics.append(metric)
        
        # 메트릭 개수 제한
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    async def _check_performance_thresholds(self, metric: PerformanceMetrics):
        """성능 임계값 확인"""
        try:
            thresholds = self.performance_policies["monitoring"]["alert_thresholds"]
            
            # 응답 시간 임계값 확인
            if metric.duration * 1000 > thresholds["response_time_ms"]:
                await self._trigger_performance_alert("high_response_time", metric)
            
            # 에러율 확인
            if not metric.success:
                error_rate = self._calculate_error_rate()
                if error_rate > thresholds["error_rate"]:
                    await self._trigger_performance_alert("high_error_rate", metric)
                    
        except Exception as e:
            logger.warning(f"Performance threshold check failed: {str(e)}")
    
    def _calculate_error_rate(self) -> float:
        """에러율 계산"""
        if not self.metrics:
            return 0.0
        
        recent_metrics = self.metrics[-100:]  # 최근 100개
        error_count = sum(1 for m in recent_metrics if not m.success)
        return error_count / len(recent_metrics)
    
    async def _trigger_performance_alert(self, alert_type: str, metric: PerformanceMetrics):
        """성능 알림 트리거"""
        try:
            alert_data = {
                "type": alert_type,
                "timestamp": datetime.now().isoformat(),
                "metric": {
                    "operation": metric.operation,
                    "duration": metric.duration,
                    "success": metric.success
                }
            }
            
            # Redis에 알림 저장
            alert_key = f"performance_alert:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            await self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))
            
            logger.warning(f"Performance alert: {alert_data}")
            
        except Exception as e:
            logger.error(f"Failed to trigger performance alert: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def cached_function(self, func: Callable, *args, **kwargs):
        """함수 결과 캐싱"""
        return func(*args, **kwargs)
    
    async def async_cache_get(self, key: str) -> Optional[Any]:
        """비동기 캐시 조회"""
        try:
            if not self.cache_enabled:
                return None
            
            cached_data = await self.redis_client.get(f"cache:{key}")
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache get failed: {str(e)}")
            return None
    
    async def async_cache_set(self, key: str, value: Any, ttl: int = None):
        """비동기 캐시 설정"""
        try:
            if not self.cache_enabled:
                return
            
            ttl = ttl or self.cache_ttl
            await self.redis_client.setex(
                f"cache:{key}",
                ttl,
                json.dumps(value)
            )
            
        except Exception as e:
            logger.warning(f"Cache set failed: {str(e)}")
    
    async def execute_in_thread_pool(self, func: Callable, *args, **kwargs):
        """스레드 풀에서 함수 실행 (I/O 바운드 작업용)"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool, func, *args, **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Thread pool execution failed: {str(e)}")
            raise
    
    async def execute_in_process_pool(self, func: Callable, *args, **kwargs):
        """프로세스 풀에서 함수 실행 (CPU 바운드 작업용)"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_pool, func, *args, **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Process pool execution failed: {str(e)}")
            raise
    
    async def add_background_task(self, task_func: Callable, *args, **kwargs):
        """백그라운드 작업 추가"""
        try:
            task_info = {
                "func": task_func,
                "args": args,
                "kwargs": kwargs,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
            
            self.background_tasks.append(task_info)
            
            # 백그라운드에서 실행
            asyncio.create_task(self._execute_background_task(task_info))
            
        except Exception as e:
            logger.error(f"Failed to add background task: {str(e)}")
    
    async def _execute_background_task(self, task_info: Dict[str, Any]):
        """백그라운드 작업 실행"""
        try:
            task_info["status"] = "running"
            task_info["started_at"] = datetime.now().isoformat()
            
            if asyncio.iscoroutinefunction(task_info["func"]):
                result = await task_info["func"](*task_info["args"], **task_info["kwargs"])
            else:
                result = await self.execute_in_thread_pool(
                    task_info["func"], *task_info["args"], **task_info["kwargs"]
                )
            
            task_info["status"] = "completed"
            task_info["completed_at"] = datetime.now().isoformat()
            task_info["result"] = result
            
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            task_info["failed_at"] = datetime.now().isoformat()
            logger.error(f"Background task failed: {str(e)}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        try:
            if not self.metrics:
                return {}
            
            recent_metrics = self.metrics[-100:]  # 최근 100개
            
            # 응답 시간 통계
            durations = [m.duration for m in recent_metrics if m.success]
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            
            # 성공률
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            
            # 에러율
            error_rate = 1 - success_rate
            
            return {
                "total_operations": len(self.metrics),
                "recent_operations": len(recent_metrics),
                "success_rate": success_rate,
                "error_rate": error_rate,
                "avg_response_time_ms": avg_duration * 1000,
                "max_response_time_ms": max_duration * 1000,
                "min_response_time_ms": min_duration * 1000,
                "cache_enabled": self.cache_enabled,
                "background_tasks": len(self.background_tasks),
                "active_background_tasks": sum(1 for t in self.background_tasks if t["status"] == "running")
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    async def cleanup_old_metrics(self):
        """오래된 메트릭 정리"""
        try:
            cutoff_time = time.time() - (24 * 3600)  # 24시간 전
            
            # 메모리에서 오래된 메트릭 제거
            self.metrics = [m for m in self.metrics if m.end_time > cutoff_time]
            
            # Redis에서 오래된 캐시 정리
            old_cache_keys = await self.redis_client.keys("cache:*")
            for key in old_cache_keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # TTL이 설정되지 않은 키
                    await self.redis_client.delete(key)
            
            logger.info(f"Cleaned up old metrics and cache")
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {str(e)}")
    
    async def shutdown(self):
        """성능 관리자 종료"""
        try:
            # 스레드 풀과 프로세스 풀 종료
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # Redis 연결 종료
            await self.redis_client.close()
            
            logger.info("Performance manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Performance manager shutdown failed: {str(e)}")

# 전역 성능 관리자 인스턴스
performance_manager = PerformanceManager("redis://localhost:6379")

# 성능 모니터링이 적용된 검색 도구
@performance_manager.performance_monitor("web_search")
async def enhanced_search_tool(query: str) -> List[Dict[str, Any]]:
    """성능 모니터링이 적용된 웹 검색 도구"""
    try:
        # 캐시 확인
        cache_key = f"search:{query}"
        cached_result = await performance_manager.async_cache_get(cache_key)
        if cached_result:
            return cached_result
        
        # 실제 검색 수행
        search_results = search_tool.invoke({"query": query})
        
        # 결과 캐싱
        await performance_manager.async_cache_set(cache_key, search_results, 1800)  # 30분
        
        return search_results
        
    except Exception as e:
        logger.error(f"Enhanced search tool failed: {str(e)}")
        raise

# 백그라운드 정리 작업
async def start_background_cleanup():
    """백그라운드 정리 작업 시작"""
    while True:
        try:
            await asyncio.sleep(3600)  # 1시간마다
            await performance_manager.cleanup_old_metrics()
        except Exception as e:
            logger.error(f"Background cleanup failed: {str(e)}")

# 백그라운드 작업 시작 (선택사항)
# asyncio.create_task(start_background_cleanup()) 