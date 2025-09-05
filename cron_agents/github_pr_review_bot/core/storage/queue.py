"""
Task Queue - Celery 기반 작업 큐 시스템

이 모듈은 Celery를 사용한 분산 작업 큐 시스템을 제공합니다.
비동기 작업 처리, 작업 상태 추적, 재시도 로직 등을 지원합니다.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from celery import Celery, Task
from celery.result import AsyncResult
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config

logger = logging.getLogger(__name__)

class TaskQueue:
    """Celery 기반 작업 큐 관리자"""
    
    def __init__(self, broker_url: str = None, result_backend: str = None):
        """
        작업 큐 초기화
        
        Args:
            broker_url (str, optional): Celery 브로커 URL
            result_backend (str, optional): 결과 백엔드 URL
        """
        self.broker_url = broker_url or config.queue.broker_url
        self.result_backend = result_backend or config.queue.result_backend
        
        # Celery 앱 초기화
        self.celery_app = Celery(
            'github_pr_review_bot',
            broker=self.broker_url,
            backend=self.result_backend,
            include=['core.queue']
        )
        
        # Celery 설정
        self.celery_app.conf.update(
            task_serializer=config.queue.task_serializer,
            result_serializer=config.queue.result_serializer,
            accept_content=config.queue.accept_content,
            timezone=config.queue.timezone,
            enable_utc=config.queue.enable_utc,
            task_track_started=True,
            task_time_limit=30 * 60,  # 30분
            task_soft_time_limit=25 * 60,  # 25분
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_max_tasks_per_child=1000,
        )
        
        logger.info(f"TaskQueue initialized with broker: {self.broker_url}")
    
    def register_task(self, task_func: Callable, name: str = None) -> Task:
        """
        작업 함수 등록
        
        Args:
            task_func (Callable): 작업 함수
            name (str, optional): 작업 이름
            
        Returns:
            Task: 등록된 Celery 작업
        """
        task_name = name or f"{task_func.__module__}.{task_func.__name__}"
        
        @self.celery_app.task(name=task_name, bind=True)
        def wrapped_task(self_task, *args, **kwargs):
            try:
                logger.info(f"Starting task: {task_name}")
                result = task_func(*args, **kwargs)
                logger.info(f"Task completed: {task_name}")
                return result
            except Exception as e:
                logger.error(f"Task failed: {task_name}, error: {e}")
                raise
        
        return wrapped_task
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> AsyncResult:
        """
        작업 제출
        
        Args:
            task_func (Callable): 작업 함수
            *args: 위치 인수
            **kwargs: 키워드 인수
            
        Returns:
            AsyncResult: 작업 결과 객체
        """
        task = self.register_task(task_func)
        return task.delay(*args, **kwargs)
    
    def submit_review_task(self, repository: str, pr_number: int, 
                          review_type: str = "detailed", provider: str = None) -> AsyncResult:
        """
        PR 리뷰 작업 제출
        
        Args:
            repository (str): 저장소 이름
            pr_number (int): PR 번호
            review_type (str): 리뷰 유형
            provider (str, optional): LLM 제공자
            
        Returns:
            AsyncResult: 작업 결과 객체
        """
        from .review_generator import ReviewGenerator
        from .github_client import GitHubClient
        
        def review_task(repo: str, pr_num: int, r_type: str, llm_provider: str = None):
            try:
                # GitHub 클라이언트 초기화
                github_client = GitHubClient()
                
                # PR 정보 가져오기
                pr_metadata = github_client.get_pr_metadata(repo, pr_num)
                diff_content = github_client.get_pr_diff(repo, pr_num)
                
                # 리뷰 생성기 초기화
                review_generator = ReviewGenerator()
                
                # 리뷰 생성
                review_result = review_generator.generate_review(
                    diff_content=diff_content,
                    pr_metadata=pr_metadata,
                    provider=llm_provider
                )
                
                # GitHub에 리뷰 등록
                github_client.create_review(
                    repo_full_name=repo,
                    pr_number=pr_num,
                    body=review_result["review"],
                    event="COMMENT"
                )
                
                return {
                    "status": "success",
                    "repository": repo,
                    "pr_number": pr_num,
                    "review_id": review_result.get("id"),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Review task failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "repository": repo,
                    "pr_number": pr_num,
                    "timestamp": datetime.now().isoformat()
                }
        
        return self.submit_task(review_task, repository, pr_number, review_type, provider)
    
    def submit_analysis_task(self, repository: str, pr_number: int, 
                           analysis_type: str = "quality") -> AsyncResult:
        """
        코드 분석 작업 제출
        
        Args:
            repository (str): 저장소 이름
            pr_number (int): PR 번호
            analysis_type (str): 분석 유형 (quality, security)
            
        Returns:
            AsyncResult: 작업 결과 객체
        """
        from .review_generator import ReviewGenerator
        from .github_client import GitHubClient
        
        def analysis_task(repo: str, pr_num: int, a_type: str):
            try:
                # GitHub 클라이언트 초기화
                github_client = GitHubClient()
                
                # PR 파일 정보 가져오기
                pr_files = github_client.get_pr_files(repo, pr_num)
                
                # 리뷰 생성기 초기화
                review_generator = ReviewGenerator()
                
                analysis_results = []
                
                for file_info in pr_files:
                    if file_info["patch"]:
                        if a_type == "quality":
                            analysis_result = review_generator.analyze_code_quality(
                                code_content=file_info["patch"],
                                file_path=file_info["filename"]
                            )
                        elif a_type == "security":
                            analysis_result = review_generator.analyze_security(
                                code_content=file_info["patch"],
                                file_path=file_info["filename"]
                            )
                        else:
                            continue
                        
                        analysis_results.append({
                            "filename": file_info["filename"],
                            "analysis": analysis_result
                        })
                
                return {
                    "status": "success",
                    "repository": repo,
                    "pr_number": pr_num,
                    "analysis_type": a_type,
                    "results": analysis_results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Analysis task failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "repository": repo,
                    "pr_number": pr_num,
                    "analysis_type": a_type,
                    "timestamp": datetime.now().isoformat()
                }
        
        return self.submit_task(analysis_task, repository, pr_number, analysis_type)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        작업 상태 확인
        
        Args:
            task_id (str): 작업 ID
            
        Returns:
            Dict[str, Any]: 작업 상태 정보
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "ready": result.ready(),
                "successful": result.successful(),
                "failed": result.failed(),
                "result": result.result if result.ready() else None,
                "traceback": result.traceback if result.failed() else None,
                "date_done": result.date_done.isoformat() if result.date_done else None
            }
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {
                "task_id": task_id,
                "status": "ERROR",
                "error": str(e)
            }
    
    def get_task_result(self, task_id: str) -> Any:
        """
        작업 결과 가져오기
        
        Args:
            task_id (str): 작업 ID
            
        Returns:
            Any: 작업 결과
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            return result.get(timeout=30)  # 30초 타임아웃
        except Exception as e:
            logger.error(f"Error getting task result: {e}")
            raise
    
    def cancel_task(self, task_id: str) -> bool:
        """
        작업 취소
        
        Args:
            task_id (str): 작업 ID
            
        Returns:
            bool: 취소 성공 여부
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            result.revoke(terminate=True)
            return True
        except Exception as e:
            logger.error(f"Error canceling task: {e}")
            return False
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        활성 작업 목록 가져오기
        
        Returns:
            List[Dict[str, Any]]: 활성 작업 목록
        """
        try:
            inspector = self.celery_app.control.inspect()
            active_tasks = inspector.active()
            
            tasks = []
            for worker, worker_tasks in active_tasks.items():
                for task in worker_tasks:
                    tasks.append({
                        "worker": worker,
                        "task_id": task["id"],
                        "name": task["name"],
                        "args": task["args"],
                        "kwargs": task["kwargs"],
                        "time_start": task["time_start"],
                        "acknowledged": task["acknowledged"]
                    })
            
            return tasks
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return []
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        큐 통계 정보 가져오기
        
        Returns:
            Dict[str, Any]: 큐 통계
        """
        try:
            inspector = self.celery_app.control.inspect()
            
            stats = {
                "active": len(inspector.active() or {}),
                "reserved": len(inspector.reserved() or {}),
                "scheduled": len(inspector.scheduled() or {}),
                "registered": len(inspector.registered() or {}),
                "workers": len(inspector.ping() or {})
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {"error": str(e)}
    
    def purge_queue(self, queue_name: str = None) -> bool:
        """
        큐 정리
        
        Args:
            queue_name (str, optional): 큐 이름
            
        Returns:
            bool: 정리 성공 여부
        """
        try:
            if queue_name:
                self.celery_app.control.purge(queue=queue_name)
            else:
                self.celery_app.control.purge()
            return True
        except Exception as e:
            logger.error(f"Error purging queue: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        큐 상태 확인
        
        Returns:
            Dict[str, Any]: 상태 정보
        """
        try:
            inspector = self.celery_app.control.inspect()
            ping_result = inspector.ping()
            
            if ping_result:
                return {
                    "status": "healthy",
                    "workers": len(ping_result),
                    "broker": self.broker_url,
                    "backend": self.result_backend
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "No workers available",
                    "broker": self.broker_url,
                    "backend": self.result_backend
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "broker": self.broker_url,
                "backend": self.result_backend
            }
    
    def start_worker(self, concurrency: int = 4, loglevel: str = "info"):
        """
        Celery 워커 시작
        
        Args:
            concurrency (int): 동시 작업 수
            loglevel (str): 로그 레벨
        """
        self.celery_app.worker_main([
            'worker',
            '--loglevel=' + loglevel,
            '--concurrency=' + str(concurrency),
            '--hostname=worker@%h'
        ])

# 전역 작업 큐 인스턴스
task_queue = TaskQueue() 