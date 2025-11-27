"""
Fallback LLM Factory Module

Gemini API 503 오류 시 자동으로 최고 성능 모델로 fallback하는 공통 모듈
모든 agent에서 동일한 fallback 메커니즘 사용
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache
from datetime import datetime, timedelta

from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams

logger = logging.getLogger(__name__)

# 모델 목록 캐시 (1시간 유지)
_model_list_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamp: Dict[str, datetime] = {}
CACHE_DURATION = timedelta(hours=1)


def _fetch_openrouter_models(api_key: str) -> List[str]:
    """OpenRouter API에서 사용 가능한 모델 목록 가져오기 (동기)"""
    cache_key = "openrouter"
    now = datetime.now()
    
    # 캐시 확인
    if cache_key in _model_list_cache and cache_key in _cache_timestamp:
        if now - _cache_timestamp[cache_key] < CACHE_DURATION:
            logger.debug("OpenRouter 모델 목록 캐시 사용")
            return _model_list_cache[cache_key].get("models", [])
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/mcp-agent",  # OpenRouter 요구사항
            "X-Title": "MCP Agent Fallback"
        }
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            # 고성능 모델 우선순위 필터링 (특정 키워드 포함 모델 우선)
            priority_keywords = ["llama-4", "llama-3.3", "moonshot", "gemini-2.5", "qwen-2.5", "yi-1.5", "glm-4", "minimax"]
            prioritized = []
            others = []
            for model in models:
                if any(keyword in model.lower() for keyword in priority_keywords):
                    prioritized.append(model)
                else:
                    others.append(model)
            # 우선순위 모델 먼저, 나머지 나중에
            sorted_models = prioritized + others
            _model_list_cache[cache_key] = {"models": sorted_models}
            _cache_timestamp[cache_key] = now
            logger.info(f"OpenRouter에서 {len(sorted_models)}개 모델 가져옴 (우선순위: {len(prioritized)}개)")
            return sorted_models
        else:
            logger.warning(f"OpenRouter API 오류: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"OpenRouter 모델 목록 가져오기 실패: {e}")
        return []


def _fetch_groq_models(api_key: str) -> List[str]:
    """Groq API에서 사용 가능한 모델 목록 가져오기 (동기)"""
    cache_key = "groq"
    now = datetime.now()
    
    # 캐시 확인
    if cache_key in _model_list_cache and cache_key in _cache_timestamp:
        if now - _cache_timestamp[cache_key] < CACHE_DURATION:
            logger.debug("Groq 모델 목록 캐시 사용")
            return _model_list_cache[cache_key].get("models", [])
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            # 고성능 모델 우선순위 필터링
            priority_keywords = ["llama-3.3", "llama-3.1", "llama-3", "mixtral"]
            prioritized = []
            others = []
            for model in models:
                if any(keyword in model.lower() for keyword in priority_keywords):
                    prioritized.append(model)
                else:
                    others.append(model)
            sorted_models = prioritized + others
            _model_list_cache[cache_key] = {"models": sorted_models}
            _cache_timestamp[cache_key] = now
            logger.info(f"Groq에서 {len(sorted_models)}개 모델 가져옴 (우선순위: {len(prioritized)}개)")
            return sorted_models
        else:
            logger.warning(f"Groq API 오류: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Groq 모델 목록 가져오기 실패: {e}")
        return []


def get_best_fallback_models() -> List[Dict[str, Any]]:
    """
    각 서비스에서 사용 가능한 최고 성능 모델 목록 반환
    API에서 동적으로 모델 목록을 가져와서 사용
    
    Returns:
        Fallback 모델 설정 리스트 (우선순위 순)
    """
    fallback_configs = []
    
    # OpenRouter 모델 목록 가져오기
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            models = _fetch_openrouter_models(openrouter_key)
            for model in models[:20]:  # 최대 20개만 사용
                fallback_configs.append({
                    "provider": "openrouter",
                    "model": model,
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "OPENROUTER_API_KEY"
                })
        except Exception as e:
            logger.warning(f"OpenRouter 모델 목록 가져오기 실패, 빈 목록 사용: {e}")
    
    # Groq 모델 목록 가져오기
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            models = _fetch_groq_models(groq_key)
            for model in models[:10]:  # 최대 10개만 사용
                fallback_configs.append({
                    "provider": "groq",
                    "model": model,
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY"
                })
        except Exception as e:
            logger.warning(f"Groq 모델 목록 가져오기 실패, 빈 목록 사용: {e}")
    
    return fallback_configs


def _try_fallback_llm(primary_model: str, logger_instance: Optional[logging.Logger] = None) -> Optional[Any]:
    """
    503 오류 발생 시 fallback LLM 생성 시도
    
    Args:
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        
    Returns:
        Fallback LLM 인스턴스 또는 None
    """
    log = logger_instance or logger
    fallback_configs = get_best_fallback_models()
    
    for config in fallback_configs:
        try:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                continue
            
            llm = OpenAIAugmentedLLM(model=config["model"])
            if hasattr(llm, 'client'):
                llm.client.base_url = config["base_url"]
                llm.client.api_key = api_key
            
            log.info(f"Fallback to {config['provider']} model: {config['model']}")
            return llm
        except Exception as fallback_error:
            continue
    
    return None


def create_fallback_llm_factory(
    primary_model: str = "gemini-2.5-flash-lite",
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Fallback이 가능한 LLM factory 생성
    
    Args:
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        
    Returns:
        LLM factory 함수
    """
    log = logger_instance or logger
    
    def llm_factory_with_fallback(**kwargs):
        """Fallback이 가능한 LLM factory"""
        try:
            return GoogleAugmentedLLM(model=primary_model)
        except Exception as e:
            error_str = str(e).lower()
            # 503 오류나 overloaded 오류인 경우 fallback 시도
            if "503" in error_str or "overloaded" in error_str or "unavailable" in error_str:
                log.warning(f"Gemini API 오류 발생, fallback 모델로 전환: {e}")
                fallback_llm = _try_fallback_llm(primary_model, log)
                if fallback_llm:
                    return fallback_llm
                # 모든 fallback 실패 시 원래 오류 발생
                raise e
            else:
                # 503이 아닌 다른 오류는 그대로 발생
                raise
    
    return llm_factory_with_fallback


def create_fallback_llm_for_agents(
    primary_model: str = "gemini-2.5-flash-lite",
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Agent용 fallback LLM factory 생성
    
    Args:
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        
    Returns:
        Agent용 LLM factory 함수
    """
    log = logger_instance or logger
    
    def llm_factory_for_agents(**kwargs):
        try:
            return GoogleAugmentedLLM(model=primary_model)
        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "overloaded" in error_str or "unavailable" in error_str:
                # Fallback 모델 시도 (동적으로 최고 성능 모델 선택)
                log.warning(f"Agent LLM: Gemini API 오류, fallback 모델로 전환: {e}")
                fallback_llm = _try_fallback_llm(primary_model, log)
                if fallback_llm:
                    return fallback_llm
            raise
    
    return llm_factory_for_agents


def create_fallback_orchestrator_llm_factory(
    primary_model: str = "gemini-2.5-flash-lite",
    logger_instance: Optional[logging.Logger] = None
) -> Callable:
    """
    Orchestrator용 fallback LLM factory 생성
    
    Args:
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        
    Returns:
        Orchestrator용 LLM factory 함수
    """
    log = logger_instance or logger
    
    def orchestrator_llm_factory(**kwargs):
        """Orchestrator용 fallback LLM factory"""
        try:
            return GoogleAugmentedLLM(model=primary_model)
        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "overloaded" in error_str or "unavailable" in error_str:
                log.warning(f"Orchestrator LLM: Gemini API 오류, fallback 모델로 전환: {e}")
                # Fallback 모델 시도 (동적으로 최고 성능 모델 선택)
                fallback_llm = _try_fallback_llm(primary_model, log)
                if fallback_llm:
                    return fallback_llm
            raise
    
    return orchestrator_llm_factory


async def try_fallback_orchestrator_execution(
    orchestrator: Orchestrator,
    agents: List[Any],
    task: str,
    primary_model: str,
    logger_instance: Optional[logging.Logger] = None,
    max_loops: int = 30
) -> str:
    """
    Orchestrator 실행 시 503 오류 발생하면 fallback 모델로 재시도
    
    Args:
        orchestrator: 기본 orchestrator
        agents: Agent 리스트
        task: 실행할 작업
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        max_loops: 최대 반복 횟수
        
    Returns:
        실행 결과 (final_report)
        
    Raises:
        원래 오류 또는 모든 fallback 실패 시 오류
    """
    log = logger_instance or logger
    
    try:
        final_report = await orchestrator.generate_str(
            message=task,
            request_params=RequestParams(model=primary_model)
        )
        return final_report
    except Exception as api_error:
        error_str = str(api_error).lower()
        # 503 오류, overloaded 오류, 또는 iterations 실패인 경우 fallback 시도
        should_fallback = (
            "503" in error_str or 
            "overloaded" in error_str or 
            "unavailable" in error_str or
            "task failed to complete" in error_str or
            "failed to complete in" in error_str
        )
        
        if should_fallback:
            log.warning(f"Orchestrator 실행 오류 발생, fallback 모델로 재시도: {api_error}")
            
            # Fallback orchestrator 생성 (동적으로 최고 성능 모델 선택)
            fallback_configs = get_best_fallback_models()
            fallback_success = False
            
            for config in fallback_configs:
                try:
                    api_key = os.getenv(config["api_key_env"])
                    if not api_key:
                        continue
                    
                    log.info(f"{config['provider']} 모델로 fallback 시도: {config['model']}")
                    
                    def create_fallback_llm(**kwargs):
                        llm = OpenAIAugmentedLLM(model=config["model"])
                        if hasattr(llm, 'client'):
                            llm.client.base_url = config["base_url"]
                            llm.client.api_key = api_key
                        return llm
                    
                    fallback_orchestrator = Orchestrator(
                        llm_factory=create_fallback_llm,
                        available_agents=agents,
                        plan_type="full",
                        max_loops=max_loops
                    )
                    
                    final_report = await fallback_orchestrator.generate_str(
                        message=task,
                        request_params=RequestParams(model=config["model"])
                    )
                    
                    log.info(f"✅ {config['provider']} fallback 성공: {config['model']}")
                    fallback_success = True
                    return final_report
                except Exception as fallback_error:
                    log.warning(f"{config['provider']} fallback 실패: {fallback_error}")
                    continue
            
            if not fallback_success:
                log.error("모든 fallback 모델 실패, 원래 오류 발생")
                raise api_error
        elif isinstance(api_error, (BrokenPipeError, OSError)):
            # EPIPE 또는 파이프 관련 에러 처리
            error_code = getattr(api_error, 'errno', None)
            if error_code == 32 or 'EPIPE' in str(api_error):
                log.error(f"MCP server process terminated unexpectedly (EPIPE). This may indicate the server crashed or was closed.")
                raise RuntimeError(f"MCP server connection lost: {api_error}") from api_error
            raise
        else:
            # 다른 오류는 그대로 발생
            raise

