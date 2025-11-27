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
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
import httpx

logger = logging.getLogger(__name__)

# 모델 목록 캐시 (1시간 유지)
_model_list_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamp: Dict[str, datetime] = {}
CACHE_DURATION = timedelta(hours=1)


def _fetch_openrouter_models(api_key: str) -> List[str]:
    """OpenRouter API에서 사용 가능한 모델 목록을 동적으로 가져와서 순위별로 정렬"""
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
            "HTTP-Referer": "https://github.com/mcp-agent",
            "X-Title": "MCP Agent Fallback"
        }
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            # 실제 사용 가능한 모델만 필터링하고 동적 점수 계산
            model_scores = []
            current_time = datetime.now()
            
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if not model_id or model.get("moderation", {}).get("required", False):
                    continue
                
                # 동적 점수 계산 (하드코딩 없이, API 데이터만 사용)
                score = 0.0
                
                # 1. 최신성 점수 (created 날짜가 최신일수록 높은 점수)
                created_str = model.get("created", "")
                if created_str:
                    try:
                        # ISO 형식 날짜 파싱
                        created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                        if created_date.tzinfo:
                            created_date = created_date.replace(tzinfo=None)
                        # 최근 1년 기준으로 점수 계산 (최신일수록 높은 점수)
                        days_old = (current_time - created_date).days
                        if days_old >= 0:
                            recency_score = max(0, 1.0 - (days_old / 365.0))  # 1년 이상이면 0점
                            score += recency_score * 0.4  # 최신성이 가장 중요
                    except (ValueError, AttributeError):
                        pass  # 날짜 파싱 실패 시 무시
                
                # 2. 컨텍스트 길이 (긴 것이 좋음)
                context_length = model.get("context_length", 0)
                if context_length > 0:
                    # 최대 1M 토큰 기준으로 정규화
                    context_score = min(context_length / 1000000.0, 1.0)
                    score += context_score * 0.25
                
                # 3. 가격 효율성 (저렴할수록 좋음, 무료면 최고점)
                pricing = model.get("pricing", {})
                prompt_price = pricing.get("prompt", 0)
                completion_price = pricing.get("completion", 0)
                
                # 타입 변환 (문자열일 수 있음)
                try:
                    if isinstance(prompt_price, str):
                        prompt_price = float(prompt_price) if prompt_price else 0
                    if isinstance(completion_price, str):
                        completion_price = float(completion_price) if completion_price else 0
                    prompt_price = float(prompt_price) if prompt_price else 0
                    completion_price = float(completion_price) if completion_price else 0
                except (ValueError, TypeError):
                    prompt_price = 0
                    completion_price = 0
                
                total_price = (prompt_price + completion_price) / 2.0
                if total_price == 0:
                    # 무료 모델은 최고 점수
                    score += 0.2
                elif total_price > 0:
                    # 가격이 낮을수록 높은 점수 (역수 사용, 최대 $0.01 기준)
                    price_score = min(0.01 / total_price, 1.0)  # $0.01 이상이면 낮은 점수
                    score += price_score * 0.15
                
                # 4. 기본 점수 (모든 모델에 동일하게 부여)
                score += 0.0  # 기본 점수 제거, 순수하게 데이터 기반으로만
                
                model_scores.append((model_id, score))
            
            # 점수 기준으로 정렬 (높은 점수부터)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_models = [model_id for model_id, _ in model_scores]
            
            _model_list_cache[cache_key] = {"models": sorted_models}
            _cache_timestamp[cache_key] = now
            logger.info(f"OpenRouter에서 {len(sorted_models)}개 사용 가능한 모델 가져옴 (동적 점수 기준 정렬)")
            return sorted_models
        else:
            logger.warning(f"OpenRouter API 오류: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"OpenRouter 모델 목록 가져오기 실패: {e}")
        return []


def _fetch_groq_models(api_key: str) -> List[str]:
    """Groq API에서 사용 가능한 모델 목록을 동적으로 가져와서 정렬"""
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
            # 실제 사용 가능한 모델만 필터링
            available_models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id:
                    available_models.append(model_id)
            
            # Groq는 모델 정보가 제한적이므로 알파벳 순서로 정렬 (하드코딩 없이)
            # 또는 모델 ID 길이 기준 (일반적으로 더 구체적인 이름이 더 최신일 수 있음)
            # 하지만 이것도 추측이므로 단순히 정렬만 수행
            sorted_models = sorted(available_models)
            
            _model_list_cache[cache_key] = {"models": sorted_models}
            _cache_timestamp[cache_key] = now
            logger.info(f"Groq에서 {len(sorted_models)}개 사용 가능한 모델 가져옴 (동적 정렬)")
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
            # 실제로 작동하는 모델만 사용 (최대 10개로 제한하여 빠른 fallback)
            for model in models[:10]:
                # OpenRouter 모델 ID는 이미 openrouter/ prefix가 포함되어 있음
                fallback_configs.append({
                    "provider": "openrouter",
                    "model": model,  # 이미 openrouter/ prefix 포함
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
            # 실제로 작동하는 모델만 사용 (최대 5개로 제한)
            for model in models[:5]:
                fallback_configs.append({
                    "provider": "groq",
                    "model": model,  # Groq 모델 ID는 그대로 사용
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key_env": "GROQ_API_KEY"
                })
        except Exception as e:
            logger.warning(f"Groq 모델 목록 가져오기 실패, 빈 목록 사용: {e}")
    
    return fallback_configs


class DirectHTTPLLM:
    """OpenAI 클라이언트 없이 직접 HTTP 요청을 사용하는 LLM 래퍼"""
    
    def __init__(self, model: str, base_url: str, api_key: str, provider: str):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider = provider
        self._client = None
    
    async def generate_str(self, message: str, request_params=None) -> str:
        """텍스트 생성 (직접 HTTP 요청)"""
        messages = [{"role": "user", "content": message}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # OpenRouter의 경우 추가 헤더
        if self.provider == "openrouter":
            headers.update({
                "HTTP-Referer": "https://github.com/mcp-agent",
                "X-Title": "MCP Agent Fallback"
            })
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']


def _try_fallback_llm(primary_model: str, logger_instance: Optional[logging.Logger] = None) -> Optional[Any]:
    """
    503 오류 발생 시 fallback LLM 생성 시도 (OpenAI 사용 안 함)
    
    Args:
        primary_model: 기본 모델명
        logger_instance: 로거 인스턴스 (선택)
        
    Returns:
        Fallback LLM 인스턴스 또는 None
    """
    log = logger_instance or logger
    fallback_configs = get_best_fallback_models()
    
    # OpenAI API 키가 설정되어 있어도 사용하지 않음 (fallback에서 제외)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        log.debug("OPENAI_API_KEY가 설정되어 있지만 fallback에서는 사용하지 않음")
    
    for config in fallback_configs:
        try:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                continue
            
            # OpenAI 기본 API는 사용하지 않음 (fallback에서 제외)
            if config["base_url"] == "https://api.openai.com/v1" or "api.openai.com" in config["base_url"]:
                log.debug(f"OpenAI 기본 API는 fallback에서 제외: {config['base_url']}")
                continue
            
            # OpenRouter는 모델 ID를 그대로 사용 (이미 올바른 형식)
            # Groq도 모델 ID를 그대로 사용
            model_id = config["model"]
            
            # 직접 HTTP 요청을 사용하는 LLM 래퍼 생성 (OpenAI 클라이언트 사용 안 함)
            llm = DirectHTTPLLM(
                model=model_id,
                base_url=config["base_url"],
                api_key=api_key,
                provider=config["provider"]
            )
            
            log.info(f"Fallback to {config['provider']} model: {model_id} (base_url: {config['base_url']})")
            return llm
        except Exception as fallback_error:
            log.debug(f"Fallback LLM 생성 실패 ({config.get('provider', 'unknown')}): {fallback_error}")
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
            # 503, 429, overloaded, unavailable 등 모든 에러에 대해 fallback 시도
            if ("503" in error_str or "429" in error_str or "overloaded" in error_str or 
                "unavailable" in error_str or "quota" in error_str or "resource_exhausted" in error_str):
                log.warning(f"Gemini API 오류 발생, fallback 모델로 전환: {e}")
                fallback_llm = _try_fallback_llm(primary_model, log)
                if fallback_llm:
                    return fallback_llm
                # 모든 fallback 실패 시 원래 오류 발생
                raise e
            else:
                # 다른 오류는 그대로 발생
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
            # 503, 429, overloaded, unavailable 등 모든 에러에 대해 fallback 시도
            if ("503" in error_str or "429" in error_str or "overloaded" in error_str or 
                "unavailable" in error_str or "quota" in error_str or "resource_exhausted" in error_str):
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
            # "Both GOOGLE_API_KEY..." 메시지 억제를 위해 stdout 일시 리다이렉트
            import sys
            import io
            from contextlib import redirect_stdout
            
            # 메시지 필터링을 위한 StringIO
            class MessageFilter(io.StringIO):
                def write(self, s):
                    # "Both GOOGLE_API_KEY" 메시지만 필터링
                    if "Both GOOGLE_API_KEY" not in s and "GEMINI_API_KEY" not in s:
                        super().write(s)
            
            filter_stream = MessageFilter()
            # stderr도 리다이렉트 (일부 메시지가 stderr로 출력될 수 있음)
            from contextlib import redirect_stderr
            with redirect_stdout(filter_stream), redirect_stderr(filter_stream):
                llm = GoogleAugmentedLLM(model=primary_model)
            return llm
        except Exception as e:
            error_str = str(e).lower()
            # 503, 429, overloaded, unavailable 등 모든 에러에 대해 fallback 시도
            if ("503" in error_str or "429" in error_str or "overloaded" in error_str or 
                "unavailable" in error_str or "quota" in error_str or "resource_exhausted" in error_str):
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
        # 503, 429, overloaded, unavailable, iterations 실패, ClientError 등 모든 에러에 대해 fallback 시도
        should_fallback = (
            "503" in error_str or 
            "429" in error_str or
            "overloaded" in error_str or 
            "unavailable" in error_str or
            "task failed to complete" in error_str or
            "failed to complete in" in error_str or
            "quota" in error_str or
            "resource_exhausted" in error_str or
            "clienterror" in error_str or
            "candidates" in error_str
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
                        # 모델 ID는 API에서 가져온 그대로 사용 (이미 올바른 형식)
                        model_id = config["model"]
                        # DirectHTTPLLM 사용 (OpenAI 클라이언트 사용 안 함)
                        llm = DirectHTTPLLM(
                            model=model_id,
                            base_url=config["base_url"],
                            api_key=api_key,
                            provider=config["provider"]
                        )
                        return llm
                    
                    fallback_orchestrator = Orchestrator(
                        llm_factory=create_fallback_llm,
                        available_agents=agents,
                        plan_type="full",
                        max_loops=max_loops
                    )
                    
                    # 모델 ID는 그대로 사용 (API에서 가져온 형식 그대로)
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

