"""
Fallback Handler for LLM Model Management

오류 발생 시 자동으로 다음 모델로 전환하는 메커니즘
"""

import logging
from typing import Optional, Callable, Any, Dict, List
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.exceptions import LangChainException

from .model_manager import ModelManager, ModelProvider

logger = logging.getLogger(__name__)


class FallbackError(Exception):
    """Fallback 관련 오류"""
    pass


class FallbackHandler:
    """
    LLM Fallback Handler
    
    모델 오류 발생 시 자동으로 다음 모델로 전환
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        FallbackHandler 초기화
        
        Args:
            model_manager: ModelManager 인스턴스
        """
        self.model_manager = model_manager
        self.fallback_history: List[Dict[str, Any]] = []
        self.max_retries = 3
    
    def invoke_with_fallback(
        self,
        messages: List[BaseMessage],
        preferred_provider: Optional[ModelProvider] = None,
        **kwargs
    ) -> Any:
        """
        Fallback 메커니즘을 사용하여 LLM 호출
        
        Args:
            messages: 입력 메시지 리스트
            preferred_provider: 선호하는 Provider
            **kwargs: 추가 인자
        
        Returns:
            LLM 응답
        
        Raises:
            FallbackError: 모든 모델 실패 시
        """
        tried_models: List[str] = []
        last_error: Optional[Exception] = None
        
        # 선호하는 Provider부터 시도
        if preferred_provider:
            llm = self.model_manager.get_llm(preferred_provider)
            if llm:
                result = self._try_invoke(llm, messages, tried_models, **kwargs)
                if result is not None:
                    return result
        
        # Fallback 순서에 따라 시도
        for provider in self.model_manager.fallback_order:
            llm = self.model_manager.get_llm(provider)
            if llm is None:
                continue
            
            # 이미 시도한 모델은 건너뛰기
            model_name = self._get_model_name(llm)
            if model_name in tried_models:
                continue
            
            result = self._try_invoke(llm, messages, tried_models, **kwargs)
            if result is not None:
                return result
        
        # 모든 모델 실패
        error_msg = f"All models failed. Tried: {tried_models}"
        logger.error(error_msg)
        self._log_fallback_failure(tried_models, last_error)
        raise FallbackError(error_msg)
    
    def _try_invoke(
        self,
        llm: BaseChatModel,
        messages: List[BaseMessage],
        tried_models: List[str],
        **kwargs
    ) -> Optional[Any]:
        """
        LLM 호출 시도
        
        Args:
            llm: LLM 인스턴스
            messages: 입력 메시지
            tried_models: 시도한 모델 목록
            **kwargs: 추가 인자
        
        Returns:
            LLM 응답 또는 None (실패 시)
        """
        model_name = self._get_model_name(llm)
        
        try:
            logger.info(f"Attempting to invoke model: {model_name}")
            response = llm.invoke(messages, **kwargs)
            
            logger.info(f"Successfully invoked model: {model_name}")
            self._log_fallback_success(model_name)
            return response
        
        except LangChainException as e:
            logger.warning(f"LangChain error with {model_name}: {e}")
            tried_models.append(model_name)
            self._log_fallback_attempt(model_name, str(e), success=False)
            return None
        
        except Exception as e:
            logger.warning(f"Error with {model_name}: {e}")
            tried_models.append(model_name)
            self._log_fallback_attempt(model_name, str(e), success=False)
            return None
    
    def _get_model_name(self, llm: BaseChatModel) -> str:
        """LLM 인스턴스에서 모델 이름 추출"""
        # LangChain 모델의 모델 이름 추출 시도
        if hasattr(llm, "model_name"):
            return llm.model_name
        elif hasattr(llm, "model"):
            return llm.model
        else:
            return str(type(llm).__name__)
    
    def _log_fallback_attempt(
        self,
        model_name: str,
        error: str,
        success: bool
    ):
        """Fallback 시도 로깅"""
        self.fallback_history.append({
            "model": model_name,
            "error": error,
            "success": success,
        })
    
    def _log_fallback_success(self, model_name: str):
        """Fallback 성공 로깅"""
        self._log_fallback_attempt(model_name, "", success=True)
    
    def _log_fallback_failure(
        self,
        tried_models: List[str],
        last_error: Optional[Exception]
    ):
        """Fallback 실패 로깅"""
        logger.error(
            f"Fallback failed after trying {len(tried_models)} models: {tried_models}"
        )
        if last_error:
            logger.error(f"Last error: {last_error}")
    
    def get_fallback_history(self) -> List[Dict[str, Any]]:
        """Fallback 이력 반환"""
        return self.fallback_history.copy()
    
    def clear_history(self):
        """Fallback 이력 초기화"""
        self.fallback_history.clear()

