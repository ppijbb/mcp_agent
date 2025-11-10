"""
Multi-Model LLM Manager for Smart Shopping Assistant Agent

Groq/OpenRouter 우선 사용, 오류 발생 시 Gemini/OpenAI/Claude로 자동 Fallback
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI as OpenRouterChat
except ImportError:
    OpenRouterChat = None

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """LLM Provider Enum"""
    GROQ = "groq"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


class ModelConfig:
    """모델 설정 클래스"""
    
    def __init__(
        self,
        name: str,
        provider: ModelProvider,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        cost_per_1k_tokens: float = 0.0,
        api_key_env: Optional[str] = None,
    ):
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.api_key_env = api_key_env or self._get_default_api_key_env()
        self.enabled = self._check_api_key()
    
    def _get_default_api_key_env(self) -> str:
        """기본 API 키 환경 변수 이름 반환"""
        env_map = {
            ModelProvider.GROQ: "GROQ_API_KEY",
            ModelProvider.OPENROUTER: "OPENROUTER_API_KEY",
            ModelProvider.GEMINI: "GOOGLE_API_KEY",
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.CLAUDE: "ANTHROPIC_API_KEY",
        }
        return env_map.get(self.provider, "")
    
    def _check_api_key(self) -> bool:
        """API 키 존재 여부 확인"""
        api_key = os.getenv(self.api_key_env)
        return api_key is not None and api_key.strip() != ""


class ModelManager:
    """
    Multi-Model LLM Manager
    
    우선순위: Groq → OpenRouter → Gemini → OpenAI → Claude
    오류 발생 시 자동으로 다음 모델로 Fallback
    """
    
    def __init__(self, budget_limit: Optional[float] = None):
        """
        ModelManager 초기화
        
        Args:
            budget_limit: 예산 제한 (선택)
        """
        self.budget_limit = budget_limit
        self.current_cost = 0.0
        self.models: Dict[str, ModelConfig] = {}
        self.model_clients: Dict[str, BaseChatModel] = {}
        self.fallback_order: List[ModelProvider] = [
            ModelProvider.GROQ,
            ModelProvider.OPENROUTER,
            ModelProvider.GEMINI,
            ModelProvider.OPENAI,
            ModelProvider.CLAUDE,
        ]
        
        self._initialize_models()
        self._initialize_clients()
    
    def _initialize_models(self):
        """모델 설정 초기화"""
        # 1. Groq 모델 (무료, 우선순위 1)
        self.models["groq-llama-70b"] = ModelConfig(
            name="groq-llama-70b",
            provider=ModelProvider.GROQ,
            model_id="llama-3.1-70b-versatile",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0,
        )
        
        self.models["groq-mixtral"] = ModelConfig(
            name="groq-mixtral",
            provider=ModelProvider.GROQ,
            model_id="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0,
        )
        
        # 2. OpenRouter 모델 (무료 모델 우선, 우선순위 2)
        self.models["openrouter-gemini-flash"] = ModelConfig(
            name="openrouter-gemini-flash",
            provider=ModelProvider.OPENROUTER,
            model_id="google/gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0,
        )
        
        self.models["openrouter-llama-3b"] = ModelConfig(
            name="openrouter-llama-3b",
            provider=ModelProvider.OPENROUTER,
            model_id="meta-llama/llama-3.2-3b-instruct",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0,
        )
        
        # 3. Gemini 모델 (Fallback 1)
        self.models["gemini-flash-lite"] = ModelConfig(
            name="gemini-flash-lite",
            provider=ModelProvider.GEMINI,
            model_id="gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0001,
        )
        
        # 4. OpenAI 모델 (Fallback 2)
        self.models["gpt-4o-mini"] = ModelConfig(
            name="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.15,
        )
        
        # 5. Claude 모델 (Fallback 3)
        self.models["claude-sonnet"] = ModelConfig(
            name="claude-sonnet",
            provider=ModelProvider.CLAUDE,
            model_id="claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=3.0,
        )
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def _initialize_clients(self):
        """모델 클라이언트 초기화"""
        for name, config in self.models.items():
            if not config.enabled:
                logger.warning(f"Model {name} disabled - API key not found")
                continue
            
            try:
                client = self._create_client(config)
                if client:
                    self.model_clients[name] = client
                    logger.info(f"Initialized client for {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize client for {name}: {e}")
    
    def _create_client(self, config: ModelConfig) -> Optional[BaseChatModel]:
        """모델 클라이언트 생성"""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            return None
        
        try:
            if config.provider == ModelProvider.GROQ and ChatGroq:
                return ChatGroq(
                    model=config.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    groq_api_key=api_key,
                )
            
            elif config.provider == ModelProvider.OPENROUTER and OpenRouterChat:
                return ChatOpenAI(
                    model=config.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    openai_api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/your-repo",
                        "X-Title": "Smart Shopping Assistant Agent",
                    },
                )
            
            elif config.provider == ModelProvider.GEMINI and ChatGoogleGenerativeAI:
                return ChatGoogleGenerativeAI(
                    model=config.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    google_api_key=api_key,
                )
            
            elif config.provider == ModelProvider.OPENAI and ChatOpenAI:
                return ChatOpenAI(
                    model=config.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    openai_api_key=api_key,
                )
            
            elif config.provider == ModelProvider.CLAUDE and ChatAnthropic:
                return ChatAnthropic(
                    model=config.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    anthropic_api_key=api_key,
                )
            
            return None
        
        except Exception as e:
            logger.error(f"Error creating client for {config.name}: {e}")
            return None
    
    def get_llm(self, preferred_provider: Optional[ModelProvider] = None) -> Optional[BaseChatModel]:
        """
        사용 가능한 LLM 반환 (우선순위에 따라)
        
        Args:
            preferred_provider: 선호하는 Provider (선택)
        
        Returns:
            BaseChatModel 인스턴스 또는 None
        """
        # 선호하는 Provider가 있으면 해당 Provider의 모델부터 시도
        if preferred_provider:
            for name, config in self.models.items():
                if config.provider == preferred_provider and config.enabled:
                    if name in self.model_clients:
                        return self.model_clients[name]
        
        # Fallback 순서에 따라 시도
        for provider in self.fallback_order:
            for name, config in self.models.items():
                if config.provider == provider and config.enabled:
                    if name in self.model_clients:
                        logger.info(f"Using model: {name} (provider: {provider.value})")
                        return self.model_clients[name]
        
        logger.error("No available LLM models found")
        return None
    
    def get_llm_by_name(self, model_name: str) -> Optional[BaseChatModel]:
        """
        모델 이름으로 LLM 반환
        
        Args:
            model_name: 모델 이름
        
        Returns:
            BaseChatModel 인스턴스 또는 None
        """
        if model_name in self.model_clients:
            return self.model_clients[model_name]
        
        logger.warning(f"Model {model_name} not found or not initialized")
        return None
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.model_clients.keys())
    
    def estimate_cost(self, tokens: int, model_name: str) -> float:
        """
        토큰 수 기반 비용 추정
        
        Args:
            tokens: 토큰 수
            model_name: 모델 이름
        
        Returns:
            예상 비용
        """
        if model_name not in self.models:
            return 0.0
        
        config = self.models[model_name]
        cost = (tokens / 1000) * config.cost_per_1k_tokens
        return cost
    
    def check_budget(self) -> bool:
        """예산 제한 확인"""
        if self.budget_limit is None:
            return True
        
        return self.current_cost < self.budget_limit

