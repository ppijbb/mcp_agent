"""
Multi-Model LLM Manager for Financial Agent

Manages multiple LLM providers with fallback support for financial analysis tasks.
Supports Groq, OpenRouter, Gemini, OpenAI, and Claude with automatic failover.

Classes:
    ModelProvider: Enum of supported LLM providers
    ModelConfig: Configuration for individual model instances
    ModelManager: Multi-model LLM manager with fallback support

Example:
    >>> manager = ModelManager(budget_limit=100.0)
    >>> llm = manager.get_llm(preferred_provider=ModelProvider.GROQ)
    >>> cost = manager.estimate_cost(1000, "gpt-4o-mini")
"""

import os
import logging
from enum import Enum
from typing import Dict, List, Optional

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_openai import ChatOpenAI as OpenRouterChat
except ImportError:
    OpenRouterChat = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM provider types."""
    GROQ = "groq"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


class ModelConfig:
    """
    Configuration for individual model instances.
    
    Attributes:
        name: Human-readable model name
        provider: LLM provider type
        model_id: Provider-specific model identifier
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        cost_per_1k_tokens: Cost per 1000 tokens for budget tracking
        api_key_env: Environment variable name for API key
        enabled: Whether model is available (has valid API key)
    """
    
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
        """Get the default environment variable name for the API key."""
        env_map = {
            ModelProvider.GROQ: "GROQ_API_KEY",
            ModelProvider.OPENROUTER: "OPENROUTER_API_KEY",
            ModelProvider.GEMINI: "GOOGLE_API_KEY",
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.CLAUDE: "ANTHROPIC_API_KEY",
        }
        return env_map.get(self.provider, "")
    
    def _check_api_key(self) -> bool:
        """Check if the API key exists and is non-empty."""
        api_key = os.getenv(self.api_key_env)
        return api_key is not None and api_key.strip() != ""


class ModelManager:
    """
    Multi-Model LLM Manager with fallback support.
    
    Manages multiple LLM providers and automatically falls back to
    the next available provider when an error occurs.
    
    Priority order: Groq -> OpenRouter -> Gemini -> OpenAI -> Claude
    
    Attributes:
        budget_limit: Optional spending limit for token usage
        current_cost: Accumulated cost from model usage
        models: Dictionary of configured models
        model_clients: Dictionary of initialized LLM clients
        fallback_order: Priority order for provider fallback
    """
    
    def __init__(self, budget_limit: Optional[float] = None):
        """
        Initialize the ModelManager.
        
        Args:
            budget_limit: Optional budget limit for token costs
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
    
    def _initialize_models(self) -> None:
        """Initialize model configurations for all supported providers."""
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
        
        self.models["gemini-flash-lite"] = ModelConfig(
            name="gemini-flash-lite",
            provider=ModelProvider.GEMINI,
            model_id="gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.0001,
        )
        
        self.models["gpt-4o-mini"] = ModelConfig(
            name="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=0.15,
        )
        
        self.models["claude-sonnet"] = ModelConfig(
            name="claude-sonnet",
            provider=ModelProvider.CLAUDE,
            model_id="claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_tokens=4000,
            cost_per_1k_tokens=3.0,
        )
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def _initialize_clients(self) -> None:
        """Initialize LLM clients for enabled models."""
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
        """
        Create an LLM client for the given model configuration.
        
        Args:
            config: Model configuration with provider details
            
        Returns:
            Initialized BaseChatModel or None if creation fails
        """
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
                        "X-Title": "Financial Agent",
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
        Get an available LLM client based on priority order.
        
        Args:
            preferred_provider: Optional preferred provider to try first
            
        Returns:
            BaseChatModel instance or None if no models available
        """
        if preferred_provider:
            for name, config in self.models.items():
                if config.provider == preferred_provider and config.enabled:
                    if name in self.model_clients:
                        return self.model_clients[name]
        
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
        Get a specific LLM client by model name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            BaseChatModel instance or None if not found
        """
        if model_name in self.model_clients:
            return self.model_clients[model_name]
        
        logger.warning(f"Model {model_name} not found or not initialized")
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_clients.keys())
    
    def estimate_cost(self, tokens: int, model_name: str) -> float:
        """
        Estimate the cost for a given token count.
        
        Args:
            tokens: Number of tokens
            model_name: Name of the model to estimate cost for
            
        Returns:
            Estimated cost in the configured currency
        """
        if model_name not in self.models:
            return 0.0
        
        config = self.models[model_name]
        return (tokens / 1000) * config.cost_per_1k_tokens
    
    def check_budget(self) -> bool:
        """Check if current cost is within budget limit."""
        if self.budget_limit is None:
            return True
        
        return self.current_cost < self.budget_limit

