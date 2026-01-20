"""
LLM Provider Configuration Module
Unified interface for multiple LLM providers: OpenRouter, Groq, Cerebras, OpenAI, Anthropic, Google

This module provides a consistent interface for accessing different LLM providers
with support for automatic fallback and RANDOM provider selection from free providers.

Free Providers (무료): OpenRouter, Groq, Cerebras
Paid Providers (유료): OpenAI, Anthropic, Google (only if API key configured)
"""

import os
import random
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENROUTER = "openrouter"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider"""
    provider: LLMProvider
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for OpenAI-compatible API"""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.additional_params
        }


# Provider configurations with free/paid info
# is_free: True for free tier available, False for paid only
# free_models: List of models available on free tier
PROVIDER_CONFIGS = {
    LLMProvider.OPENROUTER: {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "anthropic/claude-3.5-sonnet",
        "is_free": True,  # Has free tier
        "free_models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-405b"]
    },
    LLMProvider.GROQ: {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.1-70b-versatile",
        "is_free": True,  # Has free tier
        "free_models": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    },
    LLMProvider.CEREBRAS: {
        "env_key": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "llama-3.1-70b",
        "is_free": True,  # Has free tier
        "free_models": ["llama-3.1-70b", "llama-3.1-8b"]
    },
    LLMProvider.OPENAI: {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o",
        "is_free": False,  # Paid only
        "free_models": []
    },
    LLMProvider.ANTHROPIC: {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "model_env": "ANTHROPIC_MODEL",
        "default_model": "claude-3-5-sonnet-20241022",
        "is_free": False,  # Paid only
        "free_models": []
    },
    LLMProvider.GOOGLE: {
        "env_key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_env": "GOOGLE_MODEL",
        "default_model": "gemini-1.5-pro",
        "is_free": False,  # Paid only
        "free_models": []
    }
}

# Free providers that don't require paid API keys
FREE_PROVIDERS = [LLMProvider.OPENROUTER, LLMProvider.GROQ, LLMProvider.CEREBRAS]


class LLMProviderManager:
    """
    Manages LLM provider configurations and provides a unified interface
    for accessing different LLM providers with automatic fallback support.
    """
    
    def __init__(self):
        self._configs: Dict[LLMProvider, Optional[LLMConfig]] = {}
        self._primary_provider = self._get_primary_provider()
        self._fallback_providers = self._get_fallback_providers()
        self._load_all_configs()
    
    def _get_primary_provider(self) -> LLMProvider:
        """Get primary LLM provider from environment"""
        provider_name = os.getenv("LLM_PROVIDER", "openrouter").lower()
        try:
            return LLMProvider(provider_name)
        except ValueError:
            logger.warning(f"Unknown provider: {provider_name}, using openrouter")
            return LLMProvider.OPENROUTER
    
    def _get_fallback_providers(self) -> List[LLMProvider]:
        """Get fallback providers from environment"""
        fallback = os.getenv("LLM_FALLBACK_PROVIDER", "groq").lower()
        try:
            provider = LLMProvider(fallback)
            if provider != self._primary_provider:
                return [provider]
            return []
        except ValueError:
            return []
    
    def _load_config(self, provider: LLMProvider) -> Optional[LLMConfig]:
        """Load configuration for a single provider"""
        config_info = PROVIDER_CONFIGS[provider]
        
        # Get API key
        api_key = os.getenv(str(config_info["env_key"]))
        if not api_key:
            logger.debug(f"API key not found for {provider.value}")
            return None
        
        # Get base URL
        base_url = os.getenv(f"{provider.value.upper()}_BASE_URL", str(config_info["base_url"]))
        
        # Get model
        model = os.getenv(str(config_info["model_env"]), str(config_info["default_model"]))
        
        # Get temperature
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError:
            temperature = 0.7
        
        # Get max tokens
        try:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        except ValueError:
            max_tokens = 4096
        
        # Get top_p
        try:
            top_p = float(os.getenv("LLM_TOP_P", "0.9"))
        except ValueError:
            top_p = 0.9
        
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    
    def _load_all_configs(self):
        """Load configurations for all providers"""
        for provider in LLMProvider:
            self._configs[provider] = self._load_config(provider)
    
    def get_config(self, provider: Optional[LLMProvider] = None) -> Optional[LLMConfig]:
        """Get configuration for a specific provider or primary provider"""
        if provider is None:
            provider = self._primary_provider
        
        config = self._configs.get(provider)
        if config is None:
            # Try to load on demand
            config = self._load_config(provider)
            if config:
                self._configs[provider] = config
        
        return config
    
    def get_primary_config(self) -> Optional[LLMConfig]:
        """Get primary provider configuration"""
        return self.get_config(self._primary_provider)
    
    def get_fallback_config(self) -> Optional[LLMConfig]:
        """Get fallback provider configuration"""
        for provider in self._fallback_providers:
            config = self.get_config(provider)
            if config:
                return config
        return None
    
    def get_with_fallback(self) -> Optional[LLMConfig]:
        """Get primary config with fallback support"""
        primary = self.get_primary_config()
        if primary:
            return primary
        return self.get_fallback_config()
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available (configured) providers"""
        return [p for p, config in self._configs.items() if config is not None]
    
    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is configured and available"""
        return self._configs.get(provider) is not None
    
    def get_provider_for_model(self, model: str) -> Optional[LLMProvider]:
        """Infer provider from model name"""
        model_lower = model.lower()
        
        # Check provider-specific model patterns
        if "claude" in model_lower:
            # Prefer OpenRouter or Anthropic for Claude
            if self.is_provider_available(LLMProvider.OPENROUTER):
                return LLMProvider.OPENROUTER
            return LLMProvider.ANTHROPIC
        elif "gpt-" in model_lower or "gpt4" in model_lower:
            # Prefer OpenAI or OpenRouter for GPT models
            if self.is_provider_available(LLMProvider.OPENAI):
                return LLMProvider.OPENAI
            return LLMProvider.OPENROUTER
        elif "llama" in model_lower:
            # Groq and Cerebras are great for Llama
            if self.is_provider_available(LLMProvider.GROQ):
                return LLMProvider.GROQ
            elif self.is_provider_available(LLMProvider.CEREBRAS):
                return LLMProvider.CEREBRAS
        elif "gemini" in model_lower:
            return LLMProvider.GOOGLE
        
        # Default to primary provider
        return self._primary_provider if self.is_provider_available(self._primary_provider) else None
    
    def get_openai_compatible_config(self, provider: Optional[LLMProvider] = None) -> Dict[str, Any]:
        """Get OpenAI-compatible configuration dict for use with OpenAI SDK"""
        config = self.get_config(provider)
        if config:
            return config.to_dict()
        return {}
    
    # =========================================================================
    # RANDOM FREE PROVIDER SELECTION
    # =========================================================================
    
    def get_free_providers(self) -> List[LLMProvider]:
        """Get list of configured FREE providers (OpenRouter, Groq, Cerebras)"""
        return [p for p in FREE_PROVIDERS if self.is_provider_available(p)]
    
    def get_all_configured_providers(self) -> List[LLMProvider]:
        """Get ALL configured providers including paid ones (OpenAI, Google if API key set)"""
        return self.get_available_providers()
    
    def get_random_free_provider(self) -> Optional[LLMConfig]:
        """
        Get a random FREE provider config (OpenRouter, Groq, or Cerebras)
        This is the DEFAULT behavior when no provider is specified.
        
        Returns:
            Random LLMConfig from free providers, or None if none configured
        """
        free_providers = self.get_free_providers()
        if not free_providers:
            logger.warning("No free providers configured! Check your API keys.")
            return None
        
        # Random selection
        selected_provider = random.choice(free_providers)
        config = self.get_config(selected_provider)
        
        if config:
            logger.info(f"🎲 Random free provider selected: {selected_provider.value} ({config.model})")
        
        return config
    
    def get_random_provider(self, include_paid: bool = True) -> Optional[LLMConfig]:
        """
        Get a randomly selected provider from ALL configured providers.
        
        Args:
            include_paid: If True, includes OpenAI/Google if they have API keys.
                         If False, only free providers (OpenRouter, Groq, Cerebras).
        
        Returns:
            Random LLMConfig from configured providers
        """
        if include_paid:
            providers = self.get_all_configured_providers()
        else:
            providers = self.get_free_providers()
        
        if not providers:
            logger.warning("No providers available!")
            return None
        
        selected_provider = random.choice(providers)
        config = self.get_config(selected_provider)
        
        if config:
            provider_type = "free" if selected_provider in FREE_PROVIDERS else "paid"
            logger.info(f"🎲 Random {provider_type} provider selected: {selected_provider.value} ({config.model})")
        
        return config


class LLMClientFactory:
    """Factory for creating LLM clients based on provider"""
    
    @staticmethod
    def create_client(config: LLMConfig):
        """Create an LLM client based on configuration"""
        import openai
        
        # All providers support OpenAI-compatible API
        return openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    @staticmethod
    def create_async_client(config: LLMConfig):
        """Create an async LLM client based on configuration"""
        import openai
        
        # All providers support OpenAI-compatible API
        return openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    @staticmethod
    async def create_chat_completion(
        client,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion using the provided client"""
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "provider": response.object
        }


@lru_cache(maxsize=1)
def get_llm_manager() -> LLMProviderManager:
    """Get singleton LLM provider manager instance"""
    return LLMProviderManager()


def get_llm_config(provider: Optional[str] = None) -> Optional[LLMConfig]:
    """
    Convenience function to get LLM configuration
    
    Args:
        provider: Provider name (openrouter, groq, cerebras, openai, anthropic, google)
                 If None, uses primary provider from environment
    
    Returns:
        LLMConfig or None if provider not configured
    """
    manager = get_llm_manager()
    if provider:
        try:
            return manager.get_config(LLMProvider(provider.lower()))
        except ValueError:
            return manager.get_primary_config()
    return manager.get_primary_config()


def get_available_providers() -> List[str]:
    """Get list of available provider names"""
    manager = get_llm_manager()
    return [p.value for p in manager.get_available_providers()]


def switch_provider(provider: str) -> bool:
    """
    Switch the primary LLM provider
    
    Args:
        provider: Provider name to switch to
    
    Returns:
        True if switch successful, False otherwise
    """
    try:
        provider_enum = LLMProvider(provider.lower())
        manager = get_llm_manager()
        if manager.is_provider_available(provider_enum):
            os.environ["LLM_PROVIDER"] = provider.lower()
            # Clear the cached manager to pick up new provider
            get_llm_manager.cache_clear()
            logger.info(f"Switched to provider: {provider}")
            return True
        else:
            logger.warning(f"Provider {provider} is not available")
            return False
    except ValueError:
        logger.error(f"Unknown provider: {provider}")
        return False


# =============================================================================
# RANDOM FREE PROVIDER FUNCTIONS (DEFAULT BEHAVIOR)
# =============================================================================

def get_random_free_config() -> Optional[LLMConfig]:
    """
    Get a random FREE provider config (OpenRouter, Groq, or Cerebras)
    
    This is the DEFAULT behavior when no provider is specified.
    Use this function to automatically rotate between free providers.
    
    Returns:
        Random LLMConfig from free providers (OpenRouter, Groq, Cerebras)
        Returns None if no free providers are configured.
    
    Example:
        >>> config = get_random_free_config()
        >>> if config:
        ...     print(f"Using: {config.provider.value} - {config.model}")
        ... else:
        ...     print("No free providers configured!")
    """
    manager = get_llm_manager()
    return manager.get_random_free_provider()


def get_random_config(include_paid: bool = True) -> Optional[LLMConfig]:
    """
    Get a randomly selected provider from ALL configured providers.
    
    Args:
        include_paid: If True, includes OpenAI/Google if they have API keys.
                     If False, only free providers (OpenRouter, Groq, Cerebras).
    
    Returns:
        Random LLMConfig from configured providers
    
    Example:
        >>> # Only free providers
        >>> config = get_random_config(include_paid=False)
        >>> 
        >>> # All configured providers
        >>> config = get_random_config(include_paid=True)
    """
    manager = get_llm_manager()
    return manager.get_random_provider(include_paid=include_paid)


def get_free_providers_list() -> List[str]:
    """Get list of configured free provider names"""
    manager = get_llm_manager()
    return [p.value for p in manager.get_free_providers()]


def get_all_providers_list() -> List[str]:
    """Get list of ALL configured provider names (free + paid)"""
    manager = get_llm_manager()
    return [p.value for p in manager.get_all_configured_providers()]

