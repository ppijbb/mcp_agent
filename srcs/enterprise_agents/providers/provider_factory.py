"""
Provider Factory
프로바이더 팩토리

Factory pattern for creating and managing data providers
"""

import logging
from typing import Dict, List, Optional, Any, Type
from srcs.enterprise_agents.models.providers import DataProvider, ProviderConfig, ProviderManager
from srcs.enterprise_agents.providers.openbb_provider import OpenBBProvider
from srcs.enterprise_agents.providers.korean_finance_provider import KoreanFinanceProvider
from srcs.enterprise_agents.providers.crypto_provider import CryptoProvider

logger = logging.getLogger(__name__)

class ProviderFactory:
    """Factory for creating and managing data providers"""
    
    # Registry of available provider classes
    _provider_registry: Dict[str, Type[DataProvider]] = {
        "openbb": OpenBBProvider,
        "korean_finance": KoreanFinanceProvider,
        "crypto": CryptoProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Optional[ProviderConfig] = None) -> Optional[DataProvider]:
        """Create a data provider instance"""
        provider_class = cls._provider_registry.get(provider_type.lower())
        if not provider_class:
            logger.error(f"Unknown provider type: {provider_type}")
            return None
        
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            return None
    
    @classmethod
    def create_default_manager(cls) -> ProviderManager:
        """Create provider manager with default providers"""
        manager = ProviderManager()
        
        # Default configurations for each provider
        default_configs = {
            "openbb": ProviderConfig(
                name="OpenBB Platform",
                enabled=True,
                priority=1,
                supported_markets=["KR", "US", "Global"],
                supported_categories=["stocks", "etf", "bonds"],
                timeout=30,
                cache_duration=300
            ),
            "korean_finance": ProviderConfig(
                name="Korean Finance",
                enabled=True,
                priority=2,
                supported_markets=["KR"],
                supported_categories=["savings", "real_estate", "bonds"],
                timeout=30,
                cache_duration=600
            ),
            "crypto": ProviderConfig(
                name="Crypto Markets",
                enabled=True,
                priority=3,
                supported_markets=["KR", "Global"],
                supported_categories=["crypto"],
                timeout=15,
                cache_duration=180
            )
        }
        
        # Create and register providers
        for provider_type, config in default_configs.items():
            provider = cls.create_provider(provider_type, config)
            if provider:
                manager.register_provider(provider)
                logger.info(f"Registered provider: {provider.name}")
        
        return manager
    
    @classmethod
    def create_custom_manager(cls, provider_configs: Dict[str, ProviderConfig]) -> ProviderManager:
        """Create provider manager with custom configurations"""
        manager = ProviderManager()
        
        for provider_type, config in provider_configs.items():
            provider = cls.create_provider(provider_type, config)
            if provider:
                manager.register_provider(provider)
                logger.info(f"Registered custom provider: {provider.name}")
        
        return manager
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type[DataProvider]) -> None:
        """Register a new provider class"""
        cls._provider_registry[provider_type.lower()] = provider_class
        logger.info(f"Registered new provider type: {provider_type}")
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider types"""
        return list(cls._provider_registry.keys())
    
    @classmethod
    def create_provider_for_market(cls, market: str, category: str = "stocks") -> Optional[DataProvider]:
        """Create best provider for specific market and category"""
        # Priority order for different markets
        market_priorities = {
            "KR": ["korean_finance", "openbb", "crypto"],
            "US": ["openbb", "crypto"],
            "Global": ["openbb", "crypto"]
        }
        
        category_priorities = {
            "crypto": ["crypto", "openbb"],
            "savings": ["korean_finance", "openbb"],
            "real_estate": ["korean_finance", "openbb"],
            "stocks": ["openbb", "korean_finance"],
            "etf": ["openbb", "korean_finance"]
        }
        
        # Get priority list based on market and category
        market_priority = market_priorities.get(market, ["openbb"])
        category_priority = category_priorities.get(category, ["openbb"])
        
        # Combine priorities (category first, then market)
        combined_priority = list(dict.fromkeys(category_priority + market_priority))
        
        # Try to create provider in priority order
        for provider_type in combined_priority:
            provider = cls.create_provider(provider_type)
            if provider and market in provider.get_supported_markets() and category in provider.get_supported_categories():
                logger.info(f"Selected provider {provider.name} for {market}/{category}")
                return provider
        
        # Fallback to any available provider
        logger.warning(f"No optimal provider found for {market}/{category}, using default")
        return cls.create_provider("openbb")
    
    @classmethod
    async def health_check_all(cls) -> Dict[str, bool]:
        """Perform health check on all provider types"""
        results = {}
        
        for provider_type in cls._provider_registry.keys():
            provider = cls.create_provider(provider_type)
            if provider:
                try:
                    is_healthy = await provider.health_check()
                    results[provider_type] = is_healthy
                    logger.info(f"Provider {provider_type} health: {'OK' if is_healthy else 'FAILED'}")
                except Exception as e:
                    results[provider_type] = False
                    logger.error(f"Provider {provider_type} health check failed: {e}")
            else:
                results[provider_type] = False
        
        return results 