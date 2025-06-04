"""
Provider Models
데이터 제공자 모델

Pydantic models for data provider configurations and abstractions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, SecretStr
from datetime import datetime

from srcs.enterprise_agents.models.financial_data import DynamicFinancialData, DynamicProduct, MarketInsight

class APICredentials(BaseModel):
    """API credentials with secure storage"""
    provider_name: str = Field(..., description="Provider name")
    api_key: Optional[SecretStr] = Field(None, description="API key (encrypted)")
    api_secret: Optional[SecretStr] = Field(None, description="API secret (encrypted)")
    base_url: Optional[str] = Field(None, description="Base API URL")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }

class ProviderConfig(BaseModel):
    """Data provider configuration"""
    name: str = Field(..., description="Provider name")
    enabled: bool = Field(True, description="Whether provider is enabled")
    priority: int = Field(1, ge=1, le=10, description="Provider priority (1=highest)")
    supported_markets: List[str] = Field(default_factory=list, description="Supported market codes")
    supported_categories: List[str] = Field(default_factory=list, description="Supported asset categories")
    credentials: Optional[APICredentials] = Field(None, description="API credentials")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")
    retry_count: int = Field(3, ge=0, description="Number of retries")
    cache_duration: int = Field(300, ge=0, description="Cache duration in seconds")
    
    class Config:
        validate_assignment = True

class DataProvider(ABC):
    """Abstract base class for data providers with configuration"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self._cache: Dict[str, Any] = {}
        self._last_cache_time: Dict[str, datetime] = {}
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> List[DynamicFinancialData]:
        """Get market data for given symbols"""
        pass
    
    @abstractmethod
    async def search_products(self, category: str, criteria: Dict[str, Any]) -> List[DynamicProduct]:
        """Search for financial products based on criteria"""
        pass
    
    @abstractmethod 
    async def get_market_insights(self, market: str = "KR") -> List[MarketInsight]:
        """Get market insights and analysis"""
        pass
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._last_cache_time:
            return False
        
        time_diff = (datetime.now() - self._last_cache_time[cache_key]).total_seconds()
        return time_diff < self.config.cache_duration
    
    def set_cache(self, cache_key: str, data: Any) -> None:
        """Set cached data"""
        self._cache[cache_key] = data
        self._last_cache_time[cache_key] = datetime.now()
    
    def get_cache(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid"""
        if self.is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None
    
    async def health_check(self) -> bool:
        """Check if provider is healthy and accessible"""
        try:
            # Basic health check - try to get minimal data
            test_data = await self.get_market_data(["005930.KS"])  # Samsung Electronics
            return len(test_data) > 0
        except Exception:
            return False
    
    def get_supported_markets(self) -> List[str]:
        """Get list of supported markets"""
        return self.config.supported_markets
    
    def get_supported_categories(self) -> List[str]:
        """Get list of supported asset categories"""
        return self.config.supported_categories

class ProviderManager(BaseModel):
    """Manager for multiple data providers"""
    providers: Dict[str, DataProvider] = Field(default_factory=dict, description="Registered providers")
    default_provider: Optional[str] = Field(None, description="Default provider name")
    failover_enabled: bool = Field(True, description="Enable failover to backup providers")
    
    class Config:
        arbitrary_types_allowed = True
    
    def register_provider(self, provider: DataProvider) -> None:
        """Register a new data provider"""
        self.providers[provider.name] = provider
        if self.default_provider is None:
            self.default_provider = provider.name
    
    def get_provider(self, name: Optional[str] = None) -> Optional[DataProvider]:
        """Get provider by name or default"""
        provider_name = name or self.default_provider
        return self.providers.get(provider_name) if provider_name else None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [name for name, provider in self.providers.items() if provider.enabled]
    
    async def get_best_provider(self, market: str = "KR", category: str = "stocks") -> Optional[DataProvider]:
        """Get best available provider for specific market and category"""
        suitable_providers = []
        
        for provider in self.providers.values():
            if (provider.enabled and 
                market in provider.get_supported_markets() and 
                category in provider.get_supported_categories()):
                suitable_providers.append(provider)
        
        if not suitable_providers:
            return None
        
        # Sort by priority (lower number = higher priority)
        suitable_providers.sort(key=lambda p: p.config.priority)
        
        # Check health of highest priority providers
        for provider in suitable_providers:
            if await provider.health_check():
                return provider
        
        # If no healthy provider found, return highest priority one
        return suitable_providers[0] if suitable_providers else None 