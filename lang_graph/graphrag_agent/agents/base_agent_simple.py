"""
Simple Base Agent Class for GraphRAG Agents

This module provides a simplified base class for all GraphRAG agents
without complex dependencies.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


class BaseAgentConfig:
    """Base configuration for all agents"""
    def __init__(self, **kwargs):
        # Default values
        self.model_name = kwargs.get("model_name", "gpt-4o-mini")
        self.openai_api_key = kwargs.get("openai_api_key", "")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        self.timeout = kwargs.get("timeout", 30)
        self.retry_attempts = kwargs.get("retry_attempts", 3)
        self.enable_caching = kwargs.get("enable_caching", True)
        self.cache_ttl = kwargs.get("cache_ttl", 3600)
        self.enable_metrics = kwargs.get("enable_metrics", True)
        self.metrics_retention_days = kwargs.get("metrics_retention_days", 30)
        self.log_level = kwargs.get("log_level", "INFO")
        
        # Set any additional kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_updated: datetime = None


class BaseAgent(ABC):
    """Base class for all GraphRAG agents"""
    
    def __init__(self, config: BaseAgentConfig):
        self.config = config
        self.metrics = AgentMetrics()
        self.logger = self._setup_logging()
        self.cache = {} if config.enable_caching else None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper(), logging.INFO))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def call_llm(self, prompt: str, **kwargs) -> str:
        """Call the LLM with the given prompt"""
        # This is a placeholder implementation
        # In a real implementation, this would call the actual LLM
        self.logger.info(f"Calling LLM with prompt: {prompt[:100]}...")
        
        # Simulate LLM call
        await asyncio.sleep(0.1)
        
        # Mock response
        return f"Mock LLM response for: {prompt[:50]}..."
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return f"{self.__class__.__name__}_{hash(prompt)}"
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.cache:
            return None
        
        if key in self.cache:
            self.metrics.cache_hits += 1
            return self.cache[key]
        
        self.metrics.cache_misses += 1
        return None
    
    def _set_cache(self, key: str, value: str):
        """Set value in cache"""
        if not self.cache:
            return
        
        self.cache[key] = value
    
    def update_metrics(self, success: bool, response_time: float):
        """Update agent metrics"""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        total_time = self.metrics.average_response_time * (self.metrics.total_requests - 1)
        self.metrics.average_response_time = (total_time + response_time) / self.metrics.total_requests
        
        self.metrics.last_updated = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "average_response_time": self.metrics.average_response_time,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": (
                self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            ),
            "last_updated": self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
        }
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Process method to be implemented by subclasses"""
        pass
