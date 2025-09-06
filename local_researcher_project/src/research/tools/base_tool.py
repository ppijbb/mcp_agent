"""
Base Search Tool

This module provides the base class for all search tools.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result data structure."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSearchTool(ABC):
    """Base class for all search tools."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the search tool.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.timeout = config.get('timeout', 30)
        self.max_results = config.get('max_results', 10)
        
        logger.info(f"Initialized search tool: {self.name}")
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Perform a search operation.
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def get_content(self, url: str) -> Optional[str]:
        """Get content from a URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Content string or None if failed
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if the tool is enabled."""
        return self.enabled
    
    def get_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return self.config
    
    async def validate_query(self, query: str) -> bool:
        """Validate search query.
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not query or not query.strip():
            return False
        
        if len(query.strip()) < 2:
            return False
        
        return True
    
    def _calculate_relevance_score(self, query: str, title: str, snippet: str) -> float:
        """Calculate relevance score for a result.
        
        Args:
            query: Original search query
            title: Result title
            snippet: Result snippet
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        snippet_words = set(snippet.lower().split())
        
        # Calculate word overlap
        title_overlap = len(query_words.intersection(title_words)) / len(query_words)
        snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words)
        
        # Weight title more heavily than snippet
        score = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        
        return min(score, 1.0)
    
    async def _make_request(self, url: str, headers: Dict[str, str] = None) -> Optional[Dict]:
        """Make HTTP request with error handling.
        
        Args:
            url: URL to request
            headers: HTTP headers
            
        Returns:
            Response data or None if failed
        """
        try:
            import aiohttp
            
            if headers is None:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()
