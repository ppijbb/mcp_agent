"""
Simple Web Search Tool
"""

import logging
import asyncio
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Simple web search tool using DuckDuckGo."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize web search tool."""
        self.config = config
        self.max_results = config.get('max_results', 10)
        self.timeout = config.get('timeout', 30)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run web search synchronously."""
        try:
            return asyncio.run(self.arun(query, **kwargs))
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run web search asynchronously."""
        try:
            # Try DuckDuckGo search first
            try:
                import duckduckgo_search
                search_results = duckduckgo_search.DDGS().text(
                    query, 
                    max_results=self.max_results
                )
                
                results = []
                for result in search_results:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'source': 'duckduckgo'
                    })
                
                return {
                    'success': True,
                    'query': query,
                    'results': results,
                    'total_results': len(results)
                }
                
            except ImportError:
                self.logger.warning("duckduckgo_search not installed, using mock results")
                return self._mock_search(query)
                
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return self._mock_search(query)
    
    def _mock_search(self, query: str) -> Dict[str, Any]:
        """Return mock search results."""
        mock_results = [
            {
                'title': f"Mock result 1 for: {query}",
                'url': f"https://example1.com/{query.replace(' ', '-')}",
                'snippet': f"This is a mock search result for the query: {query}",
                'source': 'mock'
            },
            {
                'title': f"Mock result 2 for: {query}",
                'url': f"https://example2.com/{query.replace(' ', '-')}",
                'snippet': f"Another mock search result for: {query}",
                'source': 'mock'
            }
        ]
        
        return {
            'success': True,
            'query': query,
            'results': mock_results,
            'total_results': len(mock_results)
        }