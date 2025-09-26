"""
Advanced Web Search Tool with Multiple API Support
"""

import logging
import asyncio
import os
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """Available search providers."""
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    EXA = "exa"


class WebSearchTool:
    """Advanced web search tool with multiple API support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize web search tool."""
        self.config = config
        self.max_results = config.get('max_results', 10)
        self.timeout = config.get('timeout', 30)
        self.primary_provider = config.get('primary_provider', SearchProvider.TAVILY)
        self.fallback_providers = config.get('fallback_providers', [
            SearchProvider.DUCKDUCKGO, 
            SearchProvider.GOOGLE, 
            SearchProvider.BING
        ])
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize API keys
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.exa_api_key = os.getenv('EXA_API_KEY')
    
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
        """Run web search asynchronously with provider priority."""
        try:
            # Try primary provider first
            if self.primary_provider == SearchProvider.TAVILY and self.tavily_api_key:
                result = await self._tavily_search(query)
                if result['success']:
                    return result
            elif self.primary_provider == SearchProvider.EXA and self.exa_api_key:
                result = await self._exa_search(query)
                if result['success']:
                    return result
            
            # Try fallback providers
            results = []
            for provider in self.fallback_providers:
                try:
                    if provider == SearchProvider.TAVILY and self.tavily_api_key:
                        result = await self._tavily_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == SearchProvider.EXA and self.exa_api_key:
                        result = await self._exa_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == SearchProvider.DUCKDUCKGO:
                        result = await self._duckduckgo_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == SearchProvider.GOOGLE:
                        result = await self._google_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == SearchProvider.BING:
                        result = await self._bing_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    
                    if len(results) >= self.max_results:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Search provider {provider.value} failed: {e}")
                    continue
            
            if results:
                return {
                    'success': True,
                    'query': query,
                    'results': results[:self.max_results],
                    'total_results': len(results),
                    'provider': 'multi_engine'
                }
            else:
                self.logger.warning("All search methods failed")
                return {
                    'success': False,
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'error': 'All search providers failed'
                }
                
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {
                'success': False,
                'query': query,
                'results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    async def _tavily_search(self, query: str) -> Dict[str, Any]:
        """Perform search using Tavily API."""
        try:
            if not self.tavily_api_key:
                return {'success': False, 'error': 'Tavily API key not found'}
            
            import requests
            
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
                "max_results": self.max_results
            }
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('content', ''),
                    'source': 'tavily',
                    'score': item.get('score', 0.0)
                })
            
            self.logger.info(f"Tavily search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'tavily'
            }
            
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _exa_search(self, query: str) -> Dict[str, Any]:
        """Perform search using Exa API."""
        try:
            if not self.exa_api_key:
                return {'success': False, 'error': 'Exa API key not found'}
            
            import requests
            
            url = "https://api.exa.ai/search"
            headers = {
                "x-api-key": self.exa_api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "numResults": self.max_results,
                "type": "search",
                "useAutoprompt": True
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('text', ''),
                    'source': 'exa',
                    'score': item.get('score', 0.0)
                })
            
            self.logger.info(f"Exa search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'exa'
            }
            
        except Exception as e:
            self.logger.error(f"Exa search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """Perform search using DuckDuckGo."""
        try:
            import duckduckgo_search
            ddg_results = duckduckgo_search.DDGS().text(
                query, 
                max_results=self.max_results
            )
            
            results = []
            for result in ddg_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            
            self.logger.info(f"DuckDuckGo search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'duckduckgo'
            }
            
        except ImportError:
            return {'success': False, 'error': 'duckduckgo_search not installed'}
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _google_search(self, query: str) -> Dict[str, Any]:
        """Perform Google search using requests."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            # Use Google's search API or scraping
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={self.max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML response using regex (no BeautifulSoup dependency)
            import re
            
            results = []
            # Simple regex-based parsing for Google results
            title_pattern = r'<h3[^>]*>(.*?)</h3>'
            link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>'
            
            titles = re.findall(title_pattern, response.text)
            links = re.findall(link_pattern, response.text)
            
            for i, (title, link) in enumerate(zip(titles[:self.max_results], links[:self.max_results])):
                if title and link and 'http' in link:
                    # Clean title
                    title = re.sub(r'<[^>]+>', '', title)
                    results.append({
                        'title': title.strip(),
                        'url': link,
                        'snippet': f'Search result {i+1}',
                        'source': 'google'
                    })
            
            self.logger.info(f"Google search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'google'
            }
            
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _bing_search(self, query: str) -> Dict[str, Any]:
        """Perform Bing search using requests."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={self.max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML response using regex
            import re
            
            results = []
            # Simple regex-based parsing for Bing results
            title_pattern = r'<h2[^>]*>(.*?)</h2>'
            link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>'
            
            titles = re.findall(title_pattern, response.text)
            links = re.findall(link_pattern, response.text)
            
            for i, (title, link) in enumerate(zip(titles[:self.max_results], links[:self.max_results])):
                if title and link and 'http' in link:
                    # Clean title
                    title = re.sub(r'<[^>]+>', '', title)
                    results.append({
                        'title': title.strip(),
                        'url': link,
                        'snippet': f'Bing search result {i+1}',
                        'source': 'bing'
                    })
            
            self.logger.info(f"Bing search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'bing'
            }
            
        except Exception as e:
            self.logger.error(f"Bing search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _scrape_search_results(self, query: str) -> Dict[str, Any]:
        """Scrape search results from various sources as fallback."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            # Try different search engines
            search_engines = [
                f"https://www.startpage.com/sp/search?query={quote_plus(query)}",
                f"https://search.yahoo.com/search?p={quote_plus(query)}",
                f"https://www.ecosia.org/search?q={quote_plus(query)}"
            ]
            
            results = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            for search_url in search_engines:
                try:
                    response = requests.get(search_url, headers=headers, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Generic parsing for search results using regex
                    import re
                    
                    # Simple regex-based parsing
                    title_pattern = r'<h[1-6][^>]*>(.*?)</h[1-6]>'
                    link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>'
                    
                    titles = re.findall(title_pattern, response.text)
                    links = re.findall(link_pattern, response.text)
                    
                    for i, (title, link) in enumerate(zip(titles[:5], links[:5])):
                        if title and link and 'http' in link:
                            # Clean title
                            title = re.sub(r'<[^>]+>', '', title)
                            results.append({
                                'title': title.strip(),
                                'url': link,
                                'snippet': f'Scraped result {i+1}',
                                'source': 'scraped'
                            })
                    
                    if results:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Scraping {search_url} failed: {e}")
                    continue
            
            self.logger.info(f"Web scraping completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results[:self.max_results],
                'total_results': len(results),
                'provider': 'scraped'
            }
            
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return {'success': False, 'error': str(e)}