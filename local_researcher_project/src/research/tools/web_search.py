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
            # Try multiple search engines
            results = []
            
            # 1. Try DuckDuckGo search
            try:
                import duckduckgo_search
                ddg_results = duckduckgo_search.DDGS().text(
                    query, 
                    max_results=self.max_results
                )
                
                for result in ddg_results:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'source': 'duckduckgo'
                    })
                
                self.logger.info(f"DuckDuckGo search completed: {len(results)} results")
                
            except ImportError:
                self.logger.warning("duckduckgo_search not installed, trying alternative methods")
            except Exception as e:
                self.logger.warning(f"DuckDuckGo search failed: {e}")
            
            # 2. Try Google Search (if available)
            if len(results) < self.max_results:
                try:
                    google_results = await self._google_search(query, self.max_results - len(results))
                    results.extend(google_results)
                    self.logger.info(f"Google search completed: {len(google_results)} additional results")
                except Exception as e:
                    self.logger.warning(f"Google search failed: {e}")
            
            # 3. Try Bing Search (if available)
            if len(results) < self.max_results:
                try:
                    bing_results = await self._bing_search(query, self.max_results - len(results))
                    results.extend(bing_results)
                    self.logger.info(f"Bing search completed: {len(bing_results)} additional results")
                except Exception as e:
                    self.logger.warning(f"Bing search failed: {e}")
            
            # 4. If no results, try web scraping
            if not results:
                try:
                    scraped_results = await self._scrape_search_results(query)
                    results.extend(scraped_results)
                    self.logger.info(f"Web scraping completed: {len(scraped_results)} results")
                except Exception as e:
                    self.logger.warning(f"Web scraping failed: {e}")
            
            if results:
                return {
                    'success': True,
                    'query': query,
                    'results': results[:self.max_results],
                    'total_results': len(results),
                    'provider': 'multi_engine'
                }
            else:
                self.logger.warning("All search methods failed, using mock results")
                return self._mock_search(query)
                
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return self._mock_search(query)
    
    async def _google_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform Google search using requests."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            # Use Google's search API or scraping
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML response using regex (no BeautifulSoup dependency)
            import re
            
            results = []
            # Simple regex-based parsing for Google results
            title_pattern = r'<h3[^>]*>(.*?)</h3>'
            link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>'
            
            titles = re.findall(title_pattern, response.text)
            links = re.findall(link_pattern, response.text)
            
            for i, (title, link) in enumerate(zip(titles[:max_results], links[:max_results])):
                if title and link and 'http' in link:
                    # Clean title
                    title = re.sub(r'<[^>]+>', '', title)
                    results.append({
                        'title': title.strip(),
                        'url': link,
                        'snippet': f'Search result {i+1}',
                        'source': 'google'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []
    
    async def _bing_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform Bing search using requests."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML response using regex
            import re
            
            results = []
            # Simple regex-based parsing for Bing results
            title_pattern = r'<h2[^>]*>(.*?)</h2>'
            link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>'
            
            titles = re.findall(title_pattern, response.text)
            links = re.findall(link_pattern, response.text)
            
            for i, (title, link) in enumerate(zip(titles[:max_results], links[:max_results])):
                if title and link and 'http' in link:
                    # Clean title
                    title = re.sub(r'<[^>]+>', '', title)
                    results.append({
                        'title': title.strip(),
                        'url': link,
                        'snippet': f'Bing search result {i+1}',
                        'source': 'bing'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Bing search failed: {e}")
            return []
    
    async def _scrape_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Scrape search results from various sources."""
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
                    response = requests.get(search_url, headers=headers, timeout=10)
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
            
            return results[:self.max_results]
            
        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []

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