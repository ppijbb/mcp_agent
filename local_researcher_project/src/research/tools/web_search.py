"""
Advanced Web Search Tool (v2.0 - 8대 혁신 통합)

Universal MCP Hub, Production-Grade Reliability, Multi-Model Orchestration을
통합한 고도화된 웹 검색 도구.
"""

import logging
import asyncio
import os
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

from src.research.tools.base_tool import BaseResearchTool, SearchResult, ToolType
from src.core.mcp_integration import execute_tool, get_best_tool_for_task, ToolCategory
from src.core.reliability import execute_with_reliability

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """사용 가능한 검색 제공자."""
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    EXA = "exa"
    MCP_GSEARCH = "mcp_gsearch"
    MCP_TAVILY = "mcp_tavily"
    MCP_EXA = "mcp_exa"


class WebSearchTool(BaseResearchTool):
    """8대 혁신을 통합한 고도화된 웹 검색 도구."""
    
    def __init__(self, config: Dict[str, Any]):
        """초기화."""
        super().__init__(config)
        self.tool_type = ToolType.SEARCH
        
        # 검색 제공자 설정
        self.primary_provider = SearchProvider(config.get('primary_provider', 'mcp_gsearch'))
        self.fallback_providers = [
            SearchProvider(p) for p in config.get('fallback_providers', [
                'mcp_tavily', 'mcp_exa', 'tavily', 'exa', 'duckduckgo'
            ])
        ]
        
        # API 키 설정
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.exa_api_key = os.getenv('EXA_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        # MCP 도구 매핑
        self.mcp_tool_mapping = {
            'web_search': 'g-search',
            'tavily_search': 'tavily',
            'exa_search': 'exa',
            'duckduckgo_search': 'duckduckgo'
        }
        
        logger.info(f"WebSearchTool initialized with primary provider: {self.primary_provider.value}")
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """웹 검색 실행 (Universal MCP Hub + Production-Grade Reliability)."""
        max_results = kwargs.get('max_results', self.max_results)
        search_depth = kwargs.get('search_depth', 'basic')
        include_answer = kwargs.get('include_answer', True)
        
        # Universal MCP Hub를 통한 검색
        search_result = await self.execute_with_mcp_fallback(
            operation='web_search',
            parameters={
                'query': query,
                'max_results': max_results,
                'search_depth': search_depth,
                'include_answer': include_answer
            }
        )
        
        if search_result['success']:
            return self._convert_to_search_results(search_result['data'], query)
        else:
            # Fallback 검색 실행
            return await self._fallback_search(query, max_results)
    
    async def get_content(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 가져오기 (Universal MCP Hub)."""
        # MCP fetch 도구 사용
        fetch_result = await self.execute_with_mcp_fallback(
            operation='data_fetch',
            parameters={'url': url}
        )
        
        if fetch_result['success']:
            return fetch_result['data'].get('content', '')
        
        # API Fallback
        try:
            return await self._fetch_content_api(url)
        except Exception as e:
            logger.error(f"Content fetch failed for {url}: {e}")
            return None
    
    async def _execute_api_fallback(self, operation: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """API Fallback 실행."""
        query = parameters.get('query', '')
        max_results = parameters.get('max_results', self.max_results)
        
        # 제공자별 검색 실행
        if operation == 'web_search':
            # Tavily API 시도
            if self.tavily_api_key:
                try:
                    return await self._tavily_search_api(query, max_results)
                except Exception as e:
                    logger.warning(f"Tavily API failed: {e}")
            
            # Exa API 시도
            if self.exa_api_key:
                try:
                    return await self._exa_search_api(query, max_results)
                except Exception as e:
                    logger.warning(f"Exa API failed: {e}")
            
            # DuckDuckGo 시도
            try:
                return await self._duckduckgo_search_api(query, max_results)
            except Exception as e:
                logger.warning(f"DuckDuckGo API failed: {e}")
        
        elif operation == 'data_fetch':
            url = parameters.get('url', '')
            return await self._fetch_content_api(url)
        
        return None
    
    async def _tavily_search_api(self, query: str, max_results: int) -> Dict[str, Any]:
        """Tavily API 검색."""
        import aiohttp
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'results': data.get('results', []),
                        'answer': data.get('answer', ''),
                        'provider': 'tavily'
                    }
                else:
                    raise Exception(f"Tavily API error: {response.status}")
    
    async def _exa_search_api(self, query: str, max_results: int) -> Dict[str, Any]:
        """Exa API 검색."""
        import aiohttp
        
        url = "https://api.exa.ai/search"
        headers = {
            "x-api-key": self.exa_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "numResults": max_results,
            "type": "search",
            "useAutoprompt": True
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'results': data.get('results', []),
                        'provider': 'exa'
                    }
                else:
                    raise Exception(f"Exa API error: {response.status}")
    
    async def _duckduckgo_search_api(self, query: str, max_results: int) -> Dict[str, Any]:
        """DuckDuckGo API 검색."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'content': result.get('body', ''),
                        'score': 0.8  # DuckDuckGo는 점수를 제공하지 않음
                    })
            
            return {
                'results': results,
                'provider': 'duckduckgo'
            }
        except ImportError:
            raise Exception("duckduckgo_search not installed")
    
    async def _fetch_content_api(self, url: str) -> str:
        """API를 통한 콘텐츠 가져오기."""
        import aiohttp
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise Exception(f"HTTP {response.status}")
    
    async def _fallback_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback 검색 실행."""
        results = []
        
        # 여러 제공자로 병렬 검색
        search_tasks = []
        
        # Tavily API
        if self.tavily_api_key:
            search_tasks.append(self._tavily_search_api(query, max_results))
        
        # Exa API
        if self.exa_api_key:
            search_tasks.append(self._exa_search_api(query, max_results))
        
        # DuckDuckGo
        search_tasks.append(self._duckduckgo_search_api(query, max_results))
        
        # 병렬 실행
        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for result in search_results:
                if isinstance(result, dict) and 'results' in result:
                    search_results_list = self._convert_to_search_results(result, query)
                    results.extend(search_results_list)
                    
                    if len(results) >= max_results:
                        break
                        
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
        
        return results[:max_results]
    
    def _convert_to_search_results(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """데이터를 SearchResult 객체로 변환."""
        results = []
        search_results = data.get('results', [])
        provider = data.get('provider', 'unknown')
        
        for item in search_results:
            # 관련성 점수 계산
            title = item.get('title', '')
            snippet = item.get('content', item.get('snippet', item.get('body', '')))
            relevance_score = self._calculate_relevance_score(query, title, snippet)
            
            # 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score({
                'source': provider,
                'score': item.get('score', 0.8),
                'metadata': item.get('metadata', {})
            })
            
            result = SearchResult(
                title=title,
                url=item.get('url', item.get('href', '')),
                snippet=snippet,
                source=provider,
                relevance_score=relevance_score,
                confidence_score=confidence_score,
                mcp_tool_used=provider if provider.startswith('mcp_') else None,
                execution_time=0.0,
                metadata={
                    'original_score': item.get('score', 0.8),
                    'provider': provider,
                    'search_timestamp': datetime.now().isoformat()
                }
            )
            
            results.append(result)
        
        return results
    
    async def search_with_streaming(
        self,
        query: str,
        callback: callable = None,
        **kwargs
    ) -> List[SearchResult]:
        """스트리밍 검색 (Streaming Pipeline - 혁신 5)."""
        logger.info(f"Starting streaming search for: {query}")
        
        # 즉시 초기 응답
        if callback:
            await callback({
                'status': 'started',
                'query': query,
                'timestamp': datetime.now().isoformat()
            })
        
        # 검색 실행
        results = await self.search(query, **kwargs)
        
        # 부분 결과 스트리밍
        if callback:
            for i, result in enumerate(results):
                await callback({
                    'status': 'partial',
                    'result': result,
                    'index': i,
                    'total': len(results),
                    'timestamp': datetime.now().isoformat()
                })
        
        # 최종 결과
        if callback:
            await callback({
                'status': 'completed',
                'query': query,
                'total_results': len(results),
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    async def search_with_verification(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """검증이 포함된 검색 (Continuous Verification - 혁신 4)."""
        # 기본 검색 실행
        results = await self.search(query, **kwargs)
        
        # 검증 실행
        verified_results = []
        for result in results:
            # 신뢰도 기반 검증
            if result.confidence_score >= 0.7:
                verified_results.append(result)
            else:
                # 낮은 신뢰도 결과는 추가 검증
                verification_score = await self._verify_result(result)
                if verification_score >= 0.6:
                    result.verification_score = verification_score
                    verified_results.append(result)
        
        return verified_results
    
    async def _verify_result(self, result: SearchResult) -> float:
        """결과 검증."""
        # 실제 구현에서는 더 정교한 검증 로직 사용
        # 여기서는 간단한 신뢰도 점수 반환
        return result.confidence_score
    
    def get_search_providers(self) -> List[str]:
        """사용 가능한 검색 제공자 반환."""
        providers = []
        
        # MCP 제공자
        if self.mcp_config.enabled:
            providers.extend(['mcp_gsearch', 'mcp_tavily', 'mcp_exa'])
        
        # API 제공자
        if self.tavily_api_key:
            providers.append('tavily')
        if self.exa_api_key:
            providers.append('exa')
        if self.google_api_key:
            providers.append('google')
        
        # 기본 제공자
        providers.extend(['duckduckgo', 'google', 'bing'])
        
        return list(set(providers))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 반환."""
        stats = self.get_performance_stats()
        return {
            'tool_name': self.name,
            'tool_type': self.tool_type.value,
            'status': self.status.value,
            'success_rate': f"{stats['success_rate']:.2%}",
            'average_response_time': f"{stats['average_response_time']:.2f}s",
            'total_requests': stats['total_requests'],
            'available_providers': self.get_search_providers(),
            'mcp_enabled': self.mcp_config.enabled,
            'last_used': stats['last_used']
        }


# Global web search tool instance
def create_web_search_tool(config: Dict[str, Any] = None) -> WebSearchTool:
    """웹 검색 도구 생성."""
    if config is None:
        config = {
            'name': 'web_search',
            'enabled': True,
            'timeout': 30,
            'max_results': 10,
            'primary_provider': 'mcp_gsearch',
            'fallback_providers': ['mcp_tavily', 'mcp_exa', 'tavily', 'exa', 'duckduckgo']
        }
    
    return WebSearchTool(config)


# Convenience functions
async def search_web(query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
    """웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_reliability(query, max_results=max_results, **kwargs)


async def search_web_streaming(query: str, callback: callable = None, **kwargs) -> List[SearchResult]:
    """스트리밍 웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_streaming(query, callback, **kwargs)


async def search_web_with_verification(query: str, **kwargs) -> List[SearchResult]:
    """검증이 포함된 웹 검색 실행."""
    tool = create_web_search_tool()
    return await tool.search_with_verification(query, **kwargs)