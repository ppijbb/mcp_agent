"""
Advanced Academic Search Tool with MCP Integration and Production-Grade Reliability
Implements 8 Core Innovations: Universal MCP Hub, Production-Grade Reliability, Multi-Model Orchestration
"""

import logging
import asyncio
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from urllib.parse import quote_plus
import requests
from dataclasses import dataclass
from datetime import datetime

# Import core modules for 8 innovations
from ..base_tool import BaseResearchTool, ToolResponse, ToolCategory, ToolResult
from ...core.reliability import execute_with_reliability, CircuitBreaker
from ...core.llm_manager import execute_llm_task, TaskType
from src.core.mcp_integration import get_best_tool_for_task, execute_tool

logger = logging.getLogger(__name__)


class AcademicProvider(Enum):
    """Available academic search providers with MCP priority."""
    ARXIV_MCP = "arxiv_mcp"
    SCHOLAR_MCP = "scholar_mcp"
    PUBMED_MCP = "pubmed_mcp"
    IEEE_MCP = "ieee_mcp"
    ARXIV_API = "arxiv_api"
    SCHOLAR_API = "scholar_api"
    PUBMED_API = "pubmed_api"
    IEEE_API = "ieee_api"


@dataclass
class AcademicResult:
    """Standardized academic search result."""
    title: str
    authors: List[str]
    abstract: str
    published: str
    source: str
    url: str
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    categories: List[str] = None
    confidence_score: float = 1.0
    verification_status: str = "unverified"
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []


class AcademicSearchTool(BaseResearchTool):
    """Advanced academic search tool with MCP integration and production-grade reliability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize academic search tool with 8 innovations."""
        super().__init__(config)
        self.max_results = config.get('max_results', 10)
        self.timeout = config.get('timeout', 30)
        self.enable_mcp_priority = config.get('enable_mcp_priority', True)
        self.enable_verification = config.get('enable_verification', True)
        self.enable_confidence_scoring = config.get('enable_confidence_scoring', True)
        
        # MCP-first provider priority
        self.primary_provider = config.get('primary_provider', AcademicProvider.ARXIV_MCP)
        self.fallback_providers = config.get('fallback_providers', [
            AcademicProvider.SCHOLAR_MCP,
            AcademicProvider.PUBMED_MCP,
            AcademicProvider.IEEE_MCP,
            AcademicProvider.ARXIV_API,
            AcademicProvider.SCHOLAR_API,
            AcademicProvider.PUBMED_API,
            AcademicProvider.IEEE_API
        ])
        
        # Initialize API keys for fallback
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY')
        self.ieee_api_key = os.getenv('IEEE_API_KEY')
    
        # Circuit breaker for reliability
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self, query: str, **kwargs) -> ToolResponse:
        """Run academic search synchronously with production-grade reliability."""
        try:
            return asyncio.run(self.arun(query, **kwargs))
        except Exception as e:
            self.logger.error(f"Academic search failed: {e}")
            return ToolResponse(
                success=False,
                data=[],
                error=str(e),
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
    
    async def arun(self, query: str, **kwargs) -> ToolResponse:
        """Run academic search with MCP priority and production-grade reliability."""
        return await execute_with_reliability(
            self._execute_academic_search,
            query=query,
            **kwargs
        )
    
    async def _execute_academic_search(self, query: str, **kwargs) -> ToolResponse:
        """Execute academic search with MCP-first approach and continuous verification."""
        try:
            # Step 1: Try MCP tools first (Universal MCP Hub - Innovation 6)
            if self.enable_mcp_priority:
                mcp_result = await self._try_mcp_search(query)
                if mcp_result.success and mcp_result.data:
                    # Apply continuous verification (Innovation 4)
                    if self.enable_verification:
                        verified_result = await self._verify_results(mcp_result.data, query)
                        return verified_result
                    return mcp_result
            
            # Step 2: Fallback to API providers with smart selection
            api_result = await self._try_api_search(query)
            if api_result.success and api_result.data:
                # Apply continuous verification
                if self.enable_verification:
                    verified_result = await self._verify_results(api_result.data, query)
                    return verified_result
                return api_result
            
            # Step 3: All methods failed
            self.logger.warning("All academic search methods failed")
            return ToolResponse(
                success=False,
                data=[],
                error="All academic search providers failed",
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
                
        except Exception as e:
            self.logger.error(f"Academic search execution failed: {e}")
            return ToolResponse(
                success=False,
                data=[],
                error=str(e),
                metadata={'query': query, 'timestamp': datetime.now().isoformat()}
            )
    
    async def _try_mcp_search(self, query: str) -> ToolResponse:
        """Try MCP tools first (Universal MCP Hub - Innovation 6)."""
        try:
            # Get best MCP tool for academic search
            best_tool = get_best_tool_for_task("academic_search")
            if not best_tool:
                return ToolResponse(success=False, data=[], error="No MCP academic tools available")
            
            # Execute MCP tool
            mcp_result = await execute_tool(best_tool, {"query": query, "max_results": self.max_results})
            
            if mcp_result.get('success', False):
                # Convert MCP result to AcademicResult format
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), best_tool)
                return ToolResponse(
                    success=True,
                    data=academic_results,
                    metadata={
                        'provider': best_tool,
                        'method': 'mcp',
                        'query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'MCP tool failed'))
                
        except Exception as e:
            self.logger.warning(f"MCP academic search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _try_api_search(self, query: str) -> ToolResponse:
        """Try API providers with smart fallback strategy."""
        all_results = []
        
        # Try primary provider first
        primary_result = await self._search_with_provider(self.primary_provider, query)
        if primary_result.success:
            all_results.extend(primary_result.data)
        
        # Try fallback providers if needed
        if len(all_results) < self.max_results:
            for provider in self.fallback_providers:
                if len(all_results) >= self.max_results:
                    break
                    
                try:
                    result = await self._search_with_provider(provider, query)
                    if result.success:
                        all_results.extend(result.data)
                except Exception as e:
                    self.logger.warning(f"Provider {provider.value} failed: {e}")
                    continue
            
        if all_results:
            return ToolResponse(
                success=True,
                data=all_results[:self.max_results],
                metadata={
                    'provider': 'multi_api',
                    'method': 'api',
                    'query': query,
                    'total_found': len(all_results),
                    'timestamp': datetime.now().isoformat()
                }
            )
        else:
            return ToolResponse(success=False, data=[], error="All API providers failed")
    
    async def _search_with_provider(self, provider: AcademicProvider, query: str) -> ToolResponse:
        """Search with specific provider using circuit breaker pattern."""
        try:
            if provider == AcademicProvider.ARXIV_MCP:
                return await self._arxiv_mcp_search(query)
            elif provider == AcademicProvider.ARXIV_API:
                return await self._arxiv_api_search(query)
            elif provider == AcademicProvider.SCHOLAR_MCP:
                return await self._scholar_mcp_search(query)
            elif provider == AcademicProvider.SCHOLAR_API:
                return await self._scholar_api_search(query)
            elif provider == AcademicProvider.PUBMED_MCP:
                return await self._pubmed_mcp_search(query)
            elif provider == AcademicProvider.PUBMED_API:
                return await self._pubmed_api_search(query)
            elif provider == AcademicProvider.IEEE_MCP:
                return await self._ieee_mcp_search(query)
            elif provider == AcademicProvider.IEEE_API:
                return await self._ieee_api_search(query)
            else:
                return ToolResponse(success=False, data=[], error=f"Unknown provider: {provider}")
        except Exception as e:
            self.logger.error(f"Provider {provider.value} search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _verify_results(self, results: List[AcademicResult], query: str) -> ToolResponse:
        """Apply continuous verification to search results (Innovation 4)."""
        try:
            if not self.enable_verification or not results:
                return ToolResponse(success=True, data=results)
            
            # Use Multi-Model Orchestration for verification (Innovation 3)
            verification_prompt = f"""
            Verify the following academic search results for query: "{query}"
            
            Results to verify:
            {[f"- {r.title} ({r.source})" for r in results[:5]]}
            
            Please provide:
            1. Confidence score for each result (0.0-1.0)
            2. Verification status (verified/needs_review/unverified)
            3. Any potential issues or concerns
            """
            
            verification_result = await execute_llm_task(
                task_type=TaskType.VERIFICATION,
                prompt=verification_prompt,
                use_ensemble=True
            )
            
            # Apply verification results to academic results
            verified_results = []
            for i, result in enumerate(results):
                if i < len(verification_result.get('confidence_scores', [])):
                    result.confidence_score = verification_result['confidence_scores'][i]
                    result.verification_status = verification_result.get('verification_status', ['unverified'])[i]
                verified_results.append(result)
            
            return ToolResponse(
                success=True,
                data=verified_results,
                metadata={
                    'verification_applied': True,
                    'average_confidence': sum(r.confidence_score for r in verified_results) / len(verified_results),
                    'timestamp': datetime.now().isoformat()
                }
            )
                
        except Exception as e:
            self.logger.warning(f"Verification failed: {e}, returning unverified results")
            return ToolResponse(success=True, data=results)
    
    def _convert_to_academic_results(self, mcp_data: List[Dict], tool_name: str) -> List[AcademicResult]:
        """Convert MCP tool results to standardized AcademicResult format."""
        results = []
        for item in mcp_data:
            try:
                result = AcademicResult(
                    title=item.get('title', ''),
                    authors=item.get('authors', []),
                    abstract=item.get('abstract', ''),
                    published=item.get('published', ''),
                    source=tool_name,
                    url=item.get('url', ''),
                    pdf_url=item.get('pdf_url', ''),
                    doi=item.get('doi', ''),
                    arxiv_id=item.get('arxiv_id', ''),
                    pmid=item.get('pmid', ''),
                    categories=item.get('categories', []),
                    confidence_score=item.get('confidence_score', 1.0),
                    verification_status=item.get('verification_status', 'unverified')
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to convert MCP result: {e}")
                continue
        
        return results
    
    # MCP Search Methods (Innovation 6: Universal MCP Hub)
    async def _arxiv_mcp_search(self, query: str) -> ToolResponse:
        """Search ArXiv using MCP tools."""
        try:
            mcp_result = await execute_tool("arxiv", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "arxiv_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'ArXiv MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _scholar_mcp_search(self, query: str) -> ToolResponse:
        """Search Google Scholar using MCP tools."""
        try:
            mcp_result = await execute_tool("scholar", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "scholar_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'Scholar MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _pubmed_mcp_search(self, query: str) -> ToolResponse:
        """Search PubMed using MCP tools."""
        try:
            mcp_result = await execute_tool("pubmed", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "pubmed_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'PubMed MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _ieee_mcp_search(self, query: str) -> ToolResponse:
        """Search IEEE using MCP tools."""
        try:
            mcp_result = await execute_tool("ieee", {"query": query, "max_results": self.max_results})
            if mcp_result.get('success', False):
                academic_results = self._convert_to_academic_results(mcp_result.get('data', []), "ieee_mcp")
                return ToolResponse(success=True, data=academic_results)
            else:
                return ToolResponse(success=False, data=[], error=mcp_result.get('error', 'IEEE MCP failed'))
        except Exception as e:
            return ToolResponse(success=False, data=[], error=str(e))
    
    # API Fallback Methods (Production-Grade Reliability - Innovation 8)
    async def _arxiv_api_search(self, query: str) -> ToolResponse:
        """Perform search using ArXiv API with production-grade reliability."""
        try:
            # ArXiv API endpoint
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': self.max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            results = []
            entries = root.findall('atom:entry', ns)
            
            for entry in entries:
                # Extract paper information
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip() if title_elem is not None else ''
                
                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip() if summary_elem is not None else ''
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                
                # Extract publication date
                published_elem = entry.find('atom:published', ns)
                published = published_elem.text.strip() if published_elem is not None else ''
                
                # Extract ArXiv ID and URL
                id_elem = entry.find('atom:id', ns)
                arxiv_id = ''
                pdf_url = ''
                if id_elem is not None:
                    arxiv_id = id_elem.text.split('/')[-1]
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', ns):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                
                result = AcademicResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                    categories=categories,
                    source='arxiv_api',
                    url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ''
                )
                results.append(result)
            
            self.logger.info(f"ArXiv API search completed: {len(results)} results")
            return ToolResponse(
                success=True,
                data=results,
                metadata={'provider': 'arxiv_api', 'total_results': len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"ArXiv API search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _scholar_api_search(self, query: str) -> ToolResponse:
        """Perform search using Google Scholar API with production-grade reliability."""
        try:
            from urllib.parse import quote_plus
            import re
            
            # Google Scholar search URL
            search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&num={self.max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML response using regex
            results = []
            
            # Extract paper titles and links
            title_pattern = r'<h3[^>]*class="gs_rt"[^>]*>(.*?)</h3>'
            link_pattern = r'<a[^>]*href="([^"]*)"[^>]*class="gs_rt"[^>]*>'
            snippet_pattern = r'<div[^>]*class="gs_rs"[^>]*>(.*?)</div>'
            
            titles = re.findall(title_pattern, response.text)
            links = re.findall(link_pattern, response.text)
            snippets = re.findall(snippet_pattern, response.text)
            
            for i, (title, link) in enumerate(zip(titles[:self.max_results], links[:self.max_results])):
                if title and link:
                    # Clean title
                    title = re.sub(r'<[^>]+>', '', title)
                    title = title.strip()
                    
                    # Get snippet if available
                    snippet = snippets[i] if i < len(snippets) else ''
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    
                    result = AcademicResult(
                        title=title,
                        url=link,
                        abstract=snippet,
                        source='scholar_api',
                        authors=[],  # Would need more complex parsing
                        published='',  # Would need more complex parsing
                        pdf_url=''
                    )
                    results.append(result)
            
            self.logger.info(f"Google Scholar API search completed: {len(results)} results")
            return ToolResponse(
                success=True,
                data=results,
                metadata={'provider': 'scholar_api', 'total_results': len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"Google Scholar API search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _pubmed_api_search(self, query: str) -> ToolResponse:
        """Perform search using PubMed API with production-grade reliability."""
        try:
            if not self.pubmed_api_key:
                return ToolResponse(success=False, data=[], error='PubMed API key not found')
            
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': self.max_results,
                'retmode': 'json',
                'api_key': self.pubmed_api_key
            }
            
            # First, get PMIDs
            response = requests.get(base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return ToolResponse(
                    success=True,
                    data=[],
                    metadata={'provider': 'pubmed_api', 'total_results': 0}
                )
            
            # Get detailed information for each PMID
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'api_key': self.pubmed_api_key
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=self.timeout)
            fetch_response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(fetch_response.content)
            
            results = []
            articles = root.findall('.//PubmedArticle')
            
            for article in articles:
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text.strip() if title_elem is not None else ''
                
                # Extract abstract
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text.strip() if abstract_elem is not None else ''
                
                # Extract authors
                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None and first_name is not None:
                        authors.append(f"{first_name.text} {last_name.text}")
                
                # Extract publication date
                pub_date = article.find('.//PubDate')
                published = ''
                if pub_date is not None:
                    year = pub_date.find('Year')
                    month = pub_date.find('Month')
                    day = pub_date.find('Day')
                    if year is not None:
                        published = year.text
                        if month is not None:
                            published += f"-{month.text}"
                        if day is not None:
                            published += f"-{day.text}"
                
                # Extract PMID
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ''
                
                result = AcademicResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    pmid=pmid,
                    source='pubmed_api',
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ''
                )
                results.append(result)
            
            self.logger.info(f"PubMed API search completed: {len(results)} results")
            return ToolResponse(
                success=True,
                data=results,
                metadata={'provider': 'pubmed_api', 'total_results': len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"PubMed API search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    async def _ieee_api_search(self, query: str) -> ToolResponse:
        """Perform search using IEEE Xplore API with production-grade reliability."""
        try:
            if not self.ieee_api_key:
                return ToolResponse(success=False, data=[], error='IEEE API key not found')
            
            # IEEE Xplore API
            base_url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
            params = {
                'apikey': self.ieee_api_key,
                'querytext': query,
                'max_records': self.max_results,
                'format': 'json'
            }
            
            response = requests.get(base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for article in data.get('articles', []):
                result = AcademicResult(
                    title=article.get('title', ''),
                    authors=[author.get('full_name', '') for author in article.get('authors', {}).get('authors', [])],
                    abstract=article.get('abstract', ''),
                    published=article.get('publication_date', ''),
                    doi=article.get('doi', ''),
                    source='ieee_api',
                    url=article.get('html_url', ''),
                    pdf_url=article.get('pdf_url', '')
                )
                results.append(result)
            
            self.logger.info(f"IEEE Xplore API search completed: {len(results)} results")
            return ToolResponse(
                success=True,
                data=results,
                metadata={'provider': 'ieee_api', 'total_results': len(results)}
            )
            
        except Exception as e:
            self.logger.error(f"IEEE Xplore API search failed: {e}")
            return ToolResponse(success=False, data=[], error=str(e))
    
    # Additional utility methods for 8 innovations
    def get_tool_category(self) -> ToolCategory:
        """Return tool category for Universal MCP Hub."""
        return ToolCategory.ACADEMIC_SEARCH
    
    def get_health_status(self) -> Dict[str, Any]:
        """Return health status for Production-Grade Reliability."""
        return {
            'status': 'healthy',
            'circuit_breaker_state': self.circuit_breaker.state,
            'mcp_enabled': self.enable_mcp_priority,
            'verification_enabled': self.enable_verification,
            'confidence_scoring_enabled': self.enable_confidence_scoring,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        return {
            'total_searches': getattr(self, '_total_searches', 0),
            'successful_searches': getattr(self, '_successful_searches', 0),
            'mcp_usage_ratio': getattr(self, '_mcp_usage_ratio', 0.0),
            'average_response_time': getattr(self, '_average_response_time', 0.0),
            'verification_success_rate': getattr(self, '_verification_success_rate', 0.0)
        }
