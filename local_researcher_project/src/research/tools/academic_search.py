"""
Advanced Academic Search Tool with Multiple API Support
"""

import logging
import asyncio
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from enum import Enum
from urllib.parse import quote_plus
import requests

logger = logging.getLogger(__name__)


class AcademicProvider(Enum):
    """Available academic search providers."""
    ARXIV = "arxiv"
    SCHOLAR = "scholar"
    PUBMED = "pubmed"
    IEEE = "ieee"


class AcademicSearchTool:
    """Advanced academic search tool with multiple API support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize academic search tool."""
        self.config = config
        self.max_results = config.get('max_results', 10)
        self.timeout = config.get('timeout', 30)
        self.primary_provider = config.get('primary_provider', AcademicProvider.ARXIV)
        self.fallback_providers = config.get('fallback_providers', [
            AcademicProvider.SCHOLAR,
            AcademicProvider.PUBMED
        ])
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize API keys
        self.pubmed_api_key = os.getenv('PUBMED_API_KEY')
        self.ieee_api_key = os.getenv('IEEE_API_KEY')
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run academic search synchronously."""
        try:
            return asyncio.run(self.arun(query, **kwargs))
        except Exception as e:
            self.logger.error(f"Academic search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
    
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run academic search asynchronously with provider priority."""
        try:
            # Try primary provider first
            if self.primary_provider == AcademicProvider.ARXIV:
                result = await self._arxiv_search(query)
                if result['success']:
                    return result
            elif self.primary_provider == AcademicProvider.PUBMED and self.pubmed_api_key:
                result = await self._pubmed_search(query)
                if result['success']:
                    return result
            
            # Try fallback providers
            results = []
            for provider in self.fallback_providers:
                try:
                    if provider == AcademicProvider.ARXIV:
                        result = await self._arxiv_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == AcademicProvider.SCHOLAR:
                        result = await self._scholar_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == AcademicProvider.PUBMED and self.pubmed_api_key:
                        result = await self._pubmed_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    elif provider == AcademicProvider.IEEE and self.ieee_api_key:
                        result = await self._ieee_search(query)
                        if result['success']:
                            results.extend(result['results'])
                    
                    if len(results) >= self.max_results:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Academic provider {provider.value} failed: {e}")
                    continue
            
            if results:
                return {
                    'success': True,
                    'query': query,
                    'results': results[:self.max_results],
                    'total_results': len(results),
                    'provider': 'multi_academic'
                }
            else:
                self.logger.warning("All academic search methods failed")
                return {
                    'success': False,
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'error': 'All academic search providers failed'
                }
                
        except Exception as e:
            self.logger.error(f"Academic search failed: {e}")
            return {
                'success': False,
                'query': query,
                'results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    async def _arxiv_search(self, query: str) -> Dict[str, Any]:
        """Perform search using ArXiv API."""
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
                
                results.append({
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published,
                    'arxiv_id': arxiv_id,
                    'pdf_url': pdf_url,
                    'categories': categories,
                    'source': 'arxiv',
                    'url': f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ''
                })
            
            self.logger.info(f"ArXiv search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'arxiv'
            }
            
        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _scholar_search(self, query: str) -> Dict[str, Any]:
        """Perform search using Google Scholar (via web scraping)."""
        try:
            import requests
            from urllib.parse import quote_plus
            
            # Google Scholar search URL
            search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&num={self.max_results}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML response using regex
            import re
            
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
                    
                    results.append({
                        'title': title,
                        'url': link,
                        'abstract': snippet,
                        'source': 'scholar',
                        'authors': [],  # Would need more complex parsing
                        'published': '',  # Would need more complex parsing
                        'pdf_url': ''
                    })
            
            self.logger.info(f"Google Scholar search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'scholar'
            }
            
        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _pubmed_search(self, query: str) -> Dict[str, Any]:
        """Perform search using PubMed API."""
        try:
            if not self.pubmed_api_key:
                return {'success': False, 'error': 'PubMed API key not found'}
            
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
                return {
                    'success': True,
                    'query': query,
                    'results': [],
                    'total_results': 0,
                    'provider': 'pubmed'
                }
            
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
                
                results.append({
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published,
                    'pmid': pmid,
                    'source': 'pubmed',
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ''
                })
            
            self.logger.info(f"PubMed search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'pubmed'
            }
            
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _ieee_search(self, query: str) -> Dict[str, Any]:
        """Perform search using IEEE Xplore API."""
        try:
            if not self.ieee_api_key:
                return {'success': False, 'error': 'IEEE API key not found'}
            
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
                results.append({
                    'title': article.get('title', ''),
                    'authors': [author.get('full_name', '') for author in article.get('authors', {}).get('authors', [])],
                    'abstract': article.get('abstract', ''),
                    'published': article.get('publication_date', ''),
                    'doi': article.get('doi', ''),
                    'source': 'ieee',
                    'url': article.get('html_url', ''),
                    'pdf_url': article.get('pdf_url', '')
                })
            
            self.logger.info(f"IEEE Xplore search completed: {len(results)} results")
            return {
                'success': True,
                'query': query,
                'results': results,
                'total_results': len(results),
                'provider': 'ieee'
            }
            
        except Exception as e:
            self.logger.error(f"IEEE Xplore search failed: {e}")
            return {'success': False, 'error': str(e)}
