"""
MCP (Model Context Protocol) Layer for Business Strategy Agent

This module handles communication with MCP servers for data collection
from various sources including news APIs, social media, communities, and trends.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import aiohttp
import time
from enum import Enum

from .config import get_config, APIConfig
from .architecture import DataSource, RawContent, ContentType, RegionType

logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """MCP 서버 상태"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class MCPRequest:
    """MCP 요청"""
    server_name: str
    endpoint: str
    method: str = "GET"
    params: Dict[str, Any] = None
    headers: Dict[str, str] = None
    body: Optional[str] = None
    timeout: int = 30


@dataclass
class MCPResponse:
    """MCP 응답"""
    server_name: str
    status_code: int
    data: Any = None
    error: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """요청 허용 여부 확인"""
        async with self.lock:
            now = time.time()
            # 시간 윈도우 밖의 요청들 제거
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def get_wait_time(self) -> float:
        """다음 요청까지 대기 시간"""
        if not self.requests:
            return 0.0
        
        oldest_request = min(self.requests)
        return max(0.0, self.time_window - (time.time() - oldest_request))


class MCPServerInterface(ABC):
    """MCP 서버 인터페이스"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """서버 연결"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """서버 연결 해제"""
        pass
    
    @abstractmethod
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """요청 전송"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """헬스체크"""
        pass
    
    @abstractmethod
    def get_status(self) -> MCPServerStatus:
        """상태 반환"""
        pass


class HTTPMCPServer(MCPServerInterface):
    """HTTP 기반 MCP 서버"""
    
    def __init__(self, name: str, config: APIConfig):
        self.name = name
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.status = MCPServerStatus.DISCONNECTED
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.last_error: Optional[str] = None
        self.connection_count = 0
    
    async def connect(self) -> bool:
        """HTTP 세션 생성"""
        try:
            if self.session and not self.session.closed:
                return True
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_default_headers(),
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            )
            
            self.status = MCPServerStatus.CONNECTED
            self.connection_count += 1
            logger.info(f"Connected to MCP server: {self.name}")
            return True
            
        except Exception as e:
            self.status = MCPServerStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def disconnect(self):
        """HTTP 세션 종료"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self.status = MCPServerStatus.DISCONNECTED
        logger.info(f"Disconnected from MCP server: {self.name}")
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """HTTP 요청 전송"""
        start_time = time.time()
        
        try:
            # Rate limiting 체크
            if not await self.rate_limiter.acquire():
                wait_time = self.rate_limiter.get_wait_time()
                logger.warning(f"Rate limit exceeded for {self.name}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                if not await self.rate_limiter.acquire():
                    return MCPResponse(
                        server_name=self.name,
                        status_code=429,
                        error="Rate limit exceeded",
                        response_time=time.time() - start_time
                    )
            
            # 연결 확인
            if not await self.connect():
                return MCPResponse(
                    server_name=self.name,
                    status_code=500,
                    error="Connection failed",
                    response_time=time.time() - start_time
                )
            
            # 요청 URL 구성
            url = self._build_url(request.endpoint)
            
            # 헤더 병합
            headers = self._get_default_headers()
            if request.headers:
                headers.update(request.headers)
            
            # HTTP 요청 실행
            async with self.session.request(
                method=request.method,
                url=url,
                params=request.params,
                headers=headers,
                data=request.body,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        return MCPResponse(
                            server_name=self.name,
                            status_code=response.status,
                            data=data,
                            response_time=response_time
                        )
                    except json.JSONDecodeError:
                        text_data = await response.text()
                        return MCPResponse(
                            server_name=self.name,
                            status_code=response.status,
                            data=text_data,
                            response_time=response_time
                        )
                else:
                    error_text = await response.text()
                    return MCPResponse(
                        server_name=self.name,
                        status_code=response.status,
                        error=f"HTTP {response.status}: {error_text}",
                        response_time=response_time
                    )
        
        except asyncio.TimeoutError:
            return MCPResponse(
                server_name=self.name,
                status_code=408,
                error="Request timeout",
                response_time=time.time() - start_time
            )
        except Exception as e:
            self.status = MCPServerStatus.ERROR
            self.last_error = str(e)
            return MCPResponse(
                server_name=self.name,
                status_code=500,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def health_check(self) -> bool:
        """헬스체크"""
        try:
            request = MCPRequest(
                server_name=self.name,
                endpoint="health",
                timeout=10
            )
            
            response = await self.send_request(request)
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.name}: {e}")
            return False
    
    def get_status(self) -> MCPServerStatus:
        """상태 반환"""
        return self.status
    
    def _get_default_headers(self) -> Dict[str, str]:
        """기본 헤더 생성"""
        headers = {
            'User-Agent': 'Most-Hooking-Business-Strategy-Agent/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # API 키 인증
        if self.config.api_key:
            if 'twitter' in self.name.lower():
                headers['Authorization'] = f'Bearer {self.config.api_key}'
            elif 'naver' in self.name.lower():
                headers['X-Naver-Client-Id'] = self.config.api_key
                if self.config.secret_key:
                    headers['X-Naver-Client-Secret'] = self.config.secret_key
            else:
                headers['X-API-Key'] = self.config.api_key
        
        return headers
    
    def _build_url(self, endpoint: str) -> str:
        """URL 구성"""
        base_url = self.config.base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/{endpoint}"


class MCPServerManager:
    """MCP 서버들을 관리하는 매니저"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerInterface] = {}
        self.config = get_config()
        self.health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """매니저 초기화"""
        try:
            # 설정에서 서버들 등록
            for api_name, api_config in self.config.api_configs.items():
                await self.register_server(api_name, api_config)
            
            # 헬스체크 태스크 시작
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
            
            logger.info(f"MCP Server Manager initialized with {len(self.servers)} servers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Server Manager: {e}")
            return False
    
    async def register_server(self, name: str, config: APIConfig) -> bool:
        """서버 등록"""
        try:
            server = HTTPMCPServer(name, config)
            success = await server.connect()
            
            if success:
                self.servers[name] = server
                logger.info(f"Registered MCP server: {name}")
                return True
            else:
                logger.warning(f"Failed to register MCP server: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering server {name}: {e}")
            return False
    
    async def send_request(self, server_name: str, request: MCPRequest) -> MCPResponse:
        """특정 서버로 요청 전송"""
        server = self.servers.get(server_name)
        if not server:
            return MCPResponse(
                server_name=server_name,
                status_code=404,
                error=f"Server {server_name} not found"
            )
        
        return await server.send_request(request)
    
    async def broadcast_request(self, request: MCPRequest, 
                              server_names: Optional[List[str]] = None) -> List[MCPResponse]:
        """여러 서버로 동시 요청"""
        target_servers = server_names or list(self.servers.keys())
        tasks = []
        
        for server_name in target_servers:
            if server_name in self.servers:
                task = self.send_request(server_name, request)
                tasks.append(task)
        
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in responses if isinstance(r, MCPResponse)]
        
        return []
    
    async def health_check_all(self) -> Dict[str, bool]:
        """모든 서버 헬스체크"""
        results = {}
        tasks = []
        
        for name, server in self.servers.items():
            task = server.health_check()
            tasks.append((name, task))
        
        if tasks:
            for name, task in tasks:
                try:
                    result = await task
                    results[name] = result
                except Exception as e:
                    logger.warning(f"Health check error for {name}: {e}")
                    results[name] = False
        
        return results
    
    async def _periodic_health_check(self):
        """주기적 헬스체크"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                health_results = await self.health_check_all()
                
                healthy_count = sum(health_results.values())
                total_count = len(health_results)
                
                logger.info(f"Health check completed: {healthy_count}/{total_count} servers healthy")
                
                # 불건전한 서버들 재연결 시도
                for server_name, is_healthy in health_results.items():
                    if not is_healthy and server_name in self.servers:
                        logger.info(f"Attempting to reconnect unhealthy server: {server_name}")
                        await self.servers[server_name].connect()
                        
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 서버 상태 반환"""
        status = {}
        
        for name, server in self.servers.items():
            status[name] = {
                'status': server.get_status().value,
                'last_error': getattr(server, 'last_error', None),
                'connection_count': getattr(server, 'connection_count', 0)
            }
        
        return status
    
    async def shutdown(self):
        """매니저 종료"""
        # 헬스체크 태스크 중단
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 모든 서버 연결 해제
        disconnect_tasks = []
        for server in self.servers.values():
            disconnect_tasks.append(server.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        logger.info("MCP Server Manager shutdown completed")


class DataCollectorFactory:
    """데이터 수집기 팩토리"""
    
    def __init__(self, mcp_manager: MCPServerManager):
        self.mcp_manager = mcp_manager
    
    def create_news_collector(self, server_name: str) -> 'NewsCollector':
        """뉴스 수집기 생성"""
        return NewsCollector(server_name, self.mcp_manager)
    
    def create_social_collector(self, server_name: str) -> 'SocialMediaCollector':
        """소셜 미디어 수집기 생성"""
        return SocialMediaCollector(server_name, self.mcp_manager)
    
    def create_community_collector(self, server_name: str) -> 'CommunityCollector':
        """커뮤니티 수집기 생성"""
        return CommunityCollector(server_name, self.mcp_manager)


class BaseDataCollector:
    """기본 데이터 수집기"""
    
    def __init__(self, server_name: str, mcp_manager: MCPServerManager):
        self.server_name = server_name
        self.mcp_manager = mcp_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def collect_raw_data(self, keywords: List[str], 
                              region: RegionType = RegionType.GLOBAL,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """원시 데이터 수집"""
        request = self._build_request(keywords, region, limit)
        response = await self.mcp_manager.send_request(self.server_name, request)
        
        if response.status_code == 200:
            return self._parse_response(response.data)
        else:
            self.logger.warning(f"Failed to collect data from {self.server_name}: {response.error}")
            return []
    
    def _build_request(self, keywords: List[str], region: RegionType, limit: int) -> MCPRequest:
        """요청 구성 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def _parse_response(self, data: Any) -> List[Dict[str, Any]]:
        """응답 파싱 (하위 클래스에서 구현)"""
        raise NotImplementedError
    
    def _convert_to_raw_content(self, item: Dict[str, Any], 
                               content_type: ContentType,
                               region: RegionType) -> RawContent:
        """딕셔너리를 RawContent로 변환"""
        return RawContent(
            source=self.server_name,
            content_type=content_type,
            region=region,
            title=item.get('title', ''),
            content=item.get('content', ''),
            url=item.get('url', ''),
            timestamp=datetime.now(timezone.utc),
            author=item.get('author'),
            engagement_metrics=item.get('engagement', {}),
            metadata=item.get('metadata', {})
        )


class NewsCollector(BaseDataCollector):
    """뉴스 수집기"""
    
    def _build_request(self, keywords: List[str], region: RegionType, limit: int) -> MCPRequest:
        query = ' OR '.join(keywords)
        
        if 'reuters' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="news/search",
                params={
                    'q': query,
                    'limit': limit,
                    'sortBy': 'publishedAt'
                }
            )
        elif 'naver' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="news.json",
                params={
                    'query': query,
                    'display': min(limit, 100),
                    'sort': 'date'
                }
            )
        else:
            # 기본 뉴스 API 형태
            return MCPRequest(
                server_name=self.server_name,
                endpoint="articles",
                params={
                    'q': query,
                    'pageSize': limit,
                    'sortBy': 'publishedAt'
                }
            )
    
    def _parse_response(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            if 'articles' in data:
                return data['articles']
            elif 'items' in data:  # Naver API
                return data['items']
            elif 'results' in data:
                return data['results']
        
        return []


class SocialMediaCollector(BaseDataCollector):
    """소셜 미디어 수집기"""
    
    def _build_request(self, keywords: List[str], region: RegionType, limit: int) -> MCPRequest:
        query = ' OR '.join(keywords)
        
        if 'twitter' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="tweets/search/recent",
                params={
                    'query': query,
                    'max_results': min(limit, 100),
                    'tweet.fields': 'created_at,author_id,public_metrics,lang'
                }
            )
        elif 'linkedin' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="posts",
                params={
                    'q': query,
                    'count': limit
                }
            )
        else:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="posts/search",
                params={'q': query, 'limit': limit}
            )
    
    def _parse_response(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            if 'data' in data:  # Twitter API v2
                return data['data']
            elif 'posts' in data:
                return data['posts']
        
        return []


class CommunityCollector(BaseDataCollector):
    """커뮤니티 수집기"""
    
    def _build_request(self, keywords: List[str], region: RegionType, limit: int) -> MCPRequest:
        query = ' '.join(keywords)
        
        if 'reddit' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="search",
                params={
                    'q': query,
                    'sort': 'hot',
                    'limit': limit,
                    'type': 'link'
                }
            )
        elif 'hackernews' in self.server_name:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="search",
                params={
                    'query': query,
                    'tags': 'story',
                    'hitsPerPage': limit
                }
            )
        else:
            return MCPRequest(
                server_name=self.server_name,
                endpoint="search",
                params={'q': query, 'limit': limit}
            )
    
    def _parse_response(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            if 'data' in data and 'children' in data['data']:  # Reddit
                return [child['data'] for child in data['data']['children']]
            elif 'hits' in data:  # HackerNews
                return data['hits']
        
        return []


# 글로벌 MCP 매니저 인스턴스
mcp_manager = MCPServerManager()


async def get_mcp_manager() -> MCPServerManager:
    """MCP 매니저 인스턴스 반환"""
    if not mcp_manager.servers:
        await mcp_manager.initialize()
    return mcp_manager