from typing import Dict, Any, List, Optional
import asyncio
import httpx
import time
import logging
from datetime import datetime

# Logger 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPServerManager:
    """MCP 서버들을 통합 관리하는 매니저"""
    
    def __init__(self):
        self.servers = {}
        self.initialize_hobby_mcp_servers()
    
    def initialize_hobby_mcp_servers(self):
        """취미 관련 MCP 서버들 초기화"""
        
        # 1. 구글 캘린더 MCP 서버
        self.servers["google_calendar"] = {
            "name": "Google Calendar MCP",
            "url": "https://calendar-mcp.googleapis.com",
            "capabilities": [
                "create_event",
                "list_events",
                "update_event",
                "check_availability",
                "find_free_time"
            ],
            "authentication": "oauth2",
            "description": "일정 관리 및 취미 활동 스케줄링"
        }
        
        # 2. 구글 맵스 MCP 서버
        self.servers["google_maps"] = {
            "name": "Google Maps MCP",
            "url": "https://maps-mcp.googleapis.com",
            "capabilities": [
                "search_places",
                "get_directions",
                "find_nearby",
                "geocode_address",
                "get_place_details"
            ],
            "authentication": "api_key",
            "description": "지역 기반 취미 장소 및 커뮤니티 검색"
        }
        
        # 3. 날씨 정보 MCP 서버
        self.servers["weather_api"] = {
            "name": "Weather MCP",
            "url": "https://weather-mcp.openweathermap.org",
            "capabilities": [
                "current_weather",
                "forecast",
                "weather_alerts",
                "historical_weather"
            ],
            "authentication": "api_key",
            "description": "날씨 기반 야외 취미 활동 추천"
        }
        
        # 4. 소셜 미디어 MCP 서버 (커뮤니티 검색)
        self.servers["social_search"] = {
            "name": "Social Media MCP",
            "url": "https://social-mcp.example.com",
            "capabilities": [
                "search_groups",
                "find_communities",
                "get_group_info",
                "search_events"
            ],
            "authentication": "oauth2",
            "description": "취미 관련 소셜 그룹 및 커뮤니티 검색"
        }
        
        # 5. 전자상거래 MCP 서버 (취미 용품)
        self.servers["ecommerce"] = {
            "name": "E-commerce MCP",
            "url": "https://ecommerce-mcp.example.com",
            "capabilities": [
                "search_products",
                "get_product_info",
                "compare_prices",
                "check_availability"
            ],
            "authentication": "api_key",
            "description": "취미 관련 용품 및 도구 검색"
        }
        
        # 6. 교육 플랫폼 MCP 서버
        self.servers["education"] = {
            "name": "Education MCP",
            "url": "https://education-mcp.example.com",
            "capabilities": [
                "search_courses",
                "find_tutorials",
                "get_course_info",
                "search_instructors"
            ],
            "authentication": "oauth2",
            "description": "취미 관련 강의 및 튜토리얼 검색"
        }
        
        # 7. 피트니스 트래킹 MCP 서버
        self.servers["fitness_tracker"] = {
            "name": "Fitness Tracker MCP",
            "url": "https://fitness-mcp.example.com",
            "capabilities": [
                "log_activity",
                "get_stats",
                "set_goals",
                "track_progress"
            ],
            "authentication": "oauth2",
            "description": "운동 관련 취미 활동 추적"
        }
        
        # 8. 음악 플랫폼 MCP 서버
        self.servers["music_platform"] = {
            "name": "Music Platform MCP",
            "url": "https://music-mcp.example.com",
            "capabilities": [
                "search_music",
                "create_playlist",
                "find_similar",
                "get_recommendations"
            ],
            "authentication": "oauth2",
            "description": "음악 관련 취미 활동 지원"
        }
        
        # 9. 독서 플랫폼 MCP 서버
        self.servers["reading_platform"] = {
            "name": "Reading Platform MCP",
            "url": "https://reading-mcp.example.com",
            "capabilities": [
                "search_books",
                "get_book_info",
                "find_similar",
                "track_reading"
            ],
            "authentication": "api_key",
            "description": "독서 관련 취미 활동 지원"
        }
        
        # 10. 요리 레시피 MCP 서버
        self.servers["cooking_recipes"] = {
            "name": "Cooking Recipes MCP",
            "url": "https://recipes-mcp.example.com",
            "capabilities": [
                "search_recipes",
                "get_recipe_details",
                "find_ingredients",
                "calculate_nutrition"
            ],
            "authentication": "api_key",
            "description": "요리 관련 취미 활동 지원"
        }
    
    async def call_mcp_server(self, server_name: str, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 서버 호출"""
        server = self.servers.get(server_name, {})
        if not server:
            return {"error": "Server not found", "details": f"No server named {server_name}"}
        
        if capability not in server.get("capabilities", []):
            return {"error": "Capability not supported", "details": f"Capability {capability} not available in {server_name}"}
        
        try:
            # 인증 헤더 획득
            auth_headers = await self._get_auth_headers(server_name)
            
            # HTTP 클라이언트로 MCP 서버 호출
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{server['url']}/mcp/{capability}",
                    json=params,
                    headers=auth_headers
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    return {"error": "Authentication failed", "details": "Check API credentials"}
                elif response.status_code == 404:
                    return {"error": "Endpoint not found", "details": f"MCP endpoint {capability} not available"}
                else:
                    return {"error": "Request failed", "details": f"HTTP {response.status_code}: {response.text}"}
                    
        except httpx.TimeoutException:
            return {"error": "Request timeout", "details": f"MCP server {server_name} did not respond within 30 seconds"}
        except httpx.ConnectError:
            return {"error": "Connection failed", "details": f"Cannot connect to MCP server {server_name} at {server['url']}"}
        except Exception as e:
            return {"error": "Unexpected error", "details": f"MCP call failed: {str(e)}"}
    
    async def _get_auth_headers(self, server_name: str) -> Dict[str, str]:
        """인증 헤더 생성"""
        server = self.servers.get(server_name, {})
        auth_type = server.get("authentication", "")
        
        headers = {"Content-Type": "application/json"}
        
        try:
            if auth_type == "oauth2":
                # OAuth2 토큰 획득 (실제 구현에서는 환경변수나 설정에서 가져옴)
                token = await self._get_oauth2_token(server_name)
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    
            elif auth_type == "api_key":
                # API 키 인증
                api_key = await self._get_api_key(server_name)
                if bool(api_key):
                    # 서버에 따라 다른 헤더 형식 사용
                    if server_name == "google_maps":
                        headers["X-API-Key"] = api_key
                    elif server_name == "weather_api":
                        headers["Authorization"] = f"Bearer {api_key}"
                    else:
                        headers["X-API-Key"] = api_key
                        
        except Exception as e:
            print(f"Authentication header generation failed for {server_name}: {e}")
            raise ValueError(f"인증 헤더 생성 실패: {e}")
        
        return headers
        
    async def _get_oauth2_token(self, server_name: str) -> str:
        """OAuth2 토큰 획득"""
        try:
            # 실제 구현에서는 OAuth2 플로우 진행
            # 현재는 더미 토큰 반환 또는 환경변수에서 로드
            import os
            
            token_env_map = {
                "google_calendar": "GOOGLE_CALENDAR_TOKEN",
                "social_search": "SOCIAL_MEDIA_TOKEN",
                "education": "EDUCATION_PLATFORM_TOKEN",
                "fitness_tracker": "FITNESS_TRACKER_TOKEN",
                "music_platform": "MUSIC_PLATFORM_TOKEN"
            }
            
            env_var = token_env_map.get(server_name, "")
            token = os.getenv(env_var, "")
            
            # 토큰이 없으면 에러 발생
            if not token:
                raise ValueError(f"OAuth2 토큰을 찾을 수 없습니다: {server_name}")
            return token
            
        except Exception as e:
            raise ValueError(f"OAuth2 토큰 획득 실패: {e}")
    
    async def _get_api_key(self, server_name: str) -> str:
        """API 키 획득"""
        try:
            import os
            
            api_key_env_map = {
                "google_maps": "GOOGLE_MAPS_API_KEY",
                "weather_api": "OPENWEATHER_API_KEY", 
                "ecommerce": "ECOMMERCE_API_KEY",
                "reading_platform": "READING_PLATFORM_API_KEY",
                "cooking_recipes": "RECIPE_API_KEY"
            }
            
            env_var = api_key_env_map.get(server_name, "")
            api_key = os.getenv(env_var, "")
            
            # API 키가 없으면 에러 발생
            if not api_key:
                raise ValueError(f"API 키를 찾을 수 없습니다: {server_name}")
            return api_key
            
        except Exception as e:
            raise ValueError(f"API 키 획득 실패: {e}")

    def get_available_capabilities(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 MCP 서버 기능 목록"""
        capabilities = {}
        for server_name, server_info in self.servers.items():
            capabilities[server_name] = server_info.get("capabilities", [])
        return capabilities 

class MCPManager:
    """최적화된 MCP 서버 매니저"""
    
    def __init__(self):
        self.connections = {}
        self.circuit_breaker = {}  # 서킷 브레이커 패턴
        
    async def get_hobby_suggestions(self, user_profile: Dict[str, Any]) -> List[Dict]:
        """취미 제안 - 실제 MCP 서버 호출"""
        try:
            # 여러 MCP 서버에 병렬 요청
            tasks = [
                self._call_mcp_server("hobby_recommender", {
                    "action": "suggest_hobbies",
                    "profile": user_profile
                }),
                self._call_mcp_server("activity_finder", {
                    "action": "find_activities",
                    "interests": user_profile.get("interests", [])
                })
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            suggestions = []
            for result in results:
                if isinstance(result, list):
                    suggestions.extend(result)
                elif isinstance(result, dict) and "hobbies" in result:
                    suggestions.extend(result["hobbies"])
            
            if not suggestions:
                raise ValueError("취미 추천을 받을 수 없습니다")
            return suggestions
            
        except Exception as e:
            logger.error(f"취미 제안 실패: {e}")
            raise ValueError(f"취미 제안 실패: {e}")
    
    async def _call_mcp_server(self, server_name: str, request: Dict) -> Any:
        """실제 MCP 서버 호출 with 서킷 브레이커"""
        
        # 서킷 브레이커 확인
        if self._is_circuit_open(server_name):
            raise Exception(f"서킷 브레이커 열림: {server_name}")
        
        try:
            # MCP 서버 연결 시도
            if server_name not in self.connections:
                await self._establish_connection(server_name)
            
            connection = self.connections[server_name]
            
            # 실제 MCP 호출
            response = await connection.call_tool(
                name=request.get("action", "default"),
                arguments=request
            )
            
            # 서킷 브레이커 리셋
            self._reset_circuit_breaker(server_name)
            
            return response.content if hasattr(response, 'content') else response
            
        except Exception as e:
            # 서킷 브레이커 트리거
            self._trigger_circuit_breaker(server_name)
            logger.error(f"MCP 서버 호출 실패 {server_name}: {e}")
            raise
    
    async def _establish_connection(self, server_name: str):
        """MCP 서버 연결 설정"""
        try:
            import mcp
            
            # 서버별 설정
            server_configs = {
                "hobby_recommender": {
                    "command": "node",
                    "args": ["./mcp_servers/hobby_recommender.js"]
                },
                "activity_finder": {
                    "command": "python", 
                    "args": ["./mcp_servers/activity_finder.py"]
                },
                "community_scout": {
                    "command": "node",
                    "args": ["./mcp_servers/community_scout.js"]
                }
            }
            
            config = server_configs.get(server_name)
            if not config:
                raise ValueError(f"알 수 없는 서버: {server_name}")
            
            # MCP 클라이언트 생성
            transport = mcp.StdioServerTransport(
                command=config["command"],
                args=config["args"]
            )
            
            client = mcp.Client(transport)
            await client.connect()
            
            self.connections[server_name] = client
            logger.info(f"MCP 서버 연결 성공: {server_name}")
            
        except Exception as e:
            logger.error(f"MCP 서버 연결 실패 {server_name}: {e}")
            raise
    
    def _is_circuit_open(self, server_name: str) -> bool:
        """서킷 브레이커 상태 확인"""
        if server_name not in self.circuit_breaker:
            return False
        
        breaker = self.circuit_breaker[server_name]
        current_time = time.time()
        
        # 5분 후 재시도 허용
        if current_time - breaker["last_failure"] > 300:
            return False
        
        # 실패 횟수가 3회 이상이면 서킷 열림
        return breaker["failure_count"] >= 3
    
    def _trigger_circuit_breaker(self, server_name: str):
        """서킷 브레이커 트리거"""
        if server_name not in self.circuit_breaker:
            self.circuit_breaker[server_name] = {"failure_count": 0, "last_failure": 0}
        
        self.circuit_breaker[server_name]["failure_count"] += 1
        self.circuit_breaker[server_name]["last_failure"] = time.time()
    
    def _reset_circuit_breaker(self, server_name: str):
        """서킷 브레이커 리셋"""
        if server_name in self.circuit_breaker:
            self.circuit_breaker[server_name]["failure_count"] = 0
    