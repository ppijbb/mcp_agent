#!/usr/bin/env python3
"""
Brave Search MCP Server - 웹 검색

게임 규칙, 전략, 보드게임 정보 등
Brave Search API를 통한 웹 검색 기능을 제공하는 MCP 서버
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse
import hashlib

# MCP 관련 임포트
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️ MCP 패키지가 설치되지 않음. 시뮬레이션 모드로 실행")
    MCP_AVAILABLE = False


class BraveSearchMCPServer:
    """Brave Search API를 통한 웹 검색 MCP 서버"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "YOUR_BRAVE_API_KEY"  # 실제 환경에서는 환경변수 사용
        self.server = Server("brave-search-server") if MCP_AVAILABLE else None
        
        # API 설정
        self.base_url = "https://api.search.brave.com/res/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 검색 캐시 및 통계
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=1)  # 1시간 캐시
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0
        }
        
        # 게임 관련 검색 최적화
        self.game_search_patterns = {
            "rules": [
                "{game_name} rules",
                "{game_name} how to play",
                "{game_name} official rules PDF",
                "{game_name} rulebook"
            ],
            "strategy": [
                "{game_name} strategy guide",
                "{game_name} tips and tricks",
                "{game_name} winning strategy",
                "{game_name} advanced tactics"
            ],
            "reviews": [
                "{game_name} review",
                "{game_name} boardgamegeek",
                "{game_name} board game review"
            ]
        }
        
        if MCP_AVAILABLE and self.server:
            self._register_tools()
    
    async def initialize_session(self):
        """HTTP 세션 초기화"""
        if not self.session or self.session.closed:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
    
    def _register_tools(self):
        """MCP 도구들 등록"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_web",
                    description="일반 웹 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "count": {"type": "number", "description": "결과 수 (기본값: 10, 최대: 20)"},
                            "country": {"type": "string", "description": "국가 코드 (기본값: KR)"},
                            "search_lang": {"type": "string", "description": "검색 언어 (기본값: ko)"},
                            "ui_lang": {"type": "string", "description": "UI 언어 (기본값: ko-KR)"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_game_rules",
                    description="게임 규칙 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름"},
                            "language": {"type": "string", "description": "언어 (ko, en, 기본값: ko)"},
                            "include_pdf": {"type": "boolean", "description": "PDF 룰북 포함 (기본값: true)"}
                        },
                        "required": ["game_name"]
                    }
                ),
                Tool(
                    name="search_game_strategy",
                    description="게임 전략 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름"},
                            "strategy_type": {"type": "string", "description": "전략 타입 (beginner, advanced, competitive)"},
                            "language": {"type": "string", "description": "언어 (ko, en, 기본값: ko)"}
                        },
                        "required": ["game_name"]
                    }
                ),
                Tool(
                    name="search_game_reviews",
                    description="게임 리뷰 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "game_name": {"type": "string", "description": "게임 이름"},
                            "review_source": {"type": "string", "description": "리뷰 소스 (boardgamegeek, youtube, blog)"},
                            "language": {"type": "string", "description": "언어 (ko, en, 기본값: ko)"}
                        },
                        "required": ["game_name"]
                    }
                ),
                Tool(
                    name="search_news",
                    description="뉴스 검색",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "count": {"type": "number", "description": "결과 수 (기본값: 10)"},
                            "country": {"type": "string", "description": "국가 코드 (기본값: KR)"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_search_suggestions",
                    description="검색 제안",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "country": {"type": "string", "description": "국가 코드 (기본값: KR)"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="analyze_search_results",
                    description="검색 결과 분석",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "검색 쿼리"},
                            "analysis_type": {"type": "string", "description": "분석 타입 (summary, sources, relevance)"}
                        },
                        "required": ["query", "analysis_type"]
                    }
                ),
                Tool(
                    name="get_search_stats",
                    description="검색 통계 조회",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {"type": "boolean", "description": "상세 통계 포함 (기본값: false)"}
                        }
                    }
                ),
                Tool(
                    name="clear_search_cache",
                    description="검색 캐시 정리",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "older_than_hours": {"type": "number", "description": "N시간 이전 캐시 정리 (기본값: 24)"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """도구 호출 처리"""
            
            try:
                if name == "search_web":
                    result = await self.search_web(**arguments)
                elif name == "search_game_rules":
                    result = await self.search_game_rules(**arguments)
                elif name == "search_game_strategy":
                    result = await self.search_game_strategy(**arguments)
                elif name == "search_game_reviews":
                    result = await self.search_game_reviews(**arguments)
                elif name == "search_news":
                    result = await self.search_news(**arguments)
                elif name == "get_search_suggestions":
                    result = await self.get_search_suggestions(**arguments)
                elif name == "analyze_search_results":
                    result = await self.analyze_search_results(**arguments)
                elif name == "get_search_stats":
                    result = await self.get_search_stats(**arguments)
                elif name == "clear_search_cache":
                    result = await self.clear_search_cache(**arguments)
                else:
                    result = {"error": f"알 수 없는 도구: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )]
                
            except Exception as e:
                error_result = {
                    "error": f"도구 실행 실패: {str(e)}",
                    "tool": name,
                    "arguments": arguments
                }
                return [TextContent(
                    type="text",
                    text=json.dumps(error_result, ensure_ascii=False, indent=2)
                )]
    
    def _get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        cache_data = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """캐시 유효성 확인"""
        cached_time = datetime.fromisoformat(cache_entry["cached_at"])
        return datetime.now() - cached_time < self.cache_ttl
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Brave Search API 요청"""
        
        await self.initialize_session()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                self.search_stats["api_calls"] += 1
                
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"API 오류 {response.status}: {error_text}",
                        "status_code": response.status
                    }
                    
        except Exception as e:
            self.search_stats["errors"] += 1
            return {"success": False, "error": f"요청 실패: {str(e)}"}
    
    async def search_web(
        self,
        query: str,
        count: int = 10,
        country: str = "KR",
        search_lang: str = "ko",
        ui_lang: str = "ko-KR"
    ) -> Dict[str, Any]:
        """일반 웹 검색"""
        
        try:
            # 캐시 확인
            params = {
                "q": query,
                "count": min(count, 20),
                "country": country,
                "search_lang": search_lang,
                "ui_lang": ui_lang
            }
            
            cache_key = self._get_cache_key(query, params)
            
            if cache_key in self.search_cache and self._is_cache_valid(self.search_cache[cache_key]):
                self.search_stats["cache_hits"] += 1
                return self.search_cache[cache_key]["data"]
            
            # API 요청
            result = await self._make_api_request("web/search", params)
            
            if not result["success"]:
                return result
            
            # 결과 처리
            search_data = result["data"]
            processed_results = {
                "success": True,
                "query": query,
                "total_results": search_data.get("web", {}).get("totalCount", 0),
                "results": [],
                "search_metadata": {
                    "country": country,
                    "language": search_lang,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # 웹 결과 처리
            web_results = search_data.get("web", {}).get("results", [])
            for item in web_results:
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("published", ""),
                    "domain": urlparse(item.get("url", "")).netloc
                }
                processed_results["results"].append(processed_result)
            
            # 캐시 저장
            cache_entry = {
                "data": processed_results,
                "cached_at": datetime.now().isoformat()
            }
            self.search_cache[cache_key] = cache_entry
            
            self.search_stats["total_searches"] += 1
            
            return processed_results
            
        except Exception as e:
            return {"error": f"웹 검색 실패: {str(e)}"}
    
    async def search_game_rules(
        self,
        game_name: str,
        language: str = "ko",
        include_pdf: bool = True
    ) -> Dict[str, Any]:
        """게임 규칙 검색"""
        
        try:
            search_queries = []
            
            # 언어별 검색 쿼리 생성
            if language == "ko":
                search_queries = [
                    f"{game_name} 룰북",
                    f"{game_name} 게임 방법",
                    f"{game_name} 규칙",
                    f"{game_name} 플레이 방법"
                ]
            else:
                search_queries = [
                    f"{game_name} rules",
                    f"{game_name} rulebook",
                    f"{game_name} how to play",
                    f"{game_name} official rules"
                ]
            
            if include_pdf:
                search_queries.extend([
                    f"{game_name} rules PDF",
                    f"{game_name} rulebook PDF"
                ])
            
            all_results = []
            
            # 각 쿼리로 검색 실행
            for query in search_queries[:3]:  # 최대 3개 쿼리
                search_result = await self.search_web(
                    query=query,
                    count=5,
                    search_lang=language
                )
                
                if search_result.get("success"):
                    for result in search_result.get("results", []):
                        # 중복 제거
                        if not any(r["url"] == result["url"] for r in all_results):
                            result["search_query"] = query
                            result["relevance_score"] = self._calculate_rule_relevance(result, game_name)
                            all_results.append(result)
            
            # 관련성 점수로 정렬
            all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # 상위 10개만 반환
            top_results = all_results[:10]
            
            return {
                "success": True,
                "game_name": game_name,
                "language": language,
                "include_pdf": include_pdf,
                "total_results": len(top_results),
                "results": top_results,
                "search_metadata": {
                    "queries_used": search_queries[:3],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"게임 규칙 검색 실패: {str(e)}"}
    
    def _calculate_rule_relevance(self, result: Dict[str, Any], game_name: str) -> float:
        """규칙 검색 결과의 관련성 점수 계산"""
        
        score = 0.0
        title = result.get("title", "").lower()
        description = result.get("description", "").lower()
        url = result.get("url", "").lower()
        game_lower = game_name.lower()
        
        # 게임 이름 일치도
        if game_lower in title:
            score += 3.0
        if game_lower in description:
            score += 2.0
        if game_lower in url:
            score += 1.0
        
        # 규칙 관련 키워드
        rule_keywords = ["rule", "rulebook", "how to play", "규칙", "룰북", "게임방법", "플레이방법"]
        for keyword in rule_keywords:
            if keyword in title:
                score += 2.0
            if keyword in description:
                score += 1.0
        
        # PDF 파일 우대
        if "pdf" in url or "pdf" in title:
            score += 1.5
        
        # 신뢰할 만한 도메인 우대
        trusted_domains = ["boardgamegeek.com", "rulebook", "official", "publisher"]
        for domain in trusted_domains:
            if domain in url:
                score += 1.0
        
        return score
    
    async def search_game_strategy(
        self,
        game_name: str,
        strategy_type: str = "general",
        language: str = "ko"
    ) -> Dict[str, Any]:
        """게임 전략 검색"""
        
        try:
            strategy_keywords = {
                "beginner": ["초보자", "beginner", "basic", "guide"],
                "advanced": ["고급", "advanced", "expert", "pro"],
                "competitive": ["경쟁", "competitive", "tournament", "championship"],
                "general": ["전략", "strategy", "tip", "tactic"]
            }
            
            keywords = strategy_keywords.get(strategy_type, strategy_keywords["general"])
            
            search_queries = []
            if language == "ko":
                for keyword in keywords:
                    search_queries.append(f"{game_name} {keyword}")
            else:
                for keyword in keywords:
                    search_queries.append(f"{game_name} {keyword}")
            
            all_results = []
            
            for query in search_queries[:2]:  # 최대 2개 쿼리
                search_result = await self.search_web(
                    query=query,
                    count=5,
                    search_lang=language
                )
                
                if search_result.get("success"):
                    for result in search_result.get("results", []):
                        if not any(r["url"] == result["url"] for r in all_results):
                            result["search_query"] = query
                            result["strategy_type"] = strategy_type
                            all_results.append(result)
            
            return {
                "success": True,
                "game_name": game_name,
                "strategy_type": strategy_type,
                "language": language,
                "total_results": len(all_results),
                "results": all_results[:8],
                "search_metadata": {
                    "queries_used": search_queries[:2],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"게임 전략 검색 실패: {str(e)}"}
    
    async def search_game_reviews(
        self,
        game_name: str,
        review_source: Optional[str] = None,
        language: str = "ko"
    ) -> Dict[str, Any]:
        """게임 리뷰 검색"""
        
        try:
            if review_source:
                if language == "ko":
                    query = f"{game_name} 리뷰 site:{review_source}"
                else:
                    query = f"{game_name} review site:{review_source}"
            else:
                if language == "ko":
                    query = f"{game_name} 리뷰 평가"
                else:
                    query = f"{game_name} review rating"
            
            search_result = await self.search_web(
                query=query,
                count=10,
                search_lang=language
            )
            
            if not search_result.get("success"):
                return search_result
            
            # 리뷰 결과 필터링 및 강화
            review_results = []
            for result in search_result.get("results", []):
                # 리뷰 관련성 점수 계산
                review_score = self._calculate_review_relevance(result, game_name)
                
                if review_score > 0.5:  # 임계값 이상만 포함
                    result["review_score"] = review_score
                    result["review_indicators"] = self._extract_review_indicators(result)
                    review_results.append(result)
            
            # 리뷰 점수로 정렬
            review_results.sort(key=lambda x: x.get("review_score", 0), reverse=True)
            
            return {
                "success": True,
                "game_name": game_name,
                "review_source": review_source,
                "language": language,
                "total_results": len(review_results),
                "results": review_results,
                "search_metadata": {
                    "query_used": query,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"게임 리뷰 검색 실패: {str(e)}"}
    
    def _calculate_review_relevance(self, result: Dict[str, Any], game_name: str) -> float:
        """리뷰 검색 결과의 관련성 점수 계산"""
        
        score = 0.0
        title = result.get("title", "").lower()
        description = result.get("description", "").lower()
        url = result.get("url", "").lower()
        
        # 게임 이름 일치도
        if game_name.lower() in title:
            score += 2.0
        if game_name.lower() in description:
            score += 1.0
        
        # 리뷰 관련 키워드
        review_keywords = ["review", "rating", "리뷰", "평가", "평점", "후기"]
        for keyword in review_keywords:
            if keyword in title:
                score += 1.5
            if keyword in description:
                score += 1.0
        
        # 신뢰할 만한 리뷰 사이트
        review_sites = ["boardgamegeek.com", "youtube.com", "blog", "naver", "daum"]
        for site in review_sites:
            if site in url:
                score += 1.0
        
        return score
    
    def _extract_review_indicators(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """리뷰 지표 추출"""
        
        indicators = {}
        text = f"{result.get('title', '')} {result.get('description', '')}"
        
        # 점수 패턴 찾기
        score_patterns = [
            r'(\d+(?:\.\d+)?)/(\d+)',  # 8.5/10
            r'(\d+(?:\.\d+)?)점',      # 8.5점
            r'★+',                     # 별점
            r'(\d+)%'                  # 85%
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, text)
            if matches:
                indicators["score_mentions"] = matches
                break
        
        # 감정 키워드
        positive_keywords = ["좋은", "훌륭한", "재미있는", "추천", "excellent", "great", "fun", "recommend"]
        negative_keywords = ["나쁜", "지루한", "실망", "비추천", "bad", "boring", "disappointing", "avoid"]
        
        indicators["positive_mentions"] = sum(1 for keyword in positive_keywords if keyword in text.lower())
        indicators["negative_mentions"] = sum(1 for keyword in negative_keywords if keyword in text.lower())
        
        return indicators
    
    async def search_news(
        self,
        query: str,
        count: int = 10,
        country: str = "KR"
    ) -> Dict[str, Any]:
        """뉴스 검색"""
        
        try:
            params = {
                "q": query,
                "count": min(count, 20),
                "country": country
            }
            
            result = await self._make_api_request("news/search", params)
            
            if not result["success"]:
                return result
            
            news_data = result["data"]
            processed_results = {
                "success": True,
                "query": query,
                "total_results": len(news_data.get("results", [])),
                "results": [],
                "search_metadata": {
                    "country": country,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            for item in news_data.get("results", []):
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("published", ""),
                    "source": item.get("source", ""),
                    "thumbnail": item.get("thumbnail", {}).get("src", "")
                }
                processed_results["results"].append(processed_result)
            
            self.search_stats["total_searches"] += 1
            
            return processed_results
            
        except Exception as e:
            return {"error": f"뉴스 검색 실패: {str(e)}"}
    
    async def get_search_suggestions(
        self,
        query: str,
        country: str = "KR"
    ) -> Dict[str, Any]:
        """검색 제안"""
        
        try:
            params = {
                "q": query,
                "country": country
            }
            
            result = await self._make_api_request("suggest", params)
            
            if not result["success"]:
                return result
            
            suggestions_data = result["data"]
            
            return {
                "success": True,
                "query": query,
                "suggestions": suggestions_data.get("results", []),
                "search_metadata": {
                    "country": country,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"검색 제안 실패: {str(e)}"}
    
    async def analyze_search_results(
        self,
        query: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """검색 결과 분석"""
        
        try:
            # 먼저 검색 실행
            search_result = await self.search_web(query, count=20)
            
            if not search_result.get("success"):
                return search_result
            
            results = search_result.get("results", [])
            analysis = {}
            
            if analysis_type == "summary":
                # 결과 요약
                domains = {}
                total_results = len(results)
                
                for result in results:
                    domain = result.get("domain", "unknown")
                    domains[domain] = domains.get(domain, 0) + 1
                
                analysis = {
                    "total_results": total_results,
                    "unique_domains": len(domains),
                    "top_domains": sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5],
                    "avg_title_length": sum(len(r.get("title", "")) for r in results) / max(1, total_results),
                    "has_recent_content": any("2024" in r.get("published", "") for r in results)
                }
            
            elif analysis_type == "sources":
                # 소스 분석
                source_analysis = {}
                for result in results:
                    domain = result.get("domain", "unknown")
                    if domain not in source_analysis:
                        source_analysis[domain] = {
                            "count": 0,
                            "titles": [],
                            "avg_description_length": 0
                        }
                    
                    source_analysis[domain]["count"] += 1
                    source_analysis[domain]["titles"].append(result.get("title", ""))
                
                analysis = {"source_breakdown": source_analysis}
            
            elif analysis_type == "relevance":
                # 관련성 분석
                query_words = query.lower().split()
                relevance_scores = []
                
                for result in results:
                    text = f"{result.get('title', '')} {result.get('description', '')}".lower()
                    score = sum(1 for word in query_words if word in text) / len(query_words)
                    relevance_scores.append(score)
                
                analysis = {
                    "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                    "high_relevance_count": sum(1 for score in relevance_scores if score > 0.7),
                    "low_relevance_count": sum(1 for score in relevance_scores if score < 0.3)
                }
            
            return {
                "success": True,
                "query": query,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "analyzed_results": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"검색 결과 분석 실패: {str(e)}"}
    
    async def get_search_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """검색 통계 조회"""
        
        try:
            basic_stats = self.search_stats.copy()
            basic_stats["cache_size"] = len(self.search_cache)
            basic_stats["cache_hit_rate"] = (
                basic_stats["cache_hits"] / max(1, basic_stats["total_searches"])
            )
            
            if not detailed:
                return {
                    "success": True,
                    "stats": basic_stats,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 상세 통계
            detailed_stats = basic_stats.copy()
            
            # 캐시 분석
            cache_analysis = {
                "total_entries": len(self.search_cache),
                "valid_entries": 0,
                "expired_entries": 0
            }
            
            for cache_entry in self.search_cache.values():
                if self._is_cache_valid(cache_entry):
                    cache_analysis["valid_entries"] += 1
                else:
                    cache_analysis["expired_entries"] += 1
            
            detailed_stats["cache_analysis"] = cache_analysis
            
            return {
                "success": True,
                "stats": detailed_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"통계 조회 실패: {str(e)}"}
    
    async def clear_search_cache(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """검색 캐시 정리"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            initial_count = len(self.search_cache)
            
            # 만료된 캐시 항목 제거
            expired_keys = []
            for key, cache_entry in self.search_cache.items():
                cached_time = datetime.fromisoformat(cache_entry["cached_at"])
                if cached_time < cutoff_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.search_cache[key]
            
            cleared_count = len(expired_keys)
            remaining_count = len(self.search_cache)
            
            return {
                "success": True,
                "cleared_entries": cleared_count,
                "remaining_entries": remaining_count,
                "initial_count": initial_count,
                "older_than_hours": older_than_hours,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"캐시 정리 실패: {str(e)}"}
    
    async def close(self):
        """리소스 정리"""
        if self.session and not self.session.closed:
            await self.session.close()


# MCP 서버 실행
async def main():
    """Brave Search MCP 서버 실행"""
    
    if not MCP_AVAILABLE:
        print("❌ MCP 패키지가 필요합니다: pip install mcp")
        return
    
    brave_search_server = BraveSearchMCPServer()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await brave_search_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="brave-search-server",
                    server_version="1.0.0",
                    capabilities={}
                )
            )
    finally:
        await brave_search_server.close()


if __name__ == "__main__":
    asyncio.run(main())