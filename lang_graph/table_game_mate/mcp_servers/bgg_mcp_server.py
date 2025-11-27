"""
BoardGameGeek MCP Server

실제 BGG XML API를 사용하여 게임 정보를 제공하는 MCP 서버
웹 검색 결과에서 학습한 올바른 MCP 패턴을 적용
"""

from mcp.server.fastmcp import FastMCP
import aiohttp
import asyncio
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import json
import logging

# MCP 서버 초기화
mcp = FastMCP("BoardGameGeek")

# BGG API 기본 설정
BGG_API_BASE = "https://boardgamegeek.com/xmlapi2"
REQUEST_DELAY = 5.0  # BGG API 요청 간격 (5초)

logger = logging.getLogger(__name__)

# BGG API 호출 제한을 위한 글로벌 상태
last_request_time = 0

async def rate_limited_request(url: str) -> str:
    """BGG API 속도 제한을 준수하는 HTTP 요청"""
    global last_request_time
    
    # 5초 간격 보장
    current_time = asyncio.get_event_loop().time()
    time_since_last = current_time - last_request_time
    if time_since_last < REQUEST_DELAY:
        await asyncio.sleep(REQUEST_DELAY - time_since_last)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            last_request_time = asyncio.get_event_loop().time()
            if response.status == 200:
                return await response.text()
            else:
                raise Exception(f"BGG API error: {response.status}")

@mcp.tool()
async def search_boardgame(name: str, exact: bool = False) -> dict:
    """
    BoardGameGeek에서 보드게임 검색
    
    Args:
        name: 검색할 게임 이름
        exact: 정확한 이름 매칭 여부
        
    Returns:
        검색 결과 (게임 ID, 이름, 년도 등)
    """
    # BGG 검색 API 호출 (URL 인코딩 필요)
    from urllib.parse import quote_plus
    search_type = "&exact=1" if exact else ""
    encoded_name = quote_plus(name)
    url = f"{BGG_API_BASE}/search?query={encoded_name}&type=boardgame{search_type}"
    
    try:
        xml_content = await rate_limited_request(url)
        root = ET.fromstring(xml_content)
        
        games = []
        for item in root.findall('.//item'):
            game = {
                "id": int(item.get("id")),
                "name": item.find("name").get("value"),
                "year": item.find("yearpublished").get("value") if item.find("yearpublished") is not None else None
            }
            games.append(game)
        
        return {
            "success": True,
            "games": games,
            "total": len(games)
        }
        
    except Exception as e:
        logger.error(f"BGG search error: {e}")
        return {
            "success": False,
            "error": str(e),
            "games": []
        }

@mcp.tool()
async def get_game_details(bgg_id: int) -> dict:
    """
    BGG에서 게임 상세 정보 조회
    
    Args:
        bgg_id: BoardGameGeek 게임 ID
        
    Returns:
        게임 상세 정보 (규칙, 메타데이터 등)
    """
    async def _get_details():
        url = f"{BGG_API_BASE}/thing?id={bgg_id}&stats=1"
        
        try:
            xml_content = await rate_limited_request(url)
            root = ET.fromstring(xml_content)
            
            item = root.find('.//item')
            if item is None:
                return {"success": False, "error": "Game not found"}
            
            # 기본 정보 추출
            game_info = {
                "id": int(item.get("id")),
                "name": item.find(".//name[@type='primary']").get("value"),
                "description": item.find("description").text if item.find("description") is not None else "",
                "year_published": int(item.find("yearpublished").get("value")) if item.find("yearpublished") is not None else None,
                "min_players": int(item.find("minplayers").get("value")) if item.find("minplayers") is not None else None,
                "max_players": int(item.find("maxplayers").get("value")) if item.find("maxplayers") is not None else None,
                "playing_time": int(item.find("playingtime").get("value")) if item.find("playingtime") is not None else None,
                "min_age": int(item.find("minage").get("value")) if item.find("minage") is not None else None,
            }
            
            # 카테고리 및 메카닉
            categories = []
            for link in item.findall(".//link[@type='boardgamecategory']"):
                categories.append(link.get("value"))
            
            mechanics = []
            for link in item.findall(".//link[@type='boardgamemechanic']"):
                mechanics.append(link.get("value"))
            
            game_info["categories"] = categories
            game_info["mechanics"] = mechanics
            
            # 통계 정보
            stats = item.find(".//statistics/ratings")
            if stats is not None:
                game_info["rating"] = {
                    "average": float(stats.find("average").get("value")) if stats.find("average") is not None else 0,
                    "complexity": float(stats.find("averageweight").get("value")) if stats.find("averageweight") is not None else 0,
                    "users_rated": int(stats.find("usersrated").get("value")) if stats.find("usersrated") is not None else 0
                }
            
            return {
                "success": True,
                "game": game_info
            }
            
        except Exception as e:
            logger.error(f"BGG details error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # FastMCP는 이미 async 컨텍스트이므로 직접 await
    return await _get_details()

@mcp.tool()
async def get_game_rules_summary(bgg_id: int) -> dict:
    """
    게임 규칙 요약 생성 (BGG 설명 기반)
    
    Args:
        bgg_id: BoardGameGeek 게임 ID
        
    Returns:
        규칙 요약 정보
    """
    details = await get_game_details(bgg_id)
    
    if not details.get("success"):
        return details
    
    game = details["game"]
    description = game.get("description", "")
    
    # HTML 태그 제거 (간단한 정규식)
    import re
    clean_description = re.sub(r'<[^>]+>', '', description)
    
    # 규칙 요약 구조화
    rules_summary = {
        "game_name": game["name"],
        "players": f"{game.get('min_players', '?')}-{game.get('max_players', '?')}",
        "playing_time": f"{game.get('playing_time', '?')} minutes",
        "complexity": game.get("rating", {}).get("complexity", 0),
        "categories": game.get("categories", []),
        "mechanics": game.get("mechanics", []),
        "description": clean_description[:500] + "..." if len(clean_description) > 500 else clean_description,
        "setup_hint": f"게임 준비: {game.get('min_players', '?')}명 이상 필요",
        "objective_hint": "승리 조건은 게임 상세 규칙을 참조하세요."
    }
    
    return {
        "success": True,
        "rules_summary": rules_summary
    }

@mcp.resource("bgg://game/{game_id}")
def get_game_resource(game_id: str) -> str:
    """
    게임 정보를 리소스로 제공
    
    Args:
        game_id: BGG 게임 ID
        
    Returns:
        게임 정보 JSON 문자열
    """
    try:
        bgg_id = int(game_id)
        details = get_game_details(bgg_id)
        return json.dumps(details, ensure_ascii=False, indent=2)
    except ValueError:
        return json.dumps({"error": "Invalid game ID"}, ensure_ascii=False)

@mcp.resource("bgg://search/{query}")
def search_game_resource(query: str) -> str:
    """
    게임 검색을 리소스로 제공
    
    Args:
        query: 검색 쿼리
        
    Returns:
        검색 결과 JSON 문자열
    """
    results = search_boardgame(query)
    return json.dumps(results, ensure_ascii=False, indent=2)

@mcp.prompt()
def game_analysis_prompt(game_name: str, player_count: int) -> str:
    """게임 분석을 위한 프롬프트"""
    return f"""
당신은 테이블게임 전문가입니다. '{game_name}' 게임을 {player_count}명이 플레이하는 상황을 분석해주세요.

분석 항목:
1. 게임 타입 및 장르
2. 핵심 메카닉
3. {player_count}명 플레이에 적합한지 평가
4. 예상 게임 시간
5. 복잡도 수준
6. 추천 플레이어 페르소나

BGG 정보를 활용하여 정확한 분석을 제공해주세요.
"""

@mcp.prompt()
def rules_parsing_prompt(description: str, mechanics: list) -> str:
    """게임 규칙 파싱을 위한 프롬프트"""
    mechanics_str = ", ".join(mechanics) if mechanics else "정보 없음"
    
    return f"""
다음 게임 정보를 바탕으로 구조화된 게임 규칙을 추출해주세요:

게임 설명: {description}
주요 메카닉: {mechanics_str}

출력 형식 (JSON):
{{
    "setup": "게임 준비 방법",
    "turn_structure": "턴 진행 순서",
    "actions": ["가능한 행동들"],
    "win_conditions": "승리 조건",
    "special_rules": "특별 규칙들"
}}
"""

# 서버 실행
if __name__ == "__main__":
    # stdio 모드로 실행 (LangGraph 통합용)
    mcp.run() 