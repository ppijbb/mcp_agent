#!/usr/bin/env python3
"""
Game Rules MCP Server

실제 BGG API 호출과 LLM 파싱을 통한 동적 게임 규칙 서버
"""

import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional

# MCP 서버 초기화
mcp = FastMCP("GameRules")

# 전역 HTTP 세션 (재사용을 위해)
http_session = None

async def get_http_session():
    """HTTP 세션 싱글톤"""
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()
    return http_session

@mcp.tool()
async def search_bgg_game(game_name: str) -> str:
    """BGG에서 게임 검색 및 기본 정보 수집"""
    
    session = await get_http_session()
    
    try:
        # BGG 검색 API
        search_url = f"https://boardgamegeek.com/xmlapi2/search?query={quote(game_name)}&type=boardgame"
        
        async with session.get(search_url) as resp:
            if resp.status != 200:
                return json.dumps({"error": f"BGG 검색 실패: {resp.status}", "success": False})
            
            xml_data = await resp.text()
            root = ET.fromstring(xml_data)
            
            # 첫 번째 게임 정보
            first_item = root.find('.//item')
            if first_item is None:
                return json.dumps({"error": f"'{game_name}' 게임을 찾을 수 없음", "success": False})
            
            game_id = first_item.get('id')
            name_elem = first_item.find('.//name')
            actual_name = name_elem.get('value') if name_elem is not None else game_name
            
            return json.dumps({
                "success": True,
                "game_id": game_id,
                "name": actual_name,
                "message": f"BGG에서 '{actual_name}' 발견 (ID: {game_id})"
            })
    
    except Exception as e:
        return json.dumps({"error": f"BGG 검색 오류: {str(e)}", "success": False})

@mcp.tool()
async def get_bgg_game_details(game_id: str) -> str:
    """BGG 게임 상세 정보 조회"""
    
    session = await get_http_session()
    
    try:
        detail_url = f"https://boardgamegeek.com/xmlapi2/thing?id={game_id}&stats=1"
        
        async with session.get(detail_url) as resp:
            if resp.status != 200:
                return json.dumps({"error": f"BGG 상세 정보 실패: {resp.status}", "success": False})
            
            detail_xml = await resp.text()
            detail_root = ET.fromstring(detail_xml)
            
            item = detail_root.find('.//item')
            if item is None:
                return json.dumps({"error": "게임 상세 정보 없음", "success": False})
            
            # 정보 추출
            name = item.find('.//name[@type="primary"]')
            description = item.find('.//description')
            min_players = item.find('.//minplayers')
            max_players = item.find('.//maxplayers')
            playing_time = item.find('.//playingtime')
            min_age = item.find('.//minage')
            
            # 메커니즘과 카테고리
            mechanics = []
            for link in item.findall('.//link[@type="boardgamemechanic"]'):
                mechanics.append(link.get('value'))
            
            categories = []
            for link in item.findall('.//link[@type="boardgamecategory"]'):
                categories.append(link.get('value'))
            
            game_details = {
                "success": True,
                "name": name.get('value') if name is not None else 'Unknown',
                "description": description.text if description is not None else '',
                "min_players": int(min_players.get('value')) if min_players is not None else 2,
                "max_players": int(max_players.get('value')) if max_players is not None else 4,
                "playing_time": int(playing_time.get('value')) if playing_time is not None else 60,
                "min_age": int(min_age.get('value')) if min_age is not None else 10,
                "mechanics": mechanics[:8],  # 상위 8개
                "categories": categories[:5]  # 상위 5개
            }
            
            return json.dumps(game_details)
    
    except Exception as e:
        return json.dumps({"error": f"BGG 상세 정보 오류: {str(e)}", "success": False})

@mcp.tool()
async def generate_game_rules_structure(game_details: str) -> str:
    """게임 상세 정보를 바탕으로 구조화된 규칙 생성"""
    
    try:
        details = json.loads(game_details)
        if not details.get("success"):
            return json.dumps({"error": "잘못된 게임 정보", "success": False})
        
        # 수집된 정보 기반으로 구조화된 규칙 생성
        structured_rules = {
            "success": True,
            "game_name": details["name"],
            "player_count": {
                "min": details["min_players"],
                "max": details["max_players"]
            },
            "playing_time": details["playing_time"],
            "complexity": "medium",  # 기본값
            "setup": f"'{details['name']}' 게임을 위한 구성품을 준비합니다.",
            "turn_actions": [],
            "win_condition": f"'{details['name']}'의 목표를 달성한 플레이어가 승리합니다.",
            "key_mechanics": details["mechanics"],
            "categories": details["categories"],
            "ai_strategy": [],
            "description": details["description"][:500] + "..." if len(details["description"]) > 500 else details["description"]
        }
        
        # 메커니즘 기반 행동 추론
        if "Card Drafting" in details["mechanics"]:
            structured_rules["turn_actions"].extend(["카드 선택", "카드 패스"])
        if "Tile Placement" in details["mechanics"]:
            structured_rules["turn_actions"].extend(["타일 배치", "위치 선택"])
        if "Worker Placement" in details["mechanics"]:
            structured_rules["turn_actions"].extend(["일꾼 배치", "액션 선택"])
        if "Set Collection" in details["mechanics"]:
            structured_rules["turn_actions"].extend(["세트 수집", "자원 관리"])
        
        # 기본 행동이 없으면 추가
        if not structured_rules["turn_actions"]:
            structured_rules["turn_actions"] = ["행동 선택", "자원 관리", "전략 실행"]
        
        # 메커니즘 기반 AI 전략
        for mechanic in details["mechanics"][:3]:  # 상위 3개 메커니즘
            strategy = f"{mechanic} 최적화"
            structured_rules["ai_strategy"].append(strategy)
        
        if not structured_rules["ai_strategy"]:
            structured_rules["ai_strategy"] = ["균형잡힌 플레이", "효율성 추구"]
        
        return json.dumps(structured_rules)
    
    except Exception as e:
        return json.dumps({"error": f"규칙 구조화 오류: {str(e)}", "success": False})

@mcp.tool()
async def get_game_ai_guidance(game_name: str, mechanics: str) -> str:
    """게임별 AI 가이던스 생성"""
    
    try:
        mechanics_list = json.loads(mechanics) if isinstance(mechanics, str) else mechanics
        
        guidance = {
            "success": True,
            "game_name": game_name,
            "decision_points": [],
            "strategy_tips": [],
            "evaluation_criteria": []
        }
        
        # 메커니즘별 가이던스
        mechanic_guidance = {
            "Card Drafting": {
                "decision_points": ["카드 우선순위", "상대방 견제"],
                "strategy_tips": ["강력한 조합 추구", "상대방 전략 차단"],
                "evaluation_criteria": ["카드 시너지", "점수 효율성"]
            },
            "Tile Placement": {
                "decision_points": ["최적 위치 선택", "미래 확장성"],
                "strategy_tips": ["연결성 고려", "상대방 방해"],
                "evaluation_criteria": ["지역 장악력", "점수 잠재력"]
            },
            "Worker Placement": {
                "decision_points": ["액션 우선순위", "타이밍"],
                "strategy_tips": ["필수 액션 우선", "블로킹 고려"],
                "evaluation_criteria": ["자원 효율성", "액션 가치"]
            },
            "Set Collection": {
                "decision_points": ["수집 목표", "세트 완성 순서"],
                "strategy_tips": ["다양성 vs 집중", "타이밍 조절"],
                "evaluation_criteria": ["세트 완성도", "보너스 점수"]
            }
        }
        
        # 해당 메커니즘의 가이던스 추가
        for mechanic in mechanics_list:
            if mechanic in mechanic_guidance:
                guide = mechanic_guidance[mechanic]
                guidance["decision_points"].extend(guide["decision_points"])
                guidance["strategy_tips"].extend(guide["strategy_tips"])
                guidance["evaluation_criteria"].extend(guide["evaluation_criteria"])
        
        # 기본 가이던스 (메커니즘이 없거나 알 수 없는 경우)
        if not guidance["decision_points"]:
            guidance["decision_points"] = ["최적 행동 선택", "상황 분석"]
            guidance["strategy_tips"] = ["균형잡힌 접근", "적응형 전략"]
            guidance["evaluation_criteria"] = ["즉시 이익", "장기 전략"]
        
        return json.dumps(guidance)
    
    except Exception as e:
        return json.dumps({"error": f"AI 가이던스 생성 오류: {str(e)}", "success": False})

# 프롬프트 정의
@mcp.prompt()
def game_analysis_prompt(game_name: str, game_details: str) -> str:
    """게임 분석을 위한 프롬프트"""
    return f"""
다음 보드게임 정보를 분석하여 AI 플레이어가 효과적으로 플레이할 수 있도록 도와주세요.

게임 이름: {game_name}
게임 정보: {game_details}

분석해야 할 요소:
1. 핵심 전략 요소
2. 중요한 결정 포인트
3. AI가 고려해야 할 상호작용
4. 승리 전략

구체적이고 실용적인 조언을 제공해주세요.
"""

if __name__ == "__main__":
    # stdio로 MCP 서버 실행
    mcp.run() 