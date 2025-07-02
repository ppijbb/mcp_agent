#!/usr/bin/env python3
"""
수정된 MCP 게임 마스터

실제 BGG API + 동적 규칙 파싱 + 게임 진행
"""

import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import Dict, List, Any, Optional
from datetime import datetime


class FixedMCPGameMaster:
    """수정된 MCP 게임 마스터 - 실제 동작"""
    
    def __init__(self):
        self.session = None
        self.current_games = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_bgg_game(self, game_name: str) -> Dict[str, Any]:
        """BGG에서 실제 게임 검색"""
        
        try:
            search_url = f"https://boardgamegeek.com/xmlapi2/search?query={quote(game_name)}&type=boardgame"
            
            async with self.session.get(search_url) as resp:
                if resp.status != 200:
                    return {"error": f"BGG 검색 실패: {resp.status}", "success": False}
                
                xml_data = await resp.text()
                root = ET.fromstring(xml_data)
                
                first_item = root.find('.//item')
                if first_item is None:
                    return {"error": f"'{game_name}' 게임을 찾을 수 없음", "success": False}
                
                game_id = first_item.get('id')
                name_elem = first_item.find('.//name')
                actual_name = name_elem.get('value') if name_elem is not None else game_name
                
                return {
                    "success": True,
                    "game_id": game_id,
                    "name": actual_name,
                    "message": f"BGG에서 '{actual_name}' 발견 (ID: {game_id})"
                }
        
        except Exception as e:
            return {"error": f"BGG 검색 오류: {str(e)}", "success": False}
    
    async def get_bgg_game_details(self, game_id: str) -> Dict[str, Any]:
        """BGG 게임 상세 정보 조회"""
        
        try:
            detail_url = f"https://boardgamegeek.com/xmlapi2/thing?id={game_id}&stats=1"
            
            async with self.session.get(detail_url) as resp:
                if resp.status != 200:
                    return {"error": f"BGG 상세 정보 실패: {resp.status}", "success": False}
                
                detail_xml = await resp.text()
                detail_root = ET.fromstring(detail_xml)
                
                item = detail_root.find('.//item')
                if item is None:
                    return {"error": "게임 상세 정보 없음", "success": False}
                
                # 정보 추출
                name = item.find('.//name[@type="primary"]')
                description = item.find('.//description')
                min_players = item.find('.//minplayers')
                max_players = item.find('.//maxplayers')
                playing_time = item.find('.//playingtime')
                
                # 메커니즘과 카테고리
                mechanics = []
                for link in item.findall('.//link[@type="boardgamemechanic"]'):
                    mechanics.append(link.get('value'))
                
                categories = []
                for link in item.findall('.//link[@type="boardgamecategory"]'):
                    categories.append(link.get('value'))
                
                return {
                    "success": True,
                    "name": name.get('value') if name is not None else 'Unknown',
                    "description": description.text if description is not None else '',
                    "min_players": int(min_players.get('value')) if min_players is not None else 2,
                    "max_players": int(max_players.get('value')) if max_players is not None else 4,
                    "playing_time": int(playing_time.get('value')) if playing_time is not None else 60,
                    "mechanics": mechanics[:8],
                    "categories": categories[:5]
                }
        
        except Exception as e:
            return {"error": f"BGG 상세 정보 오류: {str(e)}", "success": False}
    
    async def generate_game_rules(self, game_details: Dict) -> Dict[str, Any]:
        """게임 상세 정보를 바탕으로 구조화된 규칙 생성"""
        
        if not game_details.get("success"):
            return {"error": "잘못된 게임 정보", "success": False}
        
        try:
            # 메커니즘 기반 행동 생성
            turn_actions = []
            ai_strategy = []
            
            mechanics = game_details.get("mechanics", [])
            
            # 메커니즘별 행동 매핑
            mechanic_actions = {
                "Card Drafting": ["카드 선택", "카드 패스"],
                "Tile Placement": ["타일 배치", "위치 선택"],
                "Worker Placement": ["일꾼 배치", "액션 선택"],
                "Set Collection": ["세트 수집", "자원 관리"],
                "Pattern Building": ["패턴 구축", "배치 계획"],
                "Grid Coverage": ["그리드 채우기", "영역 확장"],
                "Open Drafting": ["공개 선택", "순서 고려"]
            }
            
            # 메커니즘별 전략 매핑
            mechanic_strategies = {
                "Card Drafting": "카드 시너지 최적화",
                "Tile Placement": "위치 우위 확보", 
                "Worker Placement": "액션 효율성 극대화",
                "Set Collection": "세트 완성 우선순위",
                "Pattern Building": "패턴 완성 전략",
                "Grid Coverage": "영역 장악 전략",
                "End Game Bonuses": "종료 보너스 준비"
            }
            
            # 실제 메커니즘 기반 매핑
            for mechanic in mechanics:
                if mechanic in mechanic_actions:
                    turn_actions.extend(mechanic_actions[mechanic])
                if mechanic in mechanic_strategies:
                    ai_strategy.append(mechanic_strategies[mechanic])
            
            # 기본값 설정
            if not turn_actions:
                turn_actions = ["행동 선택", "자원 관리", "전략 실행"]
            if not ai_strategy:
                ai_strategy = ["균형잡힌 플레이", "효율성 추구"]
            
            structured_rules = {
                "success": True,
                "game_name": game_details["name"],
                "player_count": {
                    "min": game_details["min_players"],
                    "max": game_details["max_players"]
                },
                "playing_time": game_details["playing_time"],
                "setup": f"'{game_details['name']}' 게임을 위한 구성품을 준비합니다.",
                "turn_actions": turn_actions,
                "win_condition": f"'{game_details['name']}'의 목표를 달성한 플레이어가 승리합니다.",
                "key_mechanics": mechanics,
                "categories": game_details["categories"],
                "ai_strategy": ai_strategy,
                "description": game_details["description"][:300] + "..." if len(game_details["description"]) > 300 else game_details["description"]
            }
            
            return structured_rules
        
        except Exception as e:
            return {"error": f"규칙 구조화 오류: {str(e)}", "success": False}
    
    async def create_game_session(self, game_name: str, player_count: int) -> Dict[str, Any]:
        """게임 세션 생성"""
        
        print(f"🎮 '{game_name}' 게임 세션 생성 중... ({player_count}명)")
        
        # 1. BGG에서 게임 검색
        search_result = await self.search_bgg_game(game_name)
        if not search_result["success"]:
            return {"success": False, "error": search_result["error"]}
        
        print(f"✅ {search_result['message']}")
        
        # 2. 게임 상세 정보
        details_result = await self.get_bgg_game_details(search_result["game_id"])
        if not details_result["success"]:
            return {"success": False, "error": details_result["error"]}
        
        print(f"✅ 게임 정보 수집 완료")
        print(f"   플레이어: {details_result['min_players']}-{details_result['max_players']}명")
        print(f"   시간: {details_result['playing_time']}분")
        print(f"   메커니즘: {', '.join(details_result['mechanics'][:3])}")
        
        # 3. 플레이어 수 검증
        if player_count < details_result["min_players"] or player_count > details_result["max_players"]:
            return {
                "success": False, 
                "error": f"플레이어 수 {player_count}명은 허용 범위({details_result['min_players']}-{details_result['max_players']}명)를 벗어남"
            }
        
        # 4. 규칙 구조화
        rules_result = await self.generate_game_rules(details_result)
        if not rules_result["success"]:
            return {"success": False, "error": rules_result["error"]}
        
        print(f"✅ 규칙 구조화 완료")
        print(f"   행동: {', '.join(rules_result['turn_actions'][:3])}")
        print(f"   전략: {', '.join(rules_result['ai_strategy'][:2])}")
        
        # 5. 게임 세션 초기화
        session_id = f"{game_name}_{player_count}_{datetime.now().strftime('%H%M%S')}"
        
        game_session = {
            "session_id": session_id,
            "game_name": game_name,
            "rules": rules_result,
            "player_count": player_count,
            "players": {},
            "current_player": 0,
            "turn_count": 0,
            "game_state": "ready",
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        
        # 플레이어 생성
        for i in range(player_count):
            game_session["players"][f"player_{i}"] = {
                "id": f"player_{i}",
                "name": f"Player {i+1}",
                "type": "ai",
                "score": 0,
                "resources": {},
                "status": "active"
            }
        
        # 세션 저장
        self.current_games[session_id] = game_session
        
        print(f"✅ 게임 세션 '{session_id}' 생성 완료!")
        
        return {
            "success": True,
            "session_id": session_id,
            "game_session": game_session,
            "message": f"'{game_name}' 게임이 {player_count}명으로 준비되었습니다!"
        }
    
    async def process_turn(self, session_id: str, action: str = None) -> Dict[str, Any]:
        """게임 턴 처리"""
        
        if session_id not in self.current_games:
            return {"success": False, "error": "게임 세션을 찾을 수 없음"}
        
        session = self.current_games[session_id]
        
        if session["game_state"] != "ready" and session["game_state"] != "playing":
            return {"success": False, "error": f"게임 상태가 올바르지 않음: {session['game_state']}"}
        
        # 게임 시작
        if session["game_state"] == "ready":
            session["game_state"] = "playing"
            print(f"🚀 '{session['game_name']}' 게임 시작!")
        
        current_player_id = f"player_{session['current_player']}"
        current_player = session["players"][current_player_id]
        rules = session["rules"]
        
        # AI 행동 선택 (간단한 버전)
        available_actions = rules["turn_actions"]
        ai_strategies = rules["ai_strategy"]
        
        if not action:
            # AI가 행동 선택
            import random
            chosen_action = random.choice(available_actions)
            strategy = random.choice(ai_strategies)
            reasoning = f"{strategy}에 기반하여 {chosen_action}을 선택"
        else:
            chosen_action = action
            reasoning = "사용자 지정 행동"
        
        # 턴 기록
        turn_record = {
            "turn": session["turn_count"],
            "player": current_player_id,
            "action": chosen_action,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        session["history"].append(turn_record)
        
        # 점수 업데이트 (간단한 시뮬레이션)
        if "수집" in chosen_action or "배치" in chosen_action:
            current_player["score"] += 1
        
        print(f"🤖 {current_player['name']}: {chosen_action}")
        print(f"   이유: {reasoning}")
        print(f"   점수: {current_player['score']}")
        
        # 다음 플레이어로
        session["current_player"] = (session["current_player"] + 1) % session["player_count"]
        if session["current_player"] == 0:
            session["turn_count"] += 1
        
        # 간단한 승리 조건 (10점 먼저 달성)
        if current_player["score"] >= 10:
            session["game_state"] = "finished"
            session["winner"] = current_player_id
            print(f"🏆 {current_player['name']} 승리! (점수: {current_player['score']})")
        
        return {
            "success": True,
            "turn_result": turn_record,
            "current_state": {
                "turn": session["turn_count"],
                "current_player": session["players"][f"player_{session['current_player']}"]["name"],
                "game_state": session["game_state"],
                "scores": {p["name"]: p["score"] for p in session["players"].values()}
            },
            "game_finished": session["game_state"] == "finished",
            "winner": session.get("winner")
        }
    
    async def get_game_status(self, session_id: str) -> Dict[str, Any]:
        """게임 상태 조회"""
        
        if session_id not in self.current_games:
            return {"success": False, "error": "게임 세션을 찾을 수 없음"}
        
        session = self.current_games[session_id]
        
        return {
            "success": True,
            "session_id": session_id,
            "game_name": session["game_name"],
            "game_state": session["game_state"],
            "turn_count": session["turn_count"],
            "current_player": session["players"][f"player_{session['current_player']}"]["name"],
            "players": session["players"],
            "rules_summary": {
                "actions": session["rules"]["turn_actions"],
                "strategies": session["rules"]["ai_strategy"]
            },
            "recent_history": session["history"][-5:]  # 최근 5턴
        }


# 실제 사용 예시
async def demo_fixed_game_master():
    """수정된 게임 마스터 데모"""
    
    print("🚀 수정된 MCP 게임 마스터 데모 시작")
    print("=" * 60)
    
    async with FixedMCPGameMaster() as game_master:
        
        # 게임 세션 생성
        create_result = await game_master.create_game_session("Azul", 3)
        
        if create_result["success"]:
            session_id = create_result["session_id"]
            print(f"\n✅ {create_result['message']}")
            
            # 게임 상태 확인
            status = await game_master.get_game_status(session_id)
            print(f"\n📊 게임 상태:")
            print(f"   현재 플레이어: {status['current_player']}")
            print(f"   가능한 행동: {', '.join(status['rules_summary']['actions'][:3])}")
            
            # 몇 턴 플레이
            print(f"\n🎯 게임 진행:")
            for turn in range(8):
                turn_result = await game_master.process_turn(session_id)
                
                if turn_result["success"]:
                    if turn_result["game_finished"]:
                        print(f"\n🎉 게임 종료! 승자: {turn_result['winner']}")
                        break
                    
                    await asyncio.sleep(0.5)  # 진행 속도 조절
                else:
                    print(f"❌ 턴 처리 실패: {turn_result['error']}")
                    break
            
            # 최종 상태
            final_status = await game_master.get_game_status(session_id)
            print(f"\n📈 최종 점수:")
            for player_id, player_data in final_status["players"].items():
                print(f"   {player_data['name']}: {player_data['score']}점")
                
        else:
            print(f"❌ 게임 생성 실패: {create_result['error']}")


if __name__ == "__main__":
    asyncio.run(demo_fixed_game_master()) 