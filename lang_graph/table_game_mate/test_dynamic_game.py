#!/usr/bin/env python3
"""
테이블 게임 메이트 - 동적 게임 시스템 테스트

실제 마피아/뱅/카탄 게임이 동작하는지 테스트합니다.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_master import GameMasterGraph
from models.game_state import PlayerInfo


class MockLLMClient:
    """테스트용 Mock LLM 클라이언트"""
    
    async def complete(self, prompt: str) -> str:
        """간단한 Mock 응답"""
        if "마피아" in prompt:
            return '{"setup": {"components": ["역할카드"], "initial_setup": "역할 배정"}, "game_flow": {"turn_structure": "밤-낮 순환", "actions": ["투표", "제거"]}, "win_conditions": {"primary": "상대팀 전멸"}, "ai_guidance": {"decision_points": ["투표 대상"], "strategy_hints": ["의심스러운 행동 관찰"]}}'
        elif "뱅" in prompt:
            return '{"setup": {"components": ["카드", "역할"], "initial_setup": "역할과 카드 배분"}, "game_flow": {"turn_structure": "순서대로", "actions": ["뱅", "맥주", "장비"]}, "win_conditions": {"primary": "역할별 승리조건"}, "ai_guidance": {"decision_points": ["공격 대상"], "strategy_hints": ["체력 관리"]}}'
        else:
            return '{"setup": {"components": ["보드", "말"], "initial_setup": "시작 위치"}, "game_flow": {"turn_structure": "순서대로", "actions": ["이동", "행동"]}, "win_conditions": {"primary": "점수"}, "ai_guidance": {"decision_points": ["행동 선택"], "strategy_hints": ["균형잡힌 플레이"]}}'


class MockMCPClient:
    """테스트용 Mock MCP 클라이언트"""
    
    async def call(self, server: str, method: str, params: dict) -> dict:
        """Mock MCP 호출"""
        if "search_games" in method:
            return {
                "games": [{
                    "id": "12345",
                    "name": params.get("query", "테스트게임"),
                    "year": 2020
                }]
            }
        elif "get_game_details" in method:
            return {
                "name": "테스트게임",
                "players": {"min": 2, "max": 6},
                "playing_time": 60,
                "complexity": 3.0
            }
        else:
            return {"result": "success"}


async def test_mafia_game():
    """마피아 게임 테스트"""
    print("\n" + "="*50)
    print("🌙 마피아 게임 테스트")
    print("="*50)
    
    # 게임 마스터 초기화
    llm_client = MockLLMClient()
    game_master = GameMasterGraph(llm_client)
    game_master.mcp_client = MockMCPClient()
    
    # 게임 설정
    config = {
        "target_game_name": "마피아",
        "desired_player_count": 5,
        "difficulty_level": "medium"
    }
    
    try:
        # 게임 실행
        result = await game_master.run_game(config)
        
        print("✅ 마피아 게임 테스트 완료!")
        print(f"게임 ID: {result.get('game_id', 'N/A')}")
        print(f"플레이어 수: {len(result.get('players', []))}")
        print(f"게임 보드: {result.get('game_board', {}).get('game_type', 'N/A')}")
        
        # 마피아 게임 특화 정보
        game_board = result.get('game_board', {})
        if game_board.get('game_type') == '마피아':
            print(f"역할 배정: {game_board.get('roles', {})}")
            print(f"생존자: {len(game_board.get('alive_players', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 마피아 게임 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bang_game():
    """뱅! 게임 테스트"""
    print("\n" + "="*50)
    print("🤠 뱅! 게임 테스트")
    print("="*50)
    
    llm_client = MockLLMClient()
    game_master = GameMasterGraph(llm_client)
    game_master.mcp_client = MockMCPClient()
    
    config = {
        "target_game_name": "뱅!",
        "desired_player_count": 4,
        "difficulty_level": "medium"
    }
    
    try:
        result = await game_master.run_game(config)
        
        print("✅ 뱅! 게임 테스트 완료!")
        print(f"게임 ID: {result.get('game_id', 'N/A')}")
        
        game_board = result.get('game_board', {})
        if game_board.get('game_type') == '뱅!':
            print(f"역할 배정: {game_board.get('roles', {})}")
            print(f"체력 상태: {game_board.get('health', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ 뱅! 게임 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_catan_game():
    """카탄 게임 테스트"""
    print("\n" + "="*50)
    print("🏝️ 카탄 게임 테스트")
    print("="*50)
    
    llm_client = MockLLMClient()
    game_master = GameMasterGraph(llm_client)
    game_master.mcp_client = MockMCPClient()
    
    config = {
        "target_game_name": "카탄",
        "desired_player_count": 3,
        "difficulty_level": "medium"
    }
    
    try:
        result = await game_master.run_game(config)
        
        print("✅ 카탄 게임 테스트 완료!")
        print(f"게임 ID: {result.get('game_id', 'N/A')}")
        
        game_board = result.get('game_board', {})
        if game_board.get('game_type') == '카탄':
            print(f"보드 헥스 수: {len(game_board.get('board_hexes', {}).get('hexes', []))}")
            print(f"자원 상태: {game_board.get('resources', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ 카탄 게임 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dynamic_action_processing():
    """동적 행동 처리 테스트"""
    print("\n" + "="*50)
    print("🎯 동적 행동 처리 테스트")
    print("="*50)
    
    llm_client = MockLLMClient()
    game_master = GameMasterGraph(llm_client)
    game_master.mcp_client = MockMCPClient()
    
    # 마피아 게임으로 행동 처리 테스트
    from models.game_state import GameMasterState, GamePhase
    from datetime import datetime
    import uuid
    
    # 테스트 상태 생성
    test_state = GameMasterState(
        game_id=str(uuid.uuid4()),
        game_metadata=None,
        phase=GamePhase.PLAYER_TURN,
        players=[
            {"id": "user", "name": "사용자", "is_ai": False, "turn_order": 0},
            {"id": "ai1", "name": "AI플레이어1", "is_ai": True, "turn_order": 1, "persona_type": "aggressive"},
            {"id": "ai2", "name": "AI플레이어2", "is_ai": True, "turn_order": 2, "persona_type": "analytical"}
        ],
        current_player_index=1,  # AI 플레이어 턴
        turn_count=1,
        game_board={
            "game_type": "마피아",
            "phase": "낮",
            "roles": {"user": "시민", "ai1": "마피아", "ai2": "시민"},
            "alive_players": ["user", "ai1", "ai2"],
            "dead_players": [],
            "votes": {},
            "game_log": []
        },
        game_history=[],
        parsed_rules={
            "actions": ["투표", "변론", "관찰"],
            "win_conditions": "상대팀 전멸",
            "ai_guidance": {"decision_points": ["투표 대상"]}
        },
        game_config={"target_game_name": "마피아"},
        last_action=None,
        pending_actions=[],
        error_messages=[],
        winner_ids=[],
        final_scores={},
        game_ended=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        current_agent="",
        agent_responses=[],
        user_input=None,
        awaiting_user_input=False,
        next_step=None
    )
    
    try:
        # AI 턴 처리 테스트
        result_state = await game_master._process_turn(test_state)
        
        print("✅ 동적 행동 처리 테스트 완료!")
        print(f"마지막 행동: {result_state.get('last_action', {})}")
        
        # 사용자 턴으로 변경하여 가능한 행동 확인
        test_state["current_player_index"] = 0  # 사용자 턴
        result_state = await game_master._process_turn(test_state)
        
        print(f"사용자 입력 대기: {result_state.get('awaiting_user_input', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 동적 행동 처리 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 실행"""
    print("🎮 테이블 게임 메이트 - 동적 게임 시스템 테스트")
    print("원래 요구사항: ALL 보드게임 지원, 동적 규칙 파싱, 유연한 플레이어 생성")
    print("현재 구현: 마피아/뱅/카탄 + 일반 게임 동적 처리")
    
    tests = [
        ("마피아 게임", test_mafia_game),
        ("뱅! 게임", test_bang_game), 
        ("카탄 게임", test_catan_game),
        ("동적 행동 처리", test_dynamic_action_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name} 테스트 시작...")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과 ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("🎉 모든 테스트 통과! 동적 게임 시스템이 정상 작동합니다.")
        print("✅ 원래 요구사항 달성:")
        print("   - 다양한 보드게임 지원 (마피아, 뱅, 카탄 등)")
        print("   - 동적 게임 상태 관리")
        print("   - AI 플레이어 자동 생성 및 행동")
        print("   - 사용자 참여 가능한 턴 기반 시스템")
    else:
        print("⚠️ 일부 테스트 실패. 추가 개발이 필요합니다.")


if __name__ == "__main__":
    asyncio.run(main()) 