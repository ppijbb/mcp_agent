"""
테이블게임 메이트 시스템 테스트 실행기
기본 동작과 워크플로우 검증
"""

import asyncio
import sys
import os
from typing import Dict, Any

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from table_game_mate.core.game_master import GameMasterGraph
from table_game_mate.models.game_state import GameType

class TableGameMateTestRunner:
    """테스트 실행기"""
    
    def __init__(self):
        self.game_master = GameMasterGraph()
    
    async def run_basic_test(self):
        """기본 테스트 실행"""
        
        print("=" * 50)
        print("🎮 테이블게임 메이트 시스템 테스트 시작")
        print("=" * 50)
        
        # 테스트 설정
        test_config = {
            "target_game_name": "틱택토",
            "desired_player_count": 3,
            "difficulty_level": "medium",
            "ai_creativity": 0.7,
            "ai_aggression": 0.5,
            "enable_persona_chat": True,
            "auto_progress": True,
            "turn_timeout_seconds": 30,
            "enable_hints": False,
            "verbose_logging": True,
            "save_game_history": True
        }
        
        try:
            print(f"📋 테스트 구성:")
            for key, value in test_config.items():
                print(f"  {key}: {value}")
            print()
            
            # 게임 실행
            result = await self.game_master.run_game(test_config)
            
            print("\n✅ 테스트 완료!")
            print(f"게임 ID: {result.get('game_id', '알 수 없음')}")
            print(f"최종 단계: {result.get('phase', '알 수 없음')}")
            print(f"총 턴 수: {result.get('turn_count', 0)}")
            print(f"참여 플레이어: {len(result.get('players', []))}")
            
            if result.get('game_ended'):
                print(f"🏆 게임 결과:")
                print(f"  승자: {result.get('winner_ids', [])}")
                print(f"  최종 점수: {result.get('final_scores', {})}")
            
            return True
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_persona_test(self):
        """페르소나 생성 테스트"""
        
        print("\n" + "=" * 50)
        print("🎭 페르소나 시스템 테스트")
        print("=" * 50)
        
        try:
            from table_game_mate.models.persona import PersonaGenerator, PersonaArchetype
            
            # 다양한 게임 타입별 페르소나 생성 테스트
            test_games = [
                ("카탄", "strategy", 4),
                ("마피아", "social", 6), 
                ("UNO", "card", 3),
                ("체커", "board", 2)
            ]
            
            for game_name, game_type, player_count in test_games:
                print(f"\n🎲 {game_name} ({game_type}) - {player_count}명:")
                
                personas = PersonaGenerator.generate_for_game(
                    game_name=game_name,
                    game_type=game_type,
                    count=player_count,
                    difficulty="medium"
                )
                
                for i, persona in enumerate(personas, 1):
                    print(f"  플레이어 {i}: {persona['name']}")
                    print(f"    원형: {persona['archetype'].value}")
                    print(f"    소통 스타일: {persona['communication_style'].value}")
                    print(f"    배경: {persona['background_story']}")
                    
                    # 특성 중 일부만 출력
                    traits = persona['traits']
                    print(f"    특성: 공격성={traits.aggression:.1f}, 논리성={traits.logic:.1f}, 협력성={traits.cooperation:.1f}")
                    print()
            
            print("✅ 페르소나 테스트 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 페르소나 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_state_management_test(self):
        """상태 관리 테스트"""
        
        print("\n" + "=" * 50)
        print("💾 상태 관리 시스템 테스트")
        print("=" * 50)
        
        try:
            from table_game_mate.models.game_state import (
                GameState, PlayerInfo, GameAction, GameMetadata, GameType, GamePhase
            )
            from datetime import datetime
            import uuid
            
            # 샘플 게임 상태 생성
            game_id = str(uuid.uuid4())
            now = datetime.now()
            
            # 플레이어 생성
            players = [
                PlayerInfo(
                    id="user",
                    name="사용자",
                    is_ai=False,
                    turn_order=0
                ),
                PlayerInfo(
                    id="ai_1",
                    name="AI플레이어1",
                    is_ai=True,
                    persona_type="aggressive",
                    turn_order=1
                ),
                PlayerInfo(
                    id="ai_2", 
                    name="AI플레이어2",
                    is_ai=True,
                    persona_type="analytical",
                    turn_order=2
                )
            ]
            
            # 게임 메타데이터
            metadata = GameMetadata(
                name="테스트 게임",
                min_players=2,
                max_players=4,
                estimated_duration=30,
                complexity=2.5,
                game_type=GameType.STRATEGY,
                description="상태 관리 테스트용 게임"
            )
            
            # 게임 액션
            action = GameAction(
                player_id="user",
                action_type="test_move",
                action_data={"position": "A1", "value": "X"}
            )
            action.is_valid = True
            
            print("📊 생성된 테스트 데이터:")
            print(f"  게임 ID: {game_id}")
            print(f"  플레이어 수: {len(players)}")
            print(f"  게임 메타데이터: {metadata.name} ({metadata.game_type.value})")
            print(f"  샘플 액션: {action.action_type} by {action.player_id}")
            
            # 상태 딕셔너리 생성 (TypedDict 스타일)
            game_state = {
                "game_id": game_id,
                "game_metadata": metadata,
                "phase": GamePhase.GAME_START,
                "players": players,
                "current_player_index": 0,
                "turn_count": 1,
                "game_board": {},
                "game_history": [action],
                "parsed_rules": {"test_rule": "test_value"},
                "game_config": {"test_config": True},
                "last_action": action,
                "pending_actions": [],
                "error_messages": [],
                "winner_ids": [],
                "final_scores": {},
                "game_ended": False,
                "created_at": now,
                "updated_at": now
            }
            
            print(f"  현재 단계: {game_state['phase'].value}")
            print(f"  현재 플레이어: {game_state['players'][game_state['current_player_index']].name}")
            print(f"  게임 히스토리: {len(game_state['game_history'])}개 액션")
            
            print("✅ 상태 관리 테스트 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 상태 관리 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        
        print("🧪 전체 테스트 스위트 실행")
        print("=" * 60)
        
        tests = [
            ("페르소나 시스템", self.run_persona_test),
            ("상태 관리", self.run_state_management_test),
            ("기본 워크플로우", self.run_basic_test),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔄 {test_name} 테스트 시작...")
            result = await test_func()
            results.append((test_name, result))
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📋 테스트 결과 요약")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✅ 통과" if result else "❌ 실패"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")
        
        if failed == 0:
            print("🎉 모든 테스트 통과!")
        else:
            print("⚠️  일부 테스트 실패")
        
        return failed == 0

async def main():
    """메인 실행 함수"""
    
    runner = TableGameMateTestRunner()
    
    print("테이블게임 메이트 시스템을 테스트합니다...")
    print("시스템 준비 중...\n")
    
    success = await runner.run_all_tests()
    
    if success:
        print("\n🚀 시스템이 정상적으로 작동합니다!")
        print("이제 실제 게임을 시작할 수 있습니다.")
    else:
        print("\n🔧 시스템에 문제가 있습니다. 수정이 필요합니다.")
    
    return success

if __name__ == "__main__":
    # 이벤트 루프 실행
    success = asyncio.run(main()) 