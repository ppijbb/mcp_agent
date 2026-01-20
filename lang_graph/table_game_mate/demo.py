"""
Table Game Mate Demo

LLM 기반 실시간 보드게임 플랫폼 시연
"""

import asyncio
import sys
import os

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    BGGRuleParser,
    LLMGameAgent,
    GameStateManager,
    DynamicGameTable,
    create_game_table,
    ChessGameEngine
)


async def demo_bgg_rule_parser():
    """BGG 규칙 파서 시연"""
    print("\n" + "="*60)
    print("🎯 BGG Rule Parser Demo")
    print("="*60)
    
    parser = BGGRuleParser()
    
    # 체스 게임 규칙 가져오기 (BGG ID: 171)
    print("\n📥 체스 게임 규칙 가져오는 중...")
    rules = await parser.fetch_game_rules(171)
    
    if rules:
        print(f"\n✅ 게임 정보:")
        print(f"  - 이름: {rules.name}")
        print(f"  - 복잡도: {rules.complexity:.2f}")
        print(f"  - 플레이어: {rules.player_count.get('min', '?')}-{rules.player_count.get('max', '?')}")
        print(f"  - 예상 시간: {rules.playing_time}분")
        print(f"  - 카테고리: {', '.join(rules.categories[:3])}")
        print(f"  - 메커닉: {', '.join(rules.mechanics[:3])}")
        
        print(f"\n📋 설정 정보:")
        print(f"  {rules.setup.board_config}")
        
        print(f"\n🎯 승리 조건:")
        for win in rules.win_conditions:
            print(f"  - {win.condition_type}: {win.description}")
        
        # LLM용 프롬프트 생성
        llm_prompt = rules.to_llm_prompt()
        print(f"\n📝 LLM 프롬프트 길이: {len(llm_prompt)} 문자")
        print("  (규칙이 LLM이 이해할 수 있도록 구조화됨)")
    else:
        print("❌ 규칙 가져오기 실패")
    
    return rules


async def demo_game_state_manager():
    """게임 상태 관리자 시연"""
    print("\n" + "="*60)
    print("📊 Game State Manager Demo")
    print("="*60)
    
    manager = GameStateManager()
    
    # 테이블 생성
    print("\n🆕 게임 테이블 생성 중...")
    table = await manager.create_table(
        game_type="Chess",
        bgg_id=171,
        max_players=4,
        min_players=2
    )
    
    print(f"✅ 테이블 생성: {table.table_id}")
    print(f"  - 게임: {table.game_type}")
    print(f"  - 상태: {table.status.value}")
    
    # 플레이어 추가
    print("\n👤 플레이어 추가 중...")
    
    await manager.join_table(
        table_id=table.table_id,
        player_id="player_1",
        player_name="Alice",
        is_human=True
    )
    
    await manager.join_table(
        table_id=table.table_id,
        player_id="player_2",
        player_name="Gemini_Bot",
        is_human=False,
        llm_model="gemini-2.5-flash-lite"
    )
    
    await manager.join_table(
        table_id=table.table_id,
        player_id="player_3",
        player_name="GPT_Bot",
        is_human=False,
        llm_model="gpt-4o"
    )
    
    table = manager.get_table(table.table_id)
    print(f"  - 플레이어 수: {len(table.players)}")
    for pid, player in table.players.items():
        print(f"    - {player.name} ({'인간' if player.is_human else 'LLM'})")
    
    # 게임 시작
    print("\n🎮 게임 시작 중...")
    success = await manager.start_game(table.table_id)
    
    if success:
        print("✅ 게임 시작됨!")
        table = manager.get_table(table.table_id)
        print(f"  - 현재 턴: {table.current_turn}")
        print(f"  - 현재 플레이어: {table.current_player_id}")
    
    return manager, table


async def demo_game_engine():
    """게임 엔진 시연"""
    print("\n" + "="*60)
    print("♟️ Chess Game Engine Demo")
    print("="*60)
    
    engine = ChessGameEngine()
    
    # 초기화
    print("\n🔧 체스 보드 초기화 중...")
    board_state = await engine.initialize(
        rules=None,
        players=[]
    )
    
    print("✅ 체스 보드 생성 완료!")
    print(f"  - 보드 크기: 8x8")
    print(f"  - 현재 플레이어: {board_state['current_player']}")
    
    # 합법적인 움직임
    legal_moves = await engine.get_legal_moves("player_1", board_state)
    print(f"\n📋 합법적인 움직임: {legal_moves}")
    
    # 움직임 적용
    print("\n🎯 움직임 적용 중...")
    move_data = {"from": "e2", "to": "e4"}
    result = await engine.apply_move("player_1", "MOVE_PIECE", move_data)
    
    print(f"✅ 움직임 적용됨:")
    print(f"  - From: {result['from_pos']}")
    print(f"  - To: {result['to_pos']}")
    print(f"  - 현재 플레이어: {result['new_board_state']['current_player']}")
    
    return engine, board_state


async def demo_llm_agent():
    """LLM 에이전트 시연"""
    print("\n" + "="*60)
    print("🤖 LLM Game Agent Demo")
    print("="*60)
    
    # LLM 에이전트 생성
    print("\n👤 LLM 에이전트 생성 중...")
    
    agent = LLMGameAgent.create_llm_agent(
        agent_id="test_agent",
        provider="google",
        model="gemini-2.5-flash-lite"
    )
    
    print(f"✅ 에이전트 생성: {agent.agent_id}")
    print(f"  - 유형: {agent.player_type.value}")
    print(f"  - 모델: {agent.llm_model}")
    
    # 통계 확인
    stats = agent.get_stats()
    print(f"\n📊 에이전트 통계:")
    print(f"  - 총 움직임: {stats['total_moves']}")
    print(f"  - 승리: {stats['wins']}")
    print(f"  - 패배: {stats['losses']}")
    print(f"  - 무승부: {stats['draws']}")
    
    return agent


async def demo_realtime_table():
    """실시간 게임 테이블 시연"""
    print("\n" + "="*60)
    print("🌐 Real-time Game Table Demo")
    print("="*60)
    
    # 테이블 생성
    print("\n🆕 실시간 게임 테이블 생성 중...")
    table = create_game_table(
        game_type="Chess",
        bgg_id=171
    )
    
    print(f"✅ 테이블 생성: {table.table_id}")
    
    # 초기화
    await table.initialize()
    print("✅ 테이블 초기화 완료")
    
    # 플레이어 추가
    print("\n👤 플레이어 추가 중...")
    
    await table.add_player(
        player_id="human_1",
        player_name="Human_Alice",
        is_human=True
    )
    
    await table.add_player(
        player_id="llm_gemini",
        player_name="Gemini_AI",
        is_human=False,
        llm_model="gemini-2.5-flash-lite"
    )
    
    await table.add_player(
        player_id="llm_claude",
        player_name="Claude_AI",
        is_human=False,
        llm_model="claude-3-5-sonnet"
    )
    
    # 테이블 상태 확인
    status = table.get_table_status()
    print(f"\n📊 테이블 상태:")
    print(f"  - 게임: {status['game_type']}")
    print(f"  - 플레이어:")
    for p in status['players']:
        print(f"    - {p['name']} ({'인간' if p['is_human'] else 'LLM'})")
    
    # 게임 시작
    print("\n🎮 게임 시작...")
    await table.start_game()
    
    status = table.get_table_status()
    print(f"✅ 게임 시작됨!")
    print(f"  - 상태: {status['status']}")
    print(f"  - 현재 턴: {status['current_turn']}")
    print(f"  - 현재 플레이어: {status['current_player']}")
    
    return table


async def main():
    """메인 데모 함수"""
    print("\n" + "="*60)
    print("🎮 Table Game Mate - LLM Gaming Platform Demo")
    print("="*60)
    print("\n실시간 멀티플레이어 LLM 보드게임 플랫폼을 시연합니다.")
    
    try:
        # 1. BGG 규칙 파서
        rules = await demo_bgg_rule_parser()
        
        # 2. 게임 엔진
        engine, board = await demo_game_engine()
        
        # 3. LLM 에이전트
        agent = await demo_llm_agent()
        
        # 4. 게임 상태 관리자
        manager, table = await demo_game_state_manager()
        
        # 5. 실시간 게임 테이블
        realtime_table = await demo_realtime_table()
        
        # 완료
        print("\n" + "="*60)
        print("✅ Demo Complete!")
        print("="*60)
        print("\n📁 생성된 파일:")
        print("  - core/bgg_rule_parser.py - BGG 게임 규칙 파서")
        print("  - core/llm_game_agent.py - LLM 게임 에이전트")
        print("  - core/game_state_manager.py - 게임 상태 관리")
        print("  - core/dynamic_game_table.py - 실시간 게임 테이블")
        print("  - realtime_dashboard.py - Streamlit 웹 대시보드")
        
        print("\n🚀 실행 방법:")
        print("  1. 웹 대시보드 실행:")
        print("     streamlit run realtime_dashboard.py")
        print("  2. API 서버 실행:")
        print("     python main.py")
        
    except Exception as e:
        print(f"\n❌ Demo 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
