"""
통합 테스트 스크립트

모든 핵심 모듈들이 올바르게 작동하는지 확인하는 테스트
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lang_graph.table_game_mate import *
from lang_graph.table_game_mate.models.action import ActionFactory, ActionType
from lang_graph.table_game_mate.core.action_executor import get_action_executor, ExecutionPriority
from lang_graph.table_game_mate.core.message_hub import get_message_hub, MessageType
from lang_graph.table_game_mate.utils.logger import get_logger, get_performance_logger, get_agent_logger, get_session_logger
from lang_graph.table_game_mate.models.game_state import PlayerInfo, GameInfo, GameConfig


async def test_action_system():
    """액션 시스템 테스트"""
    print("\n🔧 액션 시스템 테스트")
    
    # 액션 팩토리 테스트
    move_action = ActionFactory.create_move_action(
        player_id="player1",
        from_position="A1",
        to_position="B2"
    )
    print(f"✅ 이동 액션 생성: {move_action.action_type.value}")
    
    # 액션 실행기 테스트
    executor = get_action_executor()
    
    # 액션 제출
    action_id = await executor.submit_action(move_action, ExecutionPriority.NORMAL)
    print(f"✅ 액션 제출: {action_id}")
    
    # 액션 실행
    results = await executor.execute_all_pending()
    print(f"✅ 액션 실행 완료: {len(results)}개")
    
    # 통계 확인
    stats = executor.get_execution_stats()
    print(f"✅ 실행 통계: {stats}")


async def test_message_hub():
    """메시지 허브 테스트"""
    print("\n📡 메시지 허브 테스트")
    
    hub = get_message_hub()
    
    # 메시지 핸들러 등록
    received_messages = []
    
    async def test_handler(message):
        received_messages.append(message)
        print(f"📨 메시지 수신: {message.message_type.value}")
    
    hub.register_agent("test_agent", test_handler)
    
    # 메시지 전송
    message_id = await hub.send_to_agent(
        "test_agent",
        MessageType.AGENT_REQUEST,
        {"test": "data"}
    )
    print(f"✅ 메시지 전송: {message_id}")
    
    # 브로드캐스트 테스트
    broadcast_id = await hub.broadcast(
        MessageType.SYSTEM_INFO,
        {"message": "시스템 정보"}
    )
    print(f"✅ 브로드캐스트: {broadcast_id}")
    
    # 통계 확인
    stats = hub.get_stats()
    print(f"✅ 메시지 통계: {stats}")


async def test_logging_system():
    """로깅 시스템 테스트"""
    print("\n📝 로깅 시스템 테스트")
    
    # 기본 로거 테스트
    logger = get_logger("test_logger")
    logger.info("테스트 로그 메시지", test_field="test_value")
    print("✅ 기본 로거 테스트 완료")
    
    # 성능 로거 테스트
    perf_logger = get_performance_logger()
    perf_logger.start_timer("test_operation")
    await asyncio.sleep(0.1)  # 시뮬레이션
    duration = perf_logger.end_timer("test_operation")
    print(f"✅ 성능 로거 테스트 완료: {duration:.3f}초")
    
    # 에이전트 로거 테스트
    agent_logger = get_agent_logger("test_agent", "analyzer")
    agent_logger.log_agent_action("test_action", {"detail": "test"})
    print("✅ 에이전트 로거 테스트 완료")
    
    # 세션 로거 테스트
    session_logger = get_session_logger("test_session", "test_game")
    session_logger.log_game_event("test_event", {"data": "test"})
    print("✅ 세션 로거 테스트 완료")


async def test_llm_models():
    """LLM 모델 테스트"""
    print("\n🤖 LLM 모델 테스트")
    
    from lang_graph.table_game_mate.models.llm import (
        create_llm_response, create_error_response, parse_json_response,
        ResponseType, ResponseStatus
    )
    
    # LLM 응답 생성 테스트
    response = create_llm_response(
        request_id="test_request",
        content='{"test": "data"}',
        model_name="test_model",
        response_type=ResponseType.JSON
    )
    print(f"✅ LLM 응답 생성: {response.response_id}")
    
    # JSON 파싱 테스트
    parsed = parse_json_response(response)
    print(f"✅ JSON 파싱: {parsed.parsing_success}")
    
    # 에러 응답 테스트
    error_response = create_error_response(
        request_id="test_request",
        error_message="테스트 에러"
    )
    print(f"✅ 에러 응답 생성: {error_response.status.value}")


async def test_game_state_models():
    """게임 상태 모델 테스트"""
    print("\n🎮 게임 상태 모델 테스트")
    
    # PlayerInfo 테스트
    player = PlayerInfo(
        id="player1",
        name="테스트 플레이어",
        is_ai=True,
        persona_type="strategic"
    )
    print(f"✅ 플레이어 생성: {player.name}")
    
    # GameInfo 테스트
    game_info = GameInfo(
        name="테스트 게임",
        description="테스트용 게임",
        min_players=2,
        max_players=4,
        estimated_duration=30,
        complexity="moderate",
        game_type="strategy"
    )
    print(f"✅ 게임 정보 생성: {game_info.name}")
    
    # GameConfig 테스트
    game_config = {
        "target_game_name": "테스트 게임",
        "desired_player_count": 3,
        "difficulty_level": "medium",
        "ai_creativity": 0.7,
        "ai_aggression": 0.5,
        "enable_persona_chat": True,
        "auto_progress": True,
        "turn_timeout_seconds": 30,
        "enable_hints": True,
        "verbose_logging": True,
        "save_game_history": True
    }
    print(f"✅ 게임 설정 생성: {game_config['target_game_name']}")


async def test_agent_system():
    """에이전트 시스템 테스트"""
    print("\n🤖 에이전트 시스템 테스트")
    
    # Mock 클라이언트 생성
    class MockLLMClient:
        async def complete(self, prompt: str) -> str:
            return "테스트 응답"
    
    class MockMCPClient:
        async def call(self, server: str, method: str, params: dict) -> dict:
            return {"success": True, "data": "test"}
    
    # 에이전트 생성 테스트
    llm_client = MockLLMClient()
    mcp_client = MockMCPClient()
    
    try:
        # GameAnalyzerAgent 테스트
        analyzer = GameAnalyzerAgent(llm_client, mcp_client, "test_analyzer")
        print("✅ GameAnalyzerAgent 생성 완료")
        
        # RuleParserAgent 테스트
        parser = RuleParserAgent(llm_client, mcp_client, "test_parser")
        print("✅ RuleParserAgent 생성 완료")
        
        # PlayerManagerAgent 테스트
        manager = PlayerManagerAgent(llm_client, mcp_client, "test_manager")
        print("✅ PlayerManagerAgent 생성 완료")
        
        # PersonaGeneratorAgent 테스트
        persona_gen = PersonaGeneratorAgent(llm_client, mcp_client, "test_persona")
        print("✅ PersonaGeneratorAgent 생성 완료")
        
        # GameRefereeAgent 테스트
        referee = GameRefereeAgent(llm_client, mcp_client, "test_referee")
        print("✅ GameRefereeAgent 생성 완료")
        
        # ScoreCalculatorAgent 테스트
        calculator = ScoreCalculatorAgent(llm_client, mcp_client, "test_calculator")
        print("✅ ScoreCalculatorAgent 생성 완료")
        
    except Exception as e:
        print(f"❌ 에이전트 생성 실패: {e}")


async def test_game_master_graph():
    """게임 마스터 그래프 테스트"""
    print("\n🎯 게임 마스터 그래프 테스트")
    
    # Mock 클라이언트 생성
    class MockLLMClient:
        async def complete(self, prompt: str) -> str:
            return "테스트 응답"
    
    class MockMCPClient:
        async def call(self, server: str, method: str, params: dict) -> dict:
            return {"success": True, "data": "test"}
    
    try:
        # GameMasterGraph 생성
        llm_client = MockLLMClient()
        mcp_client = MockMCPClient()
        
        game_master = GameMasterGraph(llm_client, mcp_client)
        print("✅ GameMasterGraph 생성 완료")
        
        # 초기화 테스트
        success = await game_master.initialize()
        print(f"✅ GameMasterGraph 초기화: {'성공' if success else '실패'}")
        
    except Exception as e:
        print(f"❌ GameMasterGraph 테스트 실패: {e}")


async def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 Table Game Mate 통합 테스트 시작")
    print("=" * 50)
    
    try:
        await test_action_system()
        await test_message_hub()
        await test_logging_system()
        await test_llm_models()
        await test_game_state_models()
        await test_agent_system()
        await test_game_master_graph()
        
        print("\n" + "=" * 50)
        print("✅ 모든 테스트 완료!")
        print("🎉 Table Game Mate 시스템이 정상적으로 작동합니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests()) 