"""
Table Game Mate 통합 테스트

모든 테스트를 하나의 파일로 통합
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from agents import GameAgent, AnalysisAgent, MonitoringAgent
from core import GameConfig, Player, SystemState, ErrorHandler, ErrorSeverity, ErrorCategory, GameState, GameStatus, MonitoringState


class TestGameAgent:
    """게임 에이전트 테스트 클래스"""
    
    @pytest.fixture
    def game_agent(self):
        """게임 에이전트 픽스처"""
        return GameAgent()
    
    @pytest.fixture
    def sample_game_config(self):
        """샘플 게임 설정"""
        return {
            "name": "테스트 체스",
            "type": "chess",
            "min_players": 2,
            "max_players": 2,
            "estimated_duration": 30
        }
    
    @pytest.fixture
    def sample_players(self):
        """샘플 플레이어"""
        return [
            {"id": "player_1", "name": "Alice", "type": "human"},
            {"id": "player_2", "name": "Bob", "type": "ai"}
        ]
    
    def test_agent_initialization(self, game_agent):
        """에이전트 초기화 테스트"""
        assert game_agent.agent_id == "game_agent"
        assert game_agent.error_handler is not None
        assert game_agent.graph is not None
    
    @pytest.mark.asyncio
    async def test_play_game_success(self, game_agent, sample_game_config, sample_players):
        """게임 실행 성공 테스트"""
        # 게임 실행
        result = await game_agent.play_game(sample_game_config, sample_players)
        
        # 결과 검증
        assert result["success"] is True
        assert "final_state" in result
        assert result["agent_id"] == "game_agent"
    
    @pytest.mark.asyncio
    async def test_play_game_failure(self, game_agent):
        """게임 실행 실패 테스트"""
        # 잘못된 데이터로 게임 실행
        result = await game_agent.play_game({}, [])
        
        # 결과 검증
        assert result["success"] is False
        assert "error" in result
        assert result["agent_id"] == "game_agent"
    
    def test_should_continue_logic(self, game_agent):
        """게임 계속 여부 결정 로직 테스트"""
        from core import GameState, GameStatus
        
        # 정상 상태 - 계속
        from core import GamePhase
        state = GameState(
            game_config=None,
            players=[],
            game_data={},
            messages=[],
            current_phase=GamePhase.PLAYING,
            game_status=GameStatus.ACTIVE,
            current_round=1,
            max_rounds=10
        )
        assert game_agent._should_continue(state) == "continue"
        
        # 에러 상태 - 종료
        state.error = "테스트 에러"
        assert game_agent._should_continue(state) == "end"
        
        # 게임 완료 - 종료
        state.error = None
        state.game_status = GameStatus.COMPLETED
        assert game_agent._should_continue(state) == "end"
        
        # 최대 라운드 초과 - 종료
        state.game_status = GameStatus.ACTIVE
        state.current_round = 15
        assert game_agent._should_continue(state) == "end"


class TestAnalysisAgent:
    """분석 에이전트 테스트 클래스"""
    
    @pytest.fixture
    def analysis_agent(self):
        """분석 에이전트 픽스처"""
        return AnalysisAgent()
    
    @pytest.fixture
    def sample_game_data(self):
        """샘플 게임 데이터"""
        return {
            "moves": [
                {"type": "move", "player": "Alice", "duration": 5, "strategic": True, "strategy_type": "aggressive"},
                {"type": "move", "player": "Bob", "duration": 3, "strategic": False, "strategy_type": "defensive"},
                {"type": "move", "player": "Alice", "duration": 7, "strategic": True, "strategy_type": "aggressive"},
                {"type": "move", "player": "Bob", "duration": 4, "strategic": True, "strategy_type": "defensive"}
            ],
            "players": ["Alice", "Bob"],
            "duration": 120,
            "rounds": 4
        }
    
    def test_agent_initialization(self, analysis_agent):
        """에이전트 초기화 테스트"""
        assert analysis_agent.agent_id == "analysis_agent"
        assert analysis_agent.error_handler is not None
        assert analysis_agent.graph is not None
    
    @pytest.mark.asyncio
    async def test_analyze_game_success(self, analysis_agent, sample_game_data):
        """게임 분석 성공 테스트"""
        # 분석 실행
        result = await analysis_agent.analyze_game(sample_game_data)
        
        # 결과 검증
        assert result["success"] is True
        assert "analysis_result" in result
        assert result["agent_id"] == "analysis_agent"
    
    @pytest.mark.asyncio
    async def test_analyze_game_failure(self, analysis_agent):
        """게임 분석 실패 테스트"""
        # 잘못된 데이터로 분석 실행
        result = await analysis_agent.analyze_game({})
        
        # 결과 검증
        assert result["success"] is False
        assert "error" in result
        assert result["agent_id"] == "analysis_agent"
    
    def test_analyze_move_frequency(self, analysis_agent, sample_game_data):
        """움직임 빈도 분석 테스트"""
        moves = sample_game_data["moves"]
        result = analysis_agent._analyze_move_frequency(moves)
        
        # 결과 검증
        assert "total_moves" in result
        assert "unique_move_types" in result
        assert "frequency_distribution" in result
        assert result["total_moves"] == 4
        assert result["unique_move_types"] == 1  # 모두 "move" 타입
    
    def test_analyze_player_behavior(self, analysis_agent, sample_game_data):
        """플레이어 행동 분석 테스트"""
        moves = sample_game_data["moves"]
        result = analysis_agent._analyze_player_behavior(moves)
        
        # 결과 검증
        assert "Alice" in result
        assert "Bob" in result
        assert result["Alice"]["move_count"] == 2
        assert result["Bob"]["move_count"] == 2
    
    def test_calculate_efficiency(self, analysis_agent):
        """효율성 계산 테스트"""
        data = {"total_moves": 10, "game_duration": 5}
        result = analysis_agent._calculate_efficiency(data)
        
        # 결과 검증
        assert result == 2.0  # 10 moves / 5 minutes
    
    def test_calculate_engagement(self, analysis_agent):
        """참여도 계산 테스트"""
        patterns = {
            "player_behavior": {
                "Alice": {"move_count": 5},
                "Bob": {"move_count": 3},
                "Charlie": {"move_count": 0}
            }
        }
        result = analysis_agent._calculate_engagement(patterns)
        
        # 결과 검증
        assert result == 2/3  # 2명이 활성, 3명 전체


class TestMonitoringAgent:
    """모니터링 에이전트 테스트 클래스"""
    
    @pytest.fixture
    def monitoring_agent(self):
        """모니터링 에이전트 픽스처"""
        return MonitoringAgent()
    
    def test_agent_initialization(self, monitoring_agent):
        """에이전트 초기화 테스트"""
        assert monitoring_agent.agent_id == "monitoring_agent"
        assert monitoring_agent.error_handler is not None
        assert monitoring_agent.graph is not None
        assert monitoring_agent.metrics_history == []
    
    @pytest.mark.asyncio
    async def test_monitor_system_success(self, monitoring_agent):
        """시스템 모니터링 성공 테스트"""
        # psutil 모킹
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.pids', return_value=[1, 2, 3, 4, 5]):
            
            # 메모리 모킹
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.available = 1000000000
            
            # 디스크 모킹
            mock_disk.return_value.percent = 70.0
            mock_disk.return_value.free = 5000000000
            
            # 네트워크 모킹
            mock_network.return_value.bytes_sent = 1000
            mock_network.return_value.bytes_recv = 2000
            
            # 모니터링 실행
            result = await monitoring_agent.monitor_system()
            
            # 결과 검증
            assert result["success"] is True
            assert "monitoring_result" in result
            assert result["agent_id"] == "monitoring_agent"
    
    @pytest.mark.asyncio
    async def test_monitor_system_failure(self, monitoring_agent):
        """시스템 모니터링 실패 테스트"""
        # psutil 모킹 - 예외 발생
        with patch('psutil.cpu_percent', side_effect=Exception("CPU 모니터링 실패")):
            # 모니터링 실행
            result = await monitoring_agent.monitor_system()
            
            # 결과 검증
            assert result["success"] is False
            assert "error" in result
            assert result["agent_id"] == "monitoring_agent"
    
    def test_should_alert_logic(self, monitoring_agent):
        """알림 필요 여부 결정 로직 테스트"""
        from core import MonitoringState
        
        # 위반 있음 - 알림 필요
        state = MonitoringState(
            threshold_violations=[{"type": "cpu", "level": "warning"}],
            messages=[]
        )
        assert monitoring_agent._should_alert(state) == "alert"
        
        # 위반 없음 - 정상
        state.threshold_violations = []
        assert monitoring_agent._should_alert(state) == "normal"
    
    def test_calculate_performance_score(self, monitoring_agent):
        """성능 점수 계산 테스트"""
        current = {"cpu_percent": 50.0, "memory_percent": 60.0, "disk_percent": 70.0}
        avg_cpu = 45.0
        avg_memory = 55.0
        avg_disk = 65.0
        
        # 성능 점수 계산
        score = monitoring_agent._calculate_performance_score(current, avg_cpu, avg_memory, avg_disk)
        
        # 결과 검증
        assert 0 <= score <= 100


class TestCoreSystem:
    """코어 시스템 테스트 클래스"""
    
    def test_error_handler_initialization(self):
        """에러 핸들러 초기화 테스트"""
        error_handler = ErrorHandler()
        assert error_handler.error_records == []
        assert error_handler.error_counts == {}
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """에러 처리 테스트"""
        error_handler = ErrorHandler()
        
        # 에러 처리
        error_id = await error_handler.handle_error(
            Exception("테스트 에러"),
            "test_agent",
            ErrorSeverity.MEDIUM,
            ErrorCategory.SYSTEM_ERROR
        )
        
        # 결과 검증
        assert error_id.startswith("err_")
        assert len(error_handler.error_records) == 1
        assert error_handler.error_records[0].error_message == "테스트 에러"
    
    def test_error_summary(self):
        """에러 요약 테스트"""
        error_handler = ErrorHandler()
        
        # 에러 요약
        summary = error_handler.get_error_summary()
        
        # 결과 검증
        assert "total_errors" in summary
        assert "unresolved_errors" in summary
        assert "severity_distribution" in summary
        assert "category_distribution" in summary
        assert "agent_distribution" in summary
    
    def test_game_config_validation(self):
        """게임 설정 검증 테스트"""
        # 유효한 게임 설정
        config = GameConfig(
            name="테스트 게임",
            type="chess",
            min_players=2,
            max_players=4
        )
        
        assert config.name == "테스트 게임"
        assert config.type == "chess"
        assert config.min_players == 2
        assert config.max_players == 4
    
    def test_player_validation(self):
        """플레이어 검증 테스트"""
        # 유효한 플레이어
        player = Player(
            id="player_1",
            name="Alice",
            type="human"
        )
        
        assert player.id == "player_1"
        assert player.name == "Alice"
        assert player.type == "human"
        assert player.score == 0
        assert player.is_active == True


if __name__ == "__main__":
    pytest.main([__file__])
