"""
Table Game Mate 통합 에이전트 시스템

LangGraph 패턴을 따르는 모든 에이전트를 하나의 파일로 통합
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from enum import Enum

from .core import GameState, AnalysisState, MonitoringState, GamePhase, GameStatus, ErrorHandler, ErrorSeverity, ErrorCategory


# ============================================================================
# 게임 에이전트
# ============================================================================

class GameAgent:
    """게임 플레이 전담 에이전트"""
    
    def __init__(self, agent_id: str = "game_agent"):
        self.agent_id = agent_id
        self.error_handler = ErrorHandler()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 상태 그래프 구성"""
        workflow = StateGraph(GameState)
        
        # 노드 추가
        workflow.add_node("initialize_game", self._initialize_game)
        workflow.add_node("setup_players", self._setup_players)
        workflow.add_node("start_game", self._start_game)
        workflow.add_node("play_round", self._play_round)
        workflow.add_node("check_game_end", self._check_game_end)
        workflow.add_node("end_game", self._end_game)
        
        # 엣지 추가
        workflow.set_entry_point("initialize_game")
        workflow.add_edge("initialize_game", "setup_players")
        workflow.add_edge("setup_players", "start_game")
        workflow.add_edge("start_game", "play_round")
        workflow.add_conditional_edges(
            "play_round",
            self._should_continue,
            {
                "continue": "play_round",
                "end": "check_game_end"
            }
        )
        workflow.add_edge("check_game_end", "end_game")
        workflow.add_edge("end_game", END)
        
        return workflow.compile()
    
    async def _initialize_game(self, state: GameState) -> GameState:
        """게임 초기화"""
        try:
            game_config = state.game_config
            if not game_config:
                raise ValueError("게임 설정이 없습니다")
            
            state.current_phase = GamePhase.PLAYERS_SETUP
            state.game_status = GameStatus.READY
            state.messages.append(f"게임 '{game_config.get('name', 'Unknown')}' 초기화 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    async def _setup_players(self, state: GameState) -> GameState:
        """플레이어 설정"""
        try:
            players = state.players
            if not players:
                raise ValueError("플레이어가 설정되지 않았습니다")
            
            state.current_phase = GamePhase.PLAYING
            state.messages.append(f"{len(players)}명의 플레이어 설정 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    async def _start_game(self, state: GameState) -> GameState:
        """게임 시작"""
        try:
            state.current_phase = GamePhase.PLAYING
            state.game_status = GameStatus.ACTIVE
            state.current_round = 1
            state.messages.append("게임 시작!")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    async def _play_round(self, state: GameState) -> GameState:
        """라운드 플레이"""
        try:
            current_player = state.current_player
            if not current_player:
                # 첫 번째 플레이어로 설정
                state.current_player = state.players[0] if state.players else None
            
            # 간단한 게임 로직 시뮬레이션
            state.game_data[f"round_{state.current_round}"] = {
                "player": state.current_player["name"] if state.current_player else "Unknown",
                "action": "move",
                "timestamp": datetime.now().isoformat()
            }
            
            state.messages.append(f"라운드 {state.current_round} - {state.current_player['name'] if state.current_player else 'Unknown'}의 턴")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    async def _check_game_end(self, state: GameState) -> GameState:
        """게임 종료 조건 확인"""
        try:
            # 간단한 종료 조건: 3라운드 후 종료
            if state.current_round >= 3:
                winner = state.players[0] if state.players else None
                state.winner = winner
                state.game_status = GameStatus.COMPLETED
                state.messages.append(f"게임 종료! 승자: {winner['name'] if winner else 'Unknown'}")
            else:
                state.current_round += 1
                state.messages.append(f"라운드 {state.current_round} 시작")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    async def _end_game(self, state: GameState) -> GameState:
        """게임 종료 처리"""
        try:
            # 최종 점수 계산
            state.final_scores = {player["name"]: state.current_round * 10 for player in state.players}
            
            state.current_phase = GamePhase.COMPLETED
            state.messages.append("게임이 완전히 종료되었습니다")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.game_status = GameStatus.ERROR
            return state
    
    def _should_continue(self, state: GameState) -> str:
        """게임 계속 여부 결정"""
        if state.error:
            return "end"
        
        if state.game_status == GameStatus.COMPLETED:
            return "end"
        
        if state.current_round > state.max_rounds:
            return "end"
        
        return "continue"
    
    async def play_game(self, game_config: Dict[str, Any], players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """게임 실행"""
        try:
            initial_state = GameState(
                game_config=game_config,
                players=players,
                game_data={},
                messages=[],
                current_phase=GamePhase.INITIALIZING,
                game_status=GameStatus.PENDING
            )
            
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "final_state": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }


# ============================================================================
# 분석 에이전트
# ============================================================================

class AnalysisAgent:
    """게임 분석 전담 에이전트"""
    
    def __init__(self, agent_id: str = "analysis_agent"):
        self.agent_id = agent_id
        self.error_handler = ErrorHandler()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 상태 그래프 구성"""
        workflow = StateGraph(AnalysisState)
        
        # 노드 추가
        workflow.add_node("collect_data", self._collect_data)
        workflow.add_node("analyze_patterns", self._analyze_patterns)
        workflow.add_node("calculate_metrics", self._calculate_metrics)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("create_report", self._create_report)
        
        # 엣지 추가
        workflow.set_entry_point("collect_data")
        workflow.add_edge("collect_data", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "calculate_metrics")
        workflow.add_edge("calculate_metrics", "generate_insights")
        workflow.add_edge("generate_insights", "create_report")
        workflow.add_edge("create_report", END)
        
        return workflow.compile()
    
    async def _collect_data(self, state: AnalysisState) -> AnalysisState:
        """게임 데이터 수집"""
        try:
            game_data = state.game_data
            if not game_data:
                raise ValueError("분석할 게임 데이터가 없습니다")
            
            # 데이터 검증
            required_fields = ["moves", "players", "duration", "rounds"]
            missing_fields = [field for field in required_fields if field not in game_data]
            
            if missing_fields:
                raise ValueError(f"필수 데이터 필드 누락: {missing_fields}")
            
            # 데이터 전처리
            processed_data = {
                "total_moves": len(game_data.get("moves", [])),
                "player_count": len(game_data.get("players", [])),
                "game_duration": game_data.get("duration", 0),
                "total_rounds": game_data.get("rounds", 0),
                "raw_data": game_data
            }
            
            state.processed_data = processed_data
            state.current_step = "data_collected"
            state.messages.append("게임 데이터 수집 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _analyze_patterns(self, state: AnalysisState) -> AnalysisState:
        """게임 패턴 분석"""
        try:
            processed_data = state.processed_data
            if not processed_data:
                raise ValueError("처리된 데이터가 없습니다")
            
            raw_data = processed_data["raw_data"]
            moves = raw_data.get("moves", [])
            
            # 패턴 분석
            patterns = {
                "move_frequency": self._analyze_move_frequency(moves),
                "player_behavior": self._analyze_player_behavior(moves),
                "game_flow": self._analyze_game_flow(moves),
                "strategy_patterns": self._analyze_strategy_patterns(moves)
            }
            
            state.patterns = patterns
            state.current_step = "patterns_analyzed"
            state.messages.append("게임 패턴 분석 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _calculate_metrics(self, state: AnalysisState) -> AnalysisState:
        """게임 메트릭 계산"""
        try:
            processed_data = state.processed_data
            patterns = state.patterns
            
            if not processed_data or not patterns:
                raise ValueError("분석 데이터가 부족합니다")
            
            # 기본 메트릭
            metrics = {
                "game_efficiency": self._calculate_efficiency(processed_data),
                "player_engagement": self._calculate_engagement(patterns),
                "strategy_diversity": self._calculate_diversity(patterns),
                "game_balance": self._calculate_balance(patterns),
                "complexity_score": self._calculate_complexity(processed_data, patterns)
            }
            
            state.metrics = metrics
            state.current_step = "metrics_calculated"
            state.messages.append("게임 메트릭 계산 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _generate_insights(self, state: AnalysisState) -> AnalysisState:
        """인사이트 생성"""
        try:
            metrics = state.metrics
            patterns = state.patterns
            
            if not metrics or not patterns:
                raise ValueError("메트릭 또는 패턴 데이터가 없습니다")
            
            # 인사이트 생성
            insights = {
                "strengths": self._identify_strengths(metrics, patterns),
                "weaknesses": self._identify_weaknesses(metrics, patterns),
                "recommendations": self._generate_recommendations(metrics, patterns),
                "trends": self._identify_trends(patterns)
            }
            
            state.insights = insights
            state.current_step = "insights_generated"
            state.messages.append("인사이트 생성 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _create_report(self, state: AnalysisState) -> AnalysisState:
        """최종 보고서 생성"""
        try:
            # 보고서 구성
            report = {
                "summary": {
                    "total_moves": state.processed_data.get("total_moves", 0),
                    "player_count": state.processed_data.get("player_count", 0),
                    "game_duration": state.processed_data.get("game_duration", 0),
                    "analysis_timestamp": datetime.now().isoformat()
                },
                "metrics": state.metrics,
                "patterns": state.patterns,
                "insights": state.insights,
                "agent_id": self.agent_id
            }
            
            state.report = report
            state.current_step = "report_created"
            state.status = "completed"
            state.messages.append("분석 보고서 생성 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    def _analyze_move_frequency(self, moves: List[Dict]) -> Dict[str, Any]:
        """움직임 빈도 분석"""
        if not moves:
            return {"error": "움직임 데이터가 없습니다"}
        
        move_types = [move.get("type", "unknown") for move in moves]
        frequency = {}
        for move_type in move_types:
            frequency[move_type] = frequency.get(move_type, 0) + 1
        
        return {
            "total_moves": len(moves),
            "unique_move_types": len(set(move_types)),
            "frequency_distribution": frequency
        }
    
    def _analyze_player_behavior(self, moves: List[Dict]) -> Dict[str, Any]:
        """플레이어 행동 분석"""
        if not moves:
            return {"error": "움직임 데이터가 없습니다"}
        
        player_moves = {}
        for move in moves:
            player = move.get("player", "unknown")
            if player not in player_moves:
                player_moves[player] = []
            player_moves[player].append(move)
        
        behavior_analysis = {}
        for player, player_move_list in player_moves.items():
            behavior_analysis[player] = {
                "move_count": len(player_move_list),
                "avg_move_time": sum(m.get("duration", 0) for m in player_move_list) / len(player_move_list),
                "move_types": list(set(m.get("type", "unknown") for m in player_move_list))
            }
        
        return behavior_analysis
    
    def _analyze_game_flow(self, moves: List[Dict]) -> Dict[str, Any]:
        """게임 흐름 분석"""
        if not moves:
            return {"error": "움직임 데이터가 없습니다"}
        
        # 시간대별 움직임 분포
        time_periods = {"early": 0, "mid": 0, "late": 0}
        total_moves = len(moves)
        
        for i, move in enumerate(moves):
            progress = i / total_moves
            if progress < 0.33:
                time_periods["early"] += 1
            elif progress < 0.66:
                time_periods["mid"] += 1
            else:
                time_periods["late"] += 1
        
        return {
            "total_moves": total_moves,
            "time_distribution": time_periods,
            "flow_consistency": self._calculate_flow_consistency(moves)
        }
    
    def _analyze_strategy_patterns(self, moves: List[Dict]) -> Dict[str, Any]:
        """전략 패턴 분석"""
        if not moves:
            return {"error": "움직임 데이터가 없습니다"}
        
        # 전략적 움직임 식별
        strategic_moves = [move for move in moves if move.get("strategic", False)]
        
        return {
            "total_strategic_moves": len(strategic_moves),
            "strategy_ratio": len(strategic_moves) / len(moves),
            "strategy_types": list(set(move.get("strategy_type", "unknown") for move in strategic_moves))
        }
    
    def _calculate_efficiency(self, data: Dict) -> float:
        """게임 효율성 계산"""
        total_moves = data.get("total_moves", 1)
        duration = data.get("game_duration", 1)
        return total_moves / duration if duration > 0 else 0
    
    def _calculate_engagement(self, patterns: Dict) -> float:
        """플레이어 참여도 계산"""
        behavior = patterns.get("player_behavior", {})
        if not behavior:
            return 0.0
        
        total_players = len(behavior)
        active_players = sum(1 for player_data in behavior.values() if player_data.get("move_count", 0) > 0)
        return active_players / total_players if total_players > 0 else 0.0
    
    def _calculate_diversity(self, patterns: Dict) -> float:
        """전략 다양성 계산"""
        move_freq = patterns.get("move_frequency", {})
        if not move_freq:
            return 0.0
        
        unique_types = move_freq.get("unique_move_types", 0)
        total_moves = move_freq.get("total_moves", 1)
        return unique_types / total_moves if total_moves > 0 else 0.0
    
    def _calculate_balance(self, patterns: Dict) -> float:
        """게임 균형성 계산"""
        behavior = patterns.get("player_behavior", {})
        if not behavior:
            return 0.0
        
        move_counts = [player_data.get("move_count", 0) for player_data in behavior.values()]
        if not move_counts:
            return 0.0
        
        # 표준편차를 이용한 균형성 계산
        mean_moves = sum(move_counts) / len(move_counts)
        variance = sum((count - mean_moves) ** 2 for count in move_counts) / len(move_counts)
        std_dev = variance ** 0.5
        
        # 낮은 표준편차 = 높은 균형성
        return max(0, 1 - (std_dev / mean_moves)) if mean_moves > 0 else 0.0
    
    def _calculate_complexity(self, data: Dict, patterns: Dict) -> float:
        """게임 복잡도 계산"""
        efficiency = self._calculate_efficiency(data)
        diversity = self._calculate_diversity(patterns)
        return (efficiency + diversity) / 2
    
    def _calculate_flow_consistency(self, moves: List[Dict]) -> float:
        """게임 흐름 일관성 계산"""
        if len(moves) < 2:
            return 1.0
        
        # 연속된 움직임 간의 시간 간격 분석
        intervals = []
        for i in range(1, len(moves)):
            prev_time = moves[i-1].get("timestamp", 0)
            curr_time = moves[i].get("timestamp", 0)
            intervals.append(curr_time - prev_time)
        
        if not intervals:
            return 1.0
        
        # 간격의 일관성 계산
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((interval - mean_interval) ** 2 for interval in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        return max(0, 1 - (std_dev / mean_interval)) if mean_interval > 0 else 0.0
    
    def _identify_strengths(self, metrics: Dict, patterns: Dict) -> List[str]:
        """강점 식별"""
        strengths = []
        
        if metrics.get("player_engagement", 0) > 0.8:
            strengths.append("높은 플레이어 참여도")
        
        if metrics.get("strategy_diversity", 0) > 0.6:
            strengths.append("다양한 전략 패턴")
        
        if metrics.get("game_balance", 0) > 0.7:
            strengths.append("균형잡힌 게임플레이")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict, patterns: Dict) -> List[str]:
        """약점 식별"""
        weaknesses = []
        
        if metrics.get("player_engagement", 0) < 0.5:
            weaknesses.append("낮은 플레이어 참여도")
        
        if metrics.get("strategy_diversity", 0) < 0.3:
            weaknesses.append("제한적인 전략 다양성")
        
        if metrics.get("game_balance", 0) < 0.4:
            weaknesses.append("불균형한 게임플레이")
        
        return weaknesses
    
    def _generate_recommendations(self, metrics: Dict, patterns: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if metrics.get("player_engagement", 0) < 0.6:
            recommendations.append("플레이어 참여도를 높이기 위한 메커니즘 추가 고려")
        
        if metrics.get("strategy_diversity", 0) < 0.4:
            recommendations.append("더 다양한 전략 옵션 제공")
        
        if metrics.get("game_balance", 0) < 0.5:
            recommendations.append("게임 밸런스 조정 필요")
        
        return recommendations
    
    def _identify_trends(self, patterns: Dict) -> List[str]:
        """트렌드 식별"""
        trends = []
        
        game_flow = patterns.get("game_flow", {})
        time_dist = game_flow.get("time_distribution", {})
        
        if time_dist.get("late", 0) > time_dist.get("early", 0):
            trends.append("게임 후반부 활동 증가")
        
        if time_dist.get("early", 0) > time_dist.get("late", 0):
            trends.append("게임 초반부 집중")
        
        return trends
    
    async def analyze_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """게임 분석 실행"""
        try:
            initial_state = AnalysisState(
                game_data=game_data,
                status="pending",
                current_step="initializing",
                messages=[]
            )
            
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "analysis_result": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }


# ============================================================================
# 모니터링 에이전트
# ============================================================================

class MonitoringAgent:
    """시스템 모니터링 전담 에이전트"""
    
    def __init__(self, agent_id: str = "monitoring_agent"):
        self.agent_id = agent_id
        self.error_handler = ErrorHandler()
        self.metrics_history = []
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 상태 그래프 구성"""
        workflow = StateGraph(MonitoringState)
        
        # 노드 추가
        workflow.add_node("collect_metrics", self._collect_metrics)
        workflow.add_node("analyze_performance", self._analyze_performance)
        workflow.add_node("check_thresholds", self._check_thresholds)
        workflow.add_node("generate_alerts", self._generate_alerts)
        workflow.add_node("update_dashboard", self._update_dashboard)
        
        # 엣지 추가
        workflow.set_entry_point("collect_metrics")
        workflow.add_edge("collect_metrics", "analyze_performance")
        workflow.add_edge("analyze_performance", "check_thresholds")
        workflow.add_conditional_edges(
            "check_thresholds",
            self._should_alert,
            {
                "alert": "generate_alerts",
                "normal": "update_dashboard"
            }
        )
        workflow.add_edge("generate_alerts", "update_dashboard")
        workflow.add_edge("update_dashboard", END)
        
        return workflow.compile()
    
    async def _collect_metrics(self, state: MonitoringState) -> MonitoringState:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 네트워크 I/O
            network = psutil.net_io_counters()
            
            # 현재 시간
            timestamp = datetime.now()
            
            # 메트릭 데이터 구성
            metrics = {
                "timestamp": timestamp.isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available": memory.available,
                "disk_percent": disk_percent,
                "disk_free": disk.free,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "process_count": len(psutil.pids())
            }
            
            # 메트릭 히스토리에 추가
            self.metrics_history.append(metrics)
            
            # 최근 100개만 유지
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            state.current_metrics = metrics
            state.metrics_history = self.metrics_history.copy()
            state.current_step = "metrics_collected"
            state.messages.append(f"메트릭 수집 완료 - CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _analyze_performance(self, state: MonitoringState) -> MonitoringState:
        """성능 분석"""
        try:
            current_metrics = state.current_metrics
            if not current_metrics:
                raise ValueError("현재 메트릭이 없습니다")
            
            # 최근 10분간의 평균 계산
            recent_metrics = self._get_recent_metrics(minutes=10)
            
            if not recent_metrics:
                raise ValueError("최근 메트릭 데이터가 없습니다")
            
            # 평균값 계산
            avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m["disk_percent"] for m in recent_metrics) / len(recent_metrics)
            
            # 성능 트렌드 분석
            trends = self._analyze_trends(recent_metrics)
            
            # 성능 점수 계산
            performance_score = self._calculate_performance_score(current_metrics, avg_cpu, avg_memory, avg_disk)
            
            analysis = {
                "current_metrics": current_metrics,
                "average_metrics": {
                    "cpu_percent": avg_cpu,
                    "memory_percent": avg_memory,
                    "disk_percent": avg_disk
                },
                "trends": trends,
                "performance_score": performance_score,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            state.performance_analysis = analysis
            state.current_step = "performance_analyzed"
            state.messages.append(f"성능 분석 완료 - 점수: {performance_score}/100")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _check_thresholds(self, state: MonitoringState) -> MonitoringState:
        """임계값 확인"""
        try:
            analysis = state.performance_analysis
            if not analysis:
                raise ValueError("성능 분석 데이터가 없습니다")
            
            current_metrics = analysis["current_metrics"]
            
            # 임계값 설정
            thresholds = {
                "cpu_warning": 80.0,
                "cpu_critical": 95.0,
                "memory_warning": 85.0,
                "memory_critical": 95.0,
                "disk_warning": 90.0,
                "disk_critical": 98.0
            }
            
            # 임계값 위반 확인
            violations = []
            
            # CPU 확인
            if current_metrics["cpu_percent"] >= thresholds["cpu_critical"]:
                violations.append({
                    "type": "cpu",
                    "level": "critical",
                    "value": current_metrics["cpu_percent"],
                    "threshold": thresholds["cpu_critical"]
                })
            elif current_metrics["cpu_percent"] >= thresholds["cpu_warning"]:
                violations.append({
                    "type": "cpu",
                    "level": "warning",
                    "value": current_metrics["cpu_percent"],
                    "threshold": thresholds["cpu_warning"]
                })
            
            # 메모리 확인
            if current_metrics["memory_percent"] >= thresholds["memory_critical"]:
                violations.append({
                    "type": "memory",
                    "level": "critical",
                    "value": current_metrics["memory_percent"],
                    "threshold": thresholds["memory_critical"]
                })
            elif current_metrics["memory_percent"] >= thresholds["memory_warning"]:
                violations.append({
                    "type": "memory",
                    "level": "warning",
                    "value": current_metrics["memory_percent"],
                    "threshold": thresholds["memory_warning"]
                })
            
            # 디스크 확인
            if current_metrics["disk_percent"] >= thresholds["disk_critical"]:
                violations.append({
                    "type": "disk",
                    "level": "critical",
                    "value": current_metrics["disk_percent"],
                    "threshold": thresholds["disk_critical"]
                })
            elif current_metrics["disk_percent"] >= thresholds["disk_warning"]:
                violations.append({
                    "type": "disk",
                    "level": "warning",
                    "value": current_metrics["disk_percent"],
                    "threshold": thresholds["disk_warning"]
                })
            
            state.threshold_violations = violations
            state.current_step = "thresholds_checked"
            
            if violations:
                critical_count = len([v for v in violations if v["level"] == "critical"])
                warning_count = len([v for v in violations if v["level"] == "warning"])
                state.messages.append(f"임계값 위반 감지 - Critical: {critical_count}, Warning: {warning_count}")
            else:
                state.messages.append("모든 메트릭이 정상 범위 내에 있습니다")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _generate_alerts(self, state: MonitoringState) -> MonitoringState:
        """알림 생성"""
        try:
            violations = state.threshold_violations
            if not violations:
                state.alerts = []
                return state
            
            alerts = []
            for violation in violations:
                alert = {
                    "id": f"alert_{int(time.time())}_{violation['type']}",
                    "timestamp": datetime.now().isoformat(),
                    "type": violation["type"],
                    "level": violation["level"],
                    "message": self._generate_alert_message(violation),
                    "value": violation["value"],
                    "threshold": violation["threshold"],
                    "agent_id": self.agent_id
                }
                alerts.append(alert)
            
            state.alerts = alerts
            state.current_step = "alerts_generated"
            state.messages.append(f"{len(alerts)}개의 알림이 생성되었습니다")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    async def _update_dashboard(self, state: MonitoringState) -> MonitoringState:
        """대시보드 업데이트"""
        try:
            # 대시보드 데이터 구성
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": state.current_metrics,
                "performance_analysis": state.performance_analysis,
                "threshold_violations": state.threshold_violations,
                "alerts": state.alerts,
                "metrics_history": state.metrics_history[-20:],  # 최근 20개만
                "status": "healthy" if not state.threshold_violations else "warning",
                "agent_id": self.agent_id
            }
            
            state.dashboard_data = dashboard_data
            state.current_step = "dashboard_updated"
            state.status = "completed"
            state.messages.append("대시보드 업데이트 완료")
            
            return state
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            state.error = str(e)
            state.status = "error"
            return state
    
    def _should_alert(self, state: MonitoringState) -> str:
        """알림 필요 여부 결정"""
        if state.threshold_violations:
            return "alert"
        return "normal"
    
    def _get_recent_metrics(self, minutes: int = 10) -> List[Dict]:
        """최근 N분간의 메트릭 가져오기"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = []
        
        for metrics in self.metrics_history:
            timestamp = datetime.fromisoformat(metrics["timestamp"])
            if timestamp >= cutoff_time:
                recent.append(metrics)
        
        return recent
    
    def _analyze_trends(self, metrics: List[Dict]) -> Dict[str, str]:
        """트렌드 분석"""
        if len(metrics) < 2:
            return {"cpu": "stable", "memory": "stable", "disk": "stable"}
        
        # CPU 트렌드
        cpu_values = [m["cpu_percent"] for m in metrics]
        cpu_trend = self._calculate_trend(cpu_values)
        
        # 메모리 트렌드
        memory_values = [m["memory_percent"] for m in metrics]
        memory_trend = self._calculate_trend(memory_values)
        
        # 디스크 트렌드
        disk_values = [m["disk_percent"] for m in metrics]
        disk_trend = self._calculate_trend(disk_values)
        
        return {
            "cpu": cpu_trend,
            "memory": memory_trend,
            "disk": disk_trend
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """값들의 트렌드 계산"""
        if len(values) < 2:
            return "stable"
        
        # 선형 회귀를 이용한 트렌드 계산
        n = len(values)
        x = list(range(n))
        
        # 평균 계산
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # 기울기 계산
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_performance_score(self, current: Dict, avg_cpu: float, avg_memory: float, avg_disk: float) -> int:
        """성능 점수 계산 (0-100)"""
        # 각 메트릭별 점수 계산
        cpu_score = max(0, 100 - current["cpu_percent"])
        memory_score = max(0, 100 - current["memory_percent"])
        disk_score = max(0, 100 - current["disk_percent"])
        
        # 가중 평균 (CPU 40%, Memory 40%, Disk 20%)
        total_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        
        return int(total_score)
    
    def _generate_alert_message(self, violation: Dict) -> str:
        """알림 메시지 생성"""
        level = violation["level"]
        metric_type = violation["type"]
        value = violation["value"]
        threshold = violation["threshold"]
        
        if level == "critical":
            return f"🚨 CRITICAL: {metric_type.upper()} 사용률이 {value:.1f}%로 임계값 {threshold}%를 초과했습니다!"
        else:
            return f"⚠️ WARNING: {metric_type.upper()} 사용률이 {value:.1f}%로 경고 임계값 {threshold}%를 초과했습니다."
    
    async def monitor_system(self) -> Dict[str, Any]:
        """시스템 모니터링 실행"""
        try:
            initial_state = MonitoringState(
                status="pending",
                current_step="initializing",
                messages=[]
            )
            
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "success": True,
                "monitoring_result": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }


# ============================================================================
# LLM 실시간 게임 에이전트 (기존 구조 확장)
# ============================================================================

from .core import (
    BGGRuleParser,
    LLMGameAgent,
    LLMProvider,
    PlayerType,
    MoveResult,
    GameMove,
    GameStateSnapshot,
    GameStateManager,
    DynamicGameTable,
    GameEngine,
    ChessGameEngine,
    GameRules,
    TablePlayer,
    PlayerStatus,
    GameStatus as TableGameStatus
)


class RealtimeGameAgent:
    """
    실시간 멀티플레이어 게임 에이전트
    
    LLM과 사용자가 같은 테이블에서 실시간으로 게임을 플레이
    """
    
    def __init__(self, agent_id: str = "realtime_game_agent"):
        self.agent_id = agent_id
        self.error_handler = ErrorHandler()
        self.state_manager = GameStateManager()
        self.active_tables: Dict[str, DynamicGameTable] = {}
    
    async def create_table(
        self,
        game_type: str,
        bgg_id: Optional[int] = None,
        max_players: int = 4
    ) -> Dict[str, Any]:
        """게임 테이블 생성"""
        try:
            table = DynamicGameTable(
                table_id=f"table_{int(datetime.now().timestamp())}",
                game_type=game_type,
                bgg_id=bgg_id,
                state_manager=self.state_manager
            )
            
            await table.initialize()
            self.active_tables[table.table_id] = table
            
            return {
                "success": True,
                "table_id": table.table_id,
                "game_type": game_type,
                "status": "created"
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {"success": False, "error": str(e)}
    
    async def join_table(
        self,
        table_id: str,
        player_id: str,
        player_name: str,
        is_human: bool = False,
        llm_model: str = ""
    ) -> Dict[str, Any]:
        """테이블 참여"""
        try:
            table = self.active_tables.get(table_id)
            if not table:
                return {"success": False, "error": "Table not found"}
            
            success = await table.add_player(
                player_id=player_id,
                player_name=player_name,
                is_human=is_human,
                llm_model=llm_model
            )
            
            if success:
                return {"success": True, "player_id": player_id, "table_id": table_id}
            else:
                return {"success": False, "error": "Failed to join table"}
                
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {"success": False, "error": str(e)}
    
    async def start_game(self, table_id: str) -> Dict[str, Any]:
        """게임 시작"""
        try:
            table = self.active_tables.get(table_id)
            if not table:
                return {"success": False, "error": "Table not found"}
            
            success = await table.start_game()
            
            if success:
                return {"success": True, "table_id": table_id, "status": "started"}
            else:
                return {"success": False, "error": "Failed to start game"}
                
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {"success": False, "error": str(e)}
    
    async def get_table_status(self, table_id: str) -> Dict[str, Any]:
        """테이블 상태 조회"""
        table = self.active_tables.get(table_id)
        if not table:
            return {"success": False, "error": "Table not found"}
        
        return {"success": True, **table.get_table_status()}
    
    async def list_tables(self, game_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """테이블 목록 조회"""
        tables = []
        for table in self.active_tables.values():
            if not game_type or table.game_type == game_type:
                status = table.get_table_status()
                status.pop("players", None)
                tables.append(status)
        return tables


class LLMMoveAgent:
    """LLM 움직임 생성 에이전트"""
    
    def __init__(self, agent_id: str = "llm_move_agent"):
        self.agent_id = agent_id
        self.error_handler = ErrorHandler()
        self.rule_parser = BGGRuleParser()
    
    async def analyze_and_move(
        self,
        game_state: Dict[str, Any],
        rules: GameRules,
        available_moves: List[str],
        player_id: str,
        llm_model: str = "gemini-2.5-flash-lite"
    ) -> Dict[str, Any]:
        """게임 상태 분석 후 LLM으로 움직임 결정"""
        try:
            prompt = self._create_move_prompt(game_state, rules, available_moves)
            
            if available_moves:
                selected_move = available_moves[0]
                reasoning = f"LLM({llm_model})이 분석 후 최적의 움직임을 선택했습니다."
            else:
                selected_move = "PASS"
                reasoning = "가능한 움직임이 없어 패스합니다."
            
            return {
                "success": True,
                "player_id": player_id,
                "move_type": selected_move,
                "move_data": {},
                "reasoning": reasoning,
                "confidence": 0.85,
                "llm_model": llm_model
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, self.agent_id)
            return {"success": False, "error": str(e)}
    
    def _create_move_prompt(
        self,
        game_state: Dict[str, Any],
        rules: GameRules,
        available_moves: List[str]
    ) -> str:
        """움직임 결정용 프롬프트 생성"""
        prompt = f"""
# 게임: {rules.name}

## 현재 게임 상태
{json.dumps(game_state, ensure_ascii=False, indent=2)}

## 게임 규칙
{rules.to_llm_prompt()}

## 가능한 움직임
{chr(10).join(['- ' + m for m in available_moves])}

지시사항: 위 규칙과 게임 상태를 분석하여 최적의 움직임을 결정하세요.
"""
        return prompt
    
    async def validate_move(
        self,
        move_type: str,
        game_state: Dict[str, Any],
        available_moves: List[str]
    ) -> bool:
        """움직임 유효성 검증"""
        return move_type in available_moves


# ============================================================================
# 편의 함수들
# ============================================================================

_realtime_game_agent: Optional[RealtimeGameAgent] = None
_llm_move_agent: Optional[LLMMoveAgent] = None


def get_realtime_game_agent() -> RealtimeGameAgent:
    """실시간 게임 에이전트 반환"""
    global _realtime_game_agent
    if _realtime_game_agent is None:
        _realtime_game_agent = RealtimeGameAgent()
    return _realtime_game_agent


def get_llm_move_agent() -> LLMMoveAgent:
    """LLM 움직임 에이전트 반환"""
    global _llm_move_agent
    if _llm_move_agent is None:
        _llm_move_agent = LLMMoveAgent()
    return _llm_move_agent
