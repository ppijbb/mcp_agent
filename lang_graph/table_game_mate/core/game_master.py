#!/usr/bin/env python3
"""
GameMasterGraph - 완전한 멀티 에이전트 테이블 게임 오케스트레이터

6개의 전문 에이전트를 LangGraph로 연결하여 
동적으로 모든 보드게임을 플레이할 수 있는 시스템
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Annotated
from datetime import datetime
from enum import Enum

# LangGraph 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 프로젝트 내부 임포트
from ..models.game_state import GameState, GamePhase, GameMetadata, PlayerInfo, GameAction, GameConfig
from ..agents.game_analyzer import GameAnalyzerAgent
from ..agents.rule_parser import RuleParserAgent
from ..agents.player_manager import PlayerManagerAgent
from ..agents.persona_generator import PersonaGeneratorAgent
from ..agents.game_referee import GameRefereeAgent
from ..agents.score_calculator import ScoreCalculatorAgent
from ..agents.player_agent import PlayerAgent
from ..core.llm_client import LLMClient
from ..utils.mcp_client import MCPClient


class GameMasterState(GameState):
    """GameMasterGraph 전용 확장 상태"""
    # 에이전트 간 통신용 필드들
    bgg_raw_data: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    parsed_rules: Optional[Dict[str, Any]]
    generated_players: Optional[List[PlayerInfo]]
    assigned_personas: Optional[Dict[str, Any]]
    game_setup_complete: Optional[bool]
    current_turn_result: Optional[Dict[str, Any]]
    score_calculation_result: Optional[Dict[str, Any]]
    
    # 에러 핸들링
    agent_errors: List[Dict[str, Any]]
    retry_count: int
    
    # 진행 상태 추적
    workflow_step: str
    step_start_time: Optional[datetime]
    



class GameMasterGraph:
    """
    완전한 멀티 에이전트 게임 마스터
    
    6개 전문 에이전트를 오케스트레이션하여 
    어떤 보드게임이든 동적으로 플레이 가능
    """
    
    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient):
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        
        # 전문 에이전트들 초기화
        self.game_analyzer = GameAnalyzerAgent(llm_client, mcp_client, "game_analyzer")
        self.rule_parser = RuleParserAgent(llm_client, mcp_client, "rule_parser")
        self.player_manager = PlayerManagerAgent(llm_client, mcp_client, "player_manager")
        self.persona_generator = PersonaGeneratorAgent(llm_client, mcp_client, "persona_generator")
        self.game_referee = GameRefereeAgent(llm_client, mcp_client, "game_referee")
        self.score_calculator = ScoreCalculatorAgent(llm_client, mcp_client, "score_calculator")
        self.player_agents: Dict[str, PlayerAgent] = {}
        
        # 그래프 및 실행 상태
        self.graph = None
        self.current_sessions = {}
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """GameMasterGraph 초기화"""
        try:
            print("🚀 GameMasterGraph 초기화 시작...")
            
            # 그래프 구축
            self.graph = await self._build_graph()
            
            # 에이전트들 준비 상태 확인
            await self._validate_agents()
            
            self.is_initialized = True
            print("✅ GameMasterGraph 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ GameMasterGraph 초기화 실패: {e}")
            return False
    
    async def _build_graph(self) -> StateGraph:
        """LangGraph 기반 워크플로우 구축"""
        
        # 상태 그래프 생성
        workflow = StateGraph(GameMasterState)
        
        # === 노드 정의 ===
        
        # 1. 게임 분석 노드
        async def analyze_game_node(state: GameMasterState) -> GameMasterState:
            """GameAnalyzerAgent를 호출하여 게임 정보 분석"""
            print(f"🔍 게임 분석 시작: {state.get('game_config', {}).get('target_game_name', 'Unknown')}")
            
            state["workflow_step"] = "analyzing_game"
            state["step_start_time"] = datetime.now()
            
            try:
                # 환경 구성
                environment = {
                    "game_name": state["game_config"]["target_game_name"],
                    "current_state": state
                }
                
                # GameAnalyzerAgent 실행
                result = await self.game_analyzer.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["analysis_result"] = result["action_result"]
                    state["bgg_raw_data"] = result["action_result"].get("bgg_data")
                    state["phase"] = GamePhase.RULE_PARSING
                    print("✅ 게임 분석 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 게임 분석 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "game_analyzer", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 게임 분석 노드 오류: {e}")
            
            return state
        
        # 2. 규칙 파싱 노드
        async def parse_rules_node(state: GameMasterState) -> GameMasterState:
            """RuleParserAgent를 호출하여 게임 규칙 구조화"""
            print("📜 게임 규칙 파싱 시작...")
            
            state["workflow_step"] = "parsing_rules"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "analysis_result": state.get("analysis_result"),
                    "bgg_data": state.get("bgg_raw_data"),
                    "current_state": state
                }
                
                result = await self.rule_parser.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["parsed_rules"] = result["action_result"]
                    state["phase"] = GamePhase.PLAYER_GENERATION
                    print("✅ 규칙 파싱 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 규칙 파싱 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "rule_parser", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 규칙 파싱 노드 오류: {e}")
            
            return state
        
        # 3. 페르소나 생성 노드
        async def generate_personas_node(state: GameMasterState) -> GameMasterState:
            """PersonaGeneratorAgent를 호출하여 AI 플레이어 페르소나 부여"""
            print("🎭 플레이어 페르소나 생성 시작...")
            
            state["workflow_step"] = "generating_personas"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "game_analysis": state.get("analysis_result", {}),
                    "personas_needed": state["game_config"]["desired_player_count"],
                    "complexity": state.get("analysis_result", {}).get("complexity", "moderate"),
                    "suggested_types": ["strategic", "social", "aggressive"],
                    "current_state": state
                }
                
                result = await self.persona_generator.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["assigned_personas"] = result["action_result"]
                    state["phase"] = GamePhase.PERSONA_GENERATION
                    print("✅ 페르소나 생성 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 페르소나 생성 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "persona_generator", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 페르소나 생성 노드 오류: {e}")
            
            return state
        
        # 4. 플레이어 관리 노드
        async def generate_personas_node(state: GameMasterState) -> GameMasterState:
            """PersonaGeneratorAgent를 호출하여 AI 플레이어 페르소나 부여"""
            print("🎭 플레이어 페르소나 생성 시작...")
            
            state["workflow_step"] = "generating_personas"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "players": state.get("generated_players", []),
                    "parsed_rules": state.get("parsed_rules"),
                    "game_type": state.get("analysis_result", {}).get("game_type"),
                    "current_state": state
                }
                
                result = await self.persona_generator.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["assigned_personas"] = result["action_result"]
                    # 플레이어 정보에 페르소나 적용
                    for player in state["players"]:
                        if player.id in result["action_result"]["persona_assignments"]:
                            player.persona_type = result["action_result"]["persona_assignments"][player.id]
                    
                    state["phase"] = GamePhase.GAME_START
                    print("✅ 페르소나 생성 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 페르소나 생성 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "persona_generator", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 페르소나 생성 노드 오류: {e}")
            
            return state
        
        # 4.5. 플레이어 관리 노드
        async def manage_players_node(state: GameMasterState) -> GameMasterState:
            """PlayerManagerAgent를 호출하여 플레이어 생성 및 관리"""
            print("👥 플레이어 생성 시작...")
            
            state["workflow_step"] = "managing_players"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "persona_profiles": state.get("assigned_personas", {}).get("persona_profiles", []),
                    "game_config": state["game_config"],
                    "current_state": state
                }
                
                result = await self.player_manager.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["generated_players"] = result["action_result"]["players"]
                    state["players"] = result["action_result"]["players"]
                    state["phase"] = GamePhase.PLAYER_GENERATION
                    print(f"✅ 플레이어 {len(state['players'])}명 생성 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 플레이어 생성 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "player_manager", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 플레이어 관리 노드 오류: {e}")
            
            return state
        
        # 5. 플레이어 에이전트 생성 노드
        async def create_player_agents_node(state: GameMasterState) -> GameMasterState:
            """생성된 플레이어와 페르소나를 바탕으로 PlayerAgent 인스턴스 생성"""
            print("🤖 플레이어 AI 에이전트 생성 시작...")
            state["workflow_step"] = "creating_player_agents"
            
            try:
                personas = state.get("assigned_personas", {}).get("persona_profiles", {})
                for player in state["players"]:
                    if player.is_ai and player.id not in self.player_agents:
                        persona_profile = personas.get(player.id)
                        if persona_profile:
                            self.player_agents[player.id] = PlayerAgent(
                                self.llm_client,
                                self.mcp_client,
                                player_info=player,
                                persona=persona_profile
                            )
                print(f"✅ AI 에이전트 {len(self.player_agents)}명 생성 완료")
            except Exception as e:
                error_info = {"agent": "player_agent_creation", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 플레이어 AI 에이전트 생성 오류: {e}")
                
            return state
        
        # 5. 게임 시작 노드
        async def setup_game_node(state: GameMasterState) -> GameMasterState:
            """GameRefereeAgent를 호출하여 게임 초기화 및 시작"""
            print("🎯 게임 초기화 시작...")
            
            state["workflow_step"] = "setting_up_game"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "players": state["players"],
                    "parsed_rules": state.get("parsed_rules"),
                    "personas": state.get("assigned_personas"),
                    "current_state": state
                }
                
                result = await self.game_referee.run_cycle(environment)
                
                if result["cycle_complete"]:
                    state["game_setup_complete"] = True
                    state["game_board"] = result["action_result"].get("initial_board_state", {})
                    state["turn_count"] = 0
                    state["current_player_index"] = 0
                    state["phase"] = GamePhase.PLAYER_TURN
                    print("✅ 게임 초기화 완료")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 게임 초기화 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "game_referee", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 게임 초기화 노드 오류: {e}")
            
            return state
        
        # 6. 게임 턴 진행 노드
        async def play_turn_node(state: GameMasterState) -> GameMasterState:
            """게임 턴을 진행하고 승리 조건 체크"""
            print(f"🎮 턴 {state['turn_count'] + 1} 진행 중...")
            
            state["workflow_step"] = "playing_turn"
            state["step_start_time"] = datetime.now()
            
            try:
                current_player = state["players"][state["current_player_index"]]
                print(f"   현재 플레이어: {current_player.name} ({current_player.player_type})")

                # 1. 플레이어 행동 결정
                player_action = None
                if current_player.is_ai:
                    player_agent = self.player_agents.get(current_player.id)
                    if player_agent:
                        print(f"   AI ({player_agent.persona.persona_type}) 행동 결정 중...")
                        env_for_player = {
                            "game_state": state,
                            "parsed_rules": state["parsed_rules"],
                            "other_players": [p for p in state["players"] if p.id != current_player.id],
                            "is_my_turn": True,
                        }
                        action_result = await player_agent.run_cycle(env_for_player)
                        player_action = action_result.get("action_result", {})
                    else:
                        raise Exception(f"PlayerAgent for {current_player.id} not found!")
                else:
                    # TODO: 인간 플레이어의 입력을 받는 로직
                    print("   인간 플레이어 턴. (현재는 자동 패스)")
                    player_action = {"action_type": "pass", "action_data": {}}

                # 2. 행동 유효성 검증
                print(f"   심판이 행동 검증 중: {player_action}")
                env_for_referee = {
                    "player_action": player_action,
                    "game_state": state,
                    "parsed_rules": state["parsed_rules"],
                }
                validation_result = await self.game_referee.run_cycle(env_for_referee)
                validation_data = validation_result.get("action_result", {})
                
                if validation_data.get("is_valid"):
                    print(f"   ✅ 행동 유효: {validation_data.get('message')}")
                    state["current_turn_result"] = player_action
                    
                    # 3. 게임 상태 업데이트
                    # TODO: 실제 게임 로직에 따라 상태를 변경하는 더 정교한 방법 필요
                    # 현재는 PlayerAgent가 action_data에 board_changes를 포함한다고 가정
                    state["game_board"].update(player_action.get("action_data", {}).get("board_changes", {}))
                    state["game_history"].append(GameAction(
                        player_id=current_player.id,
                        action_type=player_action.get("action_type", "turn"),
                        action_data=player_action.get("action_data", {}),
                        is_valid=True
                    ))
                    
                    # 4. 승리 조건 체크 (Referee가 판단할 수도 있음)
                    if player_action.get("action_data", {}).get("game_ended", False):
                        state["phase"] = GamePhase.SCORE_CALCULATION
                        state["game_ended"] = True
                        print("🏁 게임 종료 조건 달성!")

                else:
                    print(f"   ❌ 행동 무효: {validation_data.get('message')}")
                    # 잘못된 행동에 대한 처리 (예: 턴 넘김)
                    state["game_history"].append(GameAction(
                        player_id=current_player.id,
                        action_type=player_action.get("action_type", "invalid_turn"),
                        action_data=player_action.get("action_data", {}),
                        is_valid=False,
                        reason=validation_data.get('message')
                    ))

                # 5. 다음 턴으로
                state["current_player_index"] = (state["current_player_index"] + 1) % len(state["players"])
                if state["current_player_index"] == 0:
                    state["turn_count"] += 1
                
                print(f"✅ 턴 진행 완료")
                
            except Exception as e:
                error_info = {"agent": "play_turn_node", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 턴 진행 노드 오류: {e}")
            
            return state
        
        # 7. 점수 계산 노드
        async def calculate_scores_node(state: GameMasterState) -> GameMasterState:
            """ScoreCalculatorAgent를 호출하여 최종 점수 계산"""
            print("🏆 최종 점수 계산 시작...")
            
            state["workflow_step"] = "calculating_scores"
            state["step_start_time"] = datetime.now()
            
            try:
                environment = {
                    "players": state["players"],
                    "game_board": state["game_board"],
                    "game_history": state["game_history"],
                    "parsed_rules": state["parsed_rules"],
                    "current_state": state
                }
                
                result = await self.score_calculator.run_cycle(environment)
                
                if result["cycle_complete"]:
                    score_result = result["action_result"]
                    state["score_calculation_result"] = score_result
                    state["final_scores"] = score_result.get("final_scores", {})
                    state["winner_ids"] = score_result.get("winners", [])
                    state["phase"] = GamePhase.GAME_END
                    
                    print("✅ 점수 계산 완료")
                    print(f"🏆 승자: {', '.join(state['winner_ids'])}")
                else:
                    state["agent_errors"].append(result["error"])
                    print(f"❌ 점수 계산 실패: {result['error']}")
                
            except Exception as e:
                error_info = {"agent": "score_calculator", "error": str(e), "timestamp": datetime.now()}
                state["agent_errors"].append(error_info)
                print(f"❌ 점수 계산 노드 오류: {e}")
            
            return state
        
        # === 노드 추가 ===
        workflow.add_node("analyze_game", analyze_game_node)
        workflow.add_node("parse_rules", parse_rules_node)
        workflow.add_node("manage_players", manage_players_node)
        workflow.add_node("generate_personas", generate_personas_node)
        workflow.add_node("create_player_agents", create_player_agents_node)
        workflow.add_node("setup_game", setup_game_node)
        workflow.add_node("play_turn", play_turn_node)
        workflow.add_node("calculate_scores", calculate_scores_node)
        
        # === 엣지 연결 ===
        workflow.add_edge(START, "analyze_game")
        workflow.add_edge("analyze_game", "parse_rules")
        workflow.add_edge("parse_rules", "generate_personas")
        workflow.add_edge("generate_personas", "manage_players")
        workflow.add_edge("generate_personas", "create_player_agents")
        workflow.add_edge("create_player_agents", "setup_game")
        workflow.add_edge("setup_game", "play_turn")
        
        # 조건부 엣지: 게임 계속 vs 종료
        def should_continue_game(state: GameMasterState) -> str:
            """게임 계속 여부 결정"""
            if state.get("game_ended", False):
                return "calculate_scores"
            elif state.get("turn_count", 0) >= 50:  # 무한 루프 방지
                return "calculate_scores"
            else:
                return "play_turn"
        
        workflow.add_conditional_edges(
            "play_turn",
            should_continue_game,
            {
                "play_turn": "play_turn",
                "calculate_scores": "calculate_scores"
            }
        )
        
        workflow.add_edge("calculate_scores", END)
        
        # 체크포인터로 상태 저장
        checkpointer = MemorySaver()
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _validate_agents(self) -> None:
        """모든 에이전트가 준비 상태인지 확인"""
        agents = [
            ("GameAnalyzer", self.game_analyzer),
            ("RuleParser", self.rule_parser),
            ("PlayerManager", self.player_manager),
            ("PersonaGenerator", self.persona_generator),
            ("GameReferee", self.game_referee),
            ("ScoreCalculator", self.score_calculator)
        ]
        
        for name, agent in agents:
            if not hasattr(agent, 'run_cycle'):
                raise ValueError(f"{name} 에이전트가 올바르게 구현되지 않음")
        
        print("✅ 모든 에이전트 검증 완료")
    
    async def start_game_session(self, game_config: GameConfig) -> Dict[str, Any]:
        """새로운 게임 세션 시작"""
        
        if not self.is_initialized:
            return {"success": False, "error": "GameMasterGraph가 초기화되지 않음"}
        
        session_id = str(uuid.uuid4())
        
        print(f"🎮 새 게임 세션 시작: {game_config['target_game_name']}")
        print(f"   세션 ID: {session_id}")
        print(f"   플레이어 수: {game_config['desired_player_count']}명")
        
        # 초기 상태 구성
        initial_state: GameMasterState = {
            # 기본 게임 정보 (GameState에서 상속받은 필드들)
            "game_id": session_id,
            "game_metadata": None,
            "phase": GamePhase.SETUP,
            
            # 플레이어 정보
            "players": [],
            "current_player_index": 0,
            
            # 게임 진행 상태
            "turn_count": 0,
            "game_board": {},
            "game_history": [],
            
            # 규칙 및 설정
            "parsed_rules": None,
            "game_config": game_config,
            
            # 에이전트 간 통신
            "last_action": None,
            "pending_actions": [],
            "error_messages": [],
            
            # 게임 결과
            "winner_ids": [],
            "final_scores": {},
            "game_ended": False,
            
            # 메타 정보
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            
            # 확장 필드들
            "bgg_raw_data": None,
            "analysis_result": None,
            "generated_players": None,
            "assigned_personas": None,
            "game_setup_complete": None,
            "current_turn_result": None,
            "score_calculation_result": None,
            "agent_errors": [],
            "retry_count": 0,
            "workflow_step": "initializing",
            "step_start_time": datetime.now()
        }
        
        try:
            # 그래프 실행 설정
            config = {"configurable": {"thread_id": session_id}}
            
            # 비동기로 그래프 실행 시작
            print("🚀 게임 워크플로우 실행 시작...")
            
            # 그래프 실행 (전체 게임 라이프사이클)
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # 세션 저장
            self.current_sessions[session_id] = {
                "session_id": session_id,
                "final_state": final_state,
                "config": config,
                "created_at": datetime.now(),
                "status": "completed" if final_state.get("game_ended") else "error"
            }
            
            print(f"✅ 게임 세션 완료: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "final_state": final_state,
                "game_result": {
                    "winner_ids": final_state.get("winner_ids", []),
                    "final_scores": final_state.get("final_scores", {}),
                    "turn_count": final_state.get("turn_count", 0),
                    "game_ended": final_state.get("game_ended", False)
                }
            }
            
        except Exception as e:
            print(f"❌ 게임 세션 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """게임 세션 상태 조회"""
        
        if session_id not in self.current_sessions:
            return {"success": False, "error": "세션을 찾을 수 없음"}
        
        session = self.current_sessions[session_id]
        state = session["final_state"]
        
        return {
            "success": True,
            "session_id": session_id,
            "status": session["status"],
            "game_name": state.get("game_config", {}).get("target_game_name"),
            "phase": state.get("phase"),
            "turn_count": state.get("turn_count", 0),
            "players": [
                {
                    "name": p.name,
                    "score": p.score,
                    "persona": p.persona_type
                } for p in state.get("players", [])
            ],
            "game_ended": state.get("game_ended", False),
            "winners": state.get("winner_ids", []),
            "errors": state.get("agent_errors", [])
        }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 게임 세션 목록 조회"""
        return [
            {
                "session_id": sid,
                "game_name": session["final_state"].get("game_config", {}).get("target_game_name"),
                "status": session["status"],
                "created_at": session["created_at"].isoformat()
            }
            for sid, session in self.current_sessions.items()
        ]


# === 사용 예시 ===

async def demo_game_master_graph():
    """GameMasterGraph 데모 실행"""
    
    print("🚀 GameMasterGraph 데모 시작")
    print("=" * 60)
    
    # Mock 클라이언트들 (실제 환경에서는 진짜 클라이언트 사용)
    class MockLLMClient:
        async def complete(self, prompt: str) -> str:
            return "Mock LLM response for demo"
    
    class MockMCPClient:
        async def call(self, server: str, method: str, params: Dict) -> Dict:
            return {"success": True, "result": "Mock MCP response"}
    
    # GameMasterGraph 초기화
    llm_client = MockLLMClient()
    mcp_client = MockMCPClient()
    
    game_master = GameMasterGraph(llm_client, mcp_client)
    
    # 초기화
    if not await game_master.initialize():
        print("❌ 초기화 실패")
        return
    
    # 게임 설정
    game_config: GameConfig = {
        "target_game_name": "Azul",
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
    
    # 게임 세션 시작
    result = await game_master.start_game_session(game_config)
    
    if result["success"]:
        session_id = result["session_id"]
        print(f"\n🎉 게임 완료!")
        print(f"승자: {', '.join(result['game_result']['winner_ids'])}")
        print(f"총 턴 수: {result['game_result']['turn_count']}")
        
        # 세션 상태 확인
        status = await game_master.get_session_status(session_id)
        print(f"\n📊 최종 상태:")
        for player in status["players"]:
            print(f"  {player['name']}: {player['score']}점 ({player['persona']})")
    
    else:
        print(f"❌ 게임 실행 실패: {result['error']}")


if __name__ == "__main__":
    asyncio.run(demo_game_master_graph())