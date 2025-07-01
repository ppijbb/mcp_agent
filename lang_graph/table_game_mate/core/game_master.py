"""
LangGraph 기반 게임 마스터
전체 테이블게임 시스템의 중앙 오케스트레이터

✅ 실제 MCP 통합 적용됨 (웹 검색 결과 기반)
"""

from typing import Dict, List, Any, Optional, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
import asyncio
import uuid
from datetime import datetime
import os
import sys

# MCP 통합을 위한 import (웹 검색 결과 기반)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt

from ..models.game_state import GameState, GamePhase, PlayerInfo, GameAction, GameMetadata
from ..models.persona import PersonaProfile, PersonaGenerator

# LangGraph 상태 정의
class GameMasterState(TypedDict):
    """게임 마스터의 상태 - GameState 확장"""
    # GameState 기본 필드들
    game_id: str
    game_metadata: Optional[GameMetadata]
    phase: GamePhase
    players: List[PlayerInfo]
    current_player_index: int
    turn_count: int
    game_board: Dict[str, Any]
    game_history: List[GameAction]
    parsed_rules: Optional[Dict[str, Any]]
    game_config: Dict[str, Any]
    last_action: Optional[GameAction]
    pending_actions: List[GameAction]
    error_messages: List[str]
    winner_ids: List[str]
    final_scores: Dict[str, int]
    game_ended: bool
    created_at: datetime
    updated_at: datetime
    
    # 게임 마스터 확장 필드들
    current_agent: str
    agent_responses: Annotated[List[Dict[str, Any]], add_messages]
    user_input: Optional[str]
    awaiting_user_input: bool
    next_step: Optional[str]

class GameMasterGraph:
    """LangGraph 기반 게임 마스터 (실제 MCP 통합)"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (Anthropic, OpenAI 등)
        """
        self.llm_client = llm_client
        self.memory = MemorySaver()
        
        # ✅ 실제 MCP 클라이언트 설정 (웹 검색 결과 기반)
        self.mcp_client = MultiServerMCPClient({
            "bgg": {
                "command": "python",
                "args": [os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "bgg_mcp_server.py")],
                "transport": "stdio",
            }
        })
        
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """게임 진행 그래프 구성"""
        
        # 그래프 생성
        workflow = StateGraph(GameMasterState)
        
        # 노드 추가
        workflow.add_node("initialize_game", self._initialize_game)
        workflow.add_node("analyze_game", self._analyze_game)
        workflow.add_node("parse_rules", self._parse_rules)
        workflow.add_node("generate_players", self._generate_players)
        workflow.add_node("start_game", self._start_game)
        workflow.add_node("process_turn", self._process_turn)
        workflow.add_node("validate_action", self._validate_action)
        workflow.add_node("update_state", self._update_state)
        workflow.add_node("check_end_condition", self._check_end_condition)
        workflow.add_node("calculate_scores", self._calculate_scores)
        workflow.add_node("end_game", self._end_game)
        workflow.add_node("handle_error", self._handle_error)
        
        # 엣지 정의
        workflow.set_entry_point("initialize_game")
        
        # 게임 설정 단계
        workflow.add_edge("initialize_game", "analyze_game")
        workflow.add_edge("analyze_game", "parse_rules")
        workflow.add_edge("parse_rules", "generate_players")
        workflow.add_edge("generate_players", "start_game")
        
        # 게임 진행 루프
        workflow.add_edge("start_game", "process_turn")
        workflow.add_conditional_edges(
            "process_turn",
            self._route_after_turn,
            {
                "validate": "validate_action",
                "wait_user": "process_turn",  # 사용자 입력 대기
                "error": "handle_error",
            }
        )
        
        workflow.add_conditional_edges(
            "validate_action",
            self._route_after_validation,
            {
                "update": "update_state",
                "retry": "process_turn",
                "error": "handle_error",
            }
        )
        
        workflow.add_conditional_edges(
            "update_state",
            self._route_after_update,
            {
                "continue": "process_turn",
                "check_end": "check_end_condition",
                "error": "handle_error",
            }
        )
        
        workflow.add_conditional_edges(
            "check_end_condition",
            self._route_game_end,
            {
                "continue": "process_turn",
                "end": "calculate_scores",
            }
        )
        
        workflow.add_edge("calculate_scores", "end_game")
        workflow.add_edge("end_game", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def run_game(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """게임 실행"""
        
        # 초기 상태 설정
        initial_state = self._create_initial_state(config)
        
        # 게임 실행
        result = await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )
        
        return result
    
    def _create_initial_state(self, config: Dict[str, Any]) -> GameMasterState:
        """초기 상태 생성"""
        
        game_id = str(uuid.uuid4())
        now = datetime.now()
        
        return GameMasterState(
            # 기본 게임 상태
            game_id=game_id,
            game_metadata=None,
            phase=GamePhase.SETUP,
            players=[],
            current_player_index=0,
            turn_count=0,
            game_board={},
            game_history=[],
            parsed_rules=None,
            game_config=config,
            last_action=None,
            pending_actions=[],
            error_messages=[],
            winner_ids=[],
            final_scores={},
            game_ended=False,
            created_at=now,
            updated_at=now,
            
            # 게임 마스터 확장 상태
            current_agent="",
            agent_responses=[],
            user_input=None,
            awaiting_user_input=False,
            next_step=None,
        )
    
    # 노드 구현들
    async def _initialize_game(self, state: GameMasterState) -> GameMasterState:
        """게임 초기화"""
        state["phase"] = GamePhase.SETUP
        state["current_agent"] = "game_analyzer"
        state["updated_at"] = datetime.now()
        
        print(f"🎮 게임 시스템 초기화 중...")
        print(f"요청된 게임: {state['game_config'].get('target_game_name', '미지정')}")
        
        return state
    
    async def _analyze_game(self, state: GameMasterState) -> GameMasterState:
        """✅ 실제 MCP를 사용한 게임 분석"""
        state["current_agent"] = "game_analyzer"
        
        game_name = state["game_config"]["target_game_name"]
        print(f"🔍 '{game_name}' 게임 정보 검색 중... (실제 BGG API 호출)")
        
        try:
            # ✅ 실제 MCP BGG 서버를 통한 게임 검색
            async with self.mcp_client.session("bgg") as session:
                tools = await load_mcp_tools(session)
                
                # BGG에서 게임 검색
                search_tool = next((t for t in tools if t.name == "search_boardgame"), None)
                if search_tool:
                    search_result = await search_tool.ainvoke({"name": game_name})
                    
                    if search_result.get("success") and search_result.get("games"):
                        # 첫 번째 검색 결과 사용
                        first_game = search_result["games"][0]
                        bgg_id = first_game["id"]
                        
                        # 게임 상세 정보 조회
                        details_tool = next((t for t in tools if t.name == "get_game_details"), None)
                        if details_tool:
                            details_result = await details_tool.ainvoke({"bgg_id": bgg_id})
                            
                            if details_result.get("success"):
                                game_data = details_result["game"]
                                
                                # ✅ 실제 BGG 데이터로 GameMetadata 생성
                                game_metadata = GameMetadata(
                                    name=game_data.get("name", game_name),
                                    min_players=game_data.get("min_players", 2),
                                    max_players=game_data.get("max_players", 4),
                                    estimated_duration=game_data.get("playing_time", 60),
                                    complexity=game_data.get("rating", {}).get("complexity", 2.5),
                                    description=game_data.get("description", "")[:200]
                                )
                                
                                state["game_metadata"] = game_metadata
                                print(f"✅ BGG에서 게임 정보 수집 완료: {game_metadata.name}")
                                print(f"   플레이어: {game_metadata.min_players}-{game_metadata.max_players}명")
                                print(f"   소요시간: {game_metadata.estimated_duration}분")
                                print(f"   복잡도: {game_metadata.complexity:.1f}/5")
                                
                                state["updated_at"] = datetime.now()
                                return state
        
        except Exception as e:
            print(f"⚠️ BGG API 호출 실패: {e}")
            # 폴백: 기본 메타데이터 생성
        
        # 폴백 메타데이터
        game_metadata = GameMetadata(
            name=game_name,
            min_players=2,
            max_players=4,
            estimated_duration=45,
            complexity=3.0,
            description=f"{game_name} 게임입니다. (BGG 데이터 없음)"
        )
        
        state["game_metadata"] = game_metadata
        state["updated_at"] = datetime.now()
        print(f"⚠️ 폴백 데이터 사용: {game_name}")
        
        return state
    
    async def _parse_rules(self, state: GameMasterState) -> GameMasterState:
        """규칙 파싱"""
        state["phase"] = GamePhase.RULE_PARSING
        state["current_agent"] = "rule_parser"
        
        print(f"📋 게임 규칙 분석 중...")
        
        # TODO: 실제 규칙 파서 에이전트 호출
        # 현재는 더미 규칙
        state["parsed_rules"] = {
            "setup": "게임 설정 규칙",
            "turn_structure": "턴 진행 규칙",
            "win_conditions": "승리 조건",
            "actions": ["행동1", "행동2", "행동3"]
        }
        
        return state
    
    async def _generate_players(self, state: GameMasterState) -> GameMasterState:
        """✅ 실제 페르소나 생성 시스템 사용"""
        state["phase"] = GamePhase.PLAYER_GENERATION
        state["current_agent"] = "player_generator"
        
        print(f"👥 AI 플레이어 생성 중...")
        
        config = state["game_config"]
        desired_count = config.get("desired_player_count", 3)
        game_name = config["target_game_name"]
        difficulty = config.get("difficulty_level", "medium")
        
        # 사용자 플레이어 추가
        user_player = PlayerInfo(
            id="user",
            name="사용자",
            is_ai=False,
            turn_order=0
        )
        state["players"] = [user_player]
        
        # ✅ 실제 페르소나 생성 시스템 사용
        ai_count = desired_count - 1  # 사용자 제외
        if ai_count > 0:
            try:
                # PersonaGenerator를 사용한 동적 페르소나 생성
                game_metadata = state.get("game_metadata")
                game_type = "strategy"  # 기본값, 추후 메타데이터에서 추출
                
                personas = PersonaGenerator.generate_for_game(
                    game_name=game_name,
                    game_type=game_type,
                    count=ai_count,
                    difficulty=difficulty
                )
                
                # AI 플레이어들 생성
                for i, persona in enumerate(personas):
                    ai_player = PlayerInfo(
                        id=f"ai_{i+1}",
                        name=persona["name"],
                        is_ai=True,
                        persona_type=persona["archetype"].value,
                        turn_order=i + 1
                    )
                    state["players"].append(ai_player)
                
                print(f"✅ 동적 페르소나 생성 완료: {len(personas)}명")
                for i, persona in enumerate(personas, 1):
                    print(f"   AI {i}: {persona['name']} ({persona['archetype'].value})")
                
            except Exception as e:
                print(f"⚠️ 페르소나 생성 실패: {e}, 기본 AI 사용")
                # 폴백: 기본 AI 플레이어
                for i in range(ai_count):
                    ai_player = PlayerInfo(
                        id=f"ai_{i+1}",
                        name=f"AI플레이어{i+1}",
                        is_ai=True,
                        persona_type="analytical",
                        turn_order=i + 1
                    )
                    state["players"].append(ai_player)
        
        print(f"✅ 총 {len(state['players'])}명의 플레이어 준비 완료")
        
        return state
    
    async def _start_game(self, state: GameMasterState) -> GameMasterState:
        """게임 시작"""
        state["phase"] = GamePhase.GAME_START
        state["turn_count"] = 1
        state["current_player_index"] = 0
        
        print(f"🎯 게임 시작! 첫 번째 플레이어: {state['players'][0]['name']}")
        
        return state
    
    async def _process_turn(self, state: GameMasterState) -> GameMasterState:
        """턴 처리"""
        state["phase"] = GamePhase.PLAYER_TURN
        
        current_player = state["players"][state["current_player_index"]]
        print(f"🎲 {current_player['name']}의 턴")
        
        if current_player["is_ai"]:
            # AI 플레이어 행동
            # TODO: 실제 AI 에이전트 호출
            action = GameAction(
                player_id=current_player["id"],
                action_type="test_action",
                action_data={"test": "data"}
            )
            state["last_action"] = action
        else:
            # 사용자 입력 대기
            state["awaiting_user_input"] = True
            print("사용자의 입력을 기다리는 중...")
        
        return state
    
    async def _validate_action(self, state: GameMasterState) -> GameMasterState:
        """액션 검증"""
        state["current_agent"] = "referee"
        
        action = state["last_action"]
        if action:
            # TODO: 실제 규칙 검증
            action["is_valid"] = True
            print(f"✅ 액션 검증 완료: {action['action_type']}")
        
        return state
    
    async def _update_state(self, state: GameMasterState) -> GameMasterState:
        """상태 업데이트"""
        
        action = state["last_action"]
        if action and action.get("is_valid"):
            # 게임 히스토리에 추가
            state["game_history"].append(action)
            
            # 다음 플레이어로
            state["current_player_index"] = (
                state["current_player_index"] + 1
            ) % len(state["players"])
            
            # 한 라운드 완료시 턴 카운트 증가
            if state["current_player_index"] == 0:
                state["turn_count"] += 1
        
        state["updated_at"] = datetime.now()
        return state
    
    async def _check_end_condition(self, state: GameMasterState) -> GameMasterState:
        """게임 종료 조건 확인"""
        
        # TODO: 실제 종료 조건 확인
        # 현재는 5턴 후 종료
        if state["turn_count"] >= 5:
            state["game_ended"] = True
            print("🏁 게임 종료 조건 달성!")
        
        return state
    
    async def _calculate_scores(self, state: GameMasterState) -> GameMasterState:
        """점수 계산"""
        state["current_agent"] = "score_calculator"
        
        # TODO: 실제 점수 계산
        for player in state["players"]:
            state["final_scores"][player["id"]] = player["score"]
        
        # 승자 결정
        if state["final_scores"]:
            max_score = max(state["final_scores"].values())
            state["winner_ids"] = [
                pid for pid, score in state["final_scores"].items() 
                if score == max_score
            ]
        
        return state
    
    async def _end_game(self, state: GameMasterState) -> GameMasterState:
        """게임 종료"""
        state["phase"] = GamePhase.GAME_END
        
        print("🎉 게임 종료!")
        print(f"최종 점수: {state['final_scores']}")
        print(f"승자: {state['winner_ids']}")
        
        return state
    
    async def _handle_error(self, state: GameMasterState) -> GameMasterState:
        """에러 처리"""
        
        if state["error_messages"]:
            print(f"❌ 에러 발생: {state['error_messages'][-1]}")
        
        # 에러 복구 시도 또는 게임 종료
        state["game_ended"] = True
        return state
    
    # 라우팅 함수들
    def _route_after_turn(self, state: GameMasterState) -> str:
        """턴 처리 후 라우팅"""
        
        if state["error_messages"]:
            return "error"
        
        if state["awaiting_user_input"]:
            return "wait_user"
        
        return "validate"
    
    def _route_after_validation(self, state: GameMasterState) -> str:
        """검증 후 라우팅"""
        
        if state["error_messages"]:
            return "error"
        
        last_action = state["last_action"]
        if last_action and last_action.get("is_valid"):
            return "update"
        else:
            return "retry"
    
    def _route_after_update(self, state: GameMasterState) -> str:
        """업데이트 후 라우팅"""
        
        if state["error_messages"]:
            return "error"
        
        return "check_end"
    
    def _route_game_end(self, state: GameMasterState) -> str:
        """게임 종료 라우팅"""
        
        if state["game_ended"]:
            return "end"
        else:
            return "continue" 