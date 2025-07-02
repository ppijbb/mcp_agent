 #!/usr/bin/env python3
"""
AI Agent 기반 동적 UI 생성 시스템
게임 규칙을 분석하여 실시간으로 적절한 인터페이스를 생성
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re
from datetime import datetime

# LangGraph 에이전트 import는 실제 구현시에만 활성화
# 현재는 Mock 에이전트만 사용
# from core.game_master import GameMaster
# from agents.game_analyzer import GameAnalyzer  
# from agents.ui_generator import UIGenerator

class UIComplexity(Enum):
    MINIMAL = "minimal"
    STANDARD = "standard" 
    ADVANCED = "advanced"
    CUSTOM = "custom"

@dataclass
class GameUISpec:
    """AI가 생성한 게임 UI 명세"""
    game_name: str
    board_type: str
    required_components: List[str]
    layout_structure: Dict[str, Any]
    interaction_patterns: List[str]
    complexity_level: UIComplexity
    special_features: Dict[str, Any]
    generated_at: datetime
    confidence_score: float

class AgentDrivenUIGenerator:
    """AI 에이전트가 주도하는 UI 생성기"""
    
    def __init__(self):
        # 실제로는 LangGraph 에이전트들을 사용하지만, 
        # 현재는 Mock 에이전트로 대체
        self.game_analyzer = None  # MockGameAnalyzer로 대체될 예정
        self.ui_generator = None   # MockUIGenerator로 대체될 예정
        self.game_master = None    # 실제 게임 마스터는 나중에 연동
        self.cache = {}  # UI 스펙 캐시
    
    async def analyze_and_generate_ui(self, game_description: str, game_rules: str = None) -> GameUISpec:
        """게임 설명/규칙을 분석하여 UI 명세 생성"""
        
        # 1단계: 게임 분석
        analysis_result = await self.game_analyzer.analyze_game({
            "description": game_description,
            "rules": game_rules or ""
        })
        
        # 2단계: UI 구조 결정
        ui_structure = await self.ui_generator.generate_ui_spec(analysis_result)
        
        # 3단계: UI 명세 생성
        ui_spec = GameUISpec(
            game_name=analysis_result.get("game_name", "Unknown Game"),
            board_type=analysis_result.get("board_type", "grid"),
            required_components=ui_structure.get("components", []),
            layout_structure=ui_structure.get("layout", {}),
            interaction_patterns=ui_structure.get("interactions", []),
            complexity_level=UIComplexity(ui_structure.get("complexity", "standard")),
            special_features=ui_structure.get("special_features", {}),
            generated_at=datetime.now(),
            confidence_score=analysis_result.get("confidence", 0.8)
        )
        
        return ui_spec
    
    def render_dynamic_ui(self, ui_spec: GameUISpec, game_state: Dict[str, Any]):
        """생성된 UI 명세에 따라 동적으로 인터페이스 렌더링"""
        
        st.header(f"🎲 {ui_spec.game_name}")
        
        # 신뢰도 표시
        confidence_color = "green" if ui_spec.confidence_score > 0.8 else "orange" if ui_spec.confidence_score > 0.6 else "red"
        st.markdown(f"**AI 분석 신뢰도**: :{confidence_color}[{ui_spec.confidence_score:.1%}]")
        
        # 레이아웃 구조에 따른 렌더링
        layout = ui_spec.layout_structure
        
        if layout.get("main_area"):
            self._render_main_area(ui_spec, game_state, layout["main_area"])
        
        if layout.get("sidebar"):
            with st.sidebar:
                self._render_sidebar(ui_spec, game_state, layout["sidebar"])
        
        if layout.get("bottom_panel"):
            self._render_bottom_panel(ui_spec, game_state, layout["bottom_panel"])
    
    def _render_main_area(self, ui_spec: GameUISpec, game_state: Dict[str, Any], area_config: Dict[str, Any]):
        """메인 영역 렌더링"""
        
        if ui_spec.board_type == "grid":
            self._render_grid_board(ui_spec, game_state, area_config)
        elif ui_spec.board_type == "card_layout":
            self._render_card_layout(ui_spec, game_state, area_config)
        elif ui_spec.board_type == "text_based":
            self._render_text_interface(ui_spec, game_state, area_config)
        elif ui_spec.board_type == "map":
            self._render_map_interface(ui_spec, game_state, area_config)
        else:
            # AI가 새로운 보드 타입을 발견한 경우
            self._render_custom_board(ui_spec, game_state, area_config)
    
    def _render_grid_board(self, ui_spec: GameUISpec, game_state: Dict[str, Any], config: Dict[str, Any]):
        """격자형 보드 렌더링 (AI 분석 기반)"""
        
        rows = config.get("rows", 8)
        cols = config.get("cols", 8)
        
        st.subheader(f"🎯 게임판 ({rows}x{cols})")
        
        # AI가 제안한 특수 기능들 적용
        if "coordinate_system" in ui_spec.special_features:
            st.info("좌표 시스템 활성화")
        
        if "move_preview" in ui_spec.special_features:
            st.info("이동 미리보기 활성화")
        
        # 실제 격자 렌더링
        board_data = game_state.get("board", [['' for _ in range(cols)] for _ in range(rows)])
        
        for i in range(rows):
            cols_ui = st.columns(cols)
            for j, col in enumerate(cols_ui):
                with col:
                    cell_value = board_data[i][j] if i < len(board_data) and j < len(board_data[i]) else ''
                    if st.button(f"{cell_value or '◯'}", key=f"cell_{i}_{j}"):
                        # AI 에이전트에게 이동 요청
                        self._handle_ai_move(ui_spec, (i, j), game_state)
    
    def _render_card_layout(self, ui_spec: GameUISpec, game_state: Dict[str, Any], config: Dict[str, Any]):
        """카드 레이아웃 렌더링 (AI 분석 기반)"""
        
        st.subheader(f"🃏 {ui_spec.game_name} 테이블")
        
        # AI가 분석한 카드 영역들 렌더링
        if "community_area" in config:
            st.write("**공용 카드 영역**")
            community_cards = game_state.get("community_cards", [])
            
            # AI가 제안한 카드 개수만큼 표시
            card_count = config.get("community_card_count", 5)
            card_cols = st.columns(card_count)
            
            for i, col in enumerate(card_cols):
                with col:
                    if i < len(community_cards):
                        st.info(f"🃏 {community_cards[i]}")
                    else:
                        st.empty()
        
        # 플레이어 배치 (AI가 분석한 최적 배치)
        player_arrangement = config.get("player_arrangement", "linear")
        if player_arrangement == "circle":
            self._render_circular_player_layout(game_state)
        else:
            self._render_linear_player_layout(game_state)
    
    def _render_text_interface(self, ui_spec: GameUISpec, game_state: Dict[str, Any], config: Dict[str, Any]):
        """텍스트 기반 인터페이스 (AI 분석 기반)"""
        
        st.subheader(f"🗣️ {ui_spec.game_name}")
        
        # AI가 분석한 게임 페이즈 시스템
        if "phase_system" in ui_spec.special_features:
            phase = game_state.get("phase", "시작")
            phase_color = self._get_phase_color(phase)
            st.markdown(f"**현재 페이즈**: :{phase_color}[{phase}]")
        
        # AI가 제안한 플레이어 정보 표시 방식
        if "player_status_table" in config:
            self._render_ai_player_table(game_state, config["player_status_table"])
        
        # AI가 분석한 투표/선택 시스템
        if "voting_system" in ui_spec.special_features:
            self._render_ai_voting_interface(game_state)
    
    def _render_custom_board(self, ui_spec: GameUISpec, game_state: Dict[str, Any], config: Dict[str, Any]):
        """AI가 새롭게 발견한 보드 타입 렌더링"""
        
        st.subheader(f"🎨 새로운 게임 타입: {ui_spec.board_type}")
        st.info("AI가 새로운 게임 패턴을 발견했습니다!")
        
        # AI가 제안한 커스텀 렌더링 방식 적용
        custom_rendering = ui_spec.special_features.get("custom_rendering", {})
        
        if custom_rendering.get("type") == "hybrid":
            # 여러 보드 타입의 하이브리드
            st.write("하이브리드 인터페이스")
            col1, col2 = st.columns(2)
            with col1:
                self._render_grid_board(ui_spec, game_state, {"rows": 4, "cols": 4})
            with col2:
                self._render_card_layout(ui_spec, game_state, {"community_area": True})
        else:
            # 완전히 새로운 인터페이스
            st.json(game_state)
            st.write("AI가 분석한 게임 구조:")
            st.json(ui_spec.special_features)
    
    def _handle_ai_move(self, ui_spec: GameUISpec, move: tuple, game_state: Dict[str, Any]):
        """AI 에이전트에게 플레이어 이동 전달"""
        
        # 실제로는 game_master와 연동
        move_request = {
            "game": ui_spec.game_name,
            "move": move,
            "current_state": game_state,
            "timestamp": datetime.now().isoformat()
        }
        
        # 세션 상태에 이동 요청 저장 (실제로는 에이전트로 전송)
        if "pending_moves" not in st.session_state:
            st.session_state.pending_moves = []
        
        st.session_state.pending_moves.append(move_request)
        st.success(f"이동 요청 전송: {move}")
    
    def _get_phase_color(self, phase: str) -> str:
        """페이즈에 따른 색상 결정"""
        phase_colors = {
            "낮": "orange",
            "밤": "blue", 
            "투표": "red",
            "토론": "green",
            "시작": "gray"
        }
        return phase_colors.get(phase, "gray")
    
    def _render_circular_player_layout(self, game_state: Dict[str, Any]):
        """원형 플레이어 배치"""
        players = game_state.get("players", [])
        if not players:
            return
            
        st.write("**플레이어 배치 (원형)**")
        
        # 원형 배치를 시뮬레이션하기 위한 그리드
        if len(players) <= 4:
            cols = st.columns(len(players))
            for i, (player, col) in enumerate(zip(players, cols)):
                with col:
                    self._render_player_card(player, i)
        else:
            # 더 많은 플레이어의 경우 2행으로 배치
            half = len(players) // 2
            
            top_cols = st.columns(half)
            for i, col in enumerate(top_cols):
                with col:
                    self._render_player_card(players[i], i)
            
            bottom_cols = st.columns(len(players) - half)
            for i, col in enumerate(bottom_cols):
                with col:
                    self._render_player_card(players[half + i], half + i)
    
    def _render_player_card(self, player: Dict[str, Any], index: int):
        """개별 플레이어 카드 렌더링"""
        name = player.get("name", f"플레이어{index+1}")
        status = player.get("status", "활성")
        
        if status == "생존" or status == "활성":
            st.success(f"**{name}**")
        else:
            st.error(f"~~{name}~~")
        
        # 추가 정보
        if "chips" in player:
            st.caption(f"칩: {player['chips']}")
        if "health" in player:
            st.caption(f"체력: {player['health']}")
    
    def _render_linear_player_layout(self, game_state: Dict[str, Any]):
        """선형 플레이어 배치"""
        players = game_state.get("players", [])
        if not players:
            return
            
        st.write("**플레이어 배치 (선형)**")
        cols = st.columns(len(players))
        for i, (player, col) in enumerate(zip(players, cols)):
            with col:
                self._render_player_card(player, i)
    
    def _render_sidebar(self, ui_spec: GameUISpec, game_state: Dict[str, Any], sidebar_config: Dict[str, Any]):
        """사이드바 렌더링"""
        st.header("🎮 게임 컨트롤")
        
        if "controls" in sidebar_config:
            st.subheader("액션")
            if st.button("패스"):
                st.session_state["action"] = "pass"
            if st.button("그만두기"):
                st.session_state["action"] = "quit"
    
    def _render_bottom_panel(self, ui_spec: GameUISpec, game_state: Dict[str, Any], panel_config: Dict[str, Any]):
        """하단 패널 렌더링"""
        if "hand_display" in panel_config:
            with st.expander("🃏 내 패", expanded=True):
                hand = game_state.get("player_hand", [])
                if hand:
                    hand_cols = st.columns(len(hand))
                    for card, col in zip(hand, hand_cols):
                        with col:
                            st.info(f"🃏 {card}")
                else:
                    st.write("패가 없습니다.")
        
        if "chat_interface" in panel_config:
            with st.expander("💬 채팅", expanded=True):
                messages = game_state.get("chat_messages", [])
                for msg in messages[-10:]:
                    st.write(f"**{msg['player']}**: {msg['message']}")
                
                chat_input = st.text_input("메시지 입력:", key="chat_input")
                if st.button("전송") and chat_input:
                    if "new_message" not in st.session_state:
                        st.session_state["new_message"] = chat_input
    
    def _render_map_interface(self, ui_spec: GameUISpec, game_state: Dict[str, Any], config: Dict[str, Any]):
        """맵 인터페이스 렌더링"""
        st.subheader(f"🗺️ {ui_spec.game_name} 맵")
        st.info("맵 기반 게임 인터페이스 (AI가 분석한 구조에 따라 생성)")
        
        # AI가 분석한 맵 구조 표시
        if "regions" in config:
            st.write(f"지역 수: {config['regions']}")
        if "connections" in config:
            st.write(f"연결 방식: {config['connections']}")
    
    def _render_ai_player_table(self, game_state: Dict[str, Any], table_config: Dict[str, Any]):
        """AI가 제안한 플레이어 테이블 렌더링"""
        import pandas as pd
        
        players = game_state.get("players", [])
        if players:
            st.write("**플레이어 상태**")
            player_df = pd.DataFrame([
                {
                    "이름": p.get("name", f"플레이어{i+1}"),
                    "상태": p.get("status", "활성"),
                    "역할": p.get("role", "???") if game_state.get("reveal_roles") else "숨김"
                }
                for i, p in enumerate(players)
            ])
            st.dataframe(player_df, use_container_width=True)
    
    def _render_ai_voting_interface(self, game_state: Dict[str, Any]):
        """AI가 분석한 투표 시스템 렌더링"""
        if game_state.get("voting_active", False):
            st.write("**투표 진행 중**")
            players = game_state.get("players", [])
            alive_players = [p for p in players if p.get("status") == "생존"]
            
            if alive_players:
                target = st.selectbox("투표할 대상:", [p["name"] for p in alive_players])
                if st.button("투표하기"):
                    st.success(f"{target}에게 투표했습니다!")
        else:
            st.info("투표 시간이 아닙니다.")

# 실제 AI 에이전트 시뮬레이션 (실제로는 LangGraph 연동)
class MockGameAnalyzer:
    """게임 분석 에이전트 모킹"""
    
    async def analyze_game(self, game_input: Dict[str, str]) -> Dict[str, Any]:
        """게임 설명을 분석하여 구조 파악"""
        
        description = game_input.get("description", "").lower()
        
        # 간단한 키워드 기반 분석 (실제로는 LLM 사용)
        if any(word in description for word in ["체스", "바둑", "격자", "칸"]):
            return {
                "game_name": "격자 게임",
                "board_type": "grid",
                "complexity": "standard",
                "confidence": 0.9
            }
        elif any(word in description for word in ["카드", "포커", "패"]):
            return {
                "game_name": "카드 게임", 
                "board_type": "card_layout",
                "complexity": "standard",
                "confidence": 0.85
            }
        elif any(word in description for word in ["마피아", "토론", "투표"]):
            return {
                "game_name": "소셜 게임",
                "board_type": "text_based", 
                "complexity": "advanced",
                "confidence": 0.8
            }
        else:
            return {
                "game_name": "새로운 게임",
                "board_type": "custom",
                "complexity": "custom",
                "confidence": 0.6
            }

class MockUIGenerator:
    """UI 생성 에이전트 모킹"""
    
    async def generate_ui_spec(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 바탕으로 UI 명세 생성"""
        
        board_type = analysis.get("board_type", "grid")
        
        if board_type == "grid":
            return {
                "components": ["turn_indicator", "action_buttons"],
                "layout": {
                    "main_area": {"rows": 8, "cols": 8},
                    "sidebar": {"controls": True}
                },
                "interactions": ["click_select", "move"],
                "complexity": "standard",
                "special_features": {"coordinate_system": True}
            }
        elif board_type == "card_layout":
            return {
                "components": ["player_hand", "community_area", "action_buttons"],
                "layout": {
                    "main_area": {"community_area": True, "player_arrangement": "circle"},
                    "bottom_panel": {"hand_display": True}
                },
                "interactions": ["card_select", "betting"],
                "complexity": "standard", 
                "special_features": {"chip_tracking": True}
            }
        elif board_type == "text_based":
            return {
                "components": ["chat", "player_list", "voting"],
                "layout": {
                    "main_area": {"player_status_table": True},
                    "bottom_panel": {"chat_interface": True}
                },
                "interactions": ["text_input", "voting"],
                "complexity": "advanced",
                "special_features": {"phase_system": True, "voting_system": True}
            }
        else:
            return {
                "components": ["custom_display"],
                "layout": {"main_area": {"custom": True}},
                "interactions": ["custom"],
                "complexity": "custom",
                "special_features": {"custom_rendering": {"type": "hybrid"}}
            } 