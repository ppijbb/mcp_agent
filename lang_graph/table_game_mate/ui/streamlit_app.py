#!/usr/bin/env python3
"""
테이블게임 메이트 - 범용 보드게임 UI 프로토타입
Streamlit을 사용한 동적 게임 인터페이스 시스템
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="테이블게임 메이트",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BoardType(Enum):
    GRID = "grid"
    CARD_LAYOUT = "card_layout"
    MAP = "map"
    FREE_FORM = "free_form"
    TEXT_BASED = "text_based"

class ComponentType(Enum):
    PLAYER_HAND = "player_hand"
    SCORE_BOARD = "score_board"
    ACTION_BUTTONS = "action_buttons"
    RESOURCE_TRACKER = "resource_tracker"
    TURN_INDICATOR = "turn_indicator"
    CHAT = "chat"

@dataclass
class GameConfig:
    """게임 설정 클래스"""
    name: str
    board_type: BoardType
    max_players: int
    components: List[ComponentType]
    special_rules: Dict[str, Any]
    board_config: Dict[str, Any]

class GameUIRenderer:
    """동적 게임 UI 렌더러"""
    
    def __init__(self):
        self.game_configs = self._load_game_configs()
    
    def _load_game_configs(self) -> Dict[str, GameConfig]:
        """게임 설정들을 로드"""
        return {
            "틱택토": GameConfig(
                name="틱택토",
                board_type=BoardType.GRID,
                max_players=2,
                components=[ComponentType.TURN_INDICATOR, ComponentType.ACTION_BUTTONS],
                special_rules={"win_condition": "3_in_row"},
                board_config={"rows": 3, "cols": 3}
            ),
            "체스": GameConfig(
                name="체스",
                board_type=BoardType.GRID,
                max_players=2,
                components=[ComponentType.TURN_INDICATOR, ComponentType.ACTION_BUTTONS],
                special_rules={"pieces": ["king", "queen", "rook", "bishop", "knight", "pawn"]},
                board_config={"rows": 8, "cols": 8, "alternating_colors": True}
            ),
            "포커": GameConfig(
                name="포커",
                board_type=BoardType.CARD_LAYOUT,
                max_players=8,
                components=[ComponentType.PLAYER_HAND, ComponentType.SCORE_BOARD, ComponentType.ACTION_BUTTONS],
                special_rules={"betting_rounds": 4, "hand_size": 2},
                board_config={"community_cards": 5, "deck_size": 52}
            ),
            "마피아": GameConfig(
                name="마피아",
                board_type=BoardType.TEXT_BASED,
                max_players=12,
                components=[ComponentType.CHAT, ComponentType.ACTION_BUTTONS, ComponentType.TURN_INDICATOR],
                special_rules={"day_night_cycle": True, "voting_system": True},
                board_config={"roles": ["마피아", "시민", "의사", "경찰"]}
            ),
            "방": GameConfig(
                name="방(Bang)",
                board_type=BoardType.CARD_LAYOUT,
                max_players=7,
                components=[ComponentType.PLAYER_HAND, ComponentType.RESOURCE_TRACKER, ComponentType.ACTION_BUTTONS],
                special_rules={"role_cards": True, "distance_system": True},
                board_config={"character_abilities": True, "weapon_cards": True}
            )
        }
    
    def render_board(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """게임판 렌더링"""
        if game_config.board_type == BoardType.GRID:
            self._render_grid_board(game_config, game_state)
        elif game_config.board_type == BoardType.CARD_LAYOUT:
            self._render_card_layout(game_config, game_state)
        elif game_config.board_type == BoardType.TEXT_BASED:
            self._render_text_based(game_config, game_state)
        elif game_config.board_type == BoardType.MAP:
            self._render_map_board(game_config, game_state)
    
    def _render_grid_board(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """격자형 게임판 렌더링"""
        rows = game_config.board_config.get("rows", 3)
        cols = game_config.board_config.get("cols", 3)
        
        st.subheader(f"🎯 {game_config.name} 게임판")
        
        # 격자 생성
        board_data = game_state.get("board", [['' for _ in range(cols)] for _ in range(rows)])
        
        # 체스판 스타일 시각화
        if game_config.board_config.get("alternating_colors"):
            fig = go.Figure()
            
            # 체스판 색칠
            for i in range(rows):
                for j in range(cols):
                    color = "lightgray" if (i + j) % 2 == 0 else "white"
                    fig.add_shape(
                        type="rect",
                        x0=j, x1=j+1, y0=i, y1=i+1,
                        fillcolor=color,
                        line=dict(color="black", width=1)
                    )
                    
                    # 말 표시
                    piece = board_data[i][j]
                    if piece:
                        fig.add_annotation(
                            x=j+0.5, y=i+0.5,
                            text=piece,
                            showarrow=False,
                            font=dict(size=20)
                        )
            
            fig.update_layout(
                xaxis=dict(range=[0, cols], showgrid=False, showticklabels=False),
                yaxis=dict(range=[0, rows], showgrid=False, showticklabels=False),
                width=400,
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig)
        else:
            # 간단한 격자 (틱택토용)
            cols_ui = st.columns(cols)
            for i in range(rows):
                for j, col in enumerate(cols_ui):
                    with col:
                        if st.button(f"{board_data[i][j] or '◯'}", key=f"cell_{i}_{j}"):
                            st.session_state[f"move"] = (i, j)
    
    def _render_card_layout(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """카드 기반 게임판 렌더링"""
        st.subheader(f"🃏 {game_config.name} 테이블")
        
        # 커뮤니티 카드 영역 (포커의 경우)
        if "community_cards" in game_config.board_config:
            st.write("**커뮤니티 카드**")
            community_cards = game_state.get("community_cards", [])
            card_cols = st.columns(5)
            for i, col in enumerate(card_cols):
                with col:
                    if i < len(community_cards):
                        st.info(f"🃏 {community_cards[i]}")
                    else:
                        st.empty()
        
        # 플레이어 위치 시각화
        players = game_state.get("players", [])
        if players:
            st.write("**플레이어 배치**")
            player_cols = st.columns(len(players))
            for i, (player, col) in enumerate(zip(players, player_cols)):
                with col:
                    st.metric(
                        label=f"플레이어 {i+1}",
                        value=player.get("name", f"P{i+1}"),
                        delta=f"칩: {player.get('chips', 0)}"
                    )
    
    def _render_text_based(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """텍스트 기반 게임 (마피아) 렌더링"""
        st.subheader(f"🗣️ {game_config.name} 게임")
        
        # 게임 페이즈 표시
        phase = game_state.get("phase", "준비")
        st.info(f"현재 페이즈: **{phase}**")
        
        # 플레이어 상태
        players = game_state.get("players", [])
        if players:
            st.write("**플레이어 상태**")
            player_df = pd.DataFrame([
                {
                    "이름": p.get("name", f"플레이어{i+1}"),
                    "상태": p.get("status", "생존"),
                    "역할": p.get("role", "???") if game_state.get("reveal_roles") else "숨김"
                }
                for i, p in enumerate(players)
            ])
            st.dataframe(player_df, use_container_width=True)
    
    def _render_map_board(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """맵 기반 게임판 렌더링"""
        st.subheader(f"🗺️ {game_config.name} 맵")
        st.info("맵 기반 게임은 추후 구현 예정")
    
    def render_components(self, game_config: GameConfig, game_state: Dict[str, Any]):
        """게임 컴포넌트들 렌더링"""
        
        # 사이드바에 게임 컨트롤
        with st.sidebar:
            st.header("🎮 게임 컨트롤")
            
            if ComponentType.TURN_INDICATOR in game_config.components:
                current_player = game_state.get("current_player", 0)
                st.success(f"현재 턴: 플레이어 {current_player + 1}")
            
            if ComponentType.ACTION_BUTTONS in game_config.components:
                st.subheader("액션")
                if st.button("패스"):
                    st.session_state["action"] = "pass"
                if st.button("그만두기"):
                    st.session_state["action"] = "quit"
        
        # 메인 영역 하단에 추가 컴포넌트들
        if ComponentType.PLAYER_HAND in game_config.components:
            with st.expander("🃏 내 패", expanded=True):
                hand = game_state.get("player_hand", [])
                if hand:
                    hand_cols = st.columns(len(hand))
                    for card, col in zip(hand, hand_cols):
                        with col:
                            st.info(f"🃏 {card}")
                else:
                    st.write("패가 없습니다.")
        
        if ComponentType.SCORE_BOARD in game_config.components:
            with st.expander("📊 점수판", expanded=False):
                scores = game_state.get("scores", {})
                if scores:
                    score_df = pd.DataFrame(list(scores.items()), columns=["플레이어", "점수"])
                    st.bar_chart(score_df.set_index("플레이어"))
        
        if ComponentType.CHAT in game_config.components:
            with st.expander("💬 채팅", expanded=True):
                messages = game_state.get("chat_messages", [])
                for msg in messages[-10:]:  # 최근 10개 메시지만
                    st.write(f"**{msg['player']}**: {msg['message']}")
                
                # 채팅 입력
                if "chat_input" not in st.session_state:
                    st.session_state.chat_input = ""
                
                chat_input = st.text_input("메시지 입력:", key="chat_input")
                if st.button("전송") and chat_input:
                    if "new_message" not in st.session_state:
                        st.session_state["new_message"] = chat_input
                    st.rerun()

def create_sample_game_state(game_name: str) -> Dict[str, Any]:
    """샘플 게임 상태 생성"""
    if game_name == "틱택토":
        return {
            "board": [['X', '', 'O'], ['', 'X', ''], ['O', '', '']],
            "current_player": 0,
            "scores": {"플레이어1": 2, "플레이어2": 1}
        }
    elif game_name == "포커":
        return {
            "community_cards": ["A♠", "K♦", "Q♣"],
            "players": [
                {"name": "플레이어1", "chips": 1500},
                {"name": "플레이어2", "chips": 1200},
                {"name": "플레이어3", "chips": 800}
            ],
            "player_hand": ["10♠", "J♠"],
            "current_player": 0
        }
    elif game_name == "마피아":
        return {
            "phase": "낮 토론",
            "players": [
                {"name": "알리스", "status": "생존", "role": "시민"},
                {"name": "밥", "status": "생존", "role": "마피아"},
                {"name": "찰리", "status": "사망", "role": "의사"},
                {"name": "다이애나", "status": "생존", "role": "경찰"}
            ],
            "chat_messages": [
                {"player": "알리스", "message": "밥이 수상해 보여요"},
                {"player": "밥", "message": "저는 결백합니다!"},
                {"player": "다이애나", "message": "증거가 있나요?"}
            ],
            "reveal_roles": False
        }
    else:
        return {"current_player": 0, "players": []}

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.title("🎲 테이블게임 메이트")
    st.markdown("**다양한 보드게임을 위한 범용 인터페이스 프로토타입**")
    
    # 게임 선택
    renderer = GameUIRenderer()
    
    with st.sidebar:
        st.header("⚙️ 게임 설정")
        selected_game = st.selectbox(
            "게임 선택:",
            options=list(renderer.game_configs.keys()),
            index=0
        )
        
        game_config = renderer.game_configs[selected_game]
        
        # 게임 정보 표시
        with st.expander("게임 정보"):
            st.write(f"**타입**: {game_config.board_type.value}")
            st.write(f"**최대 플레이어**: {game_config.max_players}")
            st.write(f"**컴포넌트**: {len(game_config.components)}개")
    
    # 메인 게임 영역
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 게임 상태 생성/로드
        if f"game_state_{selected_game}" not in st.session_state:
            st.session_state[f"game_state_{selected_game}"] = create_sample_game_state(selected_game)
        
        game_state = st.session_state[f"game_state_{selected_game}"]
        
        # 게임판 렌더링
        renderer.render_board(game_config, game_state)
    
    with col2:
        st.subheader("🎯 게임 상태")
        st.json(game_state, expanded=False)
    
    # 게임 컴포넌트 렌더링
    renderer.render_components(game_config, game_state)
    
    # 디버그 정보
    with st.expander("🔧 개발자 도구", expanded=False):
        st.write("**세션 상태:**")
        st.json(dict(st.session_state))
        
        if st.button("게임 상태 초기화"):
            st.session_state[f"game_state_{selected_game}"] = create_sample_game_state(selected_game)
            st.rerun()

if __name__ == "__main__":
    main() 