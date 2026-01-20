"""
Real-time Game Dashboard

LLM 게임을 위한 실시간 웹 대시보드
Streamlit 기반 UI로 게임 테이블 생성, 참여, 플레이 지원
"""

import streamlit as st
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# Streamlit 페이지 설정
st.set_page_config(
    page_title="Table Game Mate - LLM Gaming",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


class GameDashboard:
    """게임 대시보드"""
    
    def __init__(self):
        self.state_manager = None
        self.game_tables = {}
        self._init_session_state()
    
    def _init_session_state(self):
        """세션 상태 초기화"""
        if "tables" not in st.session_state:
            st.session_state.tables = {}
        if "current_table" not in st.session_state:
            st.session_state.current_table = None
        if "player_id" not in st.session_state:
            st.session_state.player_id = f"player_{datetime.now().strftime('%H%M%S')}"
        if "game_history" not in st.session_state:
            st.session_state.game_history = []
    
    def render(self):
        """대시보드 렌더링"""
        st.title("🎮 Table Game Mate - LLM Gaming")
        st.markdown("### 실시간 멀티플레이어 LLM 보드게임")
        
        # 사이드바
        self._render_sidebar()
        
        # 메인 컨텐츠
        tab1, tab2, tab3 = st.tabs(["🎯 게임 테이블", "📊 게임 결과", "⚙️ 설정"])
        
        with tab1:
            self._render_game_tables()
        
        with tab2:
            self._render_game_results()
        
        with tab3:
            self._render_settings()
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("🎛️ 제어판")
            
            # 플레이어 정보
            st.subheader("👤 플레이어")
            player_name = st.text_input(
                "이름",
                value=f"Player_{st.session_state.player_id.split('_')[1]}",
                key="player_name_input"
            )
            st.session_state.player_id = f"player_{player_name.lower().replace(' ', '_')}"
            
            # LLM 설정
            st.subheader("🤖 LLM 설정")
            llm_provider = st.selectbox(
                "LLM 제공자",
                ["google", "openai", "anthropic"],
                index=0
            )
            
            llm_model = st.text_input(
                "모델",
                value="gemini-2.5-flash-lite",
                key="llm_model_input"
            )
            
            st.divider()
            
            # 새로고침
            if st.button("🔄 새로고침", use_container_width=True):
                st.rerun()
            
            # 시스템 상태
            st.subheader("📊 시스템 상태")
            st.success("✅ 시스템 정상")
    
    def _render_game_tables(self):
        """게임 테이블 탭 렌더링"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🆕 새 게임 테이블 생성")
            
            # 게임 선택
            game_type = st.selectbox(
                "게임 선택",
                ["Chess", "체스", "Go", "바둑", "Poker", "포커"],
                index=0
            )
            
            # BGG ID (선택사항)
            bgg_id = st.number_input(
                "BGG 게임 ID (선택)",
                min_value=0,
                value=0,
                step=1,
                help="BoardGameGeek의 게임 ID를 입력하면 규칙을 자동으로 가져옵니다"
            )
            
            # 플레이어 수
            max_players = st.slider("최대 플레이어", 2, 8, 2)
            
            # 테이블 생성 버튼
            if st.button("🎮 테이블 생성", use_container_width=True):
                self._create_table(game_type, bgg_id if bgg_id > 0 else None, max_players)
                st.rerun()
        
        with col2:
            st.subheader("📋 참여 가능한 테이블")
            
            tables = st.session_state.get("tables", {})
            
            if not tables:
                st.info("생성된 테이블이 없습니다. 위에서 새 테이블을 생성하세요.")
            else:
                for table_id, table_data in tables.items():
                    with st.expander(f"🎯 {table_data['game_type']} - {table_data['player_count']}/{table_data['max_players']} 플레이어"):
                        st.write(f"**테이블 ID**: {table_id}")
                        st.write(f"**게임**: {table_data['game_type']}")
                        st.write(f"**플레이어**: {table_data['player_count']}/{table_data['max_players']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("참여", key=f"join_{table_id}", disabled=table_data['player_count'] >= table_data['max_players']):
                                self._join_table(table_id)
                                st.rerun()
                        
                        with col_b:
                            if st.button("게임 시작", key=f"start_{table_id}", disabled=table_data['player_count'] < 2):
                                self._start_game(table_id)
                                st.rerun()
    
    def _render_active_game(self, table_id: str):
        """진행 중인 게임 렌더링"""
        table_data = st.session_state.tables.get(table_id, {})
        game_state = table_data.get("game_state", {})
        
        st.divider()
        st.subheader(f"🎯 진행 중인 게임: {table_data.get('game_type', 'Unknown')}")
        
        # 현재 턴 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("턴", game_state.get("turn_number", 0))
        with col2:
            st.metric("현재 플레이어", game_state.get("current_player", ""))
        with col3:
            st.metric("총 움직임", len(game_state.get("move_history", [])))
        
        # 보드 상태 렌더링
        board_state = game_state.get("board_state", {})
        self._render_board(board_state, table_data.get("game_type", ""))
        
        # 합법적인 움직임
        legal_moves = game_state.get("legal_moves", [])
        if legal_moves:
            st.subheader("🎯 가능한 움직임")
            move_cols = st.columns(len(legal_moves))
            for i, move in enumerate(legal_moves):
                with move_cols[i]:
                    if st.button(move, key=f"move_{table_id}_{move}"):
                        self._submit_move(table_id, move, {})
        
        # 움직임 히스토리
        with st.expander("📜 움직임 히스토리"):
            for move in game_state.get("move_history", []):
                st.write(f"- {move.get('player_id', 'Unknown')}: {move.get('move_type', 'Unknown')}")
        
        # 게임 종료
        if game_state.get("game_status") == "completed":
            st.success(f"🎉 게임 종료! 승자: {game_state.get('winner_id', 'Unknown')}")
            
            if st.button("테이블로 돌아가기"):
                st.session_state.current_table = None
                st.rerun()
    
    def _render_board(self, board_state: Dict, game_type: str):
        """보드 상태 렌더링"""
        if game_type.lower() in ["chess", "체스"]:
            self._render_chess_board(board_state)
        else:
            st.json(board_state)
    
    def _render_chess_board(self, board_state: Dict):
        """체스 보드 렌더링"""
        board = board_state.get("board", {})
        
        # 8x8 체스보드
        rows = ["8", "7", "6", "5", "4", "3", "2", "1"]
        cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
        
        for row_idx, row in enumerate(rows):
            cols_list = st.columns(8)
            for col_idx, col in enumerate(cols):
                pos = f"{col}{row}"
                piece = board.get(pos, {})
                piece_symbol = self._get_chess_piece_symbol(piece)
                
                with cols_list[col_idx]:
                    is_white_square = (row_idx + col_idx) % 2 == 0
                    bg_color = "#F0D9B5" if is_white_square else "#B58863"
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {bg_color};
                            padding: 10px;
                            text-align: center;
                            font-size: 24px;
                            border-radius: 5px;
                        ">{piece_symbol}</div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def _get_chess_piece_symbol(self, piece: Dict) -> str:
        """체스 피스 심볼 반환"""
        if not piece:
            return ""
        
        symbols = {
            ("king", "white"): "♔",
            ("queen", "white"): "♕",
            ("rook", "white"): "♖",
            ("bishop", "white"): "♗",
            ("knight", "white"): "♘",
            ("pawn", "white"): "♙",
            ("king", "black"): "♚",
            ("queen", "black"): "♛",
            ("rook", "black"): "♜",
            ("bishop", "black"): "♝",
            ("knight", "black"): "♞",
            ("pawn", "black"): "♟",
        }
        
        return symbols.get((piece.get("piece", ""), piece.get("color", "")), "")
    
    def _render_game_results(self):
        """게임 결과 탭 렌더링"""
        st.subheader("📊 게임 결과")
        
        history = st.session_state.get("game_history", [])
        
        if not history:
            st.info("플레이한 게임이 없습니다.")
        else:
            for game in history:
                with st.expander(f"🎮 {game.get('game_type', 'Unknown')} - {game.get('date', '')}"):
                    st.write(f"**승자**: {game.get('winner', 'Unknown')}")
                    st.write(f"**총 움직임**: {game.get('total_moves', 0)}")
                    st.write(f"**플레이어**: {', '.join(game.get('players', []))}")
    
    def _render_settings(self):
        """설정 탭 렌더링"""
        st.subheader("⚙️ 시스템 설정")
        
        # API 키 설정
        st.text_input("Google API Key", type="password", key="google_api_key")
        st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        st.text_input("Anthropic API Key", type="password", key="anthropic_api_key")
        
        st.divider()
        
        # 게임 설정
        st.subheader("🎮 게임 기본 설정")
        default_game = st.selectbox("기본 게임", ["Chess", "체스", "Go", "바둑", "Poker", "포커"])
        default_players = st.slider("기본 플레이어 수", 2, 8, 2)
        
        if st.button("설정 저장"):
            st.success("설정이 저장되었습니다!")
    
    def _create_table(self, game_type: str, bgg_id: Optional[int], max_players: int):
        """테이블 생성"""
        import uuid
        
        table_id = f"table_{uuid.uuid4().hex[:8]}"
        
        st.session_state.tables[table_id] = {
            "table_id": table_id,
            "game_type": game_type,
            "bgg_id": bgg_id,
            "max_players": max_players,
            "player_count": 1,
            "players": [st.session_state.player_id],
            "game_state": {
                "turn_number": 0,
                "current_player": st.session_state.player_id,
                "board_state": {},
                "legal_moves": [],
                "move_history": [],
                "game_status": "waiting"
            },
            "created_at": datetime.now().isoformat()
        }
        
        st.success(f"테이블 생성 완료! (ID: {table_id})")
    
    def _join_table(self, table_id: str):
        """테이블 참여"""
        tables = st.session_state.tables
        
        if table_id in tables:
            table = tables[table_id]
            
            if st.session_state.player_id not in table["players"]:
                table["players"].append(st.session_state.player_id)
                table["player_count"] += 1
                
                st.session_state.current_table = table_id
                st.success(f"{table_id} 테이블에 참여했습니다!")
    
    def _start_game(self, table_id: str):
        """게임 시작"""
        tables = st.session_state.tables
        
        if table_id in tables:
            table = tables[table_id]
            table["game_state"]["game_status"] = "in_progress"
            table["game_state"]["turn_number"] = 1
            table["game_state"]["current_player"] = table["players"][0]
            
            # 기본 보드 상태 설정
            if table["game_type"].lower() in ["chess", "체스"]:
                table["game_state"]["board_state"] = self._create_initial_chess_board()
                table["game_state"]["legal_moves"] = ["MOVE_PIECE", "CASTLE", "CAPTURE"]
            
            st.session_state.current_table = table_id
            st.success("게임 시작!")
    
    def _create_initial_chess_board(self) -> Dict:
        """초기 체스 보드 상태 생성"""
        board = {}
        
        for col in range(8):
            board[f"a{col + 1}"] = {"piece": "pawn", "color": "white"}
            board[f"a{col + 8}"] = {"piece": "pawn", "color": "black"}
        
        pieces = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
        for i, piece in enumerate(pieces):
            board[f"{chr(97 + i)}1"] = {"piece": piece, "color": "white"}
            board[f"{chr(97 + i)}8"] = {"piece": piece, "color": "black"}
        
        return board
    
    def _submit_move(self, table_id: str, move_type: str, move_data: Dict):
        """움직임 제출"""
        tables = st.session_state.tables
        
        if table_id in tables:
            table = tables[table_id]
            game_state = table["game_state"]
            
            # 움직임 기록
            move_record = {
                "player_id": st.session_state.player_id,
                "move_type": move_type,
                "move_data": move_data,
                "timestamp": datetime.now().isoformat()
            }
            
            game_state["move_history"].append(move_record)
            
            # 다음 플레이어로 턴 전환
            players = table["players"]
            current_idx = players.index(game_state["current_player"])
            next_idx = (current_idx + 1) % len(players)
            game_state["current_player"] = players[next_idx]
            game_state["turn_number"] += 1
            
            st.success(f"움직임 적용: {move_type}")
            st.rerun()
    
    def run(self):
        """대시보드 실행"""
        self.render()


def main():
    """메인 함수"""
    dashboard = GameDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
