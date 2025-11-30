#!/usr/bin/env python3
"""
게임 시뮬레이션을 위한 A2A 실시간 통신 시스템
A2A를 통해 게임 액션을 전송하고 게임 상태를 실시간으로 업데이트
"""

import streamlit as st
import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.streamlit_a2a_runner import send_a2a_message
from srcs.common.a2a_integration import get_global_registry, get_global_broker, A2AMessage, MessagePriority

# 페이지 설정
st.set_page_config(page_title="🎮 게임 시뮬레이션", page_icon="🎮", layout="wide")

class GameSimulationUI:
    """게임 시뮬레이션 UI - A2A를 통한 실시간 게임 플레이"""
    
    def __init__(self):
        if "game_state" not in st.session_state:
            st.session_state.game_state = {
                "game_id": None,
                "players": [],
                "current_turn": 0,
                "game_phase": "waiting",
                "board_state": {},
                "hand": [],
                "last_action": None
            }
        if "game_agent_id" not in st.session_state:
            st.session_state.game_agent_id = None
    
    async def send_game_action(self, action_type: str, action_data: Dict[str, Any]) -> bool:
        """게임 액션을 A2A로 전송"""
        if not st.session_state.game_agent_id:
            st.error("게임 agent가 연결되지 않았습니다.")
            return False
        
        try:
            # game_action 메시지 타입으로 전송
            success = send_a2a_message(
                source_agent_id="streamlit_ui",
                target_agent_id=st.session_state.game_agent_id,
                message_type="game_action",
                payload={
                    "action_type": action_type,
                    "action_data": action_data,
                    "game_id": st.session_state.game_state.get("game_id"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            return success
        except Exception as e:
            logger.error(f"게임 액션 전송 실패: {e}", exc_info=True)
            return False
    
    async def check_game_state_updates(self):
        """게임 상태 업데이트 확인 (폴링 방식)"""
        # A2A 메시지 큐에서 game_state_update 메시지 확인
        # 실제 구현은 streamlit_a2a_runner의 메시지 핸들러 활용
        pass
    
    def render_game_board(self):
        """게임 보드 렌더링"""
        game_state = st.session_state.game_state
        
        st.header(f"🎮 {game_state.get('game_name', '게임')} - 실시간 플레이")
        
        # 게임 상태 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("현재 턴", f"플레이어 {game_state.get('current_turn', 0) + 1}")
        with col2:
            st.metric("게임 단계", game_state.get('game_phase', 'waiting'))
        with col3:
            st.metric("플레이어 수", len(game_state.get('players', [])))
        
        # 게임 보드
        st.subheader("🎲 게임 보드")
        board_state = game_state.get('board_state', {})
        if board_state:
            st.json(board_state)  # 임시로 JSON 표시, 나중에 UI 명세서 기반으로 렌더링
        
        # 플레이어 손패
        st.subheader("🃏 내 손패")
        hand = game_state.get('hand', [])
        if hand:
            cols = st.columns(min(len(hand), 6))
            for i, card in enumerate(hand):
                with cols[i % len(cols)]:
                    if st.button(f"카드 {i+1}", key=f"card_{i}"):
                        # 카드 플레이 액션 전송
                        asyncio.run(self.send_game_action("play_card", {"card_index": i}))
        else:
            st.info("손패가 없습니다.")
        
        # 액션 버튼
        st.subheader("⚡ 액션")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🃏 카드 뽑기", use_container_width=True):
                asyncio.run(self.send_game_action("draw_card", {}))
        
        with col2:
            if st.button("⏭️ 턴 종료", use_container_width=True):
                asyncio.run(self.send_game_action("end_turn", {}))
        
        with col3:
            if st.button("🔄 게임 상태 새로고침", use_container_width=True):
                asyncio.run(self.send_game_action("get_state", {}))
        
        with col4:
            if st.button("❌ 게임 종료", use_container_width=True):
                asyncio.run(self.send_game_action("end_game", {}))
        
        # 마지막 액션 표시
        last_action = game_state.get('last_action')
        if last_action:
            st.info(f"마지막 액션: {last_action}")
    
    def render_game_setup(self):
        """게임 설정 UI"""
        st.header("🎮 게임 시뮬레이션 시작")
        
        game_name = st.text_input("게임 이름", value="BANG!")
        player_count = st.number_input("플레이어 수", min_value=2, max_value=8, value=4)
        
        if st.button("🎮 게임 시작", type="primary"):
            # 게임 초기화 액션 전송
            asyncio.run(self.send_game_action("init_game", {
                "game_name": game_name,
                "player_count": player_count
            }))
            st.session_state.game_state["game_id"] = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()
    
    def render_main(self):
        """메인 UI 렌더링"""
        game_state = st.session_state.game_state
        
        if game_state.get("game_id"):
            self.render_game_board()
        else:
            self.render_game_setup()

def main():
    ui = GameSimulationUI()
    ui.render_main()

if __name__ == "__main__":
    main()

