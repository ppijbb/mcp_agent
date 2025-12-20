#!/usr/bin/env python3
"""
게임 시뮬레이션을 위한 A2A 실시간 통신 시스템
표준 A2A 헬퍼를 사용하여 실시간 진행 상황 표시
"""

import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

# 페이지 설정
st.set_page_config(page_title="🎮 게임 시뮬레이션", page_icon="🎮", layout="wide")

class GameSimulationUI:
    """게임 시뮬레이션 UI - 표준 A2A 방식"""
    
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
    
    def execute_game_action(self, action_type: str, action_data: Dict[str, Any]):
        """게임 액션을 표준 A2A 방식으로 실행"""
        
        # 결과 저장 경로 설정
        reports_path = Path(get_reports_path('game_simulation'))
        reports_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_json_path = reports_path / f"game_action_{action_type}_{timestamp}.json"
        
        # 입력 파라미터 준비
        input_params = {
            "action_type": action_type,
            "action_data": action_data,
            "game_id": st.session_state.game_state.get("game_id"),
            "game_state": st.session_state.game_state
        }
        
        # 결과 표시용 placeholder 생성
        result_placeholder = st.empty()
        
        # 표준화된 방식으로 agent 실행
        result = execute_standard_agent_via_a2a(
            placeholder=result_placeholder,
            agent_id=f"game_simulation_agent_{action_type}",
            agent_name=f"Game Simulation Agent ({action_type})",
            agent_type=AgentType.MCP_AGENT,
            entry_point="srcs.game_agents.game_simulation_agent",
            capabilities=["game_simulation", "real_time_gameplay", "state_management"],
            description="게임 시뮬레이션 및 실시간 상태 관리",
            input_params=input_params,
            result_json_path=result_json_path,
            use_a2a=True
        )
        
        if result and result.get("success"):
            # 게임 상태 업데이트
            result_data = result.get("data", {})
            if "game_state" in result_data:
                st.session_state.game_state.update(result_data["game_state"])
            
            st.success(f"✅ {action_type} 액션이 성공적으로 실행되었습니다!")
            return True
        elif result and result.get("error"):
            st.error(f"❌ 액션 실행 실패: {result.get('error')}")
            return False
        
        return False
    
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
            st.json(board_state)
        
        # 플레이어 손패
        st.subheader("🃏 내 손패")
        hand = game_state.get('hand', [])
        if hand:
            cols = st.columns(min(len(hand), 6))
            for i, card in enumerate(hand):
                with cols[i % len(cols)]:
                    if st.button(f"카드 {i+1}", key=f"card_{i}"):
                        self.execute_game_action("play_card", {"card_index": i})
                        st.rerun()
        else:
            st.info("손패가 없습니다.")
        
        # 액션 버튼
        st.subheader("⚡ 액션")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🃏 카드 뽑기", use_container_width=True):
                self.execute_game_action("draw_card", {})
                st.rerun()
        
        with col2:
            if st.button("⏭️ 턴 종료", use_container_width=True):
                self.execute_game_action("end_turn", {})
                st.rerun()
        
        with col3:
            if st.button("🔄 게임 상태 새로고침", use_container_width=True):
                self.execute_game_action("get_state", {})
                st.rerun()
        
        with col4:
            if st.button("❌ 게임 종료", use_container_width=True):
                self.execute_game_action("end_game", {})
                st.session_state.game_state["game_id"] = None
                st.rerun()
        
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
            # 게임 초기화
            game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.game_state.update({
                "game_id": game_id,
                "game_name": game_name,
                "players": [f"Player {i+1}" for i in range(player_count)],
                "game_phase": "setup"
            })
            
            # 게임 초기화 액션 실행
            self.execute_game_action("init_game", {
                "game_name": game_name,
                "player_count": player_count
            })
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
