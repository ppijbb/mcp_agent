#!/usr/bin/env python3
"""
진짜 LangGraph 에이전트 연동 UI 시스템
실제 AI 에이전트가 게임을 분석하고 UI를 생성하는 시스템
"""

import streamlit as st
import asyncio
import sys
import os
from typing import Dict, Any
from datetime import datetime

# 상위 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 실제 LangGraph 에이전트 import
from agents.game_ui_analyzer import get_game_ui_analyzer

# 페이지 설정
st.set_page_config(page_title="🤖 Agent-driven UI", page_icon="🤖", layout="wide")

class RealLangGraphUI:
    """실제 LangGraph 에이전트 기반 UI 시스템"""
    
    def __init__(self):
        if "ui_analyzer" not in st.session_state:
            with st.spinner("LangGraph 에이전트 초기화 중..."):
                try:
                    st.session_state.ui_analyzer = get_game_ui_analyzer()
                except Exception as e:
                    st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
                    st.session_state.ui_analyzer = None
        
        # 세션 상태 초기화
        for key, default in {
            "generated_games": {},
            "current_game_id": None,
            "analysis_log": [],
            "analysis_steps": [],
            "analysis_in_progress": False,
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def render_sidebar(self):
        """사이드바는 사용하지 않음."""
        pass

    def render_game_creator(self):
        st.subheader("1. AI에게 분석을 요청할 게임 설명하기")
        game_description = st.text_area(
            "어떤 보드게임을 플레이하고 싶으신가요? 자유롭게 설명해주세요.", 
            placeholder="예시: 친구들과 할 수 있는 마피아 같은 심리 게임인데, 너무 무겁지 않고 간단하게 한 판 할 수 있는 거 없을까? 서로 속이고 정체를 밝혀내는 요소가 있었으면 좋겠어.", 
            height=150
        )
        
        if st.button("🧠 이 설명으로 UI 생성 분석 요청", type="primary", use_container_width=True, disabled=st.session_state.analysis_in_progress or not st.session_state.ui_analyzer):
            if game_description.strip():
                game_id = f"game_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.session_state.current_game_id = game_id
                st.session_state.analysis_in_progress = True
                st.session_state.analysis_steps = []
                st.session_state.generated_games[game_id] = {"name": "분석 중...", "description": game_description, "rules": ""}
                st.rerun()
            else:
                st.error("게임 설명을 입력해주세요!")

    def render_generated_games_list(self):
        st.subheader("2. 분석된 게임 목록")
        if not st.session_state.generated_games:
            st.info("아직 분석된 게임이 없습니다.")
            return

        for game_id, game_info in st.session_state.generated_games.items():
            name = game_info.get('name', '이름 없음')
            col_name, col_button = st.columns([4, 1])
            col_name.write(f"🎮 **{name}**")
            if col_button.button("결과 보기", key=f"load_{game_id}", use_container_width=True):
                st.session_state.current_game_id = game_id
                st.session_state.analysis_in_progress = False
                st.rerun()

    async def run_analysis_and_stream_results(self):
        game_id = st.session_state.current_game_id
        game_info = st.session_state.generated_games[game_id]
        
        st.subheader(f"🧠 '{game_info['name']}' 분석 진행 중...")
        status_placeholder = st.empty()
        steps_container = st.container(border=True)
        final_result = {}

        try:
            agent_app = st.session_state.ui_analyzer.app
            input_data = {"game_description": game_info["description"], "detailed_rules": game_info.get("rules", ""), "messages": []}
            
            async for chunk in agent_app.astream(input_data):
                node_name = list(chunk.keys())[0]
                node_output = list(chunk.values())[0]
                
                status_placeholder.info(f"⏳ 현재 단계: **{node_name}**")
                with steps_container:
                    with st.expander(f"단계: **{node_name}** - 출력 확인", expanded=True):
                        st.json(node_output)
                
                final_result = node_output

            ui_spec = final_result.get("ui_spec", {})
            analysis_result = {
                "id": game_id,
                "success": "error_message" not in final_result or not final_result["error_message"],
                "name": ui_spec.get("game_name", "분석 완료"),
                "board_type": ui_spec.get("board_type", "unknown"),
                "confidence": final_result.get("confidence_score", 0.0),
                "full_spec": ui_spec,
                "analysis_summary": final_result.get("analysis_result", {}),
                "error_message": final_result.get("error_message", ""),
            }
            st.session_state.generated_games[game_id].update(analysis_result)
            st.session_state.analysis_log.append(analysis_result)

        except Exception as e:
            st.error(f"❌ 분석 중 심각한 오류 발생: {e}")
            st.session_state.generated_games[game_id].update({"success": False, "error_message": str(e)})
        
        finally:
            st.session_state.analysis_in_progress = False
            st.rerun()

    def render_text_based_interface(self):
        game_id = st.session_state.current_game_id
        game_info = st.session_state.generated_games.get(game_id)
        
        if not game_info:
            st.warning("게임 정보를 찾을 수 없습니다.")
            return

        st.header(f"🎲 {game_info.get('name', '게임')}: AI 분석 결과")

        if not game_info.get("success", True):
             st.error(f"분석 실패: {game_info.get('error_message', '알 수 없는 오류')}")
             return

        col1, col2, col3 = st.columns(3)
        col1.metric("AI 신뢰도", f"{game_info.get('confidence', 0.0):.1%}")
        col2.metric("보드 타입", game_info.get('board_type', "N/A"))
        col3.metric("복잡도", game_info.get('analysis_summary', {}).get('게임_복잡도', "N/A"))

        with st.expander("📜 AI가 생성한 전체 UI 명세서 (JSON)", expanded=True):
            st.json(game_info.get("full_spec", {}))
        
        with st.expander("🔬 AI의 핵심 분석 내용 (JSON)", expanded=False):
            st.json(game_info.get("analysis_summary", {}))

    def render_main_content(self):
        st.title("🤖 LangGraph AI Game Mate")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            with st.container(border=True):
                self.render_game_creator()
            with st.container(border=True):
                self.render_generated_games_list()

        with col2:
            with st.container(border=True):
                if st.session_state.analysis_in_progress:
                    asyncio.run(self.run_analysis_and_stream_results())
                elif st.session_state.current_game_id:
                    self.render_text_based_interface()
                else:
                    st.subheader("3. 분석 과정 및 결과")
                    st.info("게임을 새로 분석하거나, 목록에서 '결과 보기'를 선택해주세요.")

# Streamlit 앱 실행 (표준 방식)
app = RealLangGraphUI()
app.render_sidebar()
app.render_main_content()