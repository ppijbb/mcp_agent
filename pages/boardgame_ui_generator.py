#!/usr/bin/env python3
"""
진짜 LangGraph 에이전트 연동 UI 시스템
실제 AI 에이전트가 게임을 분석하고 UI를 생성하는 시스템
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lang_graph.table_game_mate.utils.mcp_client import MCPClient, MCPClientError

# 실제 LangGraph 에이전트 import
from lang_graph.table_game_mate.agents.game_ui_analyzer import get_game_ui_analyzer

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

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

        if "mcp_client" not in st.session_state:
            # MCPClient는 비동기 초기화가 필요하므로 지연 초기화
            st.session_state.mcp_client = None
            st.session_state.mcp_client_initialized = False
        
        # 세션 상태 초기화
        for key, default in {
            "generated_games": {},
            "current_game_id": None,
            "analysis_log": [],
            "analysis_steps": [],
            "analysis_in_progress": False,
            "bgg_search_results": None,
            "game_selection_needed": False,
            "bgg_game_details": None,
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default



    async def _ensure_mcp_client(self):
        """MCP 클라이언트 초기화 확인"""
        if st.session_state.mcp_client is None or not st.session_state.mcp_client_initialized:
            st.session_state.mcp_client = MCPClient()
            st.session_state.mcp_client_initialized = True
        return st.session_state.mcp_client
    
    async def handle_game_search(self, game_description: str):
        st.session_state.analysis_in_progress = True
        st.session_state.game_selection_needed = False
        st.session_state.bgg_search_results = None
        st.session_state.current_game_id = None
        
        mcp_client: MCPClient = await self._ensure_mcp_client()
        
        try:
            with st.spinner(f"'{game_description}' 게임을 BoardGameGeek에서 검색 중..."):
                # bgg_mcp_server.py의 search_boardgame tool을 호출
                search_result = await mcp_client.call(
                    server_name="bgg-api",
                    method="search_boardgame",
                    params={"name": game_description, "exact": False}
                )

            if search_result.get("success") and search_result.get("total", 0) > 0:
                games = search_result.get("games", [])
                if len(games) == 1:
                    # 결과가 하나면 바로 분석 진행
                    st.session_state.bgg_search_results = games
                    await self.handle_game_selection(games[0])
                else:
                    # 결과가 여러 개면 사용자에게 선택 요청
                    st.session_state.bgg_search_results = games
                    st.session_state.game_selection_needed = True
            else:
                st.error(f"'{game_description}'에 대한 게임을 BGG에서 찾을 수 없습니다. 더 일반적인 이름으로 시도해보세요.")
                st.session_state.analysis_in_progress = False

        except MCPClientError as e:
            st.error(f"BGG 서버 통신 오류: {e}. MCP 서버가 실행 중인지 확인하세요.")
            st.session_state.analysis_in_progress = False
        except Exception as e:
            st.error(f"게임 검색 중 오류 발생: {e}")
            st.session_state.analysis_in_progress = False
        
        st.rerun()

    async def handle_game_selection(self, selected_game: Dict[str, Any]):
        st.session_state.game_selection_needed = False
        st.session_state.analysis_in_progress = True
        
        game_id = f"bgg_{selected_game['id']}"
        st.session_state.current_game_id = game_id

        # 상세 정보 및 웹 규칙 가져오기
        try:
            mcp_client: MCPClient = await self._ensure_mcp_client()
            game_name_for_search = selected_game.get('name', 'board game')

            with st.spinner(f"'{selected_game['name']}' 상세 정보 조회 중..."):
                details_result = await mcp_client.call(
                    server_name="bgg-api",
                    method="get_game_details",
                    params={"bgg_id": selected_game['id']}
                )
            
            if not details_result.get("success"):
                raise Exception(f"BGG 상세 정보 조회 실패: {details_result.get('error')}")
            
            st.session_state.bgg_game_details = details_result["game"]

            # 웹에서 추가 규칙 검색
            web_rules_content = ""
            with st.spinner(f"'{game_name_for_search}' 공식 규칙 웹 검색 중..."):
                web_search_results = await mcp_client.search_web(
                    query=f'"{game_name_for_search}" official rules',
                    max_results=3
                )

                if web_search_results and web_search_results.get('results'):
                    # 첫 번째 검색 결과의 콘텐츠만 가져오기
                    top_result_url = web_search_results['results'][0]['url']
                    with st.spinner(f"'{top_result_url}'에서 규칙 내용 추출 중..."):
                        fetched_content = await mcp_client.fetch_content(url=top_result_url)
                        if fetched_content and fetched_content.get('content'):
                            web_rules_content = fetched_content['content'][:4000] # 토큰 제한

            # 이제 LangGraph 분석 시작
            game_name = st.session_state.bgg_game_details.get('name', '분석 중...')
            st.session_state.generated_games[game_id] = {
                "name": game_name,
                "description": st.session_state.bgg_game_details.get('description', ''),
                "rules": web_rules_content # 웹에서 가져온 규칙
            }

        except Exception as e:
            st.error(f"게임 상세 정보 및 규칙 조회 중 오류 발생: {e}")
            st.session_state.analysis_in_progress = False
        
        st.rerun()

    def render_game_creator(self):
        st.subheader("1. AI에게 분석을 요청할 게임 설명하기")
        game_description = st.text_area(
            "어떤 보드게임을 플레이하고 싶으신가요? 자유롭게 설명해주세요.", 
            placeholder="예시: 친구들과 할 수 있는 마피아 같은 심리 게임인데, 너무 무겁지 않고 간단하게 한 판 할 수 있는 거 없을까? 서로 속이고 정체를 밝혀내는 요소가 있었으면 좋겠어.", 
            height=150
        )
        
        if st.button("🧠 이 설명으로 UI 생성 분석 요청", type="primary", width='stretch', disabled=st.session_state.analysis_in_progress or not st.session_state.ui_analyzer):
            if game_description.strip():
                asyncio.run(self.handle_game_search(game_description))
            else:
                st.error("게임 설명을 입력해주세요!")

    def render_game_selection(self):
        st.subheader("BGG 검색 결과")
        st.write("분석하려는 게임을 선택하세요. 너무 많은 결과가 나온 경우 설명을 더 구체적으로 입력해주세요.")
        
        results = st.session_state.bgg_search_results
        
        for game in results:
            col1, col2 = st.columns([4, 1])
            with col1:
                year = f"({game.get('year')})" if game.get('year') else ""
                st.info(f"**{game.get('name')}** {year}")
            with col2:
                if st.button("이 게임으로 분석", key=f"select_{game.get('id')}", width='stretch'):
                    asyncio.run(self.handle_game_selection(game))

    def render_generated_games_list(self):
        st.subheader("2. 분석된 게임 목록")
        if not st.session_state.generated_games:
            st.info("아직 분석된 게임이 없습니다.")
            return

        for game_id, game_info in st.session_state.generated_games.items():
            name = game_info.get('name', '이름 없음')
            col_name, col_button = st.columns([4, 1])
            col_name.write(f"🎮 **{name}**")
            if col_button.button("결과 보기", key=f"load_{game_id}", width='stretch'):
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
            
            # 입력 데이터 구성 시 BGG 상세 정보 사용
            if st.session_state.bgg_game_details:
                input_description = (f"게임명: {st.session_state.bgg_game_details.get('name')}\n\n"
                                     f"설명: {st.session_state.bgg_game_details.get('description')}")
            else:
                input_description = game_info["description"]

            # GameUIAnalysisState 객체 생성
            from lang_graph.table_game_mate.agents.game_ui_analyzer import GameUIAnalysisState
            input_state = GameUIAnalysisState(
                game_description=input_description,
                detailed_rules=game_info.get("rules", ""),
                messages=[]
            )
            
            async for chunk in agent_app.astream(input_state):
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
                    # 만약 BGG 검색 결과가 있고 선택이 필요하다면, 선택 UI를 렌더링
                    if st.session_state.game_selection_needed:
                        self.render_game_selection()
                    else:
                        asyncio.run(self.run_analysis_and_stream_results())
                elif st.session_state.current_game_id:
                    self.render_text_based_interface()
                else:
                    st.subheader("3. 분석 과정 및 결과")
                    st.info("게임을 새로 분석하거나, 목록에서 '결과 보기'를 선택해주세요.")

# Streamlit 앱 실행 (표준 방식)
app = RealLangGraphUI()
app.render_main_content()

# 최신 Boardgame UI Generator 결과 확인
st.markdown("---")
st.markdown("## 📊 최신 Boardgame UI Generator 결과")

latest_boardgame_result = result_reader.get_latest_result("game_ui_analyzer", "ui_analysis")

if latest_boardgame_result:
    with st.expander("🎲 최신 게임 UI 분석 결과", expanded=False):
        st.subheader("🤖 최근 게임 UI 분석 결과")
        
        if isinstance(latest_boardgame_result, dict):
            # 게임 정보 표시
            game_name = latest_boardgame_result.get('game_name', 'N/A')
            st.success(f"**게임: {game_name}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("AI 신뢰도", f"{latest_boardgame_result.get('confidence_score', 0.0):.1%}")
            col2.metric("보드 타입", latest_boardgame_result.get('board_type', 'N/A'))
            col3.metric("분석 상태", "완료" if latest_boardgame_result.get('success', False) else "실패")
            
            # UI 명세서 표시
            ui_spec = latest_boardgame_result.get('ui_spec', {})
            if ui_spec:
                st.subheader("📋 UI 명세서")
                with st.expander("상세 UI 명세서", expanded=False):
                    st.json(ui_spec)
            
            # 분석 결과 표시
            analysis_result = latest_boardgame_result.get('analysis_result', {})
            if analysis_result:
                st.subheader("🔬 분석 결과")
                with st.expander("상세 분석 결과", expanded=False):
                    st.json(analysis_result)
            
            # 메타데이터 표시
            if 'timestamp' in latest_boardgame_result:
                st.caption(f"⏰ 분석 시간: {latest_boardgame_result['timestamp']}")
        else:
            st.json(latest_boardgame_result)
else:
    st.info("💡 아직 Boardgame UI Generator Agent의 결과가 없습니다. 위에서 게임 UI 분석을 실행해보세요.")