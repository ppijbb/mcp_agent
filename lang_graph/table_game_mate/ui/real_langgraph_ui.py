#!/usr/bin/env python3
"""
진짜 LangGraph 에이전트 연동 UI 시스템
실제 AI 에이전트가 게임을 분석하고 UI를 생성하는 시스템
"""

import streamlit as st
import asyncio
import sys
import os
from typing import Dict, List, Any
import json
from datetime import datetime

# 상위 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 실제 LangGraph 에이전트 import
from agents.game_ui_analyzer import get_game_ui_analyzer, GameUIAnalyzerAgent

# 페이지 설정
st.set_page_config(
    page_title="🤖 Real LangGraph UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealLangGraphUI:
    """실제 LangGraph 에이전트 기반 UI 시스템"""
    
    def __init__(self):
        # 실제 LangGraph 에이전트 초기화
        if "ui_analyzer" not in st.session_state:
            with st.spinner("LangGraph 에이전트 초기화 중..."):
                try:
                    st.session_state.ui_analyzer = get_game_ui_analyzer()
                    st.success("✅ LangGraph 에이전트 초기화 완료!")
                except Exception as e:
                    st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
                    st.session_state.ui_analyzer = None
        
        # 세션 상태 초기화
        if "generated_games" not in st.session_state:
            st.session_state.generated_games = {}
        if "current_ui_spec" not in st.session_state:
            st.session_state.current_ui_spec = None
        if "game_state" not in st.session_state:
            st.session_state.game_state = {}
        if "analysis_log" not in st.session_state:
            st.session_state.analysis_log = []
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        
        st.sidebar.header("🤖 Real LangGraph AI")
        
        # Gemini API 키 상태 확인
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_api_key_here":
            st.sidebar.error("🔑 Google API Key가 필요합니다!")
            st.sidebar.markdown("""
            **설정 방법:**
            1. [Google AI Studio](https://ai.google.dev/)에서 API 키 발급
            2. 터미널에서 설정:
            ```bash
            export GOOGLE_API_KEY="your_actual_key_here"
            ```
            """)
            st.sidebar.warning("API 키 설정 후 새로고침하세요.")
        else:
            st.sidebar.success("🔑 Google API Key 설정됨")
        
        # 에이전트 상태 표시
        if st.session_state.ui_analyzer:
            st.sidebar.success("🟢 LangGraph 에이전트 활성화 (Gemini 2.0 Flash)")
        else:
            st.sidebar.error("🔴 LangGraph 에이전트 비활성화")
            if st.sidebar.button("🔄 에이전트 재시작"):
                try:
                    st.session_state.ui_analyzer = get_game_ui_analyzer()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"재시작 실패: {str(e)}")
        
        # 게임 생성 인터페이스
        with st.sidebar.expander("🎮 새 게임 생성", expanded=True):
            self.render_game_creator()
        
        # 기존 게임 목록
        if st.session_state.generated_games:
            st.sidebar.subheader("📚 생성된 게임들")
            for game_id, game_info in st.session_state.generated_games.items():
                confidence = game_info.get('confidence', 0.0)
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                if st.sidebar.button(f"{confidence_emoji} {game_info['name']}", key=f"load_{game_id}"):
                    st.session_state.current_ui_spec = game_info
                    st.session_state.game_state = self.create_initial_game_state(game_info)
                    st.rerun()
    
    def render_game_creator(self):
        """게임 생성 인터페이스"""
        
        # 게임 설명 입력
        game_description = st.text_area(
            "게임 설명:",
            placeholder="""실제 LangGraph AI가 분석할 게임을 설명해주세요:

예시:
- "체스판에서 말을 움직여 왕을 잡는 게임"
- "카드를 교환하며 같은 숫자를 모으는 게임"  
- "마피아를 투표로 찾아내는 추리게임"
- "타일을 연결해서 길을 만드는 퍼즐게임" """,
            height=120,
            key="game_desc_input"
        )
        
        # 세부 규칙 (선택사항)
        detailed_rules = st.text_area(
            "세부 규칙 (선택):",
            placeholder="더 구체적인 규칙이 있다면 설명해주세요...",
            height=80,
            key="rules_input"
        )
        
        # 플레이어 수
        col1, col2 = st.columns(2)
        with col1:
            min_players = st.number_input("최소 플레이어", 1, 20, 2, key="min_players")
        with col2:
            max_players = st.number_input("최대 플레이어", 1, 20, 4, key="max_players")
        
        # AI 분석 버튼
        if st.button("🧠 LangGraph AI로 분석 및 생성", type="primary", disabled=not st.session_state.ui_analyzer):
            if game_description.strip():
                self.analyze_with_langgraph(game_description, detailed_rules, min_players, max_players)
            else:
                st.error("게임 설명을 입력해주세요!")
    
    def analyze_with_langgraph(self, description: str, rules: str, min_players: int, max_players: int):
        """실제 LangGraph 에이전트로 분석"""
        
        if not st.session_state.ui_analyzer:
            st.error("LangGraph 에이전트가 초기화되지 않았습니다!")
            return
        
        with st.spinner("🤖 LangGraph AI가 게임을 분석하고 있습니다..."):
            try:
                # 실제 LangGraph 에이전트 호출
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                start_time = datetime.now()
                
                # 에이전트 분석 실행
                analysis_result = loop.run_until_complete(
                    st.session_state.ui_analyzer.analyze_game_for_ui(description, rules)
                )
                
                end_time = datetime.now()
                analysis_time = (end_time - start_time).total_seconds()
                
                # 분석 로그 추가
                log_entry = {
                    "timestamp": start_time,
                    "description": description,
                    "analysis_time": analysis_time,
                    "success": analysis_result.get("success", False),
                    "confidence": analysis_result.get("confidence_score", 0.0),
                    "game_name": analysis_result.get("game_name", "Unknown")
                }
                st.session_state.analysis_log.append(log_entry)
                
                if analysis_result.get("success", False):
                    # 성공적으로 분석된 경우
                    game_id = f"game_{len(st.session_state.generated_games)}"
                    
                    # 분석 결과를 UI 스펙으로 변환
                    ui_spec = {
                        "id": game_id,
                        "name": analysis_result["game_name"],
                        "board_type": analysis_result["board_type"],
                        "required_components": analysis_result["required_components"],
                        "layout_structure": analysis_result["layout_structure"],
                        "interaction_patterns": analysis_result["interaction_patterns"],
                        "special_features": analysis_result["special_features"],
                        "complexity": analysis_result["complexity"],
                        "confidence": analysis_result["confidence_score"],
                        "analysis_result": analysis_result["analysis_result"],
                        "min_players": min_players,
                        "max_players": max_players,
                        "generated_at": analysis_result["generated_at"],
                        "analysis_time": analysis_time,
                        "original_description": description,
                        "original_rules": rules
                    }
                    
                    # 저장 및 활성화
                    st.session_state.generated_games[game_id] = ui_spec
                    st.session_state.current_ui_spec = ui_spec
                    st.session_state.game_state = self.create_initial_game_state(ui_spec)
                    
                    st.success(f"✅ '{analysis_result['game_name']}' 분석 완료! (신뢰도: {analysis_result['confidence_score']:.1%}, 소요시간: {analysis_time:.1f}초)")
                    st.rerun()
                
                else:
                    # 분석 실패한 경우
                    st.error(f"❌ 분석 실패: {analysis_result.get('error_message', '알 수 없는 오류')}")
                
            except Exception as e:
                st.error(f"❌ LangGraph 에이전트 실행 중 오류: {str(e)}")
                
                # 에러 로그 추가
                error_log = {
                    "timestamp": datetime.now(),
                    "description": description,
                    "error": str(e),
                    "success": False
                }
                st.session_state.analysis_log.append(error_log)
    
    def create_initial_game_state(self, ui_spec: Dict[str, Any]) -> Dict[str, Any]:
        """UI 스펙에 따른 초기 게임 상태 생성"""
        
        board_type = ui_spec.get("board_type", "grid")
        max_players = ui_spec.get("max_players", 4)
        
        state = {
            "current_player": 0,
            "turn_count": 0,
            "phase": "시작",
            "players": []
        }
        
        # 플레이어 생성
        for i in range(max_players):
            player = {"name": f"플레이어{i+1}", "status": "활성"}
            
            if board_type == "card_layout":
                player.update({"chips": 1000, "hand": []})
            elif board_type == "text_based":
                player.update({"role": "시민", "status": "생존"})
            
            state["players"].append(player)
        
        # 보드 타입별 초기화
        layout = ui_spec.get("layout_structure", {}).get("main_area", {})
        
        if board_type == "grid":
            rows = layout.get("rows", 8)
            cols = layout.get("cols", 8)
            state["board"] = [['' for _ in range(cols)] for _ in range(rows)]
        elif board_type == "card_layout":
            state.update({"community_cards": [], "deck": [], "pot": 0})
        elif board_type == "text_based":
            state["chat_messages"] = [
                {"player": "시스템", "message": f"{ui_spec['name']} 게임이 시작되었습니다! (LangGraph AI 생성)"}
            ]
        
        return state
    
    def render_main_content(self):
        """메인 콘텐츠 렌더링"""
        
        if not st.session_state.current_ui_spec:
            self.render_welcome_screen()
        else:
            self.render_game_interface()
    
    def render_welcome_screen(self):
        """환영 화면"""
        
        st.title("🤖 Real LangGraph AI Game Mate")
        
        st.markdown("""
        **실제 LangGraph 에이전트가 게임을 분석하고 UI를 생성합니다!**
        
        ### 🎯 특징
        - **진짜 AI 분석**: Google Gemini 2.5 Flash Lite를 사용한 LangGraph 워크플로우
        - **동적 UI 생성**: 게임 분석 결과에 따른 실시간 인터페이스 생성  
        - **신뢰도 평가**: AI 분석 결과의 신뢰도를 실시간 계산
        - **분석 로그**: 모든 AI 분석 과정을 상세히 기록
        
        ### 🚀 LangGraph 워크플로우
        1. **게임 분석**: 설명을 파싱하여 게임 구조 파악
        2. **UI 타입 결정**: 최적의 인터페이스 타입 선택  
        3. **상세 명세 생성**: 구체적인 UI 레이아웃 설계
        4. **검증 및 신뢰도 계산**: 결과 검증 후 신뢰도 점수 부여
        """)
        
        # 빠른 테스트 버튼들
        st.subheader("⚡ 빠른 테스트")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 체스 게임 분석"):
                self.analyze_with_langgraph("8x8 체스판에서 말을 움직여서 상대방 왕을 체크메이트로 잡는 전략 게임", "", 2, 2)
            
            if st.button("🃏 포커 게임 분석"):
                self.analyze_with_langgraph("각자 카드 2장을 받고 공용 카드 5장과 조합해서 최고의 패를 만들어 베팅하는 게임", "", 2, 8)
        
        with col2:
            if st.button("🕵️ 마피아 게임 분석"):
                self.analyze_with_langgraph("낮에는 토론하고 밤에는 마피아가 시민을 제거하며, 투표로 마피아를 찾아내는 심리전 게임", "", 4, 12)
            
            if st.button("🧩 퍼즐 게임 분석"):
                self.analyze_with_langgraph("다양한 모양의 타일을 회전시키고 배치해서 주어진 패턴을 완성하는 퍼즐 게임", "", 1, 4)
        
        # 분석 로그 표시
        if st.session_state.analysis_log:
            with st.expander("📊 AI 분석 로그", expanded=False):
                for i, log in enumerate(reversed(st.session_state.analysis_log[-5:])):  # 최근 5개만
                    if log.get("success", False):
                        st.success(f"✅ {log['game_name']} - 신뢰도: {log['confidence']:.1%} ({log['analysis_time']:.1f}초)")
                    else:
                        st.error(f"❌ 분석 실패: {log.get('error', '알 수 없는 오류')}")
    
    def render_game_interface(self):
        """실제 게임 인터페이스 렌더링"""
        
        ui_spec = st.session_state.current_ui_spec
        game_state = st.session_state.game_state
        
        # 헤더 정보
        st.header(f"🎲 {ui_spec['name']}")
        
        # AI 분석 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = ui_spec.get("confidence", 0.0)
            st.metric("AI 신뢰도", f"{confidence:.1%}")
        
        with col2:
            st.metric("보드 타입", ui_spec.get("board_type", "unknown"))
        
        with col3:
            st.metric("복잡도", ui_spec.get("complexity", "medium"))
        
        with col4:
            analysis_time = ui_spec.get("analysis_time", 0.0)
            st.metric("분석 시간", f"{analysis_time:.1f}초")
        
        # 실제 UI 렌더링 (LangGraph 분석 결과 기반)
        self.render_dynamic_game_ui(ui_spec, game_state)
        
        # 게임 컨트롤
        self.render_game_controls(ui_spec)
    
    def render_dynamic_game_ui(self, ui_spec: Dict[str, Any], game_state: Dict[str, Any]):
        """LangGraph 분석 결과에 따른 동적 UI 렌더링"""
        
        board_type = ui_spec.get("board_type", "grid")
        layout = ui_spec.get("layout_structure", {})
        
        # 메인 영역 렌더링
        if "main_area" in layout:
            if board_type == "grid":
                self.render_grid_interface(layout["main_area"], game_state)
            elif board_type == "card_layout":
                self.render_card_interface(layout["main_area"], game_state)
            elif board_type == "text_based":
                self.render_text_interface(layout["main_area"], game_state)
            else:
                st.info(f"🎨 새로운 보드 타입: {board_type} (LangGraph AI가 발견)")
                st.json(layout["main_area"])
        
        # 하단 패널
        if "bottom_panel" in layout:
            with st.expander("🔧 게임 패널", expanded=True):
                self.render_bottom_panel(layout["bottom_panel"], game_state)
    
    def render_grid_interface(self, config: Dict[str, Any], game_state: Dict[str, Any]):
        """격자 인터페이스 렌더링"""
        
        rows = config.get("rows", 8)
        cols = config.get("cols", 8)
        
        st.subheader(f"🎯 게임판 ({rows}×{cols}) - LangGraph 분석 결과")
        
        board_data = game_state.get("board", [['' for _ in range(cols)] for _ in range(rows)])
        
        for i in range(rows):
            cols_ui = st.columns(cols)
            for j, col in enumerate(cols_ui):
                with col:
                    cell_value = board_data[i][j] if i < len(board_data) and j < len(board_data[i]) else ''
                    if st.button(f"{cell_value or '◯'}", key=f"real_cell_{i}_{j}"):
                        # 실제 게임 로직 처리
                        board_data[i][j] = "X" if not cell_value else ""
                        st.rerun()
    
    def render_card_interface(self, config: Dict[str, Any], game_state: Dict[str, Any]):
        """카드 인터페이스 렌더링"""
        
        st.subheader("🃏 카드 테이블 - LangGraph 분석 결과")
        
        if config.get("community_area"):
            st.write("**공용 카드 영역**")
            community_cards = game_state.get("community_cards", ["A♠", "K♦", "Q♣"])
            card_cols = st.columns(5)
            
            for i, col in enumerate(card_cols):
                with col:
                    if i < len(community_cards):
                        st.info(f"🃏 {community_cards[i]}")
                    else:
                        st.empty()
        
        # 플레이어 정보
        players = game_state.get("players", [])
        if players:
            st.write("**플레이어 상태**")
            player_cols = st.columns(len(players))
            for i, (player, col) in enumerate(zip(players, player_cols)):
                with col:
                    st.metric(
                        player["name"],
                        f"칩: {player.get('chips', 0)}",
                        delta=f"핸드: {len(player.get('hand', []))}"
                    )
    
    def render_text_interface(self, config: Dict[str, Any], game_state: Dict[str, Any]):
        """텍스트 인터페이스 렌더링"""
        
        st.subheader("🗣️ 소셜 게임 인터페이스 - LangGraph 분석 결과")
        
        # 현재 페이즈
        phase = game_state.get("phase", "시작")
        st.info(f"현재 페이즈: **{phase}**")
        
        # 플레이어 상태
        if config.get("player_list"):
            players = game_state.get("players", [])
            if players:
                import pandas as pd
                
                player_df = pd.DataFrame([
                    {
                        "이름": p["name"],
                        "상태": p.get("status", "활성"),
                        "역할": p.get("role", "???")
                    }
                    for p in players
                ])
                st.dataframe(player_df, use_container_width=True)
    
    def render_bottom_panel(self, config: Dict[str, Any], game_state: Dict[str, Any]):
        """하단 패널 렌더링"""
        
        if config.get("hand_display"):
            st.write("**내 패**")
            hand = game_state.get("player_hand", ["10♠", "J♠"])
            if hand:
                hand_cols = st.columns(len(hand))
                for card, col in zip(hand, hand_cols):
                    with col:
                        st.info(f"🃏 {card}")
        
        if config.get("chat"):
            st.write("**채팅**")
            messages = game_state.get("chat_messages", [])
            for msg in messages[-5:]:
                st.write(f"**{msg['player']}**: {msg['message']}")
            
            # 채팅 입력
            chat_input = st.text_input("메시지:", key="real_chat")
            if st.button("전송") and chat_input:
                messages.append({"player": "나", "message": chat_input})
                st.rerun()
    
    def render_game_controls(self, ui_spec: Dict[str, Any]):
        """게임 컨트롤 패널"""
        
        with st.expander("🎮 게임 정보 및 컨트롤", expanded=False):
            
            # LangGraph 분석 세부 정보
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**AI 분석 결과:**")
                st.write(f"- 게임명: {ui_spec['name']}")
                st.write(f"- 보드 타입: {ui_spec['board_type']}")
                st.write(f"- 필요 컴포넌트: {len(ui_spec['required_components'])}개")
                st.write(f"- 인터랙션 패턴: {len(ui_spec.get('interaction_patterns', []))}개")
            
            with col2:
                st.write("**원본 설명:**")
                st.write(f"- 설명: {ui_spec.get('original_description', 'N/A')}")
                st.write(f"- 규칙: {ui_spec.get('original_rules', 'N/A') or '없음'}")
                st.write(f"- 플레이어: {ui_spec.get('min_players', 2)}-{ui_spec.get('max_players', 4)}명")
            
            # 상세 분석 결과
            if st.button("📊 LangGraph 분석 상세 보기"):
                st.json(ui_spec.get("analysis_result", {}))
            
            # 게임 제어
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 게임 재시작"):
                    st.session_state.game_state = self.create_initial_game_state(ui_spec)
                    st.rerun()
            
            with col2:
                if st.button("🗑️ 게임 삭제"):
                    st.session_state.current_ui_spec = None
                    st.session_state.game_state = {}
                    st.rerun()
            
            with col3:
                if st.button("🔍 재분석"):
                    desc = ui_spec.get('original_description', '')
                    rules = ui_spec.get('original_rules', '')
                    if desc:
                        self.analyze_with_langgraph(desc, rules, 
                                                  ui_spec.get('min_players', 2), 
                                                  ui_spec.get('max_players', 4))

def main():
    """메인 애플리케이션"""
    
    app = RealLangGraphUI()
    
    # 사이드바
    app.render_sidebar()
    
    # 메인 콘텐츠
    app.render_main_content()
    
    # 디버그 정보
    with st.expander("🔧 시스템 상태", expanded=False):
        st.write("**LangGraph 에이전트 상태:**")
        agent_status = "활성" if st.session_state.ui_analyzer else "비활성"
        st.write(f"- 에이전트: {agent_status}")
        st.write(f"- 생성된 게임: {len(st.session_state.generated_games)}개")
        st.write(f"- 분석 로그: {len(st.session_state.analysis_log)}개")

if __name__ == "__main__":
    main() 