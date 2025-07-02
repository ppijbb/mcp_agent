#!/usr/bin/env python3
"""
AI 기반 동적 보드게임 UI 생성 시스템
사용자가 게임을 설명하면 AI가 분석하여 적절한 인터페이스를 실시간 생성
"""

import streamlit as st
import asyncio
from typing import Dict, List, Any
import json
from datetime import datetime

# AI 기반 UI 생성기 import
from agent_driven_ui import AgentDrivenUIGenerator, MockGameAnalyzer, MockUIGenerator, GameUISpec

# 페이지 설정
st.set_page_config(
    page_title="AI 게임 메이트",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIGameMateApp:
    """AI 기반 게임 메이트 앱"""
    
    def __init__(self):
        # Mock 에이전트 사용 (실제로는 LangGraph 에이전트)
        self.ui_generator = AgentDrivenUIGenerator()
        self.ui_generator.game_analyzer = MockGameAnalyzer()
        self.ui_generator.ui_generator = MockUIGenerator()
        
        # 세션 상태 초기화
        if "generated_games" not in st.session_state:
            st.session_state.generated_games = {}
        if "current_game_spec" not in st.session_state:
            st.session_state.current_game_spec = None
        if "game_state" not in st.session_state:
            st.session_state.game_state = {}
    
    def render_game_creator(self):
        """새로운 게임 생성 인터페이스"""
        
        st.sidebar.header("🎮 AI 게임 생성기")
        
        with st.sidebar.expander("🔧 새 게임 만들기", expanded=True):
            
            # 게임 설명 입력
            game_description = st.text_area(
                "게임을 설명해주세요:",
                placeholder="""예시:
- 8x8 체스판에서 말을 움직여서 상대방 왕을 잡는 게임
- 카드 5장을 받아서 최고의 패를 만드는 포커 게임  
- 낮과 밤을 번갈아가며 투표로 마피아를 찾는 게임
- 타일을 배치해서 아름다운 패턴을 만드는 게임""",
                height=150
            )
            
            # 추가 규칙 (선택사항)
            detailed_rules = st.text_area(
                "세부 규칙 (선택사항):",
                placeholder="게임의 구체적인 규칙이나 특수 기능을 설명해주세요...",
                height=100
            )
            
            # 플레이어 수 힌트
            col1, col2 = st.columns(2)
            with col1:
                min_players = st.number_input("최소 플레이어", min_value=1, max_value=20, value=2)
            with col2:
                max_players = st.number_input("최대 플레이어", min_value=1, max_value=20, value=4)
            
            # AI 분석 및 생성 버튼
            if st.button("🤖 AI로 게임 분석 및 UI 생성", type="primary"):
                if game_description.strip():
                    with st.spinner("AI가 게임을 분석하고 UI를 생성하는 중..."):
                        self.generate_game_ui(game_description, detailed_rules, min_players, max_players)
                else:
                    st.error("게임 설명을 입력해주세요!")
        
        # 기존 생성된 게임들
        if st.session_state.generated_games:
            st.sidebar.subheader("📚 생성된 게임들")
            for game_id, game_info in st.session_state.generated_games.items():
                if st.sidebar.button(f"🎲 {game_info['name']}", key=f"load_{game_id}"):
                    st.session_state.current_game_spec = game_info['spec']
                    st.session_state.game_state = game_info['state']
                    st.rerun()
    
    def generate_game_ui(self, description: str, rules: str, min_players: int, max_players: int):
        """AI를 사용하여 게임 UI 생성"""
        
        try:
            # AI 분석 실행 (비동기를 동기로 변환)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            ui_spec = loop.run_until_complete(
                self.ui_generator.analyze_and_generate_ui(description, rules)
            )
            
            # 플레이어 수 정보 추가
            ui_spec.special_features["min_players"] = min_players
            ui_spec.special_features["max_players"] = max_players
            
            # 초기 게임 상태 생성
            initial_state = self.create_initial_game_state(ui_spec, max_players)
            
            # 생성된 게임 저장
            game_id = f"game_{len(st.session_state.generated_games)}"
            st.session_state.generated_games[game_id] = {
                "name": ui_spec.game_name,
                "spec": ui_spec,
                "state": initial_state,
                "created_at": datetime.now()
            }
            
            # 현재 게임으로 설정
            st.session_state.current_game_spec = ui_spec
            st.session_state.game_state = initial_state
            
            st.success(f"✅ '{ui_spec.game_name}' UI가 생성되었습니다! (신뢰도: {ui_spec.confidence_score:.1%})")
            st.rerun()
            
        except Exception as e:
            st.error(f"게임 생성 중 오류가 발생했습니다: {str(e)}")
    
    def create_initial_game_state(self, ui_spec: GameUISpec, max_players: int) -> Dict[str, Any]:
        """UI 명세에 따른 초기 게임 상태 생성"""
        
        state = {
            "current_player": 0,
            "turn_count": 0,
            "phase": "시작",
            "players": []
        }
        
        # 플레이어 생성
        for i in range(max_players):
            player = {
                "name": f"플레이어{i+1}",
                "status": "활성"
            }
            
            # UI 타입에 따른 플레이어 속성 추가
            if ui_spec.board_type == "card_layout":
                player["chips"] = 1000
                player["hand"] = []
            elif ui_spec.board_type == "text_based":
                player["role"] = "시민"
                player["status"] = "생존"
            
            state["players"].append(player)
        
        # 보드 타입에 따른 초기 상태
        if ui_spec.board_type == "grid":
            layout = ui_spec.layout_structure.get("main_area", {})
            rows = layout.get("rows", 8)
            cols = layout.get("cols", 8)
            state["board"] = [['' for _ in range(cols)] for _ in range(rows)]
            
        elif ui_spec.board_type == "card_layout":
            state["community_cards"] = []
            state["deck"] = []
            state["pot"] = 0
            
        elif ui_spec.board_type == "text_based":
            state["chat_messages"] = [
                {"player": "시스템", "message": f"{ui_spec.game_name} 게임이 시작되었습니다!"}
            ]
            state["voting_active"] = False
        
        return state
    
    def render_main_interface(self):
        """메인 게임 인터페이스 렌더링"""
        
        if not st.session_state.current_game_spec:
            st.title("🤖 AI 기반 게임 메이트")
            st.markdown("""
            **AI가 당신의 게임 아이디어를 분석하여 실시간으로 인터페이스를 생성합니다!**
            
            ### 🎯 특징
            - 어떤 게임이든 설명만 하면 AI가 분석
            - 게임 타입에 맞는 최적 UI 자동 생성
            - 실시간 게임 상태 관리
            - 새로운 게임 패턴 자동 감지
            
            ### 🚀 사용 방법
            1. 사이드바에서 게임을 설명해주세요
            2. AI가 게임을 분석하고 UI를 생성합니다
            3. 생성된 인터페이스에서 게임을 플레이하세요!
            """)
            
            # 예시 게임들
            st.subheader("💡 예시 게임 설명")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔥 체스 스타일 게임 생성"):
                    self.quick_generate("8x8 격자에서 말을 움직여서 상대방 왕을 잡는 전략 게임")
                
                if st.button("🃏 포커 스타일 게임 생성"):
                    self.quick_generate("카드 5장으로 최고의 패를 만들어 베팅하는 게임")
            
            with col2:
                if st.button("🕵️ 마피아 스타일 게임 생성"):
                    self.quick_generate("낮과 밤을 번갈아가며 투표로 마피아를 찾는 추리 게임")
                
                if st.button("🧩 타일 배치 게임 생성"):
                    self.quick_generate("다양한 모양의 타일을 배치해서 패턴을 만드는 퍼즐 게임")
            
        else:
            # AI가 생성한 UI 렌더링
            ui_spec = st.session_state.current_game_spec
            game_state = st.session_state.game_state
            
            # AI가 생성한 동적 UI 렌더링
            self.ui_generator.render_dynamic_ui(ui_spec, game_state)
            
            # 게임 상태 관리 패널
            self.render_game_controls(ui_spec, game_state)
    
    def quick_generate(self, description: str):
        """빠른 게임 생성"""
        with st.spinner(f"'{description}' 게임 생성 중..."):
            self.generate_game_ui(description, "", 2, 4)
    
    def render_game_controls(self, ui_spec: GameUISpec, game_state: Dict[str, Any]):
        """게임 컨트롤 패널"""
        
        with st.expander("🎮 게임 컨트롤", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 게임 재시작"):
                    max_players = ui_spec.special_features.get("max_players", 4)
                    st.session_state.game_state = self.create_initial_game_state(ui_spec, max_players)
                    st.rerun()
            
            with col2:
                if st.button("📊 AI 분석 정보"):
                    st.session_state.show_ai_info = not st.session_state.get("show_ai_info", False)
                    st.rerun()
            
            with col3:
                if st.button("🗑️ 게임 삭제"):
                    st.session_state.current_game_spec = None
                    st.session_state.game_state = {}
                    st.rerun()
        
        # AI 분석 정보 표시
        if st.session_state.get("show_ai_info", False):
            with st.expander("🤖 AI 분석 세부 정보", expanded=True):
                st.write("**게임 분석 결과:**")
                analysis_data = {
                    "게임명": ui_spec.game_name,
                    "보드 타입": ui_spec.board_type,
                    "복잡도": ui_spec.complexity_level.value,
                    "신뢰도": f"{ui_spec.confidence_score:.1%}",
                    "생성 시간": ui_spec.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "필요 컴포넌트": ui_spec.required_components,
                    "인터랙션 패턴": ui_spec.interaction_patterns,
                    "특수 기능": list(ui_spec.special_features.keys())
                }
                
                for key, value in analysis_data.items():
                    st.write(f"- **{key}**: {value}")
                
                st.write("**전체 UI 명세:**")
                st.json({
                    "layout_structure": ui_spec.layout_structure,
                    "special_features": ui_spec.special_features
                })
        
        # 대기 중인 이동 요청 처리
        if "pending_moves" in st.session_state and st.session_state.pending_moves:
            st.info(f"대기 중인 이동: {len(st.session_state.pending_moves)}개")
            
            if st.button("이동 요청 처리"):
                # 실제로는 AI 에이전트가 처리
                processed = len(st.session_state.pending_moves)
                st.session_state.pending_moves = []
                st.success(f"{processed}개 이동이 처리되었습니다!")
                st.rerun()

def main():
    """메인 애플리케이션"""
    
    app = AIGameMateApp()
    
    # 사이드바 - 게임 생성기
    app.render_game_creator()
    
    # 메인 영역 - 게임 인터페이스
    app.render_main_interface()
    
    # 디버그 정보 (개발용)
    with st.expander("🔧 개발자 디버그", expanded=False):
        st.write("**현재 세션 상태:**")
        debug_info = {
            "생성된 게임 수": len(st.session_state.generated_games),
            "현재 게임": st.session_state.current_game_spec.game_name if st.session_state.current_game_spec else "없음",
            "게임 상태 키": list(st.session_state.game_state.keys()) if st.session_state.game_state else []
        }
        st.json(debug_info)
        
        if st.button("세션 초기화"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main() 