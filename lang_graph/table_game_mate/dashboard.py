"""
Table Game Mate 대시보드

LangGraph 패턴을 따르는 간단한 Streamlit 대시보드
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import GameAgent, AnalysisAgent, MonitoringAgent
from core import GameConfig, Player


class TableGameMateDashboard:
    """Table Game Mate 대시보드"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Table Game Mate",
            page_icon="🎮",
            layout="wide"
        )
        
        # 에이전트 초기화
        self.game_agent = GameAgent()
        self.analysis_agent = AnalysisAgent()
        self.monitoring_agent = MonitoringAgent()
        
        self.render()
    
    def render(self):
        """대시보드 렌더링"""
        st.title("🎮 Table Game Mate")
        st.markdown("LangGraph 기반 멀티 에이전트 보드게임 플랫폼")
        
        # 사이드바
        self.render_sidebar()
        
        # 메인 컨텐츠
        tab1, tab2, tab3 = st.tabs(["🎯 게임", "📊 분석", "🖥️ 모니터링"])
        
        with tab1:
            self.render_game_tab()
        
        with tab2:
            self.render_analysis_tab()
        
        with tab3:
            self.render_monitoring_tab()
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        st.sidebar.title("🔧 제어판")
        
        if st.sidebar.button("🔄 새로고침", use_container_width=True):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # 시스템 상태
        st.sidebar.subheader("📊 시스템 상태")
        st.sidebar.success("✅ 시스템 정상")
        
        # 에이전트 상태
        st.sidebar.subheader("🤖 에이전트 상태")
        st.sidebar.info("🎯 게임 에이전트: 활성")
        st.sidebar.info("📊 분석 에이전트: 활성")
        st.sidebar.info("🖥️ 모니터링 에이전트: 활성")
    
    def render_game_tab(self):
        """게임 탭 렌더링"""
        st.header("🎯 게임 플레이")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("게임 설정")
            
            # 게임 선택
            game_type = st.selectbox(
                "게임 선택",
                ["체스", "체커", "바둑", "포커"],
                index=0
            )
            
            # 플레이어 수
            player_count = st.slider("플레이어 수", 2, 4, 2)
            
            # 플레이어 이름 입력
            st.subheader("플레이어 설정")
            player_names = []
            for i in range(player_count):
                name = st.text_input(f"플레이어 {i+1} 이름", value=f"Player {i+1}", key=f"player_{i}")
                player_names.append(name)
            
            # 게임 시작 버튼
            if st.button("🎮 게임 시작", use_container_width=True):
                self.start_game(game_type, player_names)
        
        with col2:
            st.subheader("게임 상태")
            
            # 게임 로그 표시
            if "game_log" in st.session_state:
                st.text_area("게임 로그", st.session_state.game_log, height=300)
            else:
                st.info("게임을 시작하면 로그가 여기에 표시됩니다")
    
    def render_analysis_tab(self):
        """분석 탭 렌더링"""
        st.header("📊 게임 분석")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("분석 설정")
            
            # 분석할 게임 데이터 업로드
            uploaded_file = st.file_uploader("게임 데이터 파일 업로드", type=["json"])
            
            if uploaded_file:
                try:
                    game_data = json.load(uploaded_file)
                    st.success("게임 데이터가 성공적으로 로드되었습니다")
                    
                    if st.button("📊 분석 시작", use_container_width=True):
                        self.analyze_game(game_data)
                except Exception as e:
                    st.error(f"파일 로드 실패: {str(e)}")
            
            # 샘플 데이터로 분석
            if st.button("🎲 샘플 데이터로 분석", use_container_width=True):
                sample_data = self.get_sample_game_data()
                self.analyze_game(sample_data)
        
        with col2:
            st.subheader("분석 결과")
            
            if "analysis_result" in st.session_state:
                result = st.session_state.analysis_result
                
                # 기본 메트릭
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("총 움직임", result.get("total_moves", 0))
                with col_b:
                    st.metric("플레이어 수", result.get("player_count", 0))
                with col_c:
                    st.metric("게임 시간", f"{result.get('game_duration', 0)}분")
                
                # 상세 분석
                st.subheader("상세 분석")
                st.json(result)
            else:
                st.info("게임을 분석하면 결과가 여기에 표시됩니다")
    
    def render_monitoring_tab(self):
        """모니터링 탭 렌더링"""
        st.header("🖥️ 시스템 모니터링")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("실시간 메트릭")
            
            # 모니터링 시작 버튼
            if st.button("📊 모니터링 시작", use_container_width=True):
                self.start_monitoring()
            
            # 시스템 메트릭 표시
            if "monitoring_data" in st.session_state:
                data = st.session_state.monitoring_data
                
                # CPU 사용률
                cpu_usage = data.get("cpu_percent", 0)
                st.metric("CPU 사용률", f"{cpu_usage}%")
                
                # 메모리 사용률
                memory_usage = data.get("memory_percent", 0)
                st.metric("메모리 사용률", f"{memory_usage}%")
                
                # 디스크 사용률
                disk_usage = data.get("disk_percent", 0)
                st.metric("디스크 사용률", f"{disk_usage}%")
        
        with col2:
            st.subheader("시스템 로그")
            
            if "system_log" in st.session_state:
                st.text_area("시스템 로그", st.session_state.system_log, height=300)
            else:
                st.info("모니터링을 시작하면 로그가 여기에 표시됩니다")
    
    def start_game(self, game_type: str, player_names: list):
        """게임 시작"""
        try:
            st.session_state.game_log = f"게임 '{game_type}' 시작 중...\n"
            
            # 게임 설정 생성
            game_config = {
                "name": game_type,
                "type": game_type.lower(),
                "min_players": 2,
                "max_players": len(player_names),
                "estimated_duration": 30
            }
            
            # 플레이어 생성
            players = []
            for i, name in enumerate(player_names):
                player = {
                    "id": f"player_{i+1}",
                    "name": name,
                    "type": "human" if i == 0 else "ai"
                }
                players.append(player)
            
            # 게임 실행 (비동기 함수를 동기적으로 실행)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.game_agent.play_game(game_config, players)
                )
                
                if result["success"]:
                    st.session_state.game_log += "✅ 게임이 성공적으로 완료되었습니다!\n"
                    st.session_state.game_log += f"결과: {result}\n"
                    st.success("게임이 성공적으로 완료되었습니다!")
                else:
                    st.session_state.game_log += f"❌ 게임 실패: {result['error']}\n"
                    st.error(f"게임 실패: {result['error']}")
            
            finally:
                loop.close()
            
        except Exception as e:
            st.session_state.game_log += f"❌ 오류 발생: {str(e)}\n"
            st.error(f"오류 발생: {str(e)}")
    
    def analyze_game(self, game_data: dict):
        """게임 분석"""
        try:
            st.session_state.analysis_result = "분석 중...\n"
            
            # 분석 실행 (비동기 함수를 동기적으로 실행)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.analysis_agent.analyze_game(game_data)
                )
                
                if result["success"]:
                    analysis_data = result["analysis_result"]
                    st.session_state.analysis_result = {
                        "total_moves": len(game_data.get("moves", [])),
                        "player_count": len(game_data.get("players", [])),
                        "game_duration": game_data.get("duration", 0),
                        "analysis_data": analysis_data
                    }
                    st.success("게임 분석이 완료되었습니다!")
                else:
                    st.session_state.analysis_result = f"분석 실패: {result['error']}"
                    st.error(f"분석 실패: {result['error']}")
            
            finally:
                loop.close()
            
        except Exception as e:
            st.session_state.analysis_result = f"오류 발생: {str(e)}"
            st.error(f"오류 발생: {str(e)}")
    
    def start_monitoring(self):
        """모니터링 시작"""
        try:
            st.session_state.system_log = "모니터링 시작 중...\n"
            
            # 모니터링 실행 (비동기 함수를 동기적으로 실행)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.monitoring_agent.monitor_system()
                )
                
                if result["success"]:
                    monitoring_data = result["monitoring_result"]
                    st.session_state.monitoring_data = monitoring_data.current_metrics or {}
                    st.session_state.system_log += "✅ 모니터링이 성공적으로 시작되었습니다!\n"
                    st.success("모니터링이 시작되었습니다!")
                else:
                    st.session_state.system_log += f"❌ 모니터링 실패: {result['error']}\n"
                    st.error(f"모니터링 실패: {result['error']}")
            
            finally:
                loop.close()
            
        except Exception as e:
            st.session_state.system_log += f"❌ 오류 발생: {str(e)}\n"
            st.error(f"오류 발생: {str(e)}")
    
    def get_sample_game_data(self) -> dict:
        """샘플 게임 데이터 반환"""
        return {
            "moves": [
                {"type": "move", "player": "Alice", "duration": 5, "strategic": True},
                {"type": "move", "player": "Bob", "duration": 3, "strategic": False},
                {"type": "move", "player": "Alice", "duration": 7, "strategic": True},
                {"type": "move", "player": "Bob", "duration": 4, "strategic": True}
            ],
            "players": ["Alice", "Bob"],
            "duration": 120,
            "rounds": 4
        }


def main():
    """메인 함수"""
    dashboard = TableGameMateDashboard()


if __name__ == "__main__":
    main()
