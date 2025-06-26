"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import json
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('business_strategy')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def main():
    """Business Strategy Agent 메인 페이지 (프로세스 모니터링)"""
    
    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    
    # 헤더 렌더링
    header_html = get_page_header("business", "🎯 Business Strategy Agent", 
                                 "AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    render_home_button()
    
    st.markdown("---")
    
    # 입력 폼 생성
    with st.form("business_strategy_form"):
        st.subheader("📝 분석 설정")
        
        # 기본 설정
        col1, col2 = st.columns(2)
        
        with col1:
            keywords_input = st.text_input(
                "🔍 핵심 키워드 (쉼표로 구분)",
                placeholder="예: AI, fintech, sustainability",
                help="분석하고자 하는 핵심 키워드들을 입력하세요"
            )
            
            business_context_input = st.text_area(
                "🏢 비즈니스 맥락",
                placeholder="예: AI 스타트업, 핀테크 회사 등",
                help="비즈니스 상황이나 배경을 설명해주세요"
            )
        
        with col2:
            objectives_input = st.text_input(
                "🎯 목표 (쉼표로 구분)",
                placeholder="예: growth, expansion, efficiency",
                help="달성하고자 하는 비즈니스 목표들을 입력하세요"
            )
            
            regions_input = st.text_input(
                "🌍 타겟 지역 (쉼표로 구분)",
                placeholder="예: North America, Europe, Asia",
                help="분석 대상 지역을 입력하세요"
            )
        
        # 고급 설정
        st.subheader("⚙️ 고급 설정")
        
        col3, col4 = st.columns(2)
        
        with col3:
            time_horizon = st.selectbox(
                "⏰ 분석 기간",
                ["3_months", "6_months", "12_months", "24_months"],
                index=2,
                help="분석 및 전략 수립 기간을 선택하세요"
            )
        
        with col4:
            analysis_mode = st.selectbox(
                "🔄 분석 모드",
                ["unified", "individual", "both"],
                index=0,
                help="unified: 통합분석(권장), individual: 개별분석, both: 전체분석"
            )
        
        # 실행 버튼
        submitted = st.form_submit_button("🚀 비즈니스 전략 분석 시작", use_container_width=True)
    
    # 폼 제출 처리
    if submitted:
        if not keywords_input.strip():
            st.error("❌ 핵심 키워드를 입력해주세요!")
            return

        # 프로세스 실행 명령어 생성
        command = [
            "python", "-u", 
            "srcs/business_strategy_agents/run_agent_script.py",
            "--keywords", keywords_input,
            "--time-horizon", time_horizon,
            "--mode", analysis_mode,
        ]

        if business_context_input.strip():
            business_context = {"description": business_context_input}
            command.extend(["--business-context", json.dumps(business_context)])

        if objectives_input.strip():
            command.extend(["--objectives", objectives_input])

        if regions_input.strip():
            command.extend(["--regions", regions_input])
        
        # 결과 파일 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(REPORTS_PATH, f"agent_output_{timestamp}.log")
        os.makedirs(REPORTS_PATH, exist_ok=True)

        # 프로세스 매니저 실행
        process = Process(
            command,
            output_file=output_file,
        ).start()

        st.info("🔄 Business Strategy MCPAgent 실행 중...")
        
        spm.st_process_monitor(
            process,
            label="비즈니스 전략 분석"
        ).loop_until_finished()

        st.success(f"✅ 분석 프로세스가 완료되었습니다. 전체 로그는 {output_file}에 저장됩니다.")

if __name__ == "__main__":
    main() 