"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime

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

# Business Strategy Agent 모듈 임포트 - 필수 의존성
try:
    from srcs.business_strategy_agents.streamlit_app import main as bs_main
except ImportError as e:
    st.error(f"❌ Business Strategy Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: Business Strategy Agent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def main():
    """Business Strategy Agent 메인 페이지"""
    
    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    
    # 헤더 렌더링
    header_html = get_page_header("business", "🎯 Business Strategy Agent", 
                                 "AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    render_home_button()
    
    st.markdown("---")
    
    st.success("🤖 Business Strategy Agent가 성공적으로 연결되었습니다!")
    
    # Business Strategy Agent 실행
    try:
        # 실제 Business Strategy Agent 실행
        bs_main()
        
    except Exception as e:
        st.error(f"❌ Business Strategy Agent 실행 실패: {e}")
        st.error("Business Strategy Agent 구현을 확인해주세요.")
        st.stop()

if __name__ == "__main__":
    main() 