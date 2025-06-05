"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# Business Strategy Agent 모듈 임포트
try:
    from srcs.business_strategy_agents.streamlit_app import main as bs_main
    from srcs.business_strategy_agents.streamlit_app import *
    BUSINESS_STRATEGY_AVAILABLE = True
except ImportError as e:
    BUSINESS_STRATEGY_AVAILABLE = False
    import_error = str(e)

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
    
    # Business Strategy Agent 실행
    if BUSINESS_STRATEGY_AVAILABLE:
        try:
            # Business Strategy Agent의 main 함수 실행
            bs_main()
            
        except Exception as e:
            st.error(f"Business Strategy Agent 실행 중 오류가 발생했습니다: {e}")
            st.info("에이전트에 연결하려면 필요한 모듈을 확인해주세요.")
            
            # 수동 접속 가이드만 제공
            st.markdown("### 🔧 수동 접속")
            st.info("Business Strategy Agent를 별도로 실행해주세요.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.code("cd srcs/business_strategy_agents")
            with col2:
                st.code("streamlit run streamlit_app.py")
                
    else:
        st.error("Business Strategy Agent를 불러올 수 없습니다.")
        st.error(f"오류 내용: {import_error}")
        
        # 에이전트 소개만 제공 (가짜 데이터 제거)
        st.markdown("### 🎯 Business Strategy Agent 소개")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 주요 기능
            - **시장 분석**: 타겟 시장 규모 및 동향 분석
            - **경쟁사 분석**: 경쟁 구도 및 포지셔닝 전략
            - **비즈니스 모델 설계**: 수익 구조 및 가치 제안
            - **SWOT 분석**: 강점, 약점, 기회, 위협 요소
            - **재무 모델링**: 매출 예측 및 투자 계획
            """)
        
        with col2:
            st.markdown("""
            #### ✨ 스페셜 기능
            - **스파클 모드**: 재미있는 비즈니스 인사이트
            - **대화형 분석**: 자연어로 질문하고 답변 받기
            - **시각화**: 차트와 그래프로 결과 표시
            - **보고서 생성**: 전문적인 분석 리포트
            - **실시간 업데이트**: 최신 시장 데이터 반영
            """)
        
        st.markdown("---")
        
        # 설치 가이드
        with st.expander("🔧 설치 및 실행 가이드"):
            st.markdown("""
            ### Business Strategy Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install streamlit plotly pandas openai
            ```
            
            2. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            3. **에이전트 실행**:
            ```bash
            cd srcs/business_strategy_agents
            streamlit run streamlit_app.py
            ```
            
            4. **포트 설정** (옵션):
            ```bash
            streamlit run streamlit_app.py --server.port 8501
            ```
            """)

if __name__ == "__main__":
    main() 