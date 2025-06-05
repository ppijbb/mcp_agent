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

# Business Strategy Agent 모듈 임포트
try:
    from srcs.business_strategy_agents.streamlit_app import main as bs_main
    from srcs.business_strategy_agents.streamlit_app import *
    BUSINESS_STRATEGY_AVAILABLE = True
except ImportError as e:
    BUSINESS_STRATEGY_AVAILABLE = False
    import_error = str(e)

# 페이지 설정
st.set_page_config(
    page_title="🎯 Business Strategy Agent",
    page_icon="🎯",
    layout="wide"
)

def main():
    """Business Strategy Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🎯 Business Strategy Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # Business Strategy Agent 실행
    if BUSINESS_STRATEGY_AVAILABLE:
        try:
            # 기존 Business Strategy Agent의 main 함수 실행
            bs_main()
            
        except Exception as e:
            st.error(f"Business Strategy Agent 실행 중 오류가 발생했습니다: {e}")
            
            # 대체 인터페이스 제공
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
        
        # 대체 UI 제공
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
        
        # 샘플 분석 결과 표시
        st.markdown("### 📋 샘플 분석 결과")
        
        tab1, tab2, tab3 = st.tabs(["시장 분석", "경쟁사 분석", "재무 모델"])
        
        with tab1:
            st.markdown("""
            #### 🎯 타겟 시장 분석
            
            **시장 규모**: 약 1,200억원 (2024년 기준)
            **성장률**: 연 15% 성장 예상
            **주요 트렌드**:
            - AI/ML 기술 도입 가속화
            - 구독 기반 서비스 모델 확산
            - 모바일 우선 전략 필수
            """)
            
            # 가상 차트
            import pandas as pd
            import plotly.express as px
            
            market_data = pd.DataFrame({
                'Year': [2022, 2023, 2024, 2025, 2026],
                'Market Size (억원)': [850, 980, 1200, 1380, 1590]
            })
            
            fig = px.line(market_data, x='Year', y='Market Size (억원)', 
                         title='시장 규모 성장 예측',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("""
            #### 🏆 경쟁사 분석
            
            | 순위 | 회사명 | 시장점유율 | 강점 | 약점 |
            |------|--------|------------|------|------|
            | 1위 | A Company | 35% | 브랜드 인지도 | 높은 가격 |
            | 2위 | B Company | 28% | 기술력 | 제한된 시장 |
            | 3위 | C Company | 15% | 가격 경쟁력 | 품질 이슈 |
            | 4위 | 우리 회사 | 12% | 혁신성 | 마케팅 부족 |
            """)
        
        with tab3:
            st.markdown("""
            #### 💰 재무 모델링
            
            **3년 매출 예측**:
            - 1년차: 5억원
            - 2년차: 12억원  
            - 3년차: 28억원
            
            **주요 가정**:
            - 월간 성장률: 8%
            - 고객 획득 비용: 50,000원
            - 고객 생애 가치: 200,000원
            """)
        
        # 수동 설치 가이드
        st.markdown("---")
        
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