"""
🤖 MCP Agent Hub - 통합 AI 에이전트 플랫폼

모든 AI 에이전트들을 한 곳에서 체험할 수 있는 Streamlit 데모
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 공통 스타일 모듈 임포트
from srcs.common.styles import get_common_styles

# 페이지 설정
st.set_page_config(
    page_title="🤖 MCP Agent Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 공통 스타일 적용
st.markdown(get_common_styles(), unsafe_allow_html=True)

def main():
    """메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MCP Agent Hub</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            차세대 AI 에이전트 플랫폼 - 비즈니스부터 개인까지
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 통계 표시
    display_platform_stats()
    
    # 에이전트 카테고리 표시
    display_agent_categories()
    
    # 최신 업데이트 및 소식
    display_latest_updates()

def display_platform_stats():
    """플랫폼 통계"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🤖 총 에이전트",
            value="25+",
            delta="5개 신규 추가"
        )
    
    with col2:
        st.metric(
            label="📊 카테고리",
            value="5개",
            delta="엔터프라이즈 확장"
        )
    
    with col3:
        st.metric(
            label="👥 활성 사용자",
            value="1.2K+",
            delta="30% 증가"
        )
    
    with col4:
        st.metric(
            label="⭐ 평균 평점",
            value="4.8/5",
            delta="0.2 상승"
        )

def display_agent_categories():
    """에이전트 카테고리별 표시"""
    
    # 비즈니스 전략 에이전트들
    st.markdown("""
    <div class="category-header">
        <h2>💼 비즈니스 전략 에이전트</h2>
        <p>비즈니스 성장과 전략 수립을 위한 AI 어시스턴트</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>🎯 Business Strategy Agent</h3>
            <p><strong>기능:</strong> 시장 분석, 경쟁사 분석, 비즈니스 모델 설계</p>
            <p><strong>특징:</strong> 스파클 모드, 재미있는 인사이트, 대화형 분석</p>
            <p><strong>사용 사례:</strong> 스타트업 전략, 신사업 기획, 투자 검토</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Business Strategy Agent 체험하기", key="bs_agent", use_container_width=True):
            st.switch_page("pages/business_strategy.py")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>🏥 SEO Doctor</h3>
            <p><strong>기능:</strong> 사이트 응급진단, 경쟁사 스파이, SEO 처방전</p>
            <p><strong>특징:</strong> 3분 진단, 모바일 최적화, 바이럴 요소</p>
            <p><strong>사용 사례:</strong> 트래픽 급락 대응, SEO 최적화, 경쟁 분석</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚨 SEO Doctor 응급진단", key="seo_doctor", use_container_width=True):
            st.switch_page("pages/seo_doctor.py")
    
    # 엔터프라이즈 에이전트들
    st.markdown("""
    <div class="category-header">
        <h2>🏢 엔터프라이즈 에이전트</h2>
        <p>기업 운영 최적화를 위한 전문 AI 솔루션</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>💰 Finance Health Agent</h3>
            <p>재무 건강도 진단 및 최적화</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💰 재무 분석하기", key="finance", use_container_width=True):
            st.switch_page("pages/finance_health.py")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>🔒 Cybersecurity Agent</h3>
            <p>사이버 보안 인프라 관리</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔒 보안 체크", key="cyber", use_container_width=True):
            st.switch_page("pages/cybersecurity.py")
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h3>👥 HR Recruitment Agent</h3>
            <p>인재 채용 및 관리 최적화</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👥 HR 관리", key="hr", use_container_width=True):
            st.switch_page("pages/hr_recruitment.py")
    
    # 라이프스타일 에이전트들
    st.markdown("""
    <div class="category-header">
        <h2>🌟 라이프스타일 에이전트</h2>
        <p>일상 생활을 더 스마트하게 만드는 AI 어시스턴트</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>🧳 Travel Scout Agent</h3>
            <p><strong>기능:</strong> 시크릿 모드 여행 검색으로 진짜 최저가 발견</p>
            <p><strong>특징:</strong> 캐시 방지, 가격 조작 차단, 고품질 숙소/항공편</p>
            <p><strong>혁신:</strong> 🔒 인코그니토 브라우징으로 공정한 가격 비교</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Travel Scout 가성비 검색", key="travel_scout", use_container_width=True):
            st.switch_page("pages/travel_scout.py")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>🔍 Research Agent</h3>
            <p><strong>기능:</strong> 정보 검색 및 종합 분석</p>
            <p><strong>특징:</strong> 다중 소스 검증, 트렌드 분석</p>
            <p><strong>사용처:</strong> 시장 조사, 경쟁사 분석, 학술 연구</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Research Agent", key="research", use_container_width=True):
            st.switch_page("pages/research.py")

    # 고급 에이전트들
    st.markdown("""
    <div class="category-header">
        <h2>🧠 고급 AI 에이전트</h2>
        <p>혁신적인 AI 기술을 활용한 차세대 솔루션</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>🏗️ AI Architect Agent</h3>
            <p><strong>기능:</strong> 진화형 AI 아키텍처 설계</p>
            <p><strong>특징:</strong> 자동 최적화, 성능 모니터링</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏗️ AI 아키텍트", key="architect", use_container_width=True):
            st.switch_page("pages/ai_architect.py")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>🤖 Decision Agent</h3>
            <p><strong>기능:</strong> 모바일 인터액션 자동 결정</p>
            <p><strong>특징:</strong> 실시간 개입, 맞춤형 추천</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Decision Agent", key="decision", use_container_width=True):
            st.switch_page("pages/decision_agent.py")
    
    # 기본 에이전트들
    st.markdown("""
    <div class="category-header">
        <h2>⚡ 기본 에이전트</h2>
        <p>일상적인 작업을 위한 실용적인 AI 도구</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>📊 Data Generator</h3>
            <p>다양한 형태의 데이터 생성</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 데이터 생성", key="data_gen", use_container_width=True):
            st.switch_page("pages/data_generator.py")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>🔍 Research Agent</h3>
            <p>정보 검색 및 분석</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 리서치", key="research_basic", use_container_width=True):
            st.switch_page("pages/research.py")
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h3>📝 RAG Agent</h3>
            <p>문서 기반 질의응답</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 문서 분석", key="rag", use_container_width=True):
            st.switch_page("pages/rag_agent.py")

def display_latest_updates():
    """최신 업데이트 및 소식"""
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔥 최신 업데이트
        
        **v2.1.0 (2024-11-15)**
        - 🏥 SEO Doctor 신규 출시 - 3분 사이트 응급진단
        - 🎯 Business Strategy Agent 스파클 모드 추가
        - 📱 모바일 최적화 완료
        - 🚀 바이럴 기능 탑재
        
        **v2.0.5 (2024-11-10)**
        - 💰 Finance Health Agent 성능 개선
        - 🔒 Cybersecurity Agent 보안 강화
        - 🏗️ AI Architect 진화형 알고리즘 업데이트
        
        **v2.0.0 (2024-11-01)**
        - 🎉 통합 플랫폼 론칭
        - 25+ 에이전트 통합 관리
        - 멀티 페이지 내비게이션 지원
        """)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🎯 이번 주 추천</h4>
            <p><strong>SEO Doctor</strong></p>
            <p>사이트 트래픽 급락?<br>3분 내 무료 진단!</p>
        </div>
        
        <div class="feature-highlight">
            <h4>🔥 인기 급상승</h4>
            <p><strong>Business Strategy</strong></p>
            <p>스파클 모드로<br>재미있는 분석 체험!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 사용 가이드
    st.markdown("---")
    
    with st.expander("📖 사용 가이드"):
        st.markdown("""
        ### 🚀 빠른 시작
        
        1. **왼쪽 사이드바**에서 원하는 에이전트 선택
        2. **카테고리별 버튼**을 클릭하여 직접 이동
        3. **각 에이전트 페이지**에서 상세 기능 체험
        
        ### 💡 추천 사용 순서
        
        **비즈니스 분석이 필요하다면:**
        1. 🎯 Business Strategy Agent로 시장 분석
        2. 🏥 SEO Doctor로 온라인 마케팅 진단
        3. 💰 Finance Health Agent로 재무 검토
        
        **개발/기술 관련 작업이라면:**
        1. 🏗️ AI Architect로 아키텍처 설계
        2. 🔄 Workflow Orchestrator로 자동화
        3. 📊 Data Generator로 테스트 데이터 생성
        
        ### 🎯 각 에이전트별 특화 기능
        
        - **실시간 분석**: Business Strategy, SEO Doctor
        - **대화형 인터페이스**: 모든 에이전트 지원
        - **데이터 내보내기**: Excel, PDF, JSON 형식
        - **모바일 최적화**: SEO Doctor, Business Strategy
        """)
    
    # 피드백 섹션
    st.markdown("---")
    
    st.markdown("### 💬 피드백 & 문의")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⭐ 평가하기", use_container_width=True):
            st.balloons()
            st.success("평가해주셔서 감사합니다!")
    
    with col2:
        if st.button("🐛 버그 신고", use_container_width=True):
            st.info("GitHub Issues 페이지로 이동합니다.")
    
    with col3:
        if st.button("💡 기능 제안", use_container_width=True):
            st.info("새로운 아이디어를 제안해주세요!")

if __name__ == "__main__":
    main() 