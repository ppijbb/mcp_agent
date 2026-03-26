"""
🤖 MCP Agent Hub - 통합 AI 에이전트 플랫폼

모든 AI 에이전트들을 한 곳에서 체험할 수 있는 Streamlit 데모
"""

import importlib
import sys
from pathlib import Path
from functools import lru_cache

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Apply compatibility patches safely
try:
    from srcs.common.compatibility import apply_all_compatibility_patches
    apply_all_compatibility_patches()
except ImportError:
    pass  # Compatibility patches not available
except Exception as e:
    # Log error but don't break startup
    print(f"Warning: Compatibility patches failed: {e}")

# Force config reload for fresh imports - optimized with better error handling
config_modules = [
    ('mcp_agent.config', '_settings'),
    ('srcs.core.config.loader', '_config')
]

for module_name, attr_name in config_modules:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        if hasattr(module, attr_name):
            setattr(module, attr_name, None)
    except ImportError:
        continue  # Module not available, skip
    except Exception as e:
        print(f"Warning: Config reload failed for {module_name}: {e}")

# Only invalidate caches if really needed (performance optimization)
if len(sys.modules) > 50:  # Reduced threshold for more aggressive cleanup
    importlib.invalidate_caches()

# Cache for expensive operations
@lru_cache(maxsize=64)
def get_cached_page_content(page_name: str) -> str:
    """Cache page content to improve performance with smaller cache size."""
    return f"Loading {page_name}..."

# Import streamlit and styles with fallback
try:
    import streamlit as st
    from srcs.common.styles import get_common_styles
    STREAMLIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Streamlit not available: {e}")
    STREAMLIT_AVAILABLE = False
    # Create dummy objects for graceful degradation
    class DummyContextManager:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class DummyStreamlit:
        def set_page_config(self, **kwargs): pass
        def markdown(self, content, unsafe_allow_html=False): pass
        def switch_page(self, page): pass
        def container(self, border=False): return DummyContextManager()
        def columns(self, n): return [DummyContextManager()] * n
        def button(self, label, key=None, use_container_width=False): return False
        def expander(self, label, expanded=False): return DummyContextManager()
    
    st = DummyStreamlit()
    def get_common_styles() -> str: return ""

# 페이지 설정
if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="🤖 MCP Agent Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)

def main():
    """
    메인 페이지 - MCP Agent Hub의 진입점
    
    에이전트 허브의 메인 화면을 표시하고 사용자가 다양한 AI 에이전트를 
    탐색하고 접근할 수 있도록 합니다.
    """
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MCP Agent Hub</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            차세대 AI 에이전트 플랫폼 - 비즈니스부터 개인까지
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 대시보드 (최신 업데이트 및 사용 가이드)
    display_dashboard()
    
    # 에이전트 카테고리 표시
    display_agent_categories()

def display_dashboard():
    """
    최신 업데이트 및 사용 가이드를 포함한 대시보드를 표시합니다.
    
    2단 컬럼 레이아웃으로 최신 업데이트 정보와 사용 가이드를 제공합니다.
    """
    with st.container():
        col1, col2 = st.columns([1., 1.])
        with col1:
            st.markdown("""
            ### 🔥 최신 업데이트        
            **v2.4.0**
            - **에이전트 성능 최적화**: 연결 풀 관리 개선
            - **문서화 개선**: 주요 모듈에 docstring 추가
            
            **v2.3.0**
            - **신규 에이전트 추가**: Product Planner, Urban Hive, Workflow Orchestrator
            - **UI/UX 개선**: 메인 화면 재구성 및 카드 디자인 통일
            """)
        
        with col2:
            st.markdown("""
            #### 📖 사용 가이드
            1. **관심 카테고리**에서 에이전트를 선택하세요.
            2. 각 에이전트 페이지로 이동하여 **기능을 체험**해보세요.
            3. 분석 결과나 생성된 데이터를 실제 업무나 프로젝트에 **활용**해보세요.
            """)

def display_agent_categories():
    """
    에이전트 카테고리를 2단 컬럼 레이아웃으로 표시합니다.
    
    비즈니스 전략, 엔터프라이즈, 라이프스타일, 고급 AI, 기본 에이전트 카테고리를
    2단 컬럼으로 구성하여 표시하고 시연환경이 필요한 에이전트들을 별도로 표시합니다.
    """
    main_col1, main_col2 = st.columns(2)
    
    with main_col1:
        display_business_strategy_agents()
        display_lifestyle_agents()
        display_basic_agents()

    with main_col2:
        display_enterprise_agents()
        display_advanced_ai_agents()
    
    # 시연환경 구성이 필요한 Agent들 별도 표시
    display_demo_environment_required_agents()

def display_business_strategy_agents():
    """
    비즈니스 전략 관련 에이전트들을 표시합니다.
    
    Business Strategy Agent, SEO Doctor, Product Planner Agent 등
    비즈니스 전략과 관련된 에이전트들을 카드 형태로 표시합니다.
    """
    st.markdown("""
    <div class="category-header">
        <h2>💼 비즈니스 전략</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
            <h3>🎯 Business Strategy Agent</h3>
            <p>시장, 경쟁사 분석 및 비즈니스 모델 설계</p>
        """, unsafe_allow_html=True)
        if st.button("Business Strategy 체험하기", key="bs_agent", use_container_width=True):
            st.switch_page("pages/business_strategy.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🏥 SEO Doctor</h3>
            <p>사이트 응급진단, 경쟁사 분석 및 SEO 처방</p>
        """, unsafe_allow_html=True)
        if st.button("SEO Doctor 응급진단", key="seo_doctor", use_container_width=True):
            st.switch_page("pages/seo_doctor.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🚀 Product Planner Agent</h3>
            <p>Figma 디자인 분석, 프로덕트 기획, 시장 조사</p>
        """, unsafe_allow_html=True)
        if st.button("Product Planner 기획 분석", key="product_planner", use_container_width=True):
            st.switch_page("pages/product_planner.py")

def display_enterprise_agents():
    """
    엔터프라이즈 관련 에이전트들을 표시합니다.
    
    Finance Health Agent, HR Recruitment Agent 등
    엔터프라이즈 환경에서 사용되는 에이전트들을 표시합니다.
    """
    st.markdown("""
    <div class="category-header">
        <h2>🏢 엔터프라이즈</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h3>💰 Finance Health Agent</h3><p>재무 건강도 진단 및 최적화</p>", unsafe_allow_html=True)
        if st.button("재무 분석하기", key="finance", use_container_width=True):
            st.switch_page("pages/finance_health.py")


    with st.container(border=True):
        st.markdown("<h3>👥 HR Recruitment Agent</h3><p>인재 채용 및 관리 최적화</p>", unsafe_allow_html=True)
        if st.button("HR 관리", key="hr", use_container_width=True):
            st.switch_page("pages/hr_recruitment.py")

def display_lifestyle_agents():
    """
    라이프스타일 관련 에이전트들을 표시합니다.
    
    Travel Scout Agent, Research Agent, Urban Hive Agent 등
    개인 생활과 관련된 에이전트들을 표시합니다.
    """
    st.markdown("""
    <div class="category-header">
        <h2>🌟 라이프스타일</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
            <h3>🧳 Travel Scout Agent</h3>
            <p>시크릿 모드 여행 검색으로 최저가 발견</p>
        """, unsafe_allow_html=True)
        if st.button("Travel Scout 가성비 검색", key="travel_scout", use_container_width=True):
            st.switch_page("pages/travel_scout.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🔍 Research Agent</h3>
            <p>정보 검색, 다중 소스 검증 및 종합 분석</p>
        """, unsafe_allow_html=True)
        if st.button("Research Agent", key="research", use_container_width=True):
            st.switch_page("pages/research.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🏙️ Urban Hive Agent</h3>
            <p>도시 데이터(교통, 안전, 부동산) 분석</p>
        """, unsafe_allow_html=True)
        if st.button("Urban Hive 도시 분석", key="urban_hive", use_container_width=True):
            st.switch_page("pages/urban_hive.py")


def display_advanced_ai_agents():
    """
    고급 AI 관련 에이전트들을 표시합니다.
    
    Decision Agent, Workflow Orchestrator, Boardgame UI Generator 등
    고급 AI 기술을 활용하는 에이전트들을 표시합니다.
    """
    st.markdown("""
    <div class="category-header">
        <h2>🧠 고급 AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    
    with st.container(border=True):
        st.markdown("""
            <h3>🤖 Decision Agent</h3>
            <p>모바일 인터랙션 자동 결정 및 실시간 개입</p>
        """, unsafe_allow_html=True)
        if st.button("Decision Agent", key="decision", use_container_width=True):
            st.switch_page("pages/decision_agent.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🔄 Workflow Orchestrator</h3>
            <p>워크플로우 자동화 및 다중 에이전트 협업</p>
        """, unsafe_allow_html=True)
        if st.button("Workflow Orchestrator 실행", key="workflow", use_container_width=True):
            st.switch_page("pages/workflow.py")

    with st.container(border=True):
        st.markdown("""
            <h3>🎲 Boardgame UI Generator</h3>
            <p>LangGraph 기반 보드게임 UI 분석 및 생성</p>
        """, unsafe_allow_html=True)
        if st.button("Boardgame UI 생성", key="boardgame_ui", use_container_width=True):
            st.switch_page("pages/boardgame_ui_generator.py")

def display_basic_agents():
    """
    기본 기능 에이전트들을 표시합니다.
    
    Data Generator, RAG Agent 등
    기본적인 기능을 제공하는 에이전트들을 표시합니다.
    """
    st.markdown("""
    <div class="category-header">
        <h2>⚡ 기본</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("""
            <h3>📊 Data Generator</h3>
            <p>다양한 형태의 테스트 및 목업 데이터 생성</p>
        """, unsafe_allow_html=True)
        if st.button("데이터 생성", key="data_gen", use_container_width=True):
            st.switch_page("pages/data_generator.py")
    
    with st.container(border=True):
        st.markdown("""
            <h3>📝 RAG Agent</h3>
            <p>문서 기반 질의응답 및 정보 추출</p>
        """, unsafe_allow_html=True)
        if st.button("문서 분석", key="rag", use_container_width=True):
            st.switch_page("pages/rag_agent.py")

def display_demo_environment_required_agents():
    """
    시연환경 구성이 필요한 에이전트들을 표시합니다.
    
    Drone Scout Agent, AIOps Orchestrator Agent, DevOps Assistant Agent 등
    실제 인프라, 하드웨어, 또는 클라우드 서비스가 필요한 에이전트들을
    별도 섹션에서 표시하고 구성 가이드를 제공합니다.
    """
    st.markdown("---")
    st.markdown("""
    <div class="category-header">
        <h2>🔧 시연환경 구성 필요</h2>
        <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
            ⚠️ 아래 Agent들은 실제 인프라, 하드웨어, 또는 클라우드 서비스가 필요합니다.
            데모를 위해서는 별도의 시연 환경 구성이 필요합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("""
                <h3>🛸 Drone Scout Agent</h3>
                <p>자연어 임무를 입력하여 자율 드론 정찰</p>
                <p style="font-size: 0.85rem; color: #ff6b6b;">
                    ⚠️ 필요: 드론 하드웨어 또는 시뮬레이터
                </p>
            """, unsafe_allow_html=True)
            if st.button("Drone Scout 미션 실행", key="drone_scout_demo", use_container_width=True):
                st.switch_page("pages/drone_scout.py")
        
        with st.container(border=True):
            st.markdown("""
                <h3>🤖 AIOps Orchestrator Agent</h3>
                <p>AI 기반 IT 운영 자동화 및 모니터링</p>
                <p style="font-size: 0.85rem; color: #ff6b6b;">
                    ⚠️ 필요: 실제 서버/인프라, Kubernetes, 모니터링 시스템
                </p>
            """, unsafe_allow_html=True)
            if st.button("AIOps 작업 실행", key="aiops_demo", use_container_width=True):
                st.switch_page("pages/aiops_orchestrator.py")
        
        with st.container(border=True):
            st.markdown("""
                <h3>🚀 DevOps Assistant Agent</h3>
                <p>GitHub, AWS, Kubernetes 등 개발자 생산성 자동화</p>
                <p style="font-size: 0.85rem; color: #ff6b6b;">
                    ⚠️ 필요: GitHub 계정, AWS/GCP/Azure, Kubernetes 클러스터
                </p>
            """, unsafe_allow_html=True)
            if st.button("DevOps 작업 실행", key="devops_demo", use_container_width=True):
                st.switch_page("pages/devops_assistant.py")
    
    with col2:
        with st.container(border=True):
            st.markdown("""
                <h3>🏗️ AI Architect Agent</h3>
                <p>진화형 AI 아키텍처 설계 및 자동 최적화</p>
                <p style="font-size: 0.85rem; color: #ff6b6b;">
                    ⚠️ 필요: AI/ML 인프라, GPU 클러스터, 성능 벤치마크 환경
                </p>
            """, unsafe_allow_html=True)
            if st.button("AI 아키텍트", key="architect_demo", use_container_width=True):
                st.switch_page("pages/ai_architect.py")
        
        with st.container(border=True):
            st.markdown("""
                <h3>🔒 Cybersecurity Agent</h3>
                <p>사이버 보안 인프라 관리 및 위협 분석</p>
                <p style="font-size: 0.85rem; color: #ff6b6b;">
                    ⚠️ 필요: 보안 인프라, 방화벽, 보안 스캐닝 도구
                </p>
            """, unsafe_allow_html=True)
            if st.button("보안 체크", key="cyber_demo", use_container_width=True):
                st.switch_page("pages/cybersecurity_agent.py")
    
    # 안내 메시지
    with st.expander("ℹ️ 시연환경 구성 가이드", expanded=False):
        st.markdown("""
        ### 시연환경 구성이 필요한 Agent들
        
        위 Agent들은 실제 인프라나 하드웨어가 필요합니다. 데모를 위해서는:
        
        1. **최소 구성**: 로컬 개발 환경 (Docker, minikube, 테스트 계정)
        2. **완전 구성**: 클라우드 계정, Kubernetes 클러스터, 모니터링 스택
        
        자세한 내용은 [시연환경 구성 가이드](docs/DEMO_ENVIRONMENT_REQUIREMENTS.md)를 참고하세요.
        
        **참고**: 일부 Agent는 모의 데이터로 기능만 시연할 수 있습니다.
        """)

if __name__ == "__main__":
    main() 