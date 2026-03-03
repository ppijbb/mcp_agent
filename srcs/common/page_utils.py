"""
Page Utilities Module

Provides common utility functions for Streamlit pages including page setup,
rendering, and agent integration. Supports both standard and A2A agent workflows.

Functions:
    setup_page: Configure Streamlit page settings
    add_project_root: Add project root to Python path
    setup_page_header: Render page header with title and subtitle
    render_page_header: Render HTML page header
    render_common_styles: Apply common CSS styles
    render_home_button: Render home navigation button
    safe_import_agent: Safely import agent modules with fallback
    render_import_error: Display agent import errors
    render_agent_intro: Render agent introduction with features
    create_agent_page: Create unified agent page
    render_demo_content: Render demo content with tabs
    render_metrics_row: Render metrics row
"""

import streamlit as st
import sys
from pathlib import Path
from .styles import get_common_styles, get_page_header


def setup_page(title, icon, layout="wide"):
    """
    페이지 기본 설정.
    
    Args:
        title: 페이지 제목
        icon: 페이지 아이콘 (emoji)
        layout: 페이지 레이아웃 ("wide" 또는 "centered")
    """
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout=layout
        )
    except Exception:
        # set_page_config가 이미 호출된 경우 무시
        pass


def add_project_root():
    """프로젝트 루트를 Python 경로에 추가하여 모듈 임포트가 가능하도록 함."""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))


def setup_page_header(title, subtitle=""):
    """
    페이지 헤더 설정 (간단 버전).
    
    Args:
        title: 페이지 제목
        subtitle: 페이지 서브타이틀 (선택)
    """
    st.title(f"🚀 {title}")
    if subtitle:
        st.subheader(subtitle)


def render_page_header(page_type, title, subtitle):
    """
    페이지 헤더를 HTML로 렌더링합니다.
    
    Args:
        page_type: 페이지 유형
        title: 제목
        subtitle: 서브타이틀
    """
    header_html = get_page_header(page_type, title, subtitle)
    st.markdown(header_html, unsafe_allow_html=True)


def render_common_styles():
    """공통 CSS 스타일을 페이지에 적용합니다."""
    st.markdown(get_common_styles(), unsafe_allow_html=True)


def render_home_button():
    """홈으로 돌아가는 Streamlit 버튼을 렌더링합니다."""
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")


def safe_import_agent(module_path, fallback_name="Agent"):
    """
    에이전트 모듈을 안전하게 임포트합니다.
    
    Args:
        module_path: 임포트할 모듈 경로
        fallback_name: 폴백 에이전트 이름
        
    Returns:
        Tuple[bool, Optional[module], Optional[str]]: (성공여부, 모듈, 오류메시지)
    """
    try:
        module = __import__(module_path, fromlist=[fallback_name])
        return True, module, None
    except ImportError as e:
        return False, None, str(e)


def render_import_error(agent_name, error_message):
    """
    에이전트 임포트 오류를 화면에 표시합니다.
    
    Args:
        agent_name: 에이전트 이름
        error_message: 오류 메시지
    """
    st.error(f"{agent_name}을 불러올 수 없습니다.")
    st.error(f"오류 내용: {error_message}")

    st.markdown("### 🔧 수동 설치 가이드")
    st.info(f"{agent_name}를 별도로 실행해주세요.")


def render_agent_intro(agent_name, features, special_features=None, use_cases=None):
    """에이전트 소개 렌더링"""
    st.markdown(f"### 🎯 {agent_name} 소개")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 주요 기능")
        for feature in features:
            st.markdown(f"- {feature}")

    if special_features:
        with col2:
            st.markdown("#### ✨ 스페셜 기능")
            for feature in special_features:
                st.markdown(f"- {feature}")

    if use_cases:
        st.markdown("#### 🎯 사용 사례")
        for use_case in use_cases:
            st.markdown(f"- {use_case}")


def create_agent_page(
    agent_name,
    page_icon,
    page_type,
    title,
    subtitle,
    module_path=None,
    main_function_name="main",
    features=None,
    special_features=None,
    use_cases=None
):
    """통합 에이전트 페이지 생성 함수"""

    # 페이지 설정
    setup_page(f"{page_icon} {agent_name}", page_icon)

    # 프로젝트 루트 추가
    add_project_root()

    # 공통 스타일 적용
    render_common_styles()

    # 헤더 렌더링
    render_page_header(page_type, title, subtitle)

    # 홈 버튼
    render_home_button()

    st.markdown("---")


def render_demo_content(demo_data):
    """데모 콘텐츠 렌더링"""
    if "tabs" in demo_data:
        tabs = st.tabs([tab["name"] for tab in demo_data["tabs"]])

        for i, tab_data in enumerate(demo_data["tabs"]):
            with tabs[i]:
                if "markdown" in tab_data:
                    st.markdown(tab_data["markdown"])
                if "chart" in tab_data:
                    st.plotly_chart(tab_data["chart"], use_container_width=True)
                if "dataframe" in tab_data:
                    st.dataframe(tab_data["dataframe"])


def render_metrics_row(metrics):
    """메트릭 행 렌더링"""
    cols = st.columns(len(metrics))

    for i, metric in enumerate(metrics):
        with cols[i]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta", None)
            )
