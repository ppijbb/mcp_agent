"""
Page Utilities Module

페이지에서 공통으로 사용하는 유틸리티 함수들
"""

import streamlit as st
import sys
from pathlib import Path
from .styles import get_common_styles, get_page_header

def setup_page(title, icon, layout="wide"):
    """페이지 기본 설정"""
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
    """프로젝트 루트를 Python 경로에 추가"""
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

def setup_page_header(title, subtitle=""):
    """페이지 헤더 설정 (간단 버전)"""
    st.title(f"🚀 {title}")
    if subtitle:
        st.subheader(subtitle)

def render_page_header(page_type, title, subtitle):
    """페이지 헤더 렌더링"""
    header_html = get_page_header(page_type, title, subtitle)
    st.markdown(header_html, unsafe_allow_html=True)

def render_common_styles():
    """공통 스타일 적용"""
    st.markdown(get_common_styles(), unsafe_allow_html=True)

def render_home_button():
    """홈으로 돌아가기 버튼 렌더링"""
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")

def safe_import_agent(module_path, fallback_name="Agent"):
    """안전한 agent 모듈 임포트"""
    try:
        module = __import__(module_path, fromlist=[fallback_name])
        return True, module, None
    except ImportError as e:
        return False, None, str(e)

def render_import_error(agent_name, error_message):
    """임포트 오류 표시"""
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
    module_path,
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
    
    # 에이전트 모듈 임포트 시도
    success, module, error = safe_import_agent(module_path)
    
    if success:
        try:
            # 메인 함수 실행
            main_func = getattr(module, main_function_name)
            main_func()
            
        except Exception as e:
            st.error(f"{agent_name} 실행 중 오류가 발생했습니다: {e}")
            render_import_error(agent_name, str(e))
            
            if features:
                render_agent_intro(agent_name, features, special_features, use_cases)
    else:
        render_import_error(agent_name, error)
        
        if features:
            render_agent_intro(agent_name, features, special_features, use_cases)

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