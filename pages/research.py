"""
🔍 Research Agent Page

정보 검색 및 분석 AI
"""

import streamlit as st
import sys
from pathlib import Path
import streamlit_process_manager as spm
from srcs.common.ui_utils import run_agent_process
import tempfile
import json
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 임포트
from configs.settings import get_reports_path

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# Research Agent 임포트 시도
try:
    from srcs.advanced_agents.researcher_v2 import (
        ResearcherAgent,
        load_research_focus_options,
        load_research_templates,
        get_research_agent_status,
        save_research_report
    )
except ImportError as e:
    st.error(f"⚠️ Research Agent를 불러올 수 없습니다: {e}")
    st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
    st.stop()

def validate_research_result(result):
    """연구 결과 검증"""
    if not result:
        raise Exception("Research Agent에서 유효한 결과를 반환하지 않았습니다")
    return result

def main():
    """Research Agent 메인 페이지"""
    
    create_agent_page(
        agent_name="Research Agent",
        page_icon="🔍",
        page_type="research",
        title="Research Agent",
        subtitle="AI 기반 정보 검색 및 분석 시스템",
        module_path="srcs.advanced_agents.researcher_v2"
    )
    
    st.success("🤖 Research Agent v2가 성공적으로 연결되었습니다!")
    
    # 에이전트 인터페이스
    render_research_agent_interface()

def render_research_agent_interface():
    """Research Agent 실행 인터페이스 (실시간 프로세스 모니터링)"""
    st.markdown("### 🚀 Research Agent 실행")
    
    process_key = "research_process"

    with st.form(key="research_form"):
        st.markdown("#### 🎯 연구 설정")
        research_topic = st.text_input(
            "연구 주제",
            placeholder="예: 인공지능이 채용 시장에 미치는 영향",
            help="조사하고 싶은 주제를 입력하세요"
        )
        try:
            focus_options = load_research_focus_options()
            research_focus = st.selectbox(
                "연구 초점",
                focus_options,
                index=None,
                placeholder="연구 초점을 선택하세요"
            )
        except Exception as e:
            st.warning(f"연구 초점 옵션 로드 실패: {e}")
            research_focus = st.text_input(
                "연구 초점",
                placeholder="연구 초점을 직접 입력하세요"
            )

        submitted = st.form_submit_button("🚀 Research Agent 실행", type="primary", use_container_width=True)

    if submitted:
        if not research_topic or not research_focus:
            st.warning("연구 주제와 초점을 모두 입력(선택)해주세요.")
            st.stop()
            
        reports_path = Path(get_reports_path('research'))
        reports_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in research_topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        result_json_path = reports_path / f"research_result_{safe_topic}_{timestamp}.json"
        
        py_executable = sys.executable
        command = [
            py_executable, "-u", "-m", "srcs.advanced_agents.run_research_agent",
            "--topic", research_topic,
            "--focus", research_focus,
            "--result-json-path", str(result_json_path),
            "--save-to-file" # Always save report file from script
        ]
        
        placeholder = st.empty()
        result = run_agent_process(
            placeholder=placeholder,
            command=command,
            process_key_prefix="logs/research",
            log_expander_title="실시간 실행 로그"
        )
        
        if result:
            if result.get('success'):
                display_research_results(result)
            else:
                st.error(f"❌ 실행 중 오류가 보고되었습니다: {result.get('message', '알 수 없는 오류')}")
                with st.expander("🔍 오류 상세 정보"):
                    st.code(result.get('error', '상세 정보 없음'))


def display_research_results(result: dict):
    """연구 결과 표시 (탭 형식으로 개선)"""
    st.markdown("---")
    st.markdown("#### 📊 실행 결과 요약")
    
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.info(f"**주제**: {result.get('topic', 'N/A')}")
    with summary_cols[1]:
        st.info(f"**초점**: {result.get('focus', 'N/A')}")

    if result.get('output_dir'):
        st.success(f"**보고서 파일 경로**: `{result['output_dir']}`")
    
    if 'content' in result and result['content']:
        st.markdown("#### 📄 생성된 연구 보고서")
        content = result['content']
        
        # Markdown 내용을 섹션별로 분리
        sections = content.split('## ')
        
        # 첫 번째 요소는 보통 제목 이전의 내용이므로, 비어있지 않으면 '소개'로 처리
        tabs_data = {}
        if sections[0].strip():
            tabs_data["소개"] = sections[0]
        
        for section in sections[1:]:
            parts = section.split('\\n', 1)
            title = parts[0].strip().replace('#', '')
            body = parts[1].strip() if len(parts) > 1 else ""
            if title:
                tabs_data[title] = "## " + section # 원래 마크다운 형식 유지

        # '전체 보고서' 탭 추가
        tabs_data["전체 보고서 보기"] = content

        tab_titles = list(tabs_data.keys())
        tabs = st.tabs(tab_titles)
        
        for i, title in enumerate(tab_titles):
            with tabs[i]:
                st.markdown(tabs_data[title])

        st.download_button(
            label="📥 연구 결과 전문 다운로드 (.md)",
            data=content,
            file_name=f"research_report_{result.get('topic', 'untitled').replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="research_download"
        )

    with st.expander("🔍 상세 실행 정보 (JSON)"):
        st.json(result)

def display_research_info():
    """연구 에이전트 정보 표시"""
    st.markdown("""
    #### 🤖 Research Agent 정보
    
    **실행되는 프로세스:**
    1. **다중 에이전트 생성** - 전문 연구 AI 에이전트들
    2. **MCP App 초기화** - MCP 프레임워크 연결
    3. **오케스트레이터 실행** - 통합 워크플로우 관리
    4. **연구 수행** - 포괄적 정보 수집 및 분석
    
    **생성되는 연구 결과:**
    - 📈 **트렌드 분석**: 현재 동향 및 발전 패턴
    - 🏢 **경쟁 분석**: 주요 업체 및 시장 현황
    - 🔮 **미래 전망**: 전략적 시사점 및 기회
    - 📋 **종합 보고서**: 실행 요약 및 권고사항
    
    **출력 옵션:**
    - 🖥️ **화면 표시**: 즉시 결과 확인 (기본값)
    - 💾 **파일 저장**: research_reports/ 디렉토리에 저장
    """)

if __name__ == "__main__":
    main()

# 최신 Research Agent 결과 확인
st.markdown("---")
st.markdown("## 📊 최신 Research Agent 결과")

latest_research_result = result_reader.get_latest_result("research_agent", "research_analysis")

if latest_research_result:
    with st.expander("🔍 최신 연구 분석 결과", expanded=False):
        st.subheader("🤖 최근 연구 분석 결과")
        
        if isinstance(latest_research_result, dict):
            # 연구 정보 표시
            topic = latest_research_result.get('topic', 'N/A')
            focus = latest_research_result.get('focus', 'N/A')
            
            st.success(f"**연구 주제: {topic}**")
            st.info(f"**연구 초점: {focus}**")
            
            # 연구 결과 요약
            col1, col2, col3 = st.columns(3)
            col1.metric("연구 상태", "완료" if latest_research_result.get('success', False) else "실패")
            col2.metric("보고서 길이", f"{len(latest_research_result.get('content', ''))} 문자")
            col3.metric("출력 디렉토리", "저장됨" if latest_research_result.get('output_dir') else "미저장")
            
            # 연구 내용 표시
            content = latest_research_result.get('content', '')
            if content:
                st.subheader("📄 연구 보고서")
                with st.expander("보고서 내용", expanded=False):
                    st.markdown(content)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 연구 보고서 다운로드 (.md)",
                    data=content,
                    file_name=f"research_report_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # 메타데이터 표시
            if 'timestamp' in latest_research_result:
                st.caption(f"⏰ 연구 시간: {latest_research_result['timestamp']}")
        else:
            st.json(latest_research_result)
else:
    st.info("💡 아직 Research Agent의 결과가 없습니다. 위에서 연구 분석을 실행해보세요.") 