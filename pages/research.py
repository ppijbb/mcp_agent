"""
🔍 Research Agent Page

정보 검색 및 분석 AI
"""

import streamlit as st
import sys
from pathlib import Path
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process
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
        <h1>🔍 Research Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 정보 검색 및 분석 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
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
            
        reports_path = get_reports_path('research')
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
        
        st.info("🔄 Research Agent 실행 중...")
        
        process = Process(command, key=process_key).start()
        
        st_process_monitor = spm.st_process_monitor(process, key=f"monitor_{process_key}")
        st_process_monitor.loop_until_finished()
        
        if process.get_return_code() == 0:
            st.success("✅ Research Agent 실행 완료!")
            try:
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if result.get('success'):
                    display_research_results(result)
                else:
                    st.error(f"❌ 실행 중 오류가 보고되었습니다: {result.get('message', '알 수 없는 오류')}")
                    with st.expander("🔍 오류 상세 정보"):
                        st.code(result.get('error', '상세 정보 없음'))

            except Exception as e:
                st.error(f"결과 파일을 읽거나 처리하는 중 오류가 발생했습니다: {e}")
        else:
            st.error(f"❌ 에이전트 실행에 실패했습니다. (Return Code: {process.get_return_code()})")
            with st.expander("에러 로그 보기"):
                st.code(process.get_stdout() + process.get_stderr(), language="log")


def display_research_results(result: dict):
    """연구 결과 표시"""
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
        
        with st.container(border=True):
            st.markdown(content)
        
        st.download_button(
            label="📥 연구 결과 전문 다운로드 (.md)",
            data=content,
            file_name=f"research_report_{result.get('topic', 'untitled').replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
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