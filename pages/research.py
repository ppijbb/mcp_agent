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
    try:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 🎯 연구 설정")
            research_topic = st.text_input(
                "연구 주제",
                value=None,
                placeholder="조사하고 싶은 주제를 입력하세요",
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
                    value=None,
                    placeholder="연구 초점을 직접 입력하세요"
                )
            save_to_file = st.checkbox(
                "파일로 저장",
                value=False,
                help=f"체크하면 {get_reports_path('research')}/ 디렉토리에 파일로 저장합니다"
            )
            if not research_topic:
                st.warning("연구 주제를 입력해주세요.")
            elif not research_focus:
                st.warning("연구 초점을 선택하거나 입력해주세요.")
            else:
                if st.button("🚀 Research Agent 실행", type="primary", use_container_width=True):
                    # 결과 파일 경로 생성
                    reports_path = get_reports_path('research')
                    os.makedirs(reports_path, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_json_path = os.path.join(reports_path, f"research_result_{timestamp}.json")
                    # 커맨드 생성
                    command = [
                        "python", "-u",
                        "srcs/advanced_agents/run_research_agent.py",
                        "--topic", research_topic,
                        "--focus", research_focus,
                        "--result-json-path", result_json_path
                    ]
                    if save_to_file:
                        command.append("--save-to-file")
                    st.session_state['research_command'] = command
                    st.session_state['research_result_json_path'] = result_json_path
        with col2:
            if 'research_command' in st.session_state:
                st.info("🔄 Research Agent 실행 중...")
                process = Process(
                    st.session_state['research_command'],
                    output_file=None
                ).start()
                spm.st_process_monitor(
                    process,
                    label="연구 분석"
                ).loop_until_finished()
                # 결과 파일 읽기 및 표시
                try:
                    with open(st.session_state['research_result_json_path'], 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    if result.get('success'):
                        st.success("✅ Research Agent 실행 완료!")
                        st.markdown("#### 📊 실행 결과")
                        st.info(f"**메시지**: {result['message']}")
                        st.info(f"**연구 주제**: {result['topic']}")
                        if result.get('save_to_file') and result.get('output_dir'):
                            st.info(f"**출력 디렉토리**: {result['output_dir']}")
                        st.info(f"**연구 초점**: {result['focus']}")
                        if 'content' in result and result['content']:
                            st.markdown("#### 📄 생성된 연구 결과")
                            content = result['content']
                            if len(content) > 1500:
                                with st.expander("📋 전체 연구 결과 보기", expanded=True):
                                    st.markdown(content)
                            else:
                                st.markdown(content)
                            st.download_button(
                                label="📥 연구 결과 다운로드",
                                data=content,
                                file_name=f"research_result_{result['topic'].replace(' ', '_').lower()}_{result['focus'].replace(' ', '_')}.md",
                                mime="text/markdown"
                            )
                        with st.expander("🔍 상세 실행 정보"):
                            st.json({
                                'success': result['success'],
                                'message': result['message'],
                                'topic': result['topic'],
                                'focus': result['focus'],
                                'save_to_file': result['save_to_file'],
                                'output_dir': result.get('output_dir'),
                                'timestamp': result.get('timestamp'),
                                'content_length': len(result.get('content', '')) if result.get('content') else 0
                            })
                    else:
                        st.error("❌ 실행 중 오류 발생")
                        st.error(f"**오류**: {result.get('message')}")
                        with st.expander("🔍 오류 상세"):
                            st.code(result.get('error', 'Unknown error'))
                except Exception as e:
                    st.error(f"결과 파일을 읽는 중 오류 발생: {e}")
                # 실행 후 상태 초기화
                del st.session_state['research_command']
                del st.session_state['research_result_json_path']
            else:
                display_research_info()
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")

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