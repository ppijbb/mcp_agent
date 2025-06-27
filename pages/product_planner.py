import streamlit as st
from pathlib import Path
import sys
import json
import os
from datetime import datetime
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# --- 상수 정의 ---
REPORTS_PATH = project_root / "planning"

def find_latest_report() -> Path | None:
    """`planning` 디렉토리에서 가장 최근에 생성된 마크다운 보고서 파일을 찾습니다."""
    if not REPORTS_PATH.exists():
        return None
    
    markdown_files = list(REPORTS_PATH.glob("*.md"))
    if not markdown_files:
        return None
        
    latest_file = max(markdown_files, key=lambda p: p.stat().st_mtime)
    return latest_file

def main():
    """Product Planner Agent 모니터링 페이지 (프로세스 모니터링)"""
    setup_page("🚀 Product Planner Agent", "🚀")
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    header_html = get_page_header("product", "🚀 Product Planner Agent", "Figma URL을 입력하여 프로덕트 기획 분석을 시작하고, 진행 상황을 실시간으로 확인합니다.")
    st.markdown(header_html, unsafe_allow_html=True)
    render_home_button()
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        with st.form("product_planner_form"):
        st.markdown("### 🎯 분석 시작하기")

        figma_url = st.text_input(
            "Figma URL", 
                placeholder="https://www.figma.com/file/your_file_id/...",
            help="분석할 Figma 파일의 전체 URL을 입력하세요. 'node-id'가 포함되어야 합니다."
        )
        figma_api_key = st.text_input(
            "Figma API Key", 
            type="password",
            help="Figma 계정 설정에서 발급받은 API 키를 입력하세요."
        )

            submitted = st.form_submit_button("🚀 분석 시작", use_container_width=True)

            if submitted:
                if not (figma_url and figma_api_key and "figma.com/file/" in figma_url and "node-id=" in figma_url):
                    st.error("올바른 Figma URL과 API 키를 모두 입력해주세요.")
            else:
                    command = [
                        "python", "-u",
                        "srcs/product_planner_agent/run_product_planner.py",
                        "--figma-url", figma_url,
                        "--figma-api-key", figma_api_key,
                    ]
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(REPORTS_PATH, f"planner_agent_output_{timestamp}.log")
                    os.makedirs(REPORTS_PATH, exist_ok=True)
                    
                    st.session_state['planner_command'] = command
                    st.session_state['planner_output_file'] = output_file

    with col2:
        if 'planner_command' in st.session_state:
            st.info("🔄 Product Planner Agent 실행 중...")
            
            process = Process(
                st.session_state['planner_command'],
                output_file=st.session_state['planner_output_file']
            ).start()
            
            spm.st_process_monitor(
                process,
                label="프로덕트 기획 분석"
            ).loop_until_finished()
            
            st.success(f"✅ 분석 프로세스가 완료되었습니다. 전체 로그는 {st.session_state['planner_output_file']}에 저장됩니다.")

            with st.container():
                st.balloons()
                st.success("🎉 모든 작업이 성공적으로 완료되었습니다!")
                st.markdown("### 📄 최종 보고서")
                
                latest_report = find_latest_report()
                if latest_report:
                    st.info(f"가장 최근에 생성된 보고서: `{latest_report.name}`")
                    try:
                        report_content = latest_report.read_text(encoding="utf-8")
                        with st.expander("보고서 내용 보기", expanded=True):
                            st.markdown(report_content)
                    except Exception as e:
                        st.error(f"보고서 파일을 읽는 중 오류 발생: {e}")
                else:
                    st.warning("생성된 보고서를 찾을 수 없습니다.")

            # 실행 후 상태 초기화
            del st.session_state['planner_command']
            del st.session_state['planner_output_file']
        else:
            st.info("좌측에 Figma 정보를 입력하고 분석을 시작하세요.")

if __name__ == "__main__":
    main() 