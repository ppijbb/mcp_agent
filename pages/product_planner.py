import streamlit as st
from pathlib import Path
import sys
import json
import os
import time
from datetime import datetime
import asyncio
import threading

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button
from srcs.product_planner_agent.product_planner_agent import run_agent_workflow

# --- 상수 정의 ---
STATUS_FILE = project_root / "srcs" / "product_planner_agent" / "utils" / "status.json"
FINAL_REPORT_DIR = project_root / "planning"
REFRESH_INTERVAL = 3  # 초 단위

def agent_runner(figma_url: str, figma_api_key: str):
    """에이전트 워크플로우를 별도의 스레드에서 실행하기 위한 래퍼 함수"""
    try:
        # 새 이벤트 루프를 생성하고 설정
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 에이전트 워크플로우 실행
        success = loop.run_until_complete(run_agent_workflow(figma_url, figma_api_key))
        if success:
            print("✅ Agent thread finished successfully.")
        else:
            print("❌ Agent thread finished with errors.")
    except Exception as e:
        print(f"💥 Critical error in agent runner thread: {e}")
    finally:
        # 세션 상태를 직접 수정하는 대신, 파일 기반 신호를 사용하거나
        # 더 복잡한 상태 관리 메커니즘을 고려할 수 있습니다.
        # 여기서는 단순화를 위해 별도 조치는 취하지 않습니다.
        # Streamlit의 재실행 루프가 상태 파일 변경을 감지할 것입니다.
        pass

def read_status_file() -> dict:
    """상태 파일을 읽어서 내용을 반환합니다."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"상태 파일을 읽는 중 오류 발생: {e}")
    return {}

def find_latest_report() -> Path | None:
    """`planning` 디렉토리에서 가장 최근에 생성된 마크다운 보고서 파일을 찾습니다."""
    if not FINAL_REPORT_DIR.exists():
        return None
    
    markdown_files = list(FINAL_REPORT_DIR.glob("*.md"))
    if not markdown_files:
        return None
        
    latest_file = max(markdown_files, key=lambda p: p.stat().st_mtime)
    return latest_file

def render_status(statuses: dict):
    """현재 진행 상태를 UI에 렌더링합니다."""
    if not statuses:
        # st.session_state.agent_running이 True인데 상태 파일이 아직 안생겼을 수 있음
        if st.session_state.get('agent_running', False):
            st.info("에이전트 초기화 중... 잠시 후 진행 상황이 표시됩니다.")
        else:
            st.info("아래에 Figma 정보를 입력하고 분석을 시작하세요.")
        return

    st.markdown("#### 📊 실시간 진행 현황")

    steps = list(statuses.keys())
    status_values = list(statuses.values())
    
    # 각 단계별 상태 표시
    cols = st.columns(len(steps))
    for i, (step, status) in enumerate(statuses.items()):
        with cols[i]:
            if status == "completed":
                st.success(f"**{i+1}. {step}**\n\n✅ 완료")
            elif status == "in_progress":
                st.info(f"**{i+1}. {step}**\n\n⏳ 진행 중...")
            elif status == "failed":
                st.error(f"**{i+1}. {step}**\n\n❌ 실패")
            else:
                st.warning(f"**{i+1}. {step}**\n\n🕒 대기 중")

    # 전체 진행률 계산
    completed_count = status_values.count("completed")
    progress = completed_count / len(steps) if steps else 0
    
    st.progress(progress, text=f"전체 진행률: {progress:.0%}")

def main():
    """Product Planner Agent 모니터링 페이지"""
    setup_page("🚀 Product Planner Agent", "🚀")
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    header_html = get_page_header("product", "🚀 Product Planner Agent", "Figma URL을 입력하여 프로덕트 기획 분석을 시작하고, 진행 상황을 실시간으로 확인합니다.")
    st.markdown(header_html, unsafe_allow_html=True)
    render_home_button()
    st.markdown("---")

    # --- 에이전트 실행 제어 ---
    with st.container(border=True):
        st.markdown("### 🎯 분석 시작하기")
        
        # 세션 상태 초기화
        if 'agent_running' not in st.session_state:
            st.session_state.agent_running = False

        figma_url = st.text_input(
            "Figma URL", 
            placeholder="https://www.figma.com/file/your_file_id/your_project_name?node-id=your_node_id",
            help="분석할 Figma 파일의 전체 URL을 입력하세요. 'node-id'가 포함되어야 합니다."
        )
        figma_api_key = st.text_input(
            "Figma API Key", 
            type="password",
            help="Figma 계정 설정에서 발급받은 API 키를 입력하세요."
        )

        if st.button("🚀 분석 시작", disabled=st.session_state.agent_running):
            if figma_url and figma_api_key and "figma.com/file/" in figma_url and "node-id=" in figma_url:
                with st.spinner("에이전트 스레드를 시작하는 중입니다..."):
                    st.session_state.agent_running = True
                    # 별도 스레드에서 에이전트 실행
                    thread = threading.Thread(
                        target=agent_runner,
                        args=(figma_url, figma_api_key),
                        daemon=True
                    )
                    thread.start()
                    st.success("에이전트가 백그라운드에서 실행을 시작했습니다. 아래에서 진행 상황을 확인하세요.")
                    st.rerun() # 즉시 재실행하여 UI 업데이트
            else:
                st.error("올바른 Figma URL과 API 키를 모두 입력해주세요.")
    
    st.markdown("---")

    # --- 실시간 모니터링 ---
    status_placeholder = st.empty()
    report_placeholder = st.empty()

    statuses = read_status_file()
    
    with status_placeholder.container():
        render_status(statuses)

    is_complete = all(s == "completed" for s in statuses.values()) if statuses else False
    is_failed = any(s == "failed" for s in statuses.values()) if statuses else False

    if statuses and (is_complete or is_failed):
        # 작업이 완료되거나 실패하면 실행 상태를 False로 변경
        st.session_state.agent_running = False
        
        if is_complete:
            with report_placeholder.container():
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
        elif is_failed:
             with report_placeholder.container():
                st.error("🚫 작업 중 오류가 발생하여 중단되었습니다. 터미널 로그를 확인해주세요.")

    # 에이전트가 실행 중인 경우에만 주기적으로 페이지를 새로고침
    if st.session_state.get('agent_running', False):
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main() 