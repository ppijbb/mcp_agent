"""
🚀 Product Planner Agent Page

실시간 제품 기획 현황 모니터링
"""

import streamlit as st
import sys
from pathlib import Path
import time
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.product_planner_agent.utils.status_logger import StatusLogger, STATUS_FILE

def get_status_icon(status):
    """상태에 따른 아이콘 반환"""
    icons = {
        "pending": "⚪",
        "in_progress": "⏳",
        "completed": "✅",
        "failed": "❌"
    }
    return icons.get(status, "❓")

def main():
    """Product Planner Agent 상태 모니터링 페이지"""
    
    st.set_page_config(
        page_title="🚀 Product Planner Status",
        page_icon="🚀",
        layout="wide"
    )

    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #36D1DC 0%, #5B86E5 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🚀 Product Planner Agent Status</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            실시간 제품 기획 워크플로우 진행 현황
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")

    st.markdown("---")

    status_placeholder = st.empty()

    while True:
        statuses = StatusLogger.read_status()

        with status_placeholder.container():
            if not statuses:
                st.info("🕒 Product Planner Agent 실행을 기다리는 중입니다...")
            else:
                st.markdown("### 📋 워크플로우 진행률")
                
                # 전체 진행률 계산
                completed_steps = sum(1 for s in statuses.values() if s == 'completed')
                total_steps = len(statuses)
                progress = (completed_steps / total_steps) if total_steps > 0 else 0
                
                st.progress(progress, text=f"{completed_steps} / {total_steps} Steps Completed")

                cols = st.columns(len(statuses))
                
                for i, (step, status) in enumerate(statuses.items()):
                    with cols[i]:
                        icon = get_status_icon(status)
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; {'background-color: #f0f9f0;' if status == 'completed' else ''}">
                            <p style="font-size: 2rem; margin: 0;">{icon}</p>
                            <h5 style="margin-bottom: 0.5rem;">{step}</h5>
                            <p style="font-weight: bold; color: {'#2ecc71' if status == 'completed' else '#3498db'};">{status.replace('_', ' ').title()}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # 모든 단계가 완료되었는지 확인
                if all(s in ['completed', 'failed'] for s in statuses.values()):
                    st.success("🎉 모든 작업이 완료되었습니다! 최종 보고서를 확인하세요.")
                    # 여기서 반복을 중단할 수 있습니다.
                    break
        
        # 5초마다 새로고침
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main() 