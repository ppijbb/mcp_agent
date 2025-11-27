"""
🐾 Petcare Physical AI Agent Page

LangGraph 기반 반려동물 케어 Agent
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Petcare Physical AI Agent",
        page_icon="🐾",
        page_type="petcare",
        title="Petcare Physical AI Agent",
        subtitle="LangGraph 기반 반려동물 건강 관리 및 케어 시스템",
        module_path="lang_graph.petcare_physical_ai_agent"
    )

    result_placeholder = st.empty()

    with st.form("petcare_form"):
        st.subheader("📝 반려동물 케어 요청")
        
        pet_info = st.text_area(
            "반려동물 정보",
            placeholder="예: 3살 된 골든 리트리버, 활동적, 최근 식욕 감소",
            height=150
        )
        
        care_type = st.selectbox(
            "케어 유형",
            options=["health_check", "nutrition", "exercise", "comprehensive"],
            format_func=lambda x: {
                "health_check": "건강 검진",
                "nutrition": "영양 관리",
                "exercise": "운동 계획",
                "comprehensive": "종합 케어"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 케어 계획 생성", width='stretch')

    if submitted:
        if not pet_info.strip():
            st.warning("반려동물 정보를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('petcare'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"petcare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "petcare_physical_ai_agent",
                "agent_name": "Petcare Physical AI Agent",
                "entry_point": "lang_graph.petcare_physical_ai_agent",
                "agent_type": "langgraph_agent",
                "capabilities": ["pet_care", "health_management", "nutrition_planning", "exercise_planning"],
                "description": "LangGraph 기반 반려동물 건강 관리 및 케어 시스템"
            }

            input_data = {
                "pet_info": pet_info,
                "care_type": care_type,
                "messages": [{"role": "user", "content": f"Pet info: {pet_info}, Care type: {care_type}"}],
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Petcare 결과")
    latest_result = result_reader.get_latest_result("petcare_agent", "petcare_analysis")
    if latest_result:
        with st.expander("🐾 최신 반려동물 케어 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 반려동물 케어 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

