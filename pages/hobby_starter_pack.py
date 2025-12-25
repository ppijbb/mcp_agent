"""
🎨 Hobby Starter Pack Agent Page

LangGraph 기반 취미 시작 가이드 Agent
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Hobby Starter Pack Agent",
        page_icon="🎨",
        page_type="hobby",
        title="Hobby Starter Pack Agent",
        subtitle="LangGraph 기반 취미 시작 가이드 및 추천 시스템",
        module_path="lang_graph.hobby_starter_pack_agent"
    )

    result_placeholder = st.empty()

    with st.form("hobby_form"):
        st.subheader("📝 취미 추천 요청")
        
        hobby_interest = st.text_area(
            "관심 있는 취미",
            placeholder="예: 그림 그리기, 요리, 운동",
            height=150
        )
        
        experience_level = st.selectbox(
            "경험 수준",
            options=["beginner", "intermediate", "advanced"],
            format_func=lambda x: {
                "beginner": "초보자",
                "intermediate": "중급자",
                "advanced": "고급자"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 취미 가이드 생성", use_container_width=True)

    if submitted:
        if not hobby_interest.strip():
            st.warning("관심 있는 취미를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('hobby_starter_pack'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"hobby_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 입력 파라미터 준비
            input_data = {
                "hobby_interest": hobby_interest,
                "experience_level": experience_level,
                "messages": [{"role": "user", "content": hobby_interest}],
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="hobby_starter_pack_agent",
                agent_name="Hobby Starter Pack Agent",
                entry_point="lang_graph.hobby_starter_pack_agent",
                agent_type=AgentType.LANGGRAPH_AGENT,
                capabilities=["hobby_recommendation", "hobby_guide_generation", "skill_learning_path"],
                description="LangGraph 기반 취미 시작 가이드 및 추천 시스템",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Hobby Starter Pack 결과")
    latest_result = result_reader.get_latest_result("hobby_agent", "hobby_guide")
    if latest_result:
        with st.expander("🎨 최신 취미 가이드 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Hobby Starter Pack Agent의 결과가 없습니다. 위에서 취미 가이드를 생성해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 취미 가이드 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'hobby_guide' in result_data:
                st.markdown("### 🎨 취미 가이드")
                st.write(result_data['hobby_guide'])
            if 'recommended_hobbies' in result_data:
                st.markdown("### 💡 추천 취미")
                hobbies = result_data['recommended_hobbies']
                if isinstance(hobbies, list):
                    for hobby in hobbies:
                        st.write(f"• {hobby}")
                else:
                    st.write(hobbies)
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

