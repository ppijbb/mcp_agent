"""
🎯 Skill Marketplace Agent Page

LangGraph 기반 스킬 마켓플레이스 Agent
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Skill Marketplace Agent",
        page_icon="🎯",
        page_type="skill_marketplace",
        title="Skill Marketplace Agent",
        subtitle="LangGraph 기반 스킬 매칭 및 마켓플레이스 시스템",
        module_path="lang_graph.skill_marketplace_agent"
    )

    result_placeholder = st.empty()

    with st.form("skill_marketplace_form"):
        st.subheader("📝 스킬 매칭 요청")
        
        skill_query = st.text_area(
            "스킬 요구사항",
            placeholder="예: Python 개발자, 3년 이상 경력, 머신러닝 경험",
            height=150
        )
        
        match_type = st.selectbox(
            "매칭 유형",
            options=["job_seeker", "employer", "skill_gap_analysis"],
            format_func=lambda x: {
                "job_seeker": "구직자 매칭",
                "employer": "고용주 매칭",
                "skill_gap_analysis": "스킬 격차 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 스킬 매칭 시작", width='stretch')

    if submitted:
        if not skill_query.strip():
            st.warning("스킬 요구사항을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('skill_marketplace'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"skill_marketplace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 입력 파라미터 준비
            input_data = {
                "skill_query": skill_query,
                "match_type": match_type,
                "messages": [{"role": "user", "content": skill_query}],
                "result_json_path": str(result_json_path)
            }

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="skill_marketplace_agent",
                agent_name="Skill Marketplace Agent",
                entry_point="lang_graph.skill_marketplace_agent",
                agent_type=AgentType.LANGGRAPH_AGENT,
                capabilities=["skill_matching", "job_matching", "skill_gap_analysis"],
                description="LangGraph 기반 스킬 매칭 및 마켓플레이스 시스템",
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Skill Marketplace 결과")
    latest_result = result_reader.get_latest_result("skill_marketplace_agent", "skill_matching")
    if latest_result:
        with st.expander("🎯 최신 스킬 매칭 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Skill Marketplace Agent의 결과가 없습니다. 위에서 스킬 매칭을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 스킬 매칭 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'matches' in result_data:
                st.markdown("### 🎯 매칭 결과")
                matches = result_data['matches']
                if isinstance(matches, list):
                    for match in matches:
                        st.write(f"• {match}")
                else:
                    st.write(matches)
            if 'skill_gap_analysis' in result_data:
                st.markdown("### 📊 스킬 격차 분석")
                st.write(result_data['skill_gap_analysis'])
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

