"""
🌱 ESG Carbon Neutral Agent Page

ESG 및 탄소 중립 관리 AI
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
        agent_name="ESG Carbon Neutral Agent",
        page_icon="🌱",
        page_type="esg",
        title="ESG Carbon Neutral Agent",
        subtitle="ESG 보고서 작성, 탄소 발자국 측정 및 중립 전략 수립",
        module_path="srcs.enterprise_agents.esg_carbon_neutral_agent"
    )

    result_placeholder = st.empty()

    with st.form("esg_form"):
        st.subheader("📝 ESG 분석 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        analysis_type = st.selectbox(
            "분석 유형",
            options=["carbon_footprint", "esg_reporting", "sustainability_planning", "comprehensive"],
            format_func=lambda x: {
                "carbon_footprint": "탄소 발자국 측정",
                "esg_reporting": "ESG 보고서 작성",
                "sustainability_planning": "지속가능성 계획",
                "comprehensive": "종합 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 ESG 분석 시작", width='stretch')

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('esg'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"esg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="esg_agent",
                agent_name="ESG Carbon Neutral Agent",
                entry_point="srcs.enterprise_agents.esg_carbon_neutral_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["esg_analysis", "carbon_footprint", "sustainability_planning", "esg_reporting"],
                description="ESG 보고서 작성, 탄소 발자국 측정 및 중립 전략 수립",
                input_params={
                    "company_name": company_name,
                    "analysis_type": analysis_type,
                    "result_json_path": str(result_json_path)
                },
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and result.get("success") and result.get("data"):
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 ESG 결과")
    latest_result = result_reader.get_latest_result("esg_agent", "esg_analysis")
    if latest_result:
        with st.expander("🌱 최신 ESG 분석 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 ESG Agent의 결과가 없습니다. 위에서 ESG 분석을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 ESG 분석 결과")
    if result_data:
        if isinstance(result_data, dict):
            if "carbon_footprint" in result_data:
                st.metric("탄소 발자국", f"{result_data['carbon_footprint']:.2f} tCO2e")
            if "esg_score" in result_data:
                st.metric("ESG 점수", f"{result_data['esg_score']:.1f}/100")
            if "recommendations" in result_data:
                st.subheader("💡 권장사항")
                for rec in result_data['recommendations']:
                    st.write(f"• {rec}")
            st.json(result_data)
        else:
            st.write(result_data)
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

