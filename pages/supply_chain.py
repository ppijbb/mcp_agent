"""
🔗 Supply Chain Orchestrator Agent Page

공급망 관리 및 최적화 AI
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Supply Chain Orchestrator Agent",
        page_icon="🔗",
        page_type="supply_chain",
        title="Supply Chain Orchestrator Agent",
        subtitle="공급망 관리, 최적화 및 리스크 분석",
        module_path="srcs.enterprise_agents.supply_chain_orchestrator_agent"
    )

    result_placeholder = st.empty()

    with st.form("supply_chain_form"):
        st.subheader("📝 공급망 분석 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        analysis_focus = st.multiselect(
            "분석 초점",
            options=["inventory_optimization", "supplier_risk", "logistics", "demand_forecast"],
            default=["inventory_optimization", "supplier_risk"]
        )
        
        submitted = st.form_submit_button("🚀 공급망 분석 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('supply_chain'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"supply_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            from srcs.common.standard_a2a_page_helper import (
                execute_standard_agent_via_a2a,
                process_standard_agent_result
            )
            from srcs.common.agent_interface import AgentType

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="supply_chain_agent",
                agent_name="Supply Chain Orchestrator Agent",
                entry_point="srcs.enterprise_agents.supply_chain_orchestrator_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["supply_chain_management", "inventory_optimization", "supplier_risk_analysis", "logistics"],
                description="공급망 관리, 최적화 및 리스크 분석",
                input_params={
                    "company_name": company_name,
                    "analysis_focus": analysis_focus
                },
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "supply_chain_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Supply Chain 결과")
    latest_result = result_reader.get_latest_result("supply_chain_agent", "supply_chain_analysis")
    if latest_result:
        with st.expander("🔗 최신 공급망 분석 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Supply Chain Agent의 결과가 없습니다. 위에서 공급망 분석을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 공급망 분석 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'analysis_result' in result_data:
                st.markdown("### 📊 분석 결과")
                st.write(result_data['analysis_result'])
            if 'recommendations' in result_data:
                st.markdown("### 💡 권장사항")
                recommendations = result_data['recommendations']
                if isinstance(recommendations, list):
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.write(recommendations)
            st.json(result_data)
        else:
            st.write(str(result_data))
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

