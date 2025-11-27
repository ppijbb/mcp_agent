"""
🔗 Supply Chain Orchestrator Agent Page

공급망 관리 및 최적화 AI
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
        
        submitted = st.form_submit_button("🚀 공급망 분석 시작", width='stretch')

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('supply_chain'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"supply_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "supply_chain_agent",
                "agent_name": "Supply Chain Orchestrator Agent",
                "entry_point": "srcs.enterprise_agents.supply_chain_orchestrator_agent",
                "agent_type": "mcp_agent",
                "capabilities": ["supply_chain_management", "inventory_optimization", "supplier_risk_analysis", "logistics"],
                "description": "공급망 관리, 최적화 및 리스크 분석"
            }

            input_data = {
                "company_name": company_name,
                "analysis_focus": analysis_focus,
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
    st.markdown("## 📊 최신 Supply Chain 결과")
    latest_result = result_reader.get_latest_result("supply_chain_agent", "supply_chain_analysis")
    if latest_result:
        with st.expander("🔗 최신 공급망 분석 결과", expanded=False):

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 공급망 분석 결과")
    if result_data:

if __name__ == "__main__":
    main()

