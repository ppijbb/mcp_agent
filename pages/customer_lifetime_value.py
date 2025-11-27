"""
💰 Customer Lifetime Value Agent Page

고객 생애 가치 분석 AI
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
        agent_name="Customer Lifetime Value Agent",
        page_icon="💰",
        page_type="clv",
        title="Customer Lifetime Value Agent",
        subtitle="고객 생애 가치 분석 및 예측",
        module_path="srcs.enterprise_agents.customer_lifetime_value_agent"
    )

    result_placeholder = st.empty()

    with st.form("clv_form"):
        st.subheader("📝 고객 데이터 분석")
        
        customer_data = st.text_area(
            "고객 데이터 (JSON 형식)",
            placeholder='{"customer_id": "123", "purchase_history": [...]}',
            height=150
        )
        
        submitted = st.form_submit_button("🚀 CLV 분석 시작", width='stretch')

    if submitted:
        if not customer_data.strip():
            st.warning("고객 데이터를 입력해주세요.")
        else:
            try:
                json.loads(customer_data)
            except json.JSONDecodeError:
                st.error("유효한 JSON 형식이 아닙니다.")
                st.stop()
            
            reports_path = Path(get_reports_path('clv'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"clv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "customer_lifetime_value_agent",
                "agent_name": "Customer Lifetime Value Agent",
                "entry_point": "srcs.enterprise_agents.customer_lifetime_value_agent",
                "agent_type": "mcp_agent",
                "capabilities": ["customer_analysis", "lifetime_value_prediction", "customer_segmentation"],
                "description": "고객 생애 가치 분석 및 예측"
            }

            input_data = {
                "customer_data": json.loads(customer_data),
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
    st.markdown("## 📊 최신 CLV 결과")
    latest_result = result_reader.get_latest_result("clv_agent", "clv_analysis")
    if latest_result:
        with st.expander("💰 최신 고객 생애 가치 분석", expanded=False):
            display_results(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 CLV 분석 결과")
    if result_data:
        if isinstance(result_data, dict):
            # 결과가 딕셔너리인 경우
            if "clv" in result_data:
                st.metric("고객 생애 가치 (CLV)", f"${result_data['clv']:,.2f}")
            if "segments" in result_data:
                st.write("고객 세그먼트:", result_data["segments"])
            if "recommendations" in result_data:
                st.write("추천 사항:", result_data["recommendations"])
            # 전체 결과 표시
            st.json(result_data)
        else:
            st.write(result_data)
    else:
        st.warning("결과 데이터가 없습니다.")

if __name__ == "__main__":
    main()

