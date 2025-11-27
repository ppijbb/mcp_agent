"""
💰 Financial Agent Page

LangGraph 기반 금융 분석 Agent
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
        agent_name="Financial Agent",
        page_icon="💰",
        page_type="financial",
        title="Financial Agent",
        subtitle="LangGraph 기반 금융 분석 및 조언 시스템",
        module_path="lang_graph.financial_agent"
    )

    result_placeholder = st.empty()

    with st.form("financial_form"):
        st.subheader("📝 금융 분석 요청")
        
        query = st.text_area(
            "금융 질의",
            placeholder="예: 내 포트폴리오를 분석하고 리밸런싱 제안을 해줘",
            height=150
        )
        
        submitted = st.form_submit_button("🚀 금융 분석 시작", width='stretch')

    if submitted:
        if not query.strip():
            st.warning("금융 질의를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('financial_agent'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"financial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # A2A를 통한 agent 실행
            agent_metadata = {
                "agent_id": "financial_agent",
                "agent_name": "Financial Agent",
                "entry_point": "lang_graph.financial_agent",
                "agent_type": "langgraph_agent",
                "capabilities": ["financial_analysis", "portfolio_analysis", "investment_advice"],
                "description": "LangGraph 기반 금융 분석 및 조언 시스템"
            }

            input_data = {
                "query": query,
                "messages": [{"role": "user", "content": query}],
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and result.get("success") and result.get("data"):
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Financial Agent 결과")
    latest_result = result_reader.get_latest_result("financial_agent", "financial_analysis")
    if latest_result:
        with st.expander("💰 최신 금융 분석 결과", expanded=False):

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 금융 분석 결과")
    if result_data:

if __name__ == "__main__":
    main()

