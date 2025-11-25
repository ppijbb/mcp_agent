"""
🏠 Real Estate Agent Page

LangGraph 기반 부동산 분석 Agent
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
        agent_name="Real Estate Agent",
        page_icon="🏠",
        page_type="real_estate",
        title="Real Estate Agent",
        subtitle="LangGraph 기반 부동산 분석 및 추천 시스템",
        module_path="lang_graph.real_estate_agent"
    )

    result_placeholder = st.empty()

    with st.form("real_estate_form"):
        st.subheader("📝 부동산 분석 요청")
        
        property_query = st.text_area(
            "부동산 질의",
            placeholder="예: 서울 강남구 아파트 투자 가치 분석",
            height=150
        )
        
        analysis_type = st.selectbox(
            "분석 유형",
            options=["investment", "market_analysis", "property_search", "comprehensive"],
            format_func=lambda x: {
                "investment": "투자 분석",
                "market_analysis": "시장 분석",
                "property_search": "매물 검색",
                "comprehensive": "종합 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 부동산 분석 시작", use_container_width=True)

    if submitted:
        if not property_query.strip():
            st.warning("부동산 질의를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('real_estate'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"real_estate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "real_estate_agent",
                "agent_name": "Real Estate Agent",
                "entry_point": "lang_graph.real_estate_agent",
                "agent_type": "langgraph_agent",
                "capabilities": ["real_estate_analysis", "property_search", "market_analysis", "investment_analysis"],
                "description": "LangGraph 기반 부동산 분석 및 추천 시스템"
            }

            input_data = {
                "query": property_query,
                "analysis_type": analysis_type,
                "messages": [{"role": "user", "content": property_query}],
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
    st.markdown("## 📊 최신 Real Estate 결과")
    latest_result = result_reader.get_latest_result("real_estate_agent", "real_estate_analysis")
    if latest_result:
        with st.expander("🏠 최신 부동산 분석 결과", expanded=False):
            st.json(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 부동산 분석 결과")
    if result_data:
        st.json(result_data)

if __name__ == "__main__":
    main()

