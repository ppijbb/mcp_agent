"""
🛒 Smart Shopping Assistant Agent Page

LangGraph 기반 스마트 쇼핑 어시스턴트 Agent
표준 A2A 패턴 적용
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_a2a_page_helper import (
    execute_standard_agent_via_a2a,
    process_standard_agent_result
)
from srcs.common.agent_interface import AgentType
from srcs.common.page_utils import create_agent_page
from configs.settings import get_reports_path

try:
    from srcs.utils.result_reader import result_reader
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 쇼핑 분석 결과")
    if result_data:
        st.json(result_data)

def main():
    create_agent_page(
        agent_name="Smart Shopping Assistant Agent",
        page_icon="🛒",
        page_type="shopping",
        title="Smart Shopping Assistant Agent",
        subtitle="LangGraph 기반 스마트 쇼핑 추천 및 가격 비교 시스템",
        module_path="lang_graph.smart_shopping_assistant"
    )

    result_placeholder = st.empty()

    with st.form("shopping_form"):
        st.subheader("📝 쇼핑 요청")
        
        shopping_query = st.text_area(
            "쇼핑 질의",
            placeholder="예: 노트북 추천해줘, 가격은 100만원 이하로",
            height=150
        )
        
        shopping_type = st.selectbox(
            "쇼핑 유형",
            options=["product_search", "price_comparison", "recommendation", "comprehensive"],
            format_func=lambda x: {
                "product_search": "제품 검색",
                "price_comparison": "가격 비교",
                "recommendation": "추천",
                "comprehensive": "종합 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 쇼핑 분석 시작", use_container_width=True)

    if submitted:
        if not shopping_query.strip():
            st.warning("쇼핑 질의를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('smart_shopping'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"smart_shopping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행 (LangGraph)
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="smart_shopping_assistant_agent",
                agent_name="Smart Shopping Assistant Agent",
                entry_point="lang_graph.smart_shopping_assistant",
                agent_type=AgentType.LANGGRAPH_AGENT,
                capabilities=["product_search", "price_comparison", "shopping_recommendation"],
                description="LangGraph 기반 스마트 쇼핑 추천 및 가격 비교 시스템",
                input_params={
                    "query": shopping_query,
                    "shopping_type": shopping_type,
                    "messages": [{"role": "user", "content": shopping_query}]
                },
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "smart_shopping_assistant_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Smart Shopping Assistant 결과")
    latest_result = result_reader.get_latest_result("shopping_agent", "shopping_analysis")
    if latest_result:
        with st.expander("🛒 최신 쇼핑 분석 결과", expanded=False):
            st.json(latest_result)
    else:
        st.info("💡 아직 Smart Shopping Assistant Agent의 결과가 없습니다. 위에서 쇼핑 분석을 실행해보세요.")

if __name__ == "__main__":
    main()

