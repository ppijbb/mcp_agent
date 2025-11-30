"""
💡 Product Innovation Accelerator Agent Page

제품 혁신 가속화 AI
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
    st.subheader("📊 제품 혁신 분석 결과")
    if result_data:
        st.json(result_data)

def main():
    create_agent_page(
        agent_name="Product Innovation Accelerator Agent",
        page_icon="💡",
        page_type="innovation",
        title="Product Innovation Accelerator Agent",
        subtitle="제품 혁신 아이디어 생성 및 개발 가속화",
        module_path="srcs.enterprise_agents.product_innovation_accelerator_agent"
    )

    result_placeholder = st.empty()

    with st.form("innovation_form"):
        st.subheader("📝 제품 혁신 분석 설정")
        
        product_domain = st.text_input("제품 도메인", placeholder="예: AI 기반 헬스케어")
        
        innovation_focus = st.selectbox(
            "혁신 초점",
            options=["market_opportunity", "technology_trend", "user_needs", "competitive_analysis"],
            format_func=lambda x: {
                "market_opportunity": "시장 기회",
                "technology_trend": "기술 트렌드",
                "user_needs": "사용자 니즈",
                "competitive_analysis": "경쟁사 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 혁신 분석 시작", use_container_width=True)

    if submitted:
        if not product_domain.strip():
            st.warning("제품 도메인을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('innovation'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"innovation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행 (클래스 기반)
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="product_innovation_agent",
                agent_name="Product Innovation Accelerator Agent",
                entry_point="srcs.enterprise_agents.product_innovation_accelerator_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["product_innovation", "market_analysis", "technology_trend_analysis"],
                description="제품 혁신 아이디어 생성 및 개발 가속화",
                input_params={
                    "product_domain": product_domain,
                    "innovation_focus": innovation_focus
                },
                class_name="ProductInnovationAcceleratorAgent",
                method_name="analyze_innovation",
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "product_innovation_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Innovation 결과")
    latest_result = result_reader.get_latest_result("innovation_agent", "innovation_analysis")
    if latest_result:
        with st.expander("💡 최신 제품 혁신 분석 결과", expanded=False):
            st.json(latest_result)
    else:
        st.info("💡 아직 Product Innovation Accelerator Agent의 결과가 없습니다. 위에서 혁신 분석을 실행해보세요.")

if __name__ == "__main__":
    main()

