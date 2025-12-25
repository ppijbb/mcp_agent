"""
📈 Revenue Operations Intelligence Agent Page

매출 운영 인텔리전스 AI
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
        agent_name="Revenue Operations Intelligence Agent",
        page_icon="📈",
        page_type="revenue",
        title="Revenue Operations Intelligence Agent",
        subtitle="매출 예측, 파이프라인 분석 및 최적화",
        module_path="srcs.enterprise_agents.revenue_operations_intelligence_agent"
    )

    result_placeholder = st.empty()

    with st.form("revenue_form"):
        st.subheader("📝 매출 운영 분석 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        analysis_type = st.selectbox(
            "분석 유형",
            options=["revenue_forecast", "pipeline_analysis", "conversion_optimization", "comprehensive"],
            format_func=lambda x: {
                "revenue_forecast": "매출 예측",
                "pipeline_analysis": "파이프라인 분석",
                "conversion_optimization": "전환 최적화",
                "comprehensive": "종합 분석"
            }.get(x, x)
        )
        
        submitted = st.form_submit_button("🚀 매출 분석 시작", use_container_width=True)

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('revenue'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"revenue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            from srcs.common.standard_a2a_page_helper import (
                execute_standard_agent_via_a2a,
                process_standard_agent_result
            )
            from srcs.common.agent_interface import AgentType

            # 표준화된 방식으로 agent 실행 (클래스 기반)
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="revenue_operations_agent",
                agent_name="Revenue Operations Intelligence Agent",
                entry_point="srcs.enterprise_agents.revenue_operations_intelligence_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["revenue_forecast", "pipeline_analysis", "conversion_optimization"],
                description="매출 예측, 파이프라인 분석 및 최적화",
                input_params={
                    "company_name": company_name,
                    "analysis_type": analysis_type
                },
                class_name="RevenueOperationsIntelligenceAgent",
                method_name="analyze_revenue",
                result_json_path=result_json_path
            )

            # 결과 처리
            processed = process_standard_agent_result(result, "revenue_operations_agent")
            if processed["success"] and processed["has_data"]:
                display_results(processed["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Revenue Operations 결과")
    latest_result = result_reader.get_latest_result("revenue_agent", "revenue_analysis")
    if latest_result:
        with st.expander("📈 최신 매출 운영 분석 결과", expanded=False):
            display_results(latest_result)
    else:
        st.info("💡 아직 Revenue Operations Agent의 결과가 없습니다. 위에서 매출 분석을 실행해보세요.")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 매출 운영 분석 결과")
    if result_data:
        if isinstance(result_data, dict):
            if 'revenue_forecast' in result_data:
                st.markdown("### 📈 매출 예측")
                st.write(result_data['revenue_forecast'])
            if 'pipeline_analysis' in result_data:
                st.markdown("### 🔄 파이프라인 분석")
                st.write(result_data['pipeline_analysis'])
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

