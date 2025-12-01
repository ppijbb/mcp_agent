"""
⚖️ Legal Compliance Agent Page

법률 준수 및 규정 관리 AI
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
        agent_name="Legal Compliance Agent",
        page_icon="⚖️",
        page_type="legal",
        title="Legal Compliance Agent",
        subtitle="법률 준수 검토, 규정 관리 및 리스크 분석",
        module_path="srcs.enterprise_agents.legal_compliance_agent"
    )

    result_placeholder = st.empty()

    with st.form("legal_form"):
        st.subheader("📝 법률 준수 검토 설정")
        
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        
        compliance_areas = st.multiselect(
            "준수 영역",
            options=["GDPR", "CCPA", "HIPAA", "SOX", "PCI-DSS", "ISO27001"],
            default=["GDPR", "CCPA"]
        )
        
        submitted = st.form_submit_button("🚀 법률 준수 검토 시작", width='stretch')

    if submitted:
        if not company_name.strip():
            st.warning("회사명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('legal'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"legal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="legal_compliance_agent",
                agent_name="Legal Compliance Agent",
                entry_point="srcs.enterprise_agents.legal_compliance_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["legal_compliance", "gdpr_compliance", "ccpa_compliance", "regulatory_analysis"],
                description="법률 준수 검토, 규정 관리 및 리스크 분석",
                input_params={
                    "company_name": company_name,
                    "compliance_areas": compliance_areas,
                    "result_json_path": str(result_json_path)
                },
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    st.markdown("---")
    st.markdown("## 📊 최신 Legal Compliance 결과")
    latest_result = result_reader.get_latest_result("legal_agent", "compliance_review")
    if latest_result:
        with st.expander("⚖️ 최신 법률 준수 검토 결과", expanded=False):
            display_results(latest_result)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 법률 준수 검토 결과")

    if not result_data:
        st.warning("준수 검토 결과를 찾을 수 없습니다.")
        return

    # result_data가 중첩된 구조일 수 있음
    actual_data = result_data.get('data', result_data)

    # 기본 결과 표시
    if isinstance(actual_data, dict):
        if 'company_name' in actual_data:
            st.info(f"**회사명**: {actual_data['company_name']}")

        if 'compliance_areas' in actual_data:
            st.markdown("### 📋 준수 영역")
            areas = actual_data['compliance_areas']
            if isinstance(areas, list):
                for area in areas:
                    st.write(f"• {area}")
            else:
                st.write(areas)

        if 'assessment_result' in actual_data:
            st.markdown("### ✅ 준수 평가 결과")
            st.write(actual_data['assessment_result'])

        if 'recommendations' in actual_data:
            st.markdown("### 💡 권장 사항")
            recommendations = actual_data['recommendations']
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.write(recommendations)

        if 'risk_score' in actual_data:
            risk_score = actual_data['risk_score']
            if isinstance(risk_score, (int, float)):
                st.metric("위험 점수", f"{risk_score:.1f}")

        # 전체 결과 JSON 표시
        with st.expander("📄 전체 결과 (JSON)", expanded=False):
            st.json(actual_data)
    else:
        # 문자열이나 다른 형식의 결과
        st.write(str(actual_data))

if __name__ == "__main__":
    main()

