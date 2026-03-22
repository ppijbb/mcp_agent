"""
Decision Agent Page Module.

Provides a Streamlit UI for the Decision Agent that analyzes complex situations
and recommends optimal decisions based on interaction context.

Module:
    - Displays decision analysis results with confidence scores and risk levels
    - Supports multiple interaction types (ProductPurchase, ContractSigning, ServiceCancellation, etc.)
    - Executes agent via A2A protocol for standardized communication
"""

import streamlit as st
from pathlib import Path
import sys
import json
import os
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.settings import get_reports_path
from srcs.advanced_agents.decision_agent import (
    InteractionType,
)
from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# 경로 설정
REPORTS_PATH = get_reports_path('decision')
os.makedirs(REPORTS_PATH, exist_ok=True)

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 의사결정 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return
    
    decision = result_data.get('decision', {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("신뢰도", f"{decision.get('confidence_score', 0):.1%}")
    col2.metric("위험 수준", decision.get('risk_level', 'N/A'))
    col3.metric("상호작용 유형", result_data.get('interaction', {}).get('interaction_type', 'N/A'))

    st.success(f"**추천**: {decision.get('recommendation', 'N/A')}")

    with st.expander("상세 분석 내용 보기", expanded=True):
        st.markdown("#### 근거")
        st.write(decision.get('reasoning', ''))

        st.markdown("#### 대안")
        
        st.markdown("#### 데이터 소스")
        
        st.markdown("#### 전체 결과 (JSON)")

def main():
    create_agent_page(
        agent_name="Decision Agent",
        page_icon="🧠",
        page_type="decision",
        title="Decision Agent",
        subtitle="복잡한 상황을 분석하고 최적의 결정을 내리는 AI 에이전트",
        module_path="srcs.advanced_agents.decision_agent"
    )

    result_placeholder = st.empty()

    with st.form("decision_form"):
        st.subheader("📝 분석 시나리오 설정")

        user_id = st.text_input("사용자 ID", "user_12345")
        
        interaction_type = st.selectbox(
            "상호작용 유형", 
            options=[it.value for it in InteractionType],
            format_func=lambda x: InteractionType(x).name
        )

        context_text = st.text_area(
            "상호작용 컨텍스트 (JSON)",
            '{"product_id": "prod_abc", "price": 99.99, "currency": "USD"}'
        )
        
        submitted = st.form_submit_button("🚀 의사결정 분석 시작", use_container_width=True)

    if submitted:
        if not user_id.strip() or not context_text.strip():
            st.warning("사용자 ID와 컨텍스트를 모두 입력해주세요.")
        else:
            try:
                # Validate context JSON
                json.loads(context_text)
            except json.JSONDecodeError:
                st.error("상호작용 컨텍스트가 유효한 JSON 형식이 아닙니다.")
                st.stop()

            reports_path = Path(get_reports_path('decision'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"decision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Get the enum key from its value for the command line
            interaction_enum_key = InteractionType(interaction_type).name

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="decision_agent",
                agent_name="Decision Agent",
                entry_point="srcs.advanced_agents.run_decision_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["decision_making", "scenario_analysis", "risk_assessment"],
                description="복잡한 상황을 분석하고 최적의 결정을 내리는 AI 에이전트",
                input_params={
                    "user_id": user_id,
                    "interaction_type": interaction_enum_key,
                    "context_json": context_text,
                    "result_json_path": str(result_json_path)
                },
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    # 최신 Decision Agent 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Decision Agent 결과")
    
    latest_decision_result = result_reader.get_latest_result("decision_agent", "decision_analysis")
    
    if latest_decision_result:
        with st.expander("🧠 최신 의사결정 분석 결과", expanded=False):
            st.subheader("🤖 최근 의사결정 분석 결과")
            
            if isinstance(latest_decision_result, dict):
                # 의사결정 정보 표시
                decision = latest_decision_result.get('decision', {})
                user_id = latest_decision_result.get('user_id', 'N/A')
                
                st.success(f"**사용자: {user_id}**")
                st.info(f"**상호작용 유형: {latest_decision_result.get('interaction_type', 'N/A')}**")
                
                # 의사결정 결과 요약
                col1, col2, col3 = st.columns(3)
                col1.metric("신뢰도", f"{decision.get('confidence_score', 0):.1%}")
                col2.metric("위험 수준", decision.get('risk_level', 'N/A'))
                col3.metric("분석 상태", "완료" if latest_decision_result.get('success', False) else "실패")
                
                # 추천 사항 표시
                recommendation = decision.get('recommendation', 'N/A')
                if recommendation:
                    st.subheader("💡 추천 사항")
                    st.write(recommendation)
                
                # 근거 표시
                reasoning = decision.get('reasoning', '')
                if reasoning:
                    st.subheader("🔍 분석 근거")
                    with st.expander("상세 근거", expanded=False):
                        st.write(reasoning)
                
                # 대안 표시
                alternatives = decision.get('alternatives', [])
                if alternatives:
                    st.subheader("🔄 고려된 대안")
                    with st.expander("대안 목록", expanded=False):
                        for i, alt in enumerate(alternatives, 1):
                            st.write(f"{i}. {alt}")
                
                # 메타데이터 표시
                if 'timestamp' in latest_decision_result:
                    st.caption(f"⏰ 분석 시간: {latest_decision_result['timestamp']}")
            else:
                st.write("결과 데이터 형식이 예상과 다릅니다.")
    else:
        st.info("💡 아직 Decision Agent의 결과가 없습니다. 위에서 의사결정 분석을 실행해보세요.")

if __name__ == "__main__":
    main() 