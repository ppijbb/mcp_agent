"""
🧠 Mental Care Agent Page

심리 건강 관리 AI 시스템
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="Mental Care Agent",
        page_icon="🧠",
        page_type="mental",
        title="Mental Care Agent",
        subtitle="심리도식치료 기반 심리 건강 관리 및 분석 시스템",
        module_path="srcs.enterprise_agents.mental"
    )

    result_placeholder = st.empty()

    with st.form("mental_care_form"):
        st.subheader("📝 상담 세션 시작")

        user_message = st.text_area(
            "어떤 고민이 있으신가요?",
            placeholder="예: 최근 업무 스트레스가 심해서 잠을 잘 못 자고 있어요.",
            height=150,
            help="자유롭게 말씀해주세요. AI가 심리 상태를 분석하고 도움을 드립니다."
        )

        session_type = st.selectbox(
            "상담 유형",
            options=["일반 상담", "감정 분석", "심리 도식 분석", "종합 분석"],
            help="원하는 상담 유형을 선택하세요"
        )

        submitted = st.form_submit_button("🚀 상담 시작", width='stretch')

    if submitted:
        if not user_message.strip():
            st.warning("고민을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('mental_care'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"mental_care_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="mental_care_agent",
                agent_name="Mental Care Agent",
                entry_point="srcs.enterprise_agents.mental",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["mental_health_analysis", "emotion_analysis", "psychological_schema_analysis"],
                description="심리도식치료 기반 심리 건강 관리 및 분석 시스템",
                input_params={
                    "user_message": user_message,
                    "session_type": session_type,
                    "result_json_path": str(result_json_path)
                },
                class_name="MentalCareOrchestrator",
                method_name="start_conversation_session",
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    # 최신 Mental Care 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Mental Care 결과")
    
    latest_mental_result = result_reader.get_latest_result("mental_care_agent", "mental_analysis")
    
    if latest_mental_result:
        with st.expander("🧠 최신 심리 분석 결과", expanded=False):
            st.subheader("🤖 최근 심리 분석 결과")
            
            if isinstance(latest_mental_result, dict):
                session_id = latest_mental_result.get('session_id', 'N/A')
                st.success(f"**세션 ID: {session_id}**")
                
                emotions = latest_mental_result.get('emotions', [])
                if emotions:
                    st.subheader("😊 감정 분석")
                    for emotion in emotions:
                        st.write(f"• {emotion.get('emotion', 'N/A')}: {emotion.get('severity', 'N/A')}")
                
                schemas = latest_mental_result.get('psychological_schemas', [])
                if schemas:
                    st.subheader("🧠 심리 도식")
                    for schema in schemas:
                        st.write(f"• {schema.get('schema_name', 'N/A')}")
                
                if latest_mental_result.get('analysis_results'):
                    st.subheader("📋 분석 결과")
            else:
    else:
        st.info("💡 아직 Mental Care Agent의 결과가 없습니다. 위에서 상담을 시작해보세요.")

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 심리 분석 결과")
    
    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return
    
    session_id = result_data.get('session_id', 'N/A')
    st.success(f"**세션 ID: {session_id}**")
    
    emotions = result_data.get('emotions', [])
    if emotions:
        st.subheader("😊 감정 분석")
        for emotion in emotions:
            with st.expander(f"{emotion.get('emotion', 'N/A')} - 심각도: {emotion.get('severity', 'N/A')}"):
                st.write(f"**트리거**: {', '.join(emotion.get('triggers', []))}")
                st.write(f"**지속 기간**: {emotion.get('duration', 'N/A')}")
                st.write(f"**컨텍스트**: {emotion.get('context', 'N/A')}")
    
    schemas = result_data.get('psychological_schemas', [])
    if schemas:
        st.subheader("🧠 심리 도식 분석")
        for schema in schemas:
            with st.expander(schema.get('schema_name', 'N/A')):
                st.write(f"**설명**: {schema.get('description', 'N/A')}")
                st.write(f"**트리거**: {', '.join(schema.get('triggers', []))}")
                st.write(f"**적응적 반응**: {', '.join(schema.get('adaptive_responses', []))}")
    
    if result_data.get('analysis_results'):
        st.subheader("📋 종합 분석 결과")

if __name__ == "__main__":
    main()

