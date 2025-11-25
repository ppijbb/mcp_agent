"""
🏗️ AI Architect Agent Page

진화형 AI 아키텍처 설계 및 최적화
"""

import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime
import os
import pandas as pd
import plotly.express as px

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from srcs.core.config.loader import settings

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def display_results(result_data):
    st.markdown("---")
    st.subheader("🧬 AI 아키텍처 진화 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return
        
    best_architecture = result_data.get('best_architecture', {})
    if not best_architecture:
        st.error("최적 아키텍처를 찾지 못했습니다.")
        st.json(result_data)
        return
        
    st.success(f"**최적 아키텍처: {best_architecture.get('name', 'N/A')}**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("최종 점수", f"{best_architecture.get('fitness_score', 0):.4f}")
    col2.metric("총 세대 수", result_data.get('generations_completed', 'N/A'))
    col3.metric("평가된 아키텍처", result_data.get('total_architectures_evaluated', 'N/A'))
    
    with st.expander("상세 아키텍처 보기", expanded=True):
        st.markdown("##### 컴포넌트")
        st.json(best_architecture.get('components', []))
        st.markdown("##### 연결")
        st.json(best_architecture.get('connections', []))

    fitness_history = result_data.get('fitness_history', [])
    if fitness_history:
        st.markdown("#### 세대별 성능 향상 그래프")
        df = pd.DataFrame(fitness_history)
        fig = px.line(df, x='generation', y='max_fitness', title='세대별 최고 적합도', markers=True)
        fig.update_layout(xaxis_title="세대", yaxis_title="최고 적합도")
        st.plotly_chart(fig, use_container_width=True)


def main():
    create_agent_page(
        agent_name="Evolutionary AI Architect",
        page_icon="🏗️",
        page_type="architect",
        title="AI Architect Agent",
        subtitle="진화 알고리즘을 사용하여 주어진 문제에 대한 최적의 AI 아키텍처를 설계합니다.",
        module_path="srcs.evolutionary_ai_architect.run_ai_architect_agent"
    )
    result_placeholder = st.empty()

    with st.form("architect_form"):
        st.subheader("📝 문제 정의")
        problem_description = st.text_area(
            "어떤 문제를 해결하기 위한 AI 아키텍처를 설계할까요?",
            height=150,
            placeholder="예: 실시간 사용자 감정 분석을 위한 소셜 미디어 모니터링 시스템"
        )
        
        col1, col2 = st.columns(2)
        max_generations = col1.slider("최대 세대 수", 1, 20, 5)
        population_size = col2.slider("인구 크기", 5, 50, 10)
        
        submitted = st.form_submit_button("🚀 아키텍처 진화 시작", use_container_width=True)

    if submitted:
        if not problem_description.strip():
            st.warning("문제 설명을 입력해주세요.")
        else:
            reports_path = settings.get_reports_path('ai_architect')
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            agent_metadata = {
                "agent_id": "ai_architect_agent",
                "agent_name": "AI Architect Agent",
                "entry_point": "srcs.evolutionary_ai_architect.run_ai_architect_agent",
                "agent_type": "mcp_agent",
                "capabilities": ["architecture_design", "evolutionary_optimization", "ai_system_planning"],
                "description": "진화형 AI 아키텍처 설계 및 자동 최적화"
            }

            input_data = {
                "problem_description": problem_description,
                "max_generations": max_generations,
                "population_size": population_size,
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result:
                display_results(result)

    # 최신 AI Architect 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 AI Architect 결과")
    
    latest_architect_result = result_reader.get_latest_result("evolutionary_ai_architect", "architecture_design")
    
    if latest_architect_result:
        with st.expander("🏗️ 최신 아키텍처 설계 결과", expanded=False):
            st.subheader("🧬 최근 아키텍처 진화 결과")
            
            if isinstance(latest_architect_result, dict):
                best_architecture = latest_architect_result.get('best_architecture', {})
                if best_architecture:
                    st.success(f"**최적 아키텍처: {best_architecture.get('name', 'N/A')}**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("최종 점수", f"{best_architecture.get('fitness_score', 0):.4f}")
                    col2.metric("총 세대 수", latest_architect_result.get('generations_completed', 'N/A'))
                    col3.metric("평가된 아키텍처", latest_architect_result.get('total_architectures_evaluated', 'N/A'))
                    
                    # 문제 설명 표시
                    if 'problem_description' in latest_architect_result:
                        st.write("**문제 설명:**")
                        st.write(latest_architect_result['problem_description'])
                    
                    # 세대별 성능 그래프
                    fitness_history = latest_architect_result.get('fitness_history', [])
                    if fitness_history:
                        st.subheader("📈 세대별 성능 향상")
                        df = pd.DataFrame(fitness_history)
                        fig = px.line(df, x='generation', y='max_fitness', title='세대별 최고 적합도', markers=True)
                        fig.update_layout(xaxis_title="세대", yaxis_title="최고 적합도")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 메타데이터 표시
                    if 'timestamp' in latest_architect_result:
                        st.caption(f"⏰ 설계 시간: {latest_architect_result['timestamp']}")
                else:
                    st.json(latest_architect_result)
            else:
                st.json(latest_architect_result)
    else:
        st.info("💡 아직 AI Architect Agent의 결과가 없습니다. 위에서 아키텍처 설계를 실행해보세요.")

if __name__ == "__main__":
    main()