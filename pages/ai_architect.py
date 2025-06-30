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

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

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
        "🧬 Evolutionary AI Architect",
        "진화 알고리즘을 사용하여 주어진 문제에 대한 최적의 AI 아키텍처를 설계합니다.",
        "pages/ai_architect.py"
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
            reports_path = get_reports_path('ai_architect')
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.evolutionary_ai_architect.run_ai_architect_agent",
                "--problem-description", problem_description,
                "--max-generations", str(max_generations),
                "--population-size", str(population_size),
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="ai_architect"
            )

            if result:
                display_results(result)

if __name__ == "__main__":
    main()