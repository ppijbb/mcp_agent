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
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

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
    
    # result_data가 중첩된 구조일 수 있음 (data.data)
    actual_data = result_data.get('data', result_data)
    if isinstance(actual_data, dict) and 'data' in actual_data:
        actual_data = actual_data.get('data', actual_data)
    
    if not actual_data:
        st.warning("분석 결과 데이터를 찾을 수 없습니다.")
        return
    
    # 간단한 요약 정보 표시
    problem_desc = actual_data.get('problem_description', 'N/A')
    best_fitness = actual_data.get('best_fitness', 0.0)
    generation_count = actual_data.get('generation_count', 0)
    processing_time = actual_data.get('processing_time', 0.0)
    result_file_path = actual_data.get('result_file_path', '')
    evolution_summary = actual_data.get('evolution_summary', {})
    
    st.success(f"✅ **문제**: {problem_desc}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 Fitness", f"{best_fitness:.4f}")
    col2.metric("총 세대 수", generation_count)
    col3.metric("처리 시간", f"{processing_time:.2f}초")
    col4.metric("최종 평균 Fitness", f"{evolution_summary.get('final_avg_fitness', 0.0):.4f}")
    
    # 진화 히스토리 그래프
    if result_file_path and Path(result_file_path).exists():
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                full_result = json.load(f)
            
            evolution_history = full_result.get('evolution_history', [])
            if evolution_history:
                st.markdown("#### 📈 세대별 성능 향상")
                df = pd.DataFrame(evolution_history)
                fig = px.line(df, x='generation', y='best_fitness', 
                            title='세대별 최고 적합도', markers=True)
                fig.add_scatter(x=df['generation'], y=df['avg_fitness'], 
                              mode='lines', name='평균 적합도', line=dict(dash='dash'))
                fig.update_layout(xaxis_title="세대", yaxis_title="적합도")
                st.plotly_chart(fig, width='stretch')
                
                # 최적화 추천 표시
                recommendations = full_result.get('optimization_recommendations', [])
                if recommendations:
                    st.markdown("#### 🚀 최적화 추천")
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
        except Exception as e:
            st.warning(f"상세 결과를 불러올 수 없습니다: {e}")
    
    if result_file_path:
        st.info(f"📄 전체 결과는 다음 파일에 저장되었습니다: `{result_file_path}`")


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
        
        # 시뮬레이션 모드 토글
        simulation_mode = st.checkbox(
            "시뮬레이션 모드 활성화",
            value=True,
            help="시뮬레이션 모드가 활성화되면 성능 모델링 시뮬레이터를 사용하여 아키텍처 성능을 추정합니다."
        )
        
        if simulation_mode:
            st.info("🔬 시뮬레이션 모드: 아키텍처 성능 모델링 시뮬레이터를 사용합니다.")
        
        submitted = st.form_submit_button("🚀 아키텍처 진화 시작", width='stretch')

    if submitted:
        if not problem_description.strip():
            st.warning("문제 설명을 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('ai_architect'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="ai_architect_agent",
                agent_name="AI Architect Agent",
                entry_point="srcs.evolutionary_ai_architect.run_ai_architect_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["architecture_design", "evolutionary_optimization", "ai_system_planning"],
                description="진화형 AI 아키텍처 설계 및 자동 최적화",
                input_params={
                    "problem_description": problem_description,
                    "max_generations": max_generations,
                    "population_size": population_size,
                    "simulation_mode": simulation_mode,
                    "result_json_path": str(result_json_path)
                },
                method_name="run_ai_architect_agent",
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
                        st.plotly_chart(fig, width='stretch')
                    
                    # 메타데이터 표시
                    if 'timestamp' in latest_architect_result:
                        st.caption(f"⏰ 설계 시간: {latest_architect_result['timestamp']}")
    else:
        st.info("💡 아직 AI Architect Agent의 결과가 없습니다. 위에서 아키텍처 설계를 실행해보세요.")

if __name__ == "__main__":
    main()