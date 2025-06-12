"""
🏗️ AI Architect Agent Page

진화형 AI 아키텍처 설계 및 최적화
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('ai_architect')
except ImportError:
    # 설정 파일이 없으면 에러 발생
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# AI Architect Agent 임포트 - 필수 의존성
try:
    from srcs.advanced_agents.evolutionary_ai_architect_agent import EvolutionaryAIArchitectAgent
except ImportError as e:
    st.error(f"❌ AI Architect Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: EvolutionaryAIArchitectAgent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

def main():
    """AI Architect Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🏗️ AI Architect Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            진화형 AI 아키텍처 설계 및 성능 최적화 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    st.success("🤖 AI Architect Agent가 성공적으로 연결되었습니다!")
    
    # 에이전트 실행 인터페이스 제공
    render_architect_agent_interface()

def render_architect_agent_interface():
    """AI Architect Agent 실행 인터페이스"""
    
    st.markdown("### 🚀 AI Architect Agent 실행")
    
    # 에이전트 초기화
    try:
        if 'architect_agent' not in st.session_state:
            st.session_state.architect_agent = EvolutionaryAIArchitectAgent("EvoAI-Streamlit", population_size=8)
        
        agent = st.session_state.architect_agent
        
        # 실행 설정
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ 아키텍처 설계 설정")
            
            problem_description = st.text_area(
                "문제 설명", 
                placeholder="해결하고자 하는 AI 아키텍처 문제를 상세히 설명하세요",
                height=100,
                help="해결하고자 하는 AI 아키텍처 문제를 상세히 설명하세요"
            )
            
            architecture_type = st.selectbox(
                "선호 아키텍처 타입",
                ["auto (자동 선택)", "transformer", "cnn", "hybrid"],
                help="자동 선택을 권장합니다"
            )
            
            population_size = st.slider(
                "진화 인구 크기", 
                min_value=4, 
                max_value=20, 
                value=8,
                help="진화 알고리즘의 인구 크기를 설정합니다"
            )
            
            generations = st.slider(
                "진화 세대 수", 
                min_value=3, 
                max_value=15, 
                value=5,
                help="진화 알고리즘의 세대 수를 설정합니다"
            )
            
            show_details = st.checkbox(
                "상세 정보 표시", 
                value=False,
                help="에이전트 상태 및 진화 과정의 상세 정보를 표시합니다"
            )
            
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help=f"체크하면 {REPORTS_PATH} 디렉토리에 설계 결과를 파일로 저장합니다"
            )
            
            if st.button("🚀 AI Architect 실행", type="primary", use_container_width=True):
                if problem_description.strip():
                    execute_architect_agent(agent, problem_description, architecture_type, generations, show_details, save_to_file)
                else:
                    st.error("문제 설명을 입력해주세요.")
        
        with col2:
            if 'architect_execution_result' in st.session_state:
                result = st.session_state['architect_execution_result']
                
                if result['success']:
                    st.success("✅ AI Architect Agent 실행 완료!")
                    
                    # 실제 에이전트 결과만 표시
                    st.markdown("#### 📊 아키텍처 설계 결과")
                    st.text_area(
                        "설계 결과",
                        value=result.get('agent_output', ''),
                        height=300,
                        disabled=True
                    )
                    
                    # 파일 저장 결과 표시
                    if result.get('save_to_file') and result.get('file_saved'):
                        st.success(f"💾 결과가 파일로 저장되었습니다: {result.get('output_path', '')}")
                    
                    # 실제 에이전트 결과 정보 표시
                    display_agent_results(result)
                    
                    # 상세 정보 표시
                    if result.get('show_details'):
                        render_detailed_results(result)
                    
                    # 해결책 다운로드
                    if result.get('agent_output'):
                        st.download_button(
                            label="📥 아키텍처 설계서 다운로드",
                            data=result['agent_output'],
                            file_name=f"ai_architecture_design_{result['timestamp']}.md",
                            mime="text/markdown"
                        )
                    
                else:
                    st.error("❌ 실행 중 오류 발생")
                    st.error(f"**오류**: {result['message']}")
                    
                    with st.expander("🔍 오류 상세"):
                        st.code(result.get('error', 'Unknown error'))
                        
            else:
                st.markdown("""
                #### 🤖 Agent 실행 정보
                
                **실행되는 프로세스:**
                1. **문제 분석** - AI 문제 유형 및 복잡도 분석
                2. **아키텍처 진화** - 진화 알고리즘으로 최적 아키텍처 탐색
                3. **성능 평가** - 적합도 점수 기반 성능 측정
                4. **해결책 생성** - 구현 가능한 아키텍처 추천
                5. **자가 개선** - 성능 기반 시스템 최적화
                
                **생성되는 결과:**
                - 🏗️ 최적 AI 아키텍처 설계
                - 📊 성능 예측 및 메트릭
                - 📋 상세 구현 가이드
                - ✨ 적응형 기능 설명
                - 📈 개선 전략 및 기회
                """)
                
    except Exception as e:
        st.error(f"❌ Agent 초기화 실패: {e}")
        st.error("EvolutionaryAIArchitectAgent 구현을 확인해주세요.")
        st.stop()

def execute_architect_agent(agent, problem_description, architecture_type, generations, show_details, save_to_file):
    """AI Architect Agent 실행"""
    
    try:
        with st.spinner("🔄 AI 아키텍처를 설계하는 중..."):
            import time
            
            # 아키텍처 타입 처리
            arch_type = None if architecture_type == "auto (자동 선택)" else architecture_type
            
            # 제약조건 설정
            constraints = {
                'architecture_type': arch_type,
                'generations': generations
            }
            
            # 실제 에이전트 실행 - 폴백 없음
            solution_result = agent.solve_problem(problem_description, constraints)
            
            if not solution_result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            # 상세 정보가 필요한 경우 에이전트 상태 가져오기
            agent_status = agent.get_status() if show_details else None
            
            # 현재 시간 타임스탬프
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 실제 에이전트 출력만 사용
            agent_output = format_agent_output(solution_result, problem_description)
            
            # 파일 저장 처리
            file_saved = False
            output_path = None
            if save_to_file:
                file_saved, output_path = save_architect_results_to_file(solution_result, problem_description, timestamp)
            
            st.session_state['architect_execution_result'] = {
                'success': True,
                'solution': solution_result.get('solution', {}),
                'problem_analysis': solution_result.get('problem_analysis', {}),
                'performance_metrics': solution_result.get('performance_metrics', {}),
                'improvement_opportunities': solution_result.get('improvement_opportunities', []),
                'improvement_strategy': solution_result.get('improvement_strategy', {}),
                'processing_time': solution_result.get('processing_time', 0),
                'generation': solution_result.get('generation', 0),
                'agent_status': agent_status,
                'show_details': show_details,
                'timestamp': timestamp,
                'agent_output': agent_output,
                'save_to_file': save_to_file,
                'file_saved': file_saved,
                'output_path': output_path
            }
            st.rerun()
            
    except Exception as e:
        st.session_state['architect_execution_result'] = {
            'success': False,
            'message': f'AI Architect Agent 실행 실패: {str(e)}',
            'error': str(e)
        }
        st.rerun()

def display_agent_results(result):
    """실제 에이전트 결과 표시"""
    
    solution = result.get('solution', {})
    
    if not solution:
        st.warning("에이전트 결과가 비어있습니다.")
        return
    
    # 추천 아키텍처 정보
    if 'recommended_architecture' in solution:
        st.markdown("#### 🏗️ 추천 아키텍처 정보")
        arch = solution['recommended_architecture']
        
        col_arch1, col_arch2 = st.columns(2)
        
        with col_arch1:
            st.metric("아키텍처 타입", arch.get('type', 'Unknown'))
            st.metric("레이어 수", len(arch.get('layers', [])))
        
        with col_arch2:  
            st.metric("적합도 점수", f"{arch.get('fitness_score', 0):.4f}")
            st.metric("복잡도", arch.get('complexity_rating', 'N/A'))
    
    # 구현 단계
    if 'implementation_steps' in solution:
        st.markdown("#### 📋 구현 단계")
        for step in solution['implementation_steps']:
            st.markdown(f"- {step}")
    
    # 예상 성능
    if 'expected_performance' in solution:
        perf = solution['expected_performance']
        st.markdown("#### 📈 예상 성능")
        
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("정확도 추정", perf.get('accuracy_estimate', 'N/A'))
        with col_perf2:
            st.metric("복잡도 등급", perf.get('complexity_rating', 'N/A'))
        with col_perf3:
            st.metric("훈련 시간 추정", perf.get('training_time_estimate', 'N/A'))
    
    # 적응형 기능
    if 'adaptive_features' in solution:
        st.markdown("#### ✨ 적응형 기능")
        for feature in solution['adaptive_features']:
            st.markdown(f"- {feature}")

def render_detailed_results(result):
    """상세 결과 렌더링"""
    
    with st.expander("🔍 상세 실행 정보", expanded=False):
        
        # 문제 분석
        if 'problem_analysis' in result and result['problem_analysis']:
            st.markdown("#### 📊 문제 분석")
            analysis = result['problem_analysis']
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.info(f"**문제 유형**: {analysis.get('problem_type', 'N/A')}")
                st.info(f"**복잡도**: {analysis.get('complexity', 'N/A')}")
            with col_a2:
                st.info(f"**권장 아키텍처**: {analysis.get('suggested_architecture_type', 'N/A')}")
        
        # 성능 메트릭
        if 'performance_metrics' in result and result['performance_metrics']:
            st.markdown("#### 📈 성능 메트릭")
            metrics = result['performance_metrics']
            st.json(metrics)
        
        # 개선 기회
        if 'improvement_opportunities' in result and result['improvement_opportunities']:
            st.markdown("#### 🚀 개선 기회")
            opportunities = result['improvement_opportunities']
            for opp in opportunities:
                st.markdown(f"- {opp}")
        
        # 개선 전략
        if 'improvement_strategy' in result and result['improvement_strategy']:
            st.markdown("#### 🎯 개선 전략")
            strategy = result['improvement_strategy']
            st.json(strategy)
        
        # 에이전트 상태
        if result.get('agent_status'):
            st.markdown("#### 🤖 에이전트 상태")
            status = result['agent_status']
            
            # 기본 정보
            agent_info = status.get('agent_info', {})
            st.markdown(f"**세대**: {agent_info.get('generation', 0)}")
            st.markdown(f"**인구 크기**: {agent_info.get('population_size', 0)}")
            st.markdown(f"**완료 태스크**: {agent_info.get('tasks_completed', 0)}")
            
            # 인구 통계
            pop_stats = status.get('population_stats', {})
            if pop_stats:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("다양성 점수", f"{pop_stats.get('diversity_score', 0):.3f}")
                with col_s2:
                    st.metric("평균 적합도", f"{pop_stats.get('average_fitness', 0):.3f}")

def format_agent_output(solution_result, problem_description):
    """실제 에이전트 출력 포맷팅"""
    
    if not solution_result:
        raise Exception("에이전트 결과가 없습니다.")
    
    solution = solution_result.get('solution', {})
    
    if not solution:
        raise Exception("에이전트 솔루션이 비어있습니다.")
    
    # 실제 에이전트 데이터만 사용하여 출력 생성
    output_lines = [
        "🏗️ AI 아키텍처 설계 결과",
        "",
        f"📝 문제 설명:",
        problem_description,
        ""
    ]
    
    # 실제 에이전트 결과만 사용
    if 'recommended_architecture' in solution:
        arch = solution['recommended_architecture']
        output_lines.extend([
            "🎯 추천 아키텍처:",
            f"- 아키텍처 ID: {arch.get('id', 'N/A')}",
            f"- 타입: {arch.get('type', 'N/A')}",
            f"- 적합도 점수: {arch.get('fitness_score', 0):.4f}",
            f"- 레이어 수: {len(arch.get('layers', []))}",
            ""
        ])
    
    if 'expected_performance' in solution:
        perf = solution['expected_performance']
        output_lines.extend([
            "📊 예상 성능:",
            f"- 정확도 추정: {perf.get('accuracy_estimate', 'N/A')}",
            f"- 복잡도 등급: {perf.get('complexity_rating', 'N/A')}",
            f"- 훈련 시간 추정: {perf.get('training_time_estimate', 'N/A')}",
            ""
        ])
    
    if 'implementation_steps' in solution:
        output_lines.append("📋 구현 단계:")
        for i, step in enumerate(solution['implementation_steps'], 1):
            output_lines.append(f"{i}. {step}")
        output_lines.append("")
    
    if 'adaptive_features' in solution:
        output_lines.append("✨ 적응형 기능:")
        for feature in solution['adaptive_features']:
            output_lines.append(f"- {feature}")
        output_lines.append("")
    
    # 실행 정보 추가
    output_lines.extend([
        "🔍 실행 정보:",
        f"- 처리 시간: {solution_result.get('processing_time', 0):.2f}초",
        f"- 진화 세대: {solution_result.get('generation', 0)}",
        "- 설계 성공: ✅"
    ])
    
    return "\n".join(output_lines)

def save_architect_results_to_file(solution_result, problem_description, timestamp):
    """AI Architect 결과를 파일로 저장"""
    
    try:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
        filename = f"ai_architect_design_{timestamp}.md"
        filepath = os.path.join(REPORTS_PATH, filename)
        
        # 실제 에이전트 출력 생성
        agent_output = format_agent_output(solution_result, problem_description)
        
        # 마크다운 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# AI 아키텍처 설계 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(agent_output)
            f.write("\n\n---\n")
            f.write("## 상세 기술 사양\n\n")
            
            # 실제 에이전트 결과의 상세 정보 추가
            solution = solution_result.get('solution', {})
            if 'recommended_architecture' in solution:
                f.write("### 아키텍처 상세\n\n")
                f.write("```json\n")
                f.write(json.dumps(solution['recommended_architecture'], indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
            
            f.write("---\n")
            f.write("*본 보고서는 AI Architect Agent에 의해 자동 생성되었습니다.*\n")
        
        return True, filepath
        
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")
        return False, None

if __name__ == "__main__":
    main() 