"""
🏗️ AI Architect Agent Page

진화형 AI 아키텍처 설계 및 최적화
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# AI Architect Agent 임포트 시도
try:
    from srcs.advanced_agents.evolutionary_ai_architect_agent import EvolutionaryAIArchitectAgent
    ARCHITECT_AGENT_AVAILABLE = True
except ImportError as e:
    ARCHITECT_AGENT_AVAILABLE = False
    import_error = str(e)

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
    
    # Agent 연동 상태 확인
    if not ARCHITECT_AGENT_AVAILABLE:
        st.error(f"⚠️ AI Architect Agent를 불러올 수 없습니다: {import_error}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### AI Architect Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install numpy pandas matplotlib seaborn
            ```
            
            2. **에이전트 모듈 확인**:
            ```bash
            ls srcs/advanced_agents/evolutionary_ai_architect_agent.py
            ls srcs/advanced_agents/architect.py
            ls srcs/advanced_agents/genome.py
            ls srcs/advanced_agents/improvement_engine.py
            ```
            
            3. **필요 시 종속성 설치**:
            ```bash
            pip install -r requirements.txt
            ```
            """)
        
        # 에이전트 소개만 제공
        render_agent_info()
        return
    else:
        st.success("🤖 AI Architect Agent가 성공적으로 연결되었습니다!")
        
        # 에이전트 실행 인터페이스 제공
        render_architect_agent_interface()

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 🏗️ AI Architect Agent 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 주요 기능
        - **진화형 아키텍처**: 자동 최적화 및 스케일링
        - **성능 모니터링**: 실시간 시스템 건강도 체크
        - **비용 최적화**: 클라우드 리소스 효율적 관리
        - **보안 강화**: AI 기반 위협 탐지 및 대응
        - **배포 자동화**: CI/CD 파이프라인 최적화
        """)
    
    with col2:
        st.markdown("""
        #### ✨ 스페셜 기능
        - **적응형 학습**: 사용 패턴 기반 자동 조정
        - **예측 분석**: 장애 예방 및 용량 계획
        - **멀티클라우드 지원**: 하이브리드 환경 최적화
        - **A/B 테스트 자동화**: 성능 비교 분석
        - **비용 예측**: ROI 기반 아키텍처 추천
        """)
    
    st.markdown("""
    #### 🎯 사용 사례
    - 대규모 AI 서비스 아키텍처 설계
    - 레거시 시스템 현대화 전략
    - 마이크로서비스 전환 계획
    - 클라우드 네이티브 최적화
    """)

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
                value="Design an AI system for real-time image processing and analysis",
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
            
            if st.button("🚀 AI Architect 실행", type="primary", use_container_width=True):
                if problem_description.strip():
                    execute_architect_agent(agent, problem_description, architecture_type, generations, show_details)
                else:
                    st.error("문제 설명을 입력해주세요.")
        
        with col2:
            if 'architect_execution_result' in st.session_state:
                result = st.session_state['architect_execution_result']
                
                if result['success']:
                    st.success("✅ AI Architect Agent 실행 완료!")
                    
                    # 결과 정보 표시
                    st.markdown("#### 📊 아키텍처 설계 결과")
                    
                    solution = result['solution']
                    
                    # 추천 아키텍처 정보
                    if 'recommended_architecture' in solution:
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
                    
                    # 상세 정보 표시
                    if result.get('show_details'):
                        render_detailed_results(result)
                    
                    # 해결책 다운로드
                    solution_text = format_solution_for_download(result)
                    st.download_button(
                        label="📥 아키텍처 설계서 다운로드",
                        data=solution_text,
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
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

def execute_architect_agent(agent, problem_description, architecture_type, generations, show_details):
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
            
            # 에이전트 실행
            solution_result = agent.solve_problem(problem_description, constraints)
            
            # 상세 정보가 필요한 경우 에이전트 상태 가져오기
            agent_status = agent.get_status() if show_details else None
            
            # 현재 시간 타임스탬프
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            st.session_state['architect_execution_result'] = {
                'success': True,
                'solution': solution_result['solution'],
                'problem_analysis': solution_result['problem_analysis'],
                'performance_metrics': solution_result['performance_metrics'],
                'improvement_opportunities': solution_result['improvement_opportunities'],
                'improvement_strategy': solution_result['improvement_strategy'],
                'processing_time': solution_result['processing_time'],
                'generation': solution_result['generation'],
                'agent_status': agent_status,
                'show_details': show_details,
                'timestamp': timestamp
            }
            st.rerun()
            
    except Exception as e:
        st.session_state['architect_execution_result'] = {
            'success': False,
            'message': f'AI Architect Agent 실행 중 오류 발생: {str(e)}',
            'error': str(e)
        }
        st.rerun()

def render_detailed_results(result):
    """상세 결과 렌더링"""
    
    with st.expander("🔍 상세 실행 정보", expanded=False):
        
        # 문제 분석
        if 'problem_analysis' in result:
            st.markdown("#### 📊 문제 분석")
            analysis = result['problem_analysis']
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.info(f"**문제 유형**: {analysis.get('problem_type', 'N/A')}")
                st.info(f"**복잡도**: {analysis.get('complexity', 'N/A')}")
            with col_a2:
                st.info(f"**권장 아키텍처**: {analysis.get('suggested_architecture_type', 'N/A')}")
        
        # 성능 메트릭
        if 'performance_metrics' in result:
            st.markdown("#### 📈 성능 메트릭")
            metrics = result['performance_metrics']
            st.json(metrics)
        
        # 개선 기회
        if 'improvement_opportunities' in result:
            st.markdown("#### 🚀 개선 기회")
            opportunities = result['improvement_opportunities']
            for opp in opportunities:
                st.markdown(f"- {opp}")
        
        # 개선 전략
        if 'improvement_strategy' in result:
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
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("다양성 점수", f"{pop_stats.get('diversity_score', 0):.3f}")
            with col_s2:
                st.metric("평균 적합도", f"{pop_stats.get('average_fitness', 0):.3f}")

def format_solution_for_download(result):
    """다운로드용 해결책 포맷팅"""
    
    solution = result['solution']
    timestamp = result['timestamp']
    
    content = f"""# AI 아키텍처 설계서
생성 시간: {timestamp}

## 📊 추천 아키텍처

"""
    
    if 'recommended_architecture' in solution:
        arch = solution['recommended_architecture']
        content += f"""
- **아키텍처 ID**: {arch.get('id', 'N/A')}
- **타입**: {arch.get('type', 'N/A')}
- **적합도 점수**: {arch.get('fitness_score', 0):.4f}
- **레이어 수**: {len(arch.get('layers', []))}

### 하이퍼파라미터
```json
{arch.get('hyperparameters', {})}
```
"""
    
    if 'implementation_steps' in solution:
        content += "\n## 📋 구현 단계\n\n"
        for i, step in enumerate(solution['implementation_steps'], 1):
            content += f"{i}. {step}\n"
    
    if 'expected_performance' in solution:
        perf = solution['expected_performance']
        content += f"""
## 📈 예상 성능

- **정확도 추정**: {perf.get('accuracy_estimate', 'N/A')}
- **복잡도 등급**: {perf.get('complexity_rating', 'N/A')}
- **훈련 시간 추정**: {perf.get('training_time_estimate', 'N/A')}
"""
    
    if 'adaptive_features' in solution:
        content += "\n## ✨ 적응형 기능\n\n"
        for feature in solution['adaptive_features']:
            content += f"- {feature}\n"
    
    content += f"""
## 🔍 실행 정보

- **처리 시간**: {result.get('processing_time', 0):.2f}초
- **진화 세대**: {result.get('generation', 0)}
- **성공**: {result.get('success', False)}
"""
    
    return content

if __name__ == "__main__":
    main() 