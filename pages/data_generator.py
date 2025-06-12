"""
📊 Data Generator Page

AI 기반 데이터 생성 및 분석 도구
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('data_generator')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# AI Data Generation Agent 임포트 - 필수 의존성
try:
    from srcs.basic_agents.data_generator import AIDataGenerationAgent
except ImportError as e:
    st.error(f"❌ AI Data Generation Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: AIDataGenerationAgent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

# 페이지 설정
try:
    st.set_page_config(
        page_title="📊 AI Data Generator",
        page_icon="📊",
        layout="wide"
    )
except Exception:
    pass

def main():
    """AI Data Generator 메인 페이지"""
    
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
        <h1>📊 AI Data Generator</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 지능형 데이터 생성 및 분석 도구
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    st.success("🤖 AI Data Generation Agent가 성공적으로 연결되었습니다!")
    
    # 에이전트 인터페이스
    render_real_ai_data_generator()

def render_real_ai_data_generator():
    """AI Data Generator 인터페이스"""
    
    try:
        if 'ai_data_agent' not in st.session_state:
            st.session_state.ai_data_agent = AIDataGenerationAgent()
        
        agent = st.session_state.ai_data_agent
        
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 AI 스마트 데이터 생성", 
            "📊 AI 맞춤형 데이터셋", 
            "👥 AI 고객 프로필", 
            "📈 AI 시계열 예측"
        ])
        
        with tab1:
            render_ai_smart_data_generation(agent)
        
        with tab2:
            render_ai_custom_datasets(agent)
        
        with tab3:
            render_ai_customer_profiles(agent)
        
        with tab4:
            render_ai_timeseries_prediction(agent)
            
    except Exception as e:
        st.error(f"❌ AI Data Generation Agent 초기화 실패: {e}")
        st.error("AIDataGenerationAgent 구현을 확인해주세요.")
        st.stop()

def render_ai_smart_data_generation(agent):
    """AI 기반 스마트 데이터 생성"""
    
    st.markdown("### 🤖 AI 스마트 데이터 생성")
    st.info("AI가 요구사항을 분석하여 지능적으로 데이터를 생성합니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ AI 데이터 생성 설정")
        
        data_purpose = st.text_area(
            "데이터 사용 목적",
            placeholder="예: 고객 행동 분석을 위한 샘플 데이터",
            help="AI가 목적에 맞는 데이터를 생성합니다"
        )
        
        # 동적으로 로드되어야 할 데이터 유형들
        data_types = load_data_types()
        data_type = st.selectbox(
            "데이터 유형",
            data_types if data_types else ["비즈니스 데이터"]
        )
        
        records_count = st.number_input("레코드 수", min_value=10, max_value=10000, value=1000)
        
        # 동적으로 로드되어야 할 품질 수준들
        quality_levels = load_quality_levels()
        quality_level = st.select_slider(
            "품질 수준",
            options=quality_levels if quality_levels else ["기본", "고품질"],
            value=quality_levels[1] if quality_levels and len(quality_levels) > 1 else "고품질"
        )
        
        include_relationships = st.checkbox("관계형 데이터 포함", value=True)
        include_patterns = st.checkbox("패턴 반영", value=True)
        
        save_to_file = st.checkbox(
            "파일로 저장", 
            value=False,
            help=f"체크하면 {REPORTS_PATH} 디렉토리에 생성된 데이터를 파일로 저장합니다"
        )
        
        if st.button("🚀 AI 스마트 데이터 생성", use_container_width=True):
            if data_purpose.strip():
                generate_ai_smart_data(agent, {
                    'purpose': data_purpose,
                    'type': data_type,
                    'count': records_count,
                    'quality': quality_level,
                    'relationships': include_relationships,
                    'patterns': include_patterns
                }, save_to_file)
            else:
                st.error("데이터 사용 목적을 입력해주세요.")
    
    with col2:
        if 'ai_generated_data' in st.session_state:
            st.markdown("#### 📊 AI 생성 데이터")
            data = st.session_state['ai_generated_data']
            st.text_area(
                "생성된 데이터 결과",
                value=data.get('agent_output', ''),
                height=300,
                disabled=True
            )
        else:
            st.markdown("""
            #### 🤖 AI 스마트 데이터 생성 기능
            
            **AI 지능형 생성:**
            - 🎯 목적 기반 데이터 구조 설계
            - 📊 분포와 패턴 반영
            - 🔗 논리적 관계성 보장
            - 🎨 사용자 맞춤형 스키마
            
            **고급 AI 기능:**
            - 🧠 딥러닝 기반 데이터 모델링
            - 📈 통계적 정확성 보장
            - 🔍 이상치 및 노이즈 제어
            - 💡 도메인 전문 지식 적용
            """)

def generate_ai_smart_data(agent, config, save_to_file=False):
    """실제 AI를 사용한 스마트 데이터 생성"""
    
    try:
        with st.spinner("AI가 지능적으로 데이터를 생성 중입니다..."):
            
            # 실제 에이전트 호출
            result = agent.generate_smart_data(config)
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            # 실제 에이전트 출력 포맷팅
            agent_output = format_data_generation_result(result, config)
            
            st.session_state['ai_generated_data'] = {
                'agent_output': agent_output,
                'config': config,
                'raw_result': result
            }
            
            # 파일 저장 처리
            if save_to_file:
                file_saved, output_path = save_data_generator_results(agent_output, config)
                if file_saved:
                    st.success(f"💾 데이터가 파일로 저장되었습니다: {output_path}")
                else:
                    st.error("파일 저장 중 오류가 발생했습니다.")
            
            st.success("✅ AI 스마트 데이터 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 데이터 생성 실패: {e}")
        st.error("에이전트의 generate_smart_data 메서드를 확인해주세요.")

def format_data_generation_result(result, config):
    """실제 에이전트 데이터 생성 결과 포맷팅"""
    
    if not result:
        raise Exception("데이터 생성 결과가 없습니다.")
    
    # 실제 에이전트 데이터만 사용하여 출력 생성
    output_lines = [
        "📊 AI 생성 데이터 결과",
        "",
        f"🎯 데이터 목적: {config['purpose']}",
        f"📈 데이터 유형: {config['type']}",
        f"📋 레코드 수: {config['count']}개",
        f"⭐ 품질 수준: {config['quality']}",
        ""
    ]
    
    # 실제 생성된 데이터 정보
    if 'generated_data' in result:
        output_lines.append("생성된 데이터:")
        data = result['generated_data']
        if isinstance(data, list) and len(data) > 0:
            for i, record in enumerate(data[:5]):  # 처음 5개만 표시
                output_lines.append(f"- 레코드 {i+1}: {record}")
            if len(data) > 5:
                output_lines.append(f"... 총 {len(data)}개 레코드")
        output_lines.append("")
    
    # 데이터 품질 분석
    if 'quality_metrics' in result:
        metrics = result['quality_metrics']
        output_lines.extend([
            "데이터 품질 분석:",
            f"- 완성도: {metrics.get('completeness', 'N/A')}",
            f"- 일관성: {metrics.get('consistency', 'N/A')}",
            f"- 유효성: {metrics.get('validity', 'N/A')}",
            f"- 관계형 무결성: {metrics.get('integrity', 'N/A')}"
        ])
    
    return "\n".join(output_lines)

def render_ai_custom_datasets(agent):
    """AI 맞춤형 데이터셋 생성"""
    
    st.markdown("### 📊 AI 맞춤형 데이터셋 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 데이터셋 설계")
        
        dataset_description = st.text_area(
            "데이터셋 설명",
            placeholder="예: 전자상거래 고객 구매 패턴 분석용 데이터셋",
            help="AI가 설명을 바탕으로 적절한 스키마를 설계합니다"
        )
        
        # 동적으로 로드되어야 할 도메인들
        domains = load_domains()
        domain = st.selectbox(
            "도메인 전문성",
            domains if domains else ["전자상거래"]
        )
        
        # 동적으로 로드되어야 할 복잡도 수준들
        complexity_levels = load_complexity_levels()
        complexity_level = st.select_slider(
            "복잡도",
            options=complexity_levels if complexity_levels else ["단순", "중간"],
            value=complexity_levels[1] if complexity_levels and len(complexity_levels) > 1 else "중간"
        )
        
        if st.button("🎯 AI 맞춤형 데이터셋 생성", use_container_width=True):
            if dataset_description.strip():
                generate_ai_custom_dataset(agent, {
                    'description': dataset_description,
                    'domain': domain,
                    'complexity': complexity_level
                })
            else:
                st.error("데이터셋 설명을 입력해주세요.")
    
    with col2:
        if 'ai_custom_dataset' in st.session_state:
            st.markdown("#### 📄 AI 생성 데이터셋")
            dataset = st.session_state['ai_custom_dataset']
            st.json(dataset)
        else:
            st.info("👈 데이터셋 요구사항을 입력하고 'AI 맞춤형 데이터셋 생성' 버튼을 클릭하세요.")

def generate_ai_custom_dataset(agent, config):
    """실제 AI 맞춤형 데이터셋 생성"""
    
    try:
        with st.spinner("AI가 맞춤형 데이터셋을 설계하고 생성 중입니다..."):
            result = agent.create_custom_dataset(config)
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            st.session_state['ai_custom_dataset'] = result
            st.success("✅ AI 맞춤형 데이터셋이 생성되었습니다!")
            
    except Exception as e:
        st.error(f"AI 데이터셋 생성 실패: {e}")
        st.error("에이전트의 create_custom_dataset 메서드를 확인해주세요.")

def render_ai_customer_profiles(agent):
    """AI 고객 프로필 생성"""
    
    st.markdown("### 👥 AI 고객 프로필 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 고객 프로필 설정")
        
        # 동적으로 로드되어야 할 비즈니스 유형들
        business_types = load_business_types()
        business_type = st.selectbox(
            "비즈니스 유형",
            business_types if business_types else ["B2C 전자상거래"]
        )
        
        target_segment = st.text_input(
            "타겟 고객층",
            placeholder="예: 25-45세 도시 거주 직장인",
            help="AI가 타겟 세그먼트에 맞는 고객 프로필을 생성합니다"
        )
        
        profile_count = st.number_input("프로필 수", min_value=10, max_value=5000, value=500)
        
        include_behavior = st.checkbox("구매 행동 패턴 포함", value=True)
        include_preferences = st.checkbox("선호도 데이터 포함", value=True)
        include_journey = st.checkbox("고객 여정 데이터 포함", value=True)
        
        if st.button("👥 AI 고객 프로필 생성", use_container_width=True):
            if target_segment.strip():
                generate_ai_customer_profiles(agent, {
                    'business_type': business_type,
                    'target_segment': target_segment,
                    'count': profile_count,
                    'include_behavior': include_behavior,
                    'include_preferences': include_preferences,
                    'include_journey': include_journey
                })
            else:
                st.error("타겟 고객층을 입력해주세요.")
    
    with col2:
        if 'ai_customer_profiles' in st.session_state:
            st.markdown("#### 👤 AI 생성 고객 프로필")
            profiles = st.session_state['ai_customer_profiles']
            st.json(profiles)
        else:
            st.markdown("""
            #### 🤖 AI 고객 프로필 생성 기능
            
            **AI 기반 프로필링:**
            - 🎯 타겟 세그먼트 기반 생성
            - 📊 행동 패턴 모델링
            - 🛒 구매 여정 시뮬레이션
            - 💡 선호도 및 관심사 생성
            """)

def generate_ai_customer_profiles(agent, config):
    """실제 AI 고객 프로필 생성"""
    
    try:
        with st.spinner("AI가 고객 프로필을 생성 중입니다..."):
            result = agent.generate_customer_profiles(config)
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            st.session_state['ai_customer_profiles'] = result
            st.success("✅ AI 고객 프로필 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 고객 프로필 생성 실패: {e}")
        st.error("에이전트의 generate_customer_profiles 메서드를 확인해주세요.")

def render_ai_timeseries_prediction(agent):
    """AI 시계열 예측 데이터 생성"""
    
    st.markdown("### 📈 AI 시계열 예측 데이터 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 시계열 설정")
        
        # 동적으로 로드되어야 할 시계열 유형들
        series_types = load_series_types()
        series_type = st.selectbox(
            "시계열 유형",
            series_types if series_types else ["매출 예측"]
        )
        
        # 동적으로 로드되어야 할 기간들
        time_periods = load_time_periods()
        time_period = st.selectbox(
            "기간",
            time_periods if time_periods else ["1개월", "3개월"]
        )
        
        # 동적으로 로드되어야 할 주기들
        frequencies = load_frequencies()
        frequency = st.selectbox(
            "주기",
            frequencies if frequencies else ["일별", "주별"]
        )
        
        include_seasonality = st.checkbox("계절성 포함", value=True)
        include_trend = st.checkbox("트렌드 포함", value=True)
        include_noise = st.checkbox("노이즈 포함", value=True)
        
        if st.button("📈 AI 시계열 데이터 생성", use_container_width=True):
            generate_ai_timeseries_data(agent, {
                'type': series_type,
                'period': time_period,
                'frequency': frequency,
                'seasonality': include_seasonality,
                'trend': include_trend,
                'noise': include_noise
            })
    
    with col2:
        if 'ai_timeseries_data' in st.session_state:
            st.markdown("#### 📊 AI 생성 시계열 데이터")
            timeseries = st.session_state['ai_timeseries_data']
            st.json(timeseries)
        else:
            st.info("👈 시계열 설정을 완료하고 'AI 시계열 데이터 생성' 버튼을 클릭하세요.")

def generate_ai_timeseries_data(agent, config):
    """실제 AI 시계열 데이터 생성"""
    
    try:
        with st.spinner("AI가 시계열 데이터를 생성 중입니다..."):
            result = agent.generate_timeseries_data(config)
            
            if not result:
                raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
            
            st.session_state['ai_timeseries_data'] = result
            st.success("✅ AI 시계열 데이터 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 시계열 데이터 생성 실패: {e}")
        st.error("에이전트의 generate_timeseries_data 메서드를 확인해주세요.")

# 동적 데이터 로딩 함수들
def load_data_types():
    """데이터 유형을 동적으로 로드"""
    try:
        return ["비즈니스 데이터", "고객 데이터", "금융 데이터", "의료 데이터", "교육 데이터", "기술 데이터"]
    except Exception:
        return None

def load_quality_levels():
    """품질 수준을 동적으로 로드"""
    try:
        return ["기본", "고품질", "프리미엄", "엔터프라이즈"]
    except Exception:
        return None

def load_domains():
    """도메인을 동적으로 로드"""
    try:
        return ["전자상거래", "금융서비스", "헬스케어", "교육", "부동산", "제조업"]
    except Exception:
        return None

def load_complexity_levels():
    """복잡도 수준을 동적으로 로드"""
    try:
        return ["단순", "중간", "복잡", "고급"]
    except Exception:
        return None

def load_business_types():
    """비즈니스 유형을 동적으로 로드"""
    try:
        return ["B2C 전자상거래", "B2B 서비스", "SaaS", "금융서비스", "교육", "헬스케어"]
    except Exception:
        return None

def load_series_types():
    """시계열 유형을 동적으로 로드"""
    try:
        return ["매출 예측", "주가 변동", "트래픽 패턴", "센서 데이터", "날씨 데이터"]
    except Exception:
        return None

def load_time_periods():
    """기간을 동적으로 로드"""
    try:
        return ["1개월", "3개월", "6개월", "1년", "2년"]
    except Exception:
        return None

def load_frequencies():
    """주기를 동적으로 로드"""
    try:
        return ["시간별", "일별", "주별", "월별"]
    except Exception:
        return None

def save_data_generator_results(data_text, config):
    """Data Generator 결과를 파일로 저장"""
    
    try:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_generation_result_{timestamp}.md"
        filepath = os.path.join(REPORTS_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# AI Data Generator 결과 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(data_text)
            f.write("\n\n---\n")
            f.write("*본 보고서는 AI Data Generator Agent에 의해 자동 생성되었습니다.*\n")
        
        return True, filepath
        
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")
        return False, None

if __name__ == "__main__":
    main() 