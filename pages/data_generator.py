"""
📊 Data Generator Page

AI 기반 데이터 생성 및 분석 도구
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# AI Data Generation Agent 임포트
try:
    from srcs.basic_agents.data_generator import AIDataGenerationAgent
    AI_DATA_AGENT_AVAILABLE = True
except ImportError as e:
    AI_DATA_AGENT_AVAILABLE = False
    import_error = str(e)

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
    
    # Agent 연동 상태 확인
    if not AI_DATA_AGENT_AVAILABLE:
        st.error(f"⚠️ AI Data Generation Agent를 불러올 수 없습니다: {import_error}")
        st.info(" 에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### AI Data Generation Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install openai pandas numpy faker transformers torch
            ```
            
            2. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            3. **에이전트 모듈 확인**:
            ```bash
            ls srcs/basic_agents/ai_data_generation_agent.py
            ```
            """)
        return
    else:
        st.success("🤖 AI Data Generation Agent가 성공적으로 연결되었습니다!")
    
    #  에이전트 인터페이스
    render_real_ai_data_generator()

def render_real_ai_data_generator():
    """ AI Data Generator 인터페이스"""
    
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
        st.error(f"AI Data Generation Agent 초기화 중 오류: {e}")
        st.info("에이전트 모듈을 확인해주세요.")

def render_ai_smart_data_generation(agent):
    """AI 기반 스마트 데이터 생성"""
    
    st.markdown("### 🤖 AI 스마트 데이터 생성")
    st.info(" AI가 요구사항을 분석하여 지능적으로 데이터를 생성합니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ AI 데이터 생성 설정")
        
        data_purpose = st.text_area(
            "데이터 사용 목적",
            value="고객 행동 분석을 위한 샘플 데이터",
            help="AI가 목적에 맞는 데이터를 생성합니다"
        )
        
        data_type = st.selectbox(
            "데이터 유형",
            ["비즈니스 데이터", "고객 데이터", "금융 데이터", "의료 데이터", "교육 데이터", "기술 데이터"]
        )
        
        records_count = st.number_input("레코드 수", min_value=10, max_value=10000, value=1000)
        
        quality_level = st.select_slider(
            "품질 수준",
            options=["기본", "고품질", "프리미엄", "엔터프라이즈"],
            value="고품질"
        )
        
        include_relationships = st.checkbox("관계형 데이터 포함", value=True)
        include_patterns = st.checkbox(" 패턴 반영", value=True)
        
        if st.button("🚀 AI 스마트 데이터 생성", use_container_width=True):
            generate_ai_smart_data(agent, {
                'purpose': data_purpose,
                'type': data_type,
                'count': records_count,
                'quality': quality_level,
                'relationships': include_relationships,
                'patterns': include_patterns
            })
    
    with col2:
        if 'ai_generated_data' in st.session_state:
            st.markdown("#### 📊 AI 생성 데이터")
            data = st.session_state['ai_generated_data']
            st.json(data)  #  결과 표시
        else:
            st.markdown("""
            #### 🤖 AI 스마트 데이터 생성 기능
            
            **AI 지능형 생성:**
            - 🎯 목적 기반 데이터 구조 설계
            - 📊  분포와 패턴 반영
            - 🔗 논리적 관계성 보장
            - 🎨 사용자 맞춤형 스키마
            
            **고급 AI 기능:**
            - 🧠 딥러닝 기반 데이터 모델링
            - 📈 통계적 정확성 보장
            - 🔍 이상치 및 노이즈 제어
            - 💡 도메인 전문 지식 적용
            """)

def generate_ai_smart_data(agent, config):
    """AI를 사용한 스마트 데이터 생성"""
    
    try:
        with st.spinner("AI가 지능적으로 데이터를 생성 중입니다..."):
            result = agent.generate_smart_data(config)
            st.session_state['ai_generated_data'] = result
            st.success("✅ AI 스마트 데이터 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 데이터 생성 중 오류: {e}")
        st.info("에이전트의 generate_smart_data 메서드를 확인해주세요.")

def render_ai_custom_datasets(agent):
    """AI 맞춤형 데이터셋 생성"""
    
    st.markdown("### 📊 AI 맞춤형 데이터셋 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 데이터셋 설계")
        
        dataset_description = st.text_area(
            "데이터셋 설명",
            value="전자상거래 고객 구매 패턴 분석용 데이터셋",
            help="AI가 설명을 바탕으로 적절한 스키마를 설계합니다"
        )
        
        domain = st.selectbox(
            "도메인 전문성",
            ["전자상거래", "금융서비스", "헬스케어", "교육", "부동산", "제조업"]
        )
        
        complexity_level = st.select_slider(
            "복잡도",
            options=["단순", "중간", "복잡", "고급"],
            value="중간"
        )
        
        if st.button("🎯 AI 맞춤형 데이터셋 생성", use_container_width=True):
            generate_ai_custom_dataset(agent, {
                'description': dataset_description,
                'domain': domain,
                'complexity': complexity_level
            })
    
    with col2:
        if 'ai_custom_dataset' in st.session_state:
            st.markdown("#### 📄 AI 생성 데이터셋")
            dataset = st.session_state['ai_custom_dataset']
            st.json(dataset)
        else:
            st.info("👈 데이터셋 요구사항을 입력하고 'AI 맞춤형 데이터셋 생성' 버튼을 클릭하세요.")

def generate_ai_custom_dataset(agent, config):
    """AI 맞춤형 데이터셋 생성"""
    
    try:
        with st.spinner("AI가 맞춤형 데이터셋을 설계하고 생성 중입니다..."):
            result = agent.create_custom_dataset(config)
            st.session_state['ai_custom_dataset'] = result
            st.success("✅ AI 맞춤형 데이터셋이 생성되었습니다!")
            
    except Exception as e:
        st.error(f"AI 데이터셋 생성 중 오류: {e}")
        st.info("에이전트의 create_custom_dataset 메서드를 확인해주세요.")

def render_ai_customer_profiles(agent):
    """AI 고객 프로필 생성"""
    
    st.markdown("### 👥 AI 고객 프로필 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 고객 프로필 설정")
        
        business_type = st.selectbox(
            "비즈니스 유형",
            ["B2C 전자상거래", "B2B 서비스", "SaaS", "금융서비스", "교육", "헬스케어"]
        )
        
        target_segment = st.text_input(
            "타겟 고객층",
            value="25-45세 도시 거주 직장인",
            help="AI가 타겟 세그먼트에 맞는 고객 프로필을 생성합니다"
        )
        
        profile_count = st.number_input("프로필 수", min_value=10, max_value=5000, value=500)
        
        include_behavior = st.checkbox("구매 행동 패턴 포함", value=True)
        include_preferences = st.checkbox("선호도 데이터 포함", value=True)
        include_journey = st.checkbox("고객 여정 데이터 포함", value=True)
        
        if st.button("👥 AI 고객 프로필 생성", use_container_width=True):
            generate_ai_customer_profiles(agent, {
                'business_type': business_type,
                'target_segment': target_segment,
                'count': profile_count,
                'include_behavior': include_behavior,
                'include_preferences': include_preferences,
                'include_journey': include_journey
            })
    
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
            - 📊  행동 패턴 모델링
            - 🛒 구매 여정 시뮬레이션
            - 💡 선호도 및 관심사 생성
            """)

def generate_ai_customer_profiles(agent, config):
    """AI 고객 프로필 생성"""
    
    try:
        with st.spinner("AI가 고객 프로필을 생성 중입니다..."):
            result = agent.generate_customer_profiles(config)
            st.session_state['ai_customer_profiles'] = result
            st.success("✅ AI 고객 프로필 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 고객 프로필 생성 중 오류: {e}")
        st.info("에이전트의 generate_customer_profiles 메서드를 확인해주세요.")

def render_ai_timeseries_prediction(agent):
    """AI 시계열 예측 데이터 생성"""
    
    st.markdown("### 📈 AI 시계열 예측 데이터 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 시계열 설정")
        
        series_type = st.selectbox(
            "시계열 유형",
            ["매출 예측", "주가 변동", "트래픽 패턴", "센서 데이터", "날씨 데이터"]
        )
        
        time_period = st.selectbox(
            "기간",
            ["1개월", "3개월", "6개월", "1년", "2년"]
        )
        
        frequency = st.selectbox(
            "주기",
            ["시간별", "일별", "주별", "월별"]
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
    """AI 시계열 데이터 생성"""
    
    try:
        with st.spinner("AI가 시계열 데이터를 생성 중입니다..."):
            result = agent.generate_timeseries_data(config)
            st.session_state['ai_timeseries_data'] = result
            st.success("✅ AI 시계열 데이터 생성이 완료되었습니다!")
            
    except Exception as e:
        st.error(f"AI 시계열 데이터 생성 중 오류: {e}")
        st.info("에이전트의 generate_timeseries_data 메서드를 확인해주세요.")

if __name__ == "__main__":
    main() 