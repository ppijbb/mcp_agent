import streamlit as st
import asyncio
import json
from datetime import datetime

from srcs.common.streamlit_log_handler import setup_streamlit_logging
from srcs.advanced_agents.decision_agent import (
    DecisionAgentMCP,
    MobileInteraction,
    UserProfile,
    InteractionType,
)

# --- Agent Initialization ---
@st.cache_resource
def get_decision_agent():
    """Initializes and caches the DecisionAgentMCP."""
    return DecisionAgentMCP(output_dir="decision_reports")

agent = get_decision_agent()

# --- Page Configuration ---
st.set_page_config(page_title="🤔 의사결정 에이전트", page_icon="🤔", layout="wide")
st.title("🤔 의사결정 에이전트")
st.caption("🚀 복잡한 상황 속 최적의 결정을 내리도록 돕는 AI 조력자")

# --- UI Layout ---
st.header("1. 사용자 프로필 설정")
st.write("의사결정의 기반이 될 사용자 프로필을 설정합니다.")

# Initialize or get user profile from session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = UserProfile(
        user_id="demo_user_01",
        age=30,
        gender="남성",
        occupation="소프트웨어 엔지니어",
        income_level="상위 20%",
        risk_tolerance="중간",
        preferences={"선호 브랜드": ["Apple", "Nike"]},
        financial_goals=["주택 구매", "은퇴 자금 마련"],
        spending_patterns={"월 평균 소비": 2000, "주요 소비 카테고리": "기술 제품"}
    )
profile = st.session_state.user_profile

# User Profile Editor
col1, col2, col3 = st.columns(3)
with col1:
    profile.age = st.slider("나이", 20, 70, profile.age)
with col2:
    profile.gender = st.selectbox("성별", ["남성", "여성", "기타"], index=0)
with col3:
    profile.risk_tolerance = st.select_slider(
        "위험 선호도",
        options=["매우 낮음", "낮음", "중간", "높음", "매우 높음"],
        value=profile.risk_tolerance,
    )

with st.expander("전체 프로필 데이터 보기 (JSON)"):
    st.json(profile.__dict__)

st.markdown("---")

st.header("2. 상호작용 시뮬레이션")
st.write("분석할 상호작용(Interaction)을 시뮬레이션하고 AI의 결정을 확인하세요.")

col1, col2 = st.columns(2)
with col1:
    interaction_type = st.selectbox(
        "상호작용 유형 선택",
        options=[e for e in InteractionType],
        format_func=lambda x: x.value
    )
    app_name = st.text_input("앱/서비스 이름", value="Amazon")

with col2:
    context_json = st.text_area(
        "상황 상세 (JSON 형식)",
        value='{"product_id": "B08N5N43WT", "price": 999.99, "category": "electronics"}',
        height=150
    )

st.markdown("---")

# --- Real-time Log Display ---
log_expander = st.expander("실시간 실행 로그", expanded=False)
log_container = log_expander.empty()
setup_streamlit_logging(["mcp_agent", "decision_agent"], log_container)
# --- End Log Display ---

if st.button("🧠 분석 시작", type="primary", use_container_width=True):
    try:
        context_dict = json.loads(context_json)
        interaction = MobileInteraction(
            interaction_type=interaction_type,
            app_name=app_name,
            timestamp=datetime.now(),
            context=context_dict
        )
        
        with st.spinner("의사결정 분석 중... ReAct 패턴을 사용하여 심층 분석을 수행합니다."):
            result = asyncio.run(agent.analyze_and_decide(
                interaction=interaction,
                user_profile=profile
            ))
        
        st.header("3. 분석 결과")
        st.success("✅ 분석 완료!")
        
        decision = result.decision
        st.subheader(f"결정: {decision.recommendation}")
        st.metric("신뢰도 점수", f"{decision.confidence_score:.2%}")
        
        with st.expander("상세 분석 결과 보기", expanded=True):
            st.markdown(f"**- 위험 수준:** {decision.risk_level}")
            st.markdown("**- 핵심 근거:**")
            st.info(decision.reasoning)
            st.markdown("**- 대안:**")
            for alt in decision.alternatives:
                st.markdown(f"  - {alt}")
            st.markdown("**- 증거 데이터:**")
            st.json(decision.evidence)
        
    except json.JSONDecodeError:
        st.error("오류: '상황 상세'에 유효한 JSON 형식의 데이터를 입력해주세요.")
    except Exception as e:
        st.error(f"분석 중 예기치 않은 오류가 발생했습니다: {e}")
        st.exception(e) # Show full traceback 