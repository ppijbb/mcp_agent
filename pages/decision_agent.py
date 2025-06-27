import streamlit as st
import json
import os
from datetime import datetime
import streamlit_process_manager as spm
from streamlit_process_manager.process import Process

from configs.settings import get_reports_path
from srcs.advanced_agents.decision_agent import (
    MobileInteraction,
    UserProfile,
    InteractionType,
)

# 경로 설정
REPORTS_PATH = get_reports_path('decision')
os.makedirs(REPORTS_PATH, exist_ok=True)

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

# 입력 양식 및 실행
with st.form("decision_form"):
    st.subheader("🚀 분석 시작하기")
    interaction_type = st.selectbox(
        "상호작용 유형 선택",
        options=[e for e in InteractionType],
        format_func=lambda x: x.value
    )
    app_name = st.text_input("앱/서비스 이름", value="Amazon")
    context_json = st.text_area("상황 상세 (JSON)",
                                value=context_json, height=150)
    submitted = st.form_submit_button("🧠 분석 시작", use_container_width=True)
    
if submitted:
    # 검증
    try:
        context_dict = json.loads(context_json)
    except json.JSONDecodeError:
        st.error("유효한 JSON 형식의 상황 상세를 입력해주세요.")
        st.stop()
    
    # 세션 상태에 경로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(REPORTS_PATH, f"decision_output_{timestamp}.log")
    result_file = os.path.join(REPORTS_PATH, f"decision_result_{timestamp}.json")
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    # CLI 명령 생성
    command = [
        "python", "-u",
        "srcs/advanced_agents/run_decision_agent.py",
        "--user-profile", json.dumps(profile.__dict__, ensure_ascii=False),
        "--interaction-type", interaction_type.name,
        "--app-name", app_name,
        "--context", json.dumps(context_dict, ensure_ascii=False),
        "--result-json-path", result_file
    ]
    
    process = Process(command, output_file=log_file).start()
    spm.st_process_monitor(process, label="의사결정 분석").loop_until_finished()
    st.success(f"✅ 분석 완료. 전체 로그: {log_file}")
    
    # 결과 표시
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('success'):
            st.header("3. 분석 결과")
            st.success("✅ 분석 완료!")
            st.subheader(f"결정: {data.get('recommendation')}")
            st.metric("신뢰도 점수", f"{data.get('confidence_score',0):.2%}")
            st.markdown(f"**위험 수준:** {data.get('risk_level')} ")
            st.markdown("**핵심 근거:**")
            st.info(data.get('reasoning',''))
            st.markdown("**대안:**")
            for alt in data.get('alternatives',[]):
                st.markdown(f"- {alt}")
            st.markdown("**증거 데이터:**")
            st.json(data.get('evidence', {}))
        else:
            st.error(f"분석 실패: {data.get('error')}")
    except FileNotFoundError:
        st.error("결과 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"결과 표시 중 오류 발생: {e}") 