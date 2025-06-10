import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from srcs.urban_hive import ResourceMatcherAgent, SocialConnectorAgent, UrbanAnalystAgent
from srcs.common.page_utils import setup_page, render_home_button

# 페이지 설정
setup_page("🏙️ Urban Hive Agent", "🏙️")

st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>🏙️ Urban Hive 스마트 도시 에이전트</h1>
    <p style='font-size: 1.2rem; color: #666;'>도시 문제 해결과 커뮤니티 연결을 위한 AI 에이전트</p>
</div>
""", unsafe_allow_html=True)

# 홈 버튼
render_home_button()

st.markdown("---")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["🤝 자원 매칭", "👥 소셜 커넥터", "📊 도시 분석"])

# 에이전트 인스턴스 생성 (session_state에 저장하여 재사용)
if 'resource_agent' not in st.session_state:
    st.session_state.resource_agent = ResourceMatcherAgent()

if 'social_agent' not in st.session_state:
    st.session_state.social_agent = SocialConnectorAgent()

if 'urban_agent' not in st.session_state:
    st.session_state.urban_agent = UrbanAnalystAgent()

# 자원 매칭 에이전트 탭
with tab1:
    st.header("🤝 자원 매칭 에이전트")
    st.markdown("""
    **자원을 공유하거나 필요한 물건을 찾아보세요!**
    - 남은 음식, 생활용품, 도구 등을 이웃과 공유
    - AI가 자동으로 매칭해드립니다
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        resource_query = st.text_area(
            "무엇을 공유하거나 필요로 하시나요?",
            placeholder="예: 빵이 남아서 나눠드리고 싶어요\n또는: 오늘 사다리가 필요해요",
            height=100
        )
        
        if st.button("🔍 매칭 찾기", key="resource_match"):
            if resource_query:
                with st.spinner("AI가 매칭을 찾고 있습니다..."):
                    try:
                        result = st.session_state.resource_agent.run(resource_query)
                        st.success("매칭 결과를 찾았습니다!")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {str(e)}")
                        st.info("MCP 서버 연결에 문제가 있을 수 있습니다. 나중에 다시 시도해주세요.")
            else:
                st.warning("검색할 내용을 입력해주세요.")
    
    with col2:
        st.markdown("### 📊 현재 상태")
        try:
            stats = st.session_state.resource_agent.get_resource_statistics()
            st.metric("등록된 자원", stats.get("total_resources_available", "N/A"))
            st.metric("요청 건수", stats.get("total_requests", "N/A"))
        except:
            st.info("통계를 불러올 수 없습니다.")
    
    # 예시 섹션
    with st.expander("💡 사용 예시"):
        st.markdown("""
        **제공하는 경우:**
        - "빵이 많이 남아서 나눠드리고 싶어요"
        - "사용하지 않는 책들이 있어요"
        - "드릴이 있는데 빌려드릴 수 있어요"
        
        **필요한 경우:**
        - "오늘 사다리가 필요해요"
        - "아이 장난감을 찾고 있어요"
        - "요리 재료가 조금 필요해요"
        """)

# 소셜 커넥터 에이전트 탭
with tab2:
    st.header("👥 소셜 커넥터 에이전트")
    st.markdown("""
    **비슷한 관심사를 가진 사람들과 연결되어 보세요!**
    - AI가 당신의 프로필을 분석하여 맞는 사람들을 찾아드립니다
    - 고립 위험도 평가 및 맞춤형 추천
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("social_profile_form"):
            st.markdown("### 📝 프로필 정보")
            
            name = st.text_input("이름", placeholder="예: 김철수")
            interests = st.text_area(
                "관심사나 취미를 자유롭게 적어주세요",
                placeholder="예: 요리, 운동, 독서, 여행, 사진 촬영을 좋아합니다. 특히 새벽 조깅을 즐기고 카페에서 책 읽는 것을 좋아해요.",
                height=100
            )
            
            col_age, col_location = st.columns(2)
            with col_age:
                age = st.number_input("나이", min_value=10, max_value=100, value=30)
            with col_location:
                location = st.text_input("거주 지역", placeholder="예: 강남구")
            
            # 추가 정보
            st.markdown("### 📋 추가 정보 (선택사항)")
            work_status = st.selectbox("직업 상태", ["직장인", "학생", "프리랜서", "주부", "은퇴", "기타"])
            social_frequency = st.select_slider(
                "평소 사람들과 만나는 빈도",
                options=["거의 없음", "월 1-2회", "주 1-2회", "거의 매일"],
                value="주 1-2회"
            )
            
            submit_social = st.form_submit_button("🔍 소셜 매칭 찾기")
            
            if submit_social:
                if name and interests:
                    user_profile = {
                        "name": name,
                        "interests": interests,
                        "age": age,
                        "location": location,
                        "work_status": work_status,
                        "social_frequency": social_frequency
                    }
                    
                    with st.spinner("AI가 맞춤형 추천을 생성하고 있습니다..."):
                        try:
                            result = st.session_state.social_agent.run(user_profile)
                            st.success("추천 결과가 준비되었습니다!")
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"오류가 발생했습니다: {str(e)}")
                            st.info("MCP 서버 연결에 문제가 있을 수 있습니다. 나중에 다시 시도해주세요.")
                else:
                    st.warning("이름과 관심사는 필수 입력 항목입니다.")
    
    with col2:
        st.markdown("### 📊 커뮤니티 현황")
        try:
            stats = st.session_state.social_agent.get_community_statistics()
            st.metric("활성 멤버", stats.get("total_active_members", "N/A"))
            st.metric("활성 그룹", stats.get("total_active_groups", "N/A"))
            st.metric("이번 달 연결", stats.get("connections_made_this_month", "N/A"))
            
            if "most_popular_activity" in stats:
                st.markdown(f"**인기 활동:** {stats['most_popular_activity']}")
        except:
            st.info("통계를 불러올 수 없습니다.")

# 도시 분석 에이전트 탭
with tab3:
    st.header("📊 도시 분석 에이전트")
    st.markdown("""
    **실시간 도시 데이터 분석과 인사이트를 제공합니다**
    - MCP 서버를 통한 실시간 데이터 연동
    - AI 기반 도시 문제 분석 및 해결책 제시
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 분석할 도시 데이터 선택")
        
        analysis_options = [
            "Illegal Dumping - 불법 투기 분석",
            "Public Safety - 공공 안전 분석", 
            "Traffic Flow - 교통 흐름 분석",
            "Community Event - 커뮤니티 이벤트 분석"
        ]
        
        selected_analysis = st.selectbox(
            "분석 유형을 선택하세요:",
            analysis_options
        )
        
        if st.button("📈 분석 시작", key="urban_analysis"):
            with st.spinner("실시간 데이터를 분석하고 있습니다..."):
                try:
                    result = st.session_state.urban_agent.run(selected_analysis)
                    st.success("분석이 완료되었습니다!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                    st.info("MCP 서버나 데이터 소스에 연결 문제가 있을 수 있습니다.")
    
    with col2:
        st.markdown("### 📊 도시 통계")
        try:
            stats = st.session_state.urban_agent.get_urban_statistics()
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    st.metric(key.replace("_", " ").title(), f"{value}")
                else:
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        except:
            st.info("통계를 불러올 수 없습니다.")
        
        # 실시간 상태 표시
        st.markdown("### 🔌 연결 상태")
        st.markdown("- 🟢 UI 인터페이스: 정상")
        st.markdown("- 🟡 MCP 서버: 연결 시도 중")
        st.markdown("- 🟡 데이터 소스: 대기")

# 하단 정보
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💡 <strong>Urban Hive Agent</strong>는 도시의 다양한 문제를 AI로 해결하는 통합 플랫폼입니다.</p>
    <p>문제가 발생하면 MCP 서버 연결 상태를 확인해주세요.</p>
</div>
""", unsafe_allow_html=True)

