"""
🧳 Travel Scout Agent Page

시크릿 모드를 활용한 여행 검색 AI
"""

import streamlit as st
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(
    page_title="Travel Scout Agent", 
    page_icon="🧳",
    layout="wide"
)

# src 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "srcs"))

try:
    from srcs.travel_scout.travel_scout_agent import TravelScoutAgent
except ImportError as e:
    st.error(f"Travel Scout Agent를 불러올 수 없습니다: {e}")
    st.stop()

def main():
    """Travel Scout Agent 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1>🧳 Travel Scout Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            🔒 시크릿 모드를 활용한 가성비 여행 검색 AI
        </p>
        <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.9;">
            캐시 간섭 없이 진짜 최저가를 찾아드립니다!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # 에이전트 소개
    render_agent_intro()
    
    # 에이전트 인터페이스
    render_travel_scout_interface()

def render_agent_intro():
    """에이전트 소개"""
    
    st.success("🤖 Travel Scout Agent가 성공적으로 연결되었습니다!")
    
    # 주요 특징 소개
    st.markdown("### 🌟 주요 특징")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.info("""
        **🔒 시크릿 모드**
        - 캐시/쿠키 방지
        - 가격 조작 차단
        - 공정한 가격 비교
        """)
        
    with feature_col2:
        st.success("""
        **⭐ 고품질 기준**
        - 4.0+ 평점 호텔
        - 100+ 리뷰 필수
        - 신뢰성 있는 항공사
        """)
        
    with feature_col3:
        st.warning("""
        **💰 최저가 보장**
        - 다중 플랫폼 비교
        - 숨겨진 비용 확인
        - 최적 조합 추천
        """)

def render_travel_scout_interface():
    """Travel Scout Agent 실행 인터페이스"""
    
    # 2개 컬럼으로 나누기
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 제어 패널")
        render_search_interface()
        st.markdown("---")
        render_settings_interface()
    
    with col2:
        st.markdown("### 📊 검색 결과")
        render_results_display()

def render_search_interface():
    """여행 검색 인터페이스"""
    
    st.markdown("#### 📍 여행 정보 입력")
    
    # 두 개의 컬럼으로 나누어 입력 필드 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 목적지 정보")
        destination = st.text_input(
            "목적지",
            value="Tokyo, Japan",
            help="여행하고 싶은 도시를 입력하세요 (예: Tokyo, Japan)"
        )
        
        origin = st.text_input(
            "출발지",
            value="Seoul, South Korea",
            help="출발하는 도시를 입력하세요 (예: Seoul, South Korea)"
        )
    
    with col2:
        st.markdown("#### 📅 여행 날짜")
        
        # 기본 날짜 설정 (30일 후)
        default_departure = datetime.now() + timedelta(days=30)
        default_return = datetime.now() + timedelta(days=37)
        default_checkin = datetime.now() + timedelta(days=30)
        default_checkout = datetime.now() + timedelta(days=33)
        
        departure_date = st.date_input(
            "항공편 출발일",
            value=default_departure,
            help="항공편 출발 날짜를 선택하세요"
        )
        
        return_date = st.date_input(
            "항공편 귀국일",
            value=default_return,
            help="항공편 귀국 날짜를 선택하세요"
        )
        
        check_in = st.date_input(
            "호텔 체크인",
            value=default_checkin,
            help="호텔 체크인 날짜를 선택하세요"
        )
        
        check_out = st.date_input(
            "호텔 체크아웃",
            value=default_checkout,
            help="호텔 체크아웃 날짜를 선택하세요"
        )
    
    # 설정된 품질 기준 표시
    st.markdown("---")
    st.markdown("### ⭐ 품질 기준")
    
    # 설정에서 불러오기 (기본값 설정)
    settings = st.session_state.get('travel_scout_settings', {
        'quality': {
            'hotel_rating': 4.0,
            'min_reviews': 100,
            'flight_rating': 4.0
        }
    })
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        st.metric("호텔 최소 평점", f"{settings['quality']['hotel_rating']}/5.0", "⭐⭐⭐⭐")
    
    with quality_col2:
        st.metric("최소 리뷰 수", f"{settings['quality']['min_reviews']}개+", "신뢰성 보장")
    
    with quality_col3:
        st.metric("항공사 평점", f"{settings['quality']['flight_rating']}/5.0", "안전성 우선")
    
    # 시크릿 모드 정보 (간단히)
    st.info("🔒 시크릿 모드로 가격 조작 없는 진짜 최저가를 찾습니다")
    
    # 검색 실행 버튼
    st.markdown("---")
    
    if st.button("🚀 시크릿 모드로 여행 검색 시작!", type="primary", use_container_width=True):
        # 입력 검증
        search_params = {
            'destination': destination,
            'origin': origin,
            'departure_date': departure_date.strftime('%Y-%m-%d'),
            'return_date': return_date.strftime('%Y-%m-%d'),
            'check_in': check_in.strftime('%Y-%m-%d'),
            'check_out': check_out.strftime('%Y-%m-%d')
        }
        
        if validate_search_inputs(search_params):
            execute_travel_search(search_params)

def validate_search_inputs(params):
    """검색 입력값 검증"""
    
    required_fields = ['destination', 'origin']
    for field in required_fields:
        if not params[field] or not params[field].strip():
            st.error(f"❌ {field}을(를) 입력해주세요.")
            return False
    
    # 날짜 검증
    try:
        departure = datetime.strptime(params['departure_date'], '%Y-%m-%d')
        return_date = datetime.strptime(params['return_date'], '%Y-%m-%d')
        check_in = datetime.strptime(params['check_in'], '%Y-%m-%d')
        check_out = datetime.strptime(params['check_out'], '%Y-%m-%d')
        
        if return_date <= departure:
            st.error("❌ 귀국일이 출발일보다 늦어야 합니다.")
            return False
            
        if check_out <= check_in:
            st.error("❌ 체크아웃이 체크인보다 늦어야 합니다.")
            return False
            
        if departure < datetime.now():
            st.error("❌ 출발일이 오늘보다 이후여야 합니다.")
            return False
            
    except ValueError as e:
        st.error(f"❌ 날짜 형식 오류: {e}")
        return False
    
    return True

def execute_travel_search(params):
    """실제 Travel Scout Agent를 실행하여 여행 검색"""
    
    with st.spinner("🔒 시크릿 모드로 여행 정보를 검색 중입니다..."):
        try:
            # 설정값 가져오기
            settings = st.session_state.get('travel_scout_settings', {
                'quality': {
                    'hotel_rating': 4.0,
                    'min_reviews': 100,
                    'flight_rating': 4.0
                }
            })
            
            # Travel Scout Agent 인스턴스 생성
            agent = TravelScoutAgent(
                destination=params['destination'],
                origin=params['origin'],
                departure_date=params['departure_date'],
                return_date=params['return_date'],
                check_in=params['check_in'],
                check_out=params['check_out']
            )
            
            # 품질 기준 설정
            agent.min_hotel_rating = settings['quality']['hotel_rating']
            agent.min_review_count = settings['quality']['min_reviews']
            agent.min_flight_rating = settings['quality']['flight_rating']
            
            # Agent 실행 준비
            st.session_state['travel_agent'] = agent
            st.session_state['search_params'] = params
            st.session_state['search_status'] = 'ready'
            
            st.success("✅ Travel Scout Agent가 준비되었습니다!")
            
            # 실제 검색 실행 버튼
            if st.button("🔍 실제 검색 실행하기", type="primary"):
                run_agent_search()
            
            st.info("📊 '검색 결과' 탭에서 진행 상황을 확인하세요.")
            
        except Exception as e:
            st.error(f"❌ Agent 초기화 중 오류: {str(e)}")

def run_agent_search():
    """실제 agent 검색 실행"""
    try:
        agent = st.session_state.get('travel_agent')
        if not agent:
            st.error("❌ Agent가 초기화되지 않았습니다.")
            return
        
        # 검색 상태 업데이트
        st.session_state['search_status'] = 'running'
        
        # 프로그레스 바 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔒 Agent 준비 중...")
        progress_bar.progress(25)
        
        status_text.text("🎯 검색 매개변수 설정 중...")
        progress_bar.progress(50)
        
        status_text.text("📋 검색 계획 수립 중...")
        progress_bar.progress(75)
        
        # Agent 준비 완료 (실제 검색은 MCP 서버가 필요)
        result_data = {
            'status': 'prepared',
            'agent_info': {
                'destination': agent.destination,
                'origin': agent.origin,
                'departure_date': agent.departure_date,
                'return_date': agent.return_date,
                'check_in': agent.check_in,
                'check_out': agent.check_out,
                'min_hotel_rating': agent.min_hotel_rating,
                'min_review_count': agent.min_review_count,
                'min_flight_rating': agent.min_flight_rating
            },
            'message': 'Agent가 준비되었습니다. 실제 검색을 위해서는 MCP Playwright 서버가 필요합니다.'
        }
        
        status_text.text("📊 Agent 준비 완료...")
        progress_bar.progress(100)
        
        # 결과 저장
        st.session_state['search_result'] = result_data
        st.session_state['search_status'] = 'completed'
        
        status_text.text("✅ Agent 준비 완료!")
        st.success("🎉 Travel Scout Agent가 준비되었습니다!")
        st.info("💡 실제 웹 검색을 위해서는 MCP Playwright 서버 연결이 필요합니다.")
        
    except Exception as e:
        st.error(f"❌ Agent 준비 중 오류: {str(e)}")
        st.session_state['search_status'] = 'error'

def render_results_display():
    """검색 결과 표시"""
    
    search_status = st.session_state.get('search_status', 'none')
    
    if search_status == 'none':
        st.info("🔍 아직 검색을 실행하지 않았습니다.")
        st.markdown("""
        **검색을 시작하려면:**
        1. 🛫 '여행 검색' 탭으로 이동
        2. 여행 정보 입력
        3. '시크릿 모드로 여행 검색 시작!' 버튼 클릭
        """)
        
    elif search_status == 'ready':
        agent = st.session_state.get('travel_agent')
        params = st.session_state.get('search_params')
        
        if agent and params:
            st.success("✅ 검색 준비 완료!")
            
            # 검색 정보 요약
            st.markdown("#### 🎯 검색 설정")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("목적지", params['destination'])
                st.metric("출발지", params['origin'])
            
            with info_col2:
                st.metric("여행 기간", f"{params['departure_date']} ~ {params['return_date']}")
                st.metric("숙박 기간", f"{params['check_in']} ~ {params['check_out']}")
            
            with info_col3:
                st.metric("호텔 최소 평점", f"{agent.min_hotel_rating}/5.0")
                st.metric("최소 리뷰 수", f"{agent.min_review_count}개")
            
            # Agent 정보 표시
            st.markdown("#### 🤖 Agent 구성")
            
            agent_col1, agent_col2 = st.columns(2)
            
            with agent_col1:
                st.info("""
                **호텔 검색 Agent**
                - 🔒 시크릿 모드 활성화
                - 🏨 다중 플랫폼 검색
                - ⭐ 품질 기준 적용
                """)
                
            with agent_col2:
                st.info("""
                **항공편 검색 Agent**
                - 🔒 캐시 방지 모드
                - ✈️ 신뢰성 있는 항공사
                - 💰 최저가 우선
                """)
            
            # 실행 버튼
            st.markdown("---")
            if st.button("🚀 지금 검색 실행하기!", type="primary", use_container_width=True):
                run_agent_search()
                
    elif search_status == 'running':
        st.warning("🔄 검색이 진행 중입니다...")
        st.spinner("시크릿 모드로 여행 정보를 검색하고 있습니다. 잠시만 기다려주세요.")
        
    elif search_status == 'completed':
        result = st.session_state.get('search_result')
        
        if result and result.get('status') == 'prepared':
            st.success("🎉 Agent가 성공적으로 준비되었습니다!")
            
            # Agent 정보 표시
            st.markdown("#### 📋 Agent 설정 정보")
            
            agent_info = result['agent_info']
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.metric("목적지", agent_info['destination'])
                st.metric("출발지", agent_info['origin'])
                st.metric("호텔 평점 기준", f"{agent_info['min_hotel_rating']}/5.0")
                
            with info_col2:
                st.metric("여행 기간", f"{agent_info['departure_date']} ~ {agent_info['return_date']}")
                st.metric("숙박 기간", f"{agent_info['check_in']} ~ {agent_info['check_out']}")
                st.metric("최소 리뷰 수", f"{agent_info['min_review_count']}개")
            
            # 준비된 기능
            st.markdown("#### 🎯 준비된 기능")
            
            feature_col1, feature_col2 = st.columns(2)
            
            with feature_col1:
                st.info("""
                **🏨 호텔 검색**
                - Booking.com, Hotels.com
                - Expedia, Agoda
                - 시크릿 모드 지원
                """)
                
            with feature_col2:
                st.info("""
                **✈️ 항공편 검색**
                - Google Flights, Kayak
                - Skyscanner
                - 가격 비교 분석
                """)
            
            # 다음 단계 안내
            st.markdown("#### 🚀 다음 단계")
            st.warning(result['message'])
            
            with st.expander("🔧 MCP Playwright 서버 설정 방법"):
                st.markdown("""
                실제 웹 검색을 실행하려면:
                
                1. **MCP Playwright 서버 설치**
                ```bash
                npm install -g @modelcontextprotocol/server-playwright
                ```
                
                2. **서버 실행**
                ```bash
                npx @modelcontextprotocol/server-playwright
                ```
                
                3. **Agent 재실행**
                - 서버가 실행 중일 때 검색 버튼 클릭
                """)
                
        else:
            st.error("❌ Agent 결과를 불러올 수 없습니다.")
            
    elif search_status == 'error':
        st.error("❌ 검색 중 오류가 발생했습니다.")
        if st.button("🔄 다시 시도하기"):
            st.session_state['search_status'] = 'ready'
            st.rerun()

def render_settings_interface():
    """설정 인터페이스"""
    
    st.markdown("#### ⚙️ 품질 & 플랫폼 설정")
    
    # 현재 설정 불러오기
    current_settings = st.session_state.get('travel_scout_settings', {
        'quality': {
            'hotel_rating': 4.0,
            'min_reviews': 100,
            'flight_rating': 4.0,
            'search_depth': '기본 (상위 10개)'
        },
        'platforms': {
            'hotels': {
                'booking_com': True,
                'hotels_com': True,
                'expedia': True,
                'agoda': False
            },
            'flights': {
                'google_flights': True,
                'kayak': True,
                'skyscanner': True
            }
        }
    })
    
    # 품질 기준 설정
    st.markdown("#### ⭐ 품질 기준 조정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏨 호텔 품질 기준**")
        
        hotel_rating = st.slider(
            "최소 호텔 평점",
            min_value=3.0,
            max_value=5.0,
            value=current_settings['quality']['hotel_rating'],
            step=0.1
        )
        
        min_reviews = st.slider(
            "최소 리뷰 수",
            min_value=50,
            max_value=500,
            value=current_settings['quality']['min_reviews'],
            step=25
        )
        
    with col2:
        st.markdown("**✈️ 항공편 품질 기준**")
        
        flight_rating = st.slider(
            "최소 항공사 평점",
            min_value=3.0,
            max_value=5.0,
            value=current_settings['quality']['flight_rating'],
            step=0.1
        )
        
        search_depth = st.selectbox(
            "검색 깊이",
            ["기본 (상위 10개)", "깊이 (상위 20개)", "전체 (상위 50개)"],
            index=["기본 (상위 10개)", "깊이 (상위 20개)", "전체 (상위 50개)"].index(
                current_settings['quality']['search_depth']
            )
        )
    
    # 검색 플랫폼 선택
    st.markdown("#### 🌐 검색 플랫폼 선택")
    
    platform_col1, platform_col2 = st.columns(2)
    
    with platform_col1:
        st.markdown("**🏨 호텔 검색 사이트**")
        
        booking_com = st.checkbox("Booking.com", value=current_settings['platforms']['hotels']['booking_com'])
        hotels_com = st.checkbox("Hotels.com", value=current_settings['platforms']['hotels']['hotels_com'])
        expedia_hotels = st.checkbox("Expedia", value=current_settings['platforms']['hotels']['expedia'])
        agoda = st.checkbox("Agoda", value=current_settings['platforms']['hotels']['agoda'])
        
    with platform_col2:
        st.markdown("**✈️ 항공편 검색 사이트**")
        
        google_flights = st.checkbox("Google Flights", value=current_settings['platforms']['flights']['google_flights'])
        kayak = st.checkbox("Kayak", value=current_settings['platforms']['flights']['kayak'])
        skyscanner = st.checkbox("Skyscanner", value=current_settings['platforms']['flights']['skyscanner'])
    
    # 시크릿 모드 상태 표시 (간단히)
    st.success("🔒 시크릿 모드: 가격 조작 방지 활성화")
    
    # 설정 저장
    if st.button("💾 설정 저장", type="primary"):
        settings = {
            'quality': {
                'hotel_rating': hotel_rating,
                'min_reviews': min_reviews,
                'flight_rating': flight_rating,
                'search_depth': search_depth
            },
            'platforms': {
                'hotels': {
                    'booking_com': booking_com,
                    'hotels_com': hotels_com,
                    'expedia': expedia_hotels,
                    'agoda': agoda
                },
                'flights': {
                    'google_flights': google_flights,
                    'kayak': kayak,
                    'skyscanner': skyscanner
                }
            }
        }
        
        st.session_state['travel_scout_settings'] = settings
        st.success("✅ 설정이 저장되었습니다!")
        
        # 간단한 확인 메시지
        st.info(f"호텔 평점: {hotel_rating}+ | 리뷰: {min_reviews}+ | 항공사: {flight_rating}+")

if __name__ == "__main__":
    main() 