#!/usr/bin/env python3
"""
Travel Scout - REAL MCP Integration (v3 - Process Manager)

✅ st.session_state 기반의 안정적인 상태 관리
✅ 분리된 프로세스 실행으로 UI 행(hang) 문제 해결
✅ 통합된 제어판 및 명확한 워크플로우
✅ 실행 후 스크린샷 갤러리 표시
"""

import streamlit as st
import sys
import os
import json
import base64
from pathlib import Path
from datetime import datetime, timedelta
import streamlit_process_manager as spm
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a

# --- 1. 프로젝트 경로 설정 ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- 2. 페이지 기본 설정 ---
st.set_page_config(
    page_title="Travel Scout - Integrated View",
    page_icon="✈️",
    layout="wide"
)

# --- 3. 공통/MCP 모듈 로드 ---
try:
    from srcs.common.page_utils import setup_page_header
    from srcs.common.styles import apply_custom_styles
    from srcs.travel_scout.travel_scout_agent import (
        load_destination_options, 
        load_origin_options
    )
    # 설정 파일에서 경로 가져오기
    from configs.settings import get_reports_path
    mcp_available = True
except ImportError as e:
    st.error(f"❌ 필수 모듈 로드 실패: {e}")
    st.info("💡 `pip install -r requirements.txt`를 실행하고, MCP 서버(`npm install @modelcontextprotocol/server-puppeteer`)가 설치되었는지 확인하세요.")
    mcp_available = False
    st.stop()

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# --- 4. 페이지 헤더 및 스타일 적용 ---
setup_page_header("Travel Scout", "Integrated Agent View")
apply_custom_styles()

# --- 5. UI 및 상태 관리 ---
if 'hotel_results' not in st.session_state:
    st.session_state.hotel_results = None
if 'flight_results' not in st.session_state:
    st.session_state.flight_results = None
if 'screenshots' not in st.session_state:
    st.session_state.screenshots = []

# --- 🎮 통합 제어판 ---
st.markdown("---")
st.markdown("## 🎮 Integrated Control Panel")

with st.form(key="travel_scout_form"):
    st.markdown("#### 🎯 Search Parameters")
    
    try:
        destination_options = load_destination_options()
        origin_options = load_origin_options()
    except Exception as e:
        st.error(f"❌ 목적지/출발지 목록 로드 실패: {e}")
        destination_options = ["Seoul", "Tokyo", "London", "New York"]
        origin_options = ["Seoul", "Busan", "New York", "London"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        destination = st.selectbox("🏖️ Destination", options=destination_options, index=0)
    with c2:
        origin = st.selectbox("✈️ Origin", options=origin_options, index=0)
    with c3:
        guests = st.number_input("👥 Guests", min_value=1, value=2)
    with c4:
        days = st.number_input("📅 Days from today", min_value=1, value=7)
    
    st.markdown("---")
    
    b1, b2 = st.columns(2)
    with b1:
        search_hotels_submitted = st.form_submit_button("🏨 Search Hotels", width='stretch')
    with b2:
        search_flights_submitted = st.form_submit_button("✈️ Search Flights", width='stretch')

# --- 🤖 에이전트 실행 로직 ---
task_to_run = None
if search_hotels_submitted:
    task_to_run = 'search_hotels'
elif search_flights_submitted:
    task_to_run = 'search_flights'

if task_to_run:
    # 새로운 검색이 시작될 때마다 이전 결과와 스크린샷 초기화
    st.session_state.screenshots = []
    if task_to_run == 'search_hotels':
        st.session_state.hotel_results = None
    else:
        st.session_state.flight_results = None

    reports_path = Path(get_reports_path('travel_scout'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = reports_path / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_txt_path = run_output_dir / "results.txt"
    
    agent_metadata = {
        "agent_id": "travel_scout_agent",
        "agent_name": "Travel Scout Agent",
        "entry_point": "srcs.travel_scout.run_travel_scout_agent",
        "agent_type": "mcp_agent",
        "capabilities": ["hotel_search", "flight_search", "travel_planning"],
        "description": "호텔 및 항공편 검색 및 여행 계획"
    }

    # 클래스 기반 실행을 위한 input_data 구성
    result_json_path = run_output_dir / "results.json"
    
    input_data = {
        "module_path": "srcs.travel_scout.run_travel_scout_agent",
        "class_name": "TravelScoutRunner",
        "result_json_path": str(result_json_path)
    }

    # 작업에 따른 인자 추가
    if task_to_run == 'search_hotels':
        check_in = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        check_out = (datetime.now() + timedelta(days=days+3)).strftime("%Y-%m-%d")
        input_data.update({
            "method_name": "run_hotels",
            "destination": destination,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests
        })
        st.info(f"🏨 {destination} 호텔 검색을 시작합니다...")

    elif task_to_run == 'search_flights':
        departure = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        ret_date = (datetime.now() + timedelta(days=days+7)).strftime("%Y-%m-%d")
        input_data.update({
            "method_name": "run_flights",
            "origin": origin,
            "destination": destination,
            "departure_date": departure,
            "return_date": ret_date
        })
        st.info(f"✈️ {origin} -> {destination} 항공편 검색을 시작합니다...")

    placeholder = st.empty()
    result = run_agent_via_a2a(
        placeholder=placeholder,
        agent_metadata=agent_metadata,
        input_data=input_data,
        result_json_path=result_json_path,
        use_a2a=True
    )
    
    if result:
        # 결과 처리 - result는 AgentExecutionResult 형태일 수 있음
        result_data = result.get('data', result) if isinstance(result, dict) else result
        
        # 결과가 dict인 경우 처리
        if isinstance(result_data, dict):
            # 성공 여부 확인
            if result_data.get('success'):
                # 실제 데이터 추출
                search_data = result_data.get('data', {})
                search_type = result_data.get('search_type', task_to_run)
                
                # 결과 텍스트 생성
                if search_type == 'hotels' or task_to_run == 'search_hotels':
                    result_text = _format_hotel_results(search_data)
                    st.session_state.hotel_results = result_text
                elif search_type == 'flights' or task_to_run == 'search_flights':
                    result_text = _format_flight_results(search_data)
                    st.session_state.flight_results = result_text
                else:
                    result_text = json.dumps(result_data, indent=2, ensure_ascii=False)
                    if task_to_run == 'search_hotels':
                        st.session_state.hotel_results = result_text
                    else:
                        st.session_state.flight_results = result_text
                
                # 스크린샷 경로 추출
                screenshots = result_data.get('screenshots', [])
                if screenshots:
                    st.session_state.screenshots = screenshots
                else:
                    # output 디렉토리에서 스크린샷 찾기
                    screenshot_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        screenshot_files.extend(Path(run_output_dir).glob(ext))
                    st.session_state.screenshots = [str(f) for f in screenshot_files]
            else:
                error_msg = result_data.get('error', 'Unknown error')
                st.error(f"❌ 검색 실패: {error_msg}")
        else:
            # 결과가 다른 형태인 경우 문자열로 변환
            result_text = json.dumps(result_data, indent=2, ensure_ascii=False) if not isinstance(result_data, str) else result_data
            if task_to_run == 'search_hotels':
                st.session_state.hotel_results = result_text
            else:
                st.session_state.flight_results = result_text
            
            # 스크린샷 찾기
            screenshot_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                screenshot_files.extend(Path(run_output_dir).glob(ext))
            st.session_state.screenshots = [str(f) for f in screenshot_files]


def _format_hotel_results(search_data: dict) -> str:
    """호텔 검색 결과를 포맷팅"""
    if not search_data:
        return "검색 결과가 없습니다."
    
    hotels = search_data.get('data', [])
    ai_analysis = search_data.get('ai_analysis', {})
    search_params = search_data.get('search_params', {})
    
    result_lines = []
    result_lines.append("=" * 50)
    result_lines.append("🏨 호텔 검색 결과")
    result_lines.append("=" * 50)
    result_lines.append(f"\n검색 조건:")
    result_lines.append(f"  - 목적지: {search_params.get('destination', 'N/A')}")
    result_lines.append(f"  - 체크인: {search_params.get('check_in', 'N/A')}")
    result_lines.append(f"  - 체크아웃: {search_params.get('check_out', 'N/A')}")
    result_lines.append(f"  - 게스트: {search_params.get('guests', 'N/A')}명")
    result_lines.append(f"\n발견된 호텔: {len(hotels)}개\n")
    
    if hotels:
        result_lines.append("호텔 목록:")
        for i, hotel in enumerate(hotels[:10], 1):  # 상위 10개만 표시
            result_lines.append(f"\n{i}. {hotel.get('name', 'N/A')}")
            result_lines.append(f"   가격: {hotel.get('price', 'N/A')}")
            result_lines.append(f"   평점: {hotel.get('rating', 'N/A')}")
            if hotel.get('location'):
                result_lines.append(f"   위치: {hotel.get('location')}")
    
    if ai_analysis:
        result_lines.append("\n" + "=" * 50)
        result_lines.append("AI 분석 결과")
        result_lines.append("=" * 50)
        analysis_text = ai_analysis.get('analysis', '')
        if analysis_text:
            result_lines.append(analysis_text)
    
    return "\n".join(result_lines)


def _format_flight_results(search_data: dict) -> str:
    """항공편 검색 결과를 포맷팅"""
    if not search_data:
        return "검색 결과가 없습니다."
    
    flights = search_data.get('data', [])
    ai_analysis = search_data.get('ai_analysis', {})
    search_params = search_data.get('search_params', {})
    
    result_lines = []
    result_lines.append("=" * 50)
    result_lines.append("✈️ 항공편 검색 결과")
    result_lines.append("=" * 50)
    result_lines.append(f"\n검색 조건:")
    result_lines.append(f"  - 출발지: {search_params.get('origin', 'N/A')}")
    result_lines.append(f"  - 목적지: {search_params.get('destination', 'N/A')}")
    result_lines.append(f"  - 출발일: {search_params.get('departure_date', 'N/A')}")
    result_lines.append(f"  - 귀국일: {search_params.get('return_date', 'N/A')}")
    result_lines.append(f"\n발견된 항공편: {len(flights)}개\n")
    
    if flights:
        result_lines.append("항공편 목록:")
        for i, flight in enumerate(flights[:10], 1):  # 상위 10개만 표시
            result_lines.append(f"\n{i}. {flight.get('airline', 'N/A')}")
            result_lines.append(f"   가격: {flight.get('price', 'N/A')}")
            result_lines.append(f"   소요시간: {flight.get('duration', 'N/A')}")
            if flight.get('departure_time'):
                result_lines.append(f"   출발시간: {flight.get('departure_time')}")
    
    if ai_analysis:
        result_lines.append("\n" + "=" * 50)
        result_lines.append("AI 분석 결과")
        result_lines.append("=" * 50)
        analysis_text = ai_analysis.get('analysis', '')
        if analysis_text:
            result_lines.append(analysis_text)
    
    return "\n".join(result_lines)
    
# --- 📊 검색 결과 표시 ---
st.markdown("---")
st.markdown("## 📊 Search Results")

res1, res2 = st.columns(2)

with res1:
    st.markdown("#### 🏨 Hotel Results")
    if st.session_state.hotel_results:
        st.text_area("검색 결과", st.session_state.hotel_results, height=300)
    else:
        st.info("호텔을 검색하여 결과를 확인하세요.")

with res2:
    st.markdown("#### ✈️ Flight Results")
    if st.session_state.flight_results:
        st.text_area("검색 결과", st.session_state.flight_results, height=300)
    else:
        st.info("항공편을 검색하여 결과를 확인하세요.")

# --- 🖼️ 스크린샷 갤러리 ---
if st.session_state.screenshots:
    st.markdown("---")
    st.markdown("## 🖼️ Screenshot History")
    
    cols = st.columns(3)
    for i, screenshot_path in enumerate(reversed(st.session_state.screenshots)):
        with cols[i % 3]:
            try:
                # 스크린샷 파일 경로로부터 이미지 표시
                st.image(screenshot_path, caption=f"Screenshot {i+1}", use_column_width=True)
            except Exception as e:
                st.warning(f"Screenshot display error: {e}")

# --- 📊 최신 Travel Scout 결과 확인 ---
st.markdown("---")
st.markdown("## 📊 최신 Travel Scout 결과")

# Travel Scout Agent의 최신 결과 확인
latest_travel_result = result_reader.get_latest_result("travel_scout_agent", "travel_search")

if latest_travel_result:
    with st.expander("🤖 최신 여행 검색 결과", expanded=False):
        st.subheader("✈️ 최근 여행 검색 결과")
        
        if isinstance(latest_travel_result, dict):
            # 검색 타입에 따른 결과 표시
            search_type = latest_travel_result.get('search_type', 'unknown')
            
            if search_type == 'hotels':
                st.write("🏨 **호텔 검색 결과**")
                if 'results' in latest_travel_result:
                    st.text_area("호텔 검색 결과", latest_travel_result['results'], height=200)
                
                # 검색 파라미터 표시
                if 'search_params' in latest_travel_result:
                    params = latest_travel_result['search_params']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("목적지", params.get('destination', 'N/A'))
                    with col2:
                        st.metric("체크인", params.get('check_in', 'N/A'))
                    with col3:
                        st.metric("게스트 수", params.get('guests', 'N/A'))
            
            elif search_type == 'flights':
                st.write("✈️ **항공편 검색 결과**")
                if 'results' in latest_travel_result:
                    st.text_area("항공편 검색 결과", latest_travel_result['results'], height=200)
                
                # 검색 파라미터 표시
                if 'search_params' in latest_travel_result:
                    params = latest_travel_result['search_params']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("출발지", params.get('origin', 'N/A'))
                    with col2:
                        st.metric("목적지", params.get('destination', 'N/A'))
                    with col3:
                        st.metric("출발일", params.get('departure_date', 'N/A'))
            
            # 메타데이터 표시
            if 'timestamp' in latest_travel_result:
                st.caption(f"⏰ 검색 시간: {latest_travel_result['timestamp']}")
            
            if 'screenshots' in latest_travel_result:
                st.info(f"📸 스크린샷 {len(latest_travel_result['screenshots'])}개 생성됨")
        else:
else:
    st.info("💡 아직 Travel Scout Agent의 결과가 없습니다. 위에서 여행 검색을 실행해보세요.")