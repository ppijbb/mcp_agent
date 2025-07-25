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
from srcs.common.ui_utils import run_agent_process

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
    from srcs.core.config.loader import settings
    from srcs.travel_scout.travel_scout_agent import (
        load_destination_options, 
        load_origin_options
    )
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
        search_hotels_submitted = st.form_submit_button("🏨 Search Hotels", use_container_width=True)
    with b2:
        search_flights_submitted = st.form_submit_button("✈️ Search Flights", use_container_width=True)

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

    reports_path = settings.get_reports_path('travel_scout')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = reports_path / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_txt_path = run_output_dir / "results.txt"
    
    py_executable = sys.executable
    command = [py_executable, "-u", "-m", "srcs.travel_scout.run_travel_scout_agent",
               "--task", task_to_run,
               "--result-txt-path", str(result_txt_path)]
    
    # 작업에 따른 인자 추가
    if task_to_run == 'search_hotels':
        check_in = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        check_out = (datetime.now() + timedelta(days=days+3)).strftime("%Y-%m-%d")
        command.extend([
            "--destination", destination,
            "--check-in", check_in,
            "--check-out", check_out,
            "--guests", str(guests)
        ])
        st.info(f"🏨 {destination} 호텔 검색을 시작합니다...")

    elif task_to_run == 'search_flights':
        departure = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        ret_date = (datetime.now() + timedelta(days=days+7)).strftime("%Y-%m-%d")
        command.extend([
            "--origin", origin,
            "--destination", destination,
            "--departure-date", departure,
            "--return-date", ret_date
        ])
        st.info(f"✈️ {origin} -> {destination} 항공편 검색을 시작합니다...")

    placeholder = st.empty()
    result = run_agent_process(
        placeholder=placeholder,
        command=command,
        process_key_prefix="logs/travel_scout",
        log_expander_title="실시간 실행 로그"
    )
    
    if result:
        # 결과에서 텍스트 추출 (결과는 {'result_text': '...'} 형태일 수 있음)
        result_text = result.get('result_text', str(result))

        if task_to_run == 'search_hotels':
            st.session_state.hotel_results = result_text
        else:
            st.session_state.flight_results = result_text
        
        # 스크린샷 경로는 output 디렉토리에서 찾기
        screenshot_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            screenshot_files.extend(Path(run_output_dir).glob(ext))
        st.session_state.screenshots = [str(f) for f in screenshot_files]
    
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
            st.json(latest_travel_result)
else:
    st.info("💡 아직 Travel Scout Agent의 결과가 없습니다. 위에서 여행 검색을 실행해보세요.")