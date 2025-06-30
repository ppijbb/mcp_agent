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
from streamlit_process_manager.process import Process

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
    from configs.settings import get_reports_path
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

    reports_path = get_reports_path('travel_scout')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = reports_path / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = run_output_dir / "results.json"
    
    py_executable = sys.executable
    command = [py_executable, "-u", "-m", "srcs.travel_scout.run_travel_scout_agent",
               "--task", task_to_run,
               "--result-json-path", str(result_json_path)]
    
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

    log_expander = st.expander("실시간 실행 로그", expanded=True)
    with log_expander:
        process_key = f"travel_scout_{timestamp}"
        process = Process(command, key=process_key).start()
        monitor_placeholder = st.empty()
        for stdout_line in process.read_stdout_live(auto_close=True):
            monitor_placeholder.code(stdout_line, language="log")
        
        return_code = process.wait()
    
    if return_code == 0:
        st.success("✅ 검색 완료!")
        try:
            with open(result_json_path) as f:
                final_result = json.load(f)
            
            if final_result.get("success"):
                if task_to_run == 'search_hotels':
                    st.session_state.hotel_results = final_result.get("data")
                else:
                    st.session_state.flight_results = final_result.get("data")
                st.session_state.screenshots = final_result.get("screenshots", [])
            else:
                st.error(f"❌ 에이전트 실행 중 오류 발생: {final_result.get('error', '알 수 없는 오류')}")
        except Exception as e:
            st.error(f"결과 파일을 읽거나 처리하는 중 오류 발생: {e}")
    else:
        st.error(f"❌ 에이전트 실행 실패 (Return Code: {return_code}). 로그를 확인하세요.")
    
    st.rerun()

# --- 📊 검색 결과 표시 ---
st.markdown("---")
st.markdown("## 📊 Search Results")

res1, res2 = st.columns(2)

with res1:
    st.markdown("#### 🏨 Hotel Results")
    if st.session_state.hotel_results:
        results = st.session_state.hotel_results
        if isinstance(results, dict):
            if 'hotels' in results and results['hotels']:
                st.dataframe(results['hotels'])
            elif 'error' in results:
                st.error(results['error'])
            else:
                st.json(results)
    else:
        st.info("호텔을 검색하여 결과를 확인하세요.")

with res2:
    st.markdown("#### ✈️ Flight Results")
    if st.session_state.flight_results:
        results = st.session_state.flight_results
        if isinstance(results, dict):
            if 'flights' in results and results['flights']:
                st.dataframe(results['flights'])
            elif 'error' in results:
                st.error(results['error'])
            else:
                st.json(results)
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