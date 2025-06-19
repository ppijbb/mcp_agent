#!/usr/bin/env python3
"""
Travel Scout - REAL MCP Integration (v2 - Normalized)

✅ st.session_state 기반의 안정적인 상태 관리
✅ 실시간 브라우저 뷰 컨테이너
✅ 통합된 제어판 및 명확한 워크플로우
✅ 스크린샷 갤러리
"""

import streamlit as st
import sys
import os
import asyncio
import base64
from pathlib import Path
from datetime import datetime, timedelta

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
    from srcs.travel_scout.mcp_browser_client import MCPBrowserClient
    from srcs.travel_scout.travel_scout_agent import (
        TravelScoutAgent, 
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

# 🖥️ 실시간 브라우저 뷰 컨테이너
st.markdown("## 🖥️ Real-time Browser View")
browser_view_container = st.container(border=True, height=600)

# ⚙️ 서비스 초기화 (session_state 사용)
if 'services_initialized' not in st.session_state:
    with st.spinner("Initializing MCP services..."):
        try:
            # 브라우저 뷰 컨테이너를 클라이언트에 전달
            client = MCPBrowserClient(
                headless=True,
                disable_gpu=True,
                streamlit_container=browser_view_container
            )
            st.session_state.browser_client = client
            st.session_state.travel_agent = TravelScoutAgent(browser_client=client)
            st.session_state.mcp_connected = False
            st.session_state.services_initialized = True
            st.session_state.hotel_results = None
            st.session_state.flight_results = None
            st.success("✅ MCP Services Initialized.")
        except Exception as e:
            st.error(f"❌ MCP 서비스 초기화 실패: {e}")
            st.session_state.services_initialized = False
            st.stop()

# --- 🎮 통합 제어판 ---
st.markdown("---")
st.markdown("## 🎮 Integrated Control Panel")

# 🔌 MCP 서버 연결 관리
if not st.session_state.get('mcp_connected', False):
    if st.button("🔌 Connect to MCP Server", type="primary", use_container_width=True):
        with st.spinner("MCP 서버에 연결 중... 브라우저를 시작합니다."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = st.session_state.browser_client
            connected = loop.run_until_complete(client.connect_to_mcp_server())
            loop.close()
            if connected:
                st.session_state.mcp_connected = True
                st.success("✅ MCP 서버 연결 성공!")
                st.rerun()
            else:
                st.error("❌ MCP 서버 연결 실패. 콘솔 로그를 확인하세요.")
                st.stop()
else:
    st.success("✅ MCP Server Connected")

# ✈️ 에이전트 작업 제어 (연결된 경우에만 표시)
if st.session_state.get('mcp_connected', False):
    with st.container(border=True):
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
            if st.button("🏨 Search Hotels", use_container_width=True, type="secondary"):
                with st.spinner(f"🏨 {destination} 호텔 검색 중... (브라우저 뷰를 확인하세요)"):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        agent = st.session_state.travel_agent
                        check_in = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                        check_out = (datetime.now() + timedelta(days=days+3)).strftime("%Y-%m-%d")
                        
                        results = loop.run_until_complete(
                            agent.search_hotels(destination, check_in, check_out, guests)
                        )
                        loop.close()
                        st.session_state.hotel_results = results
                        st.success("호텔 검색 완료!")
                    except Exception as e:
                        st.error(f"❌ 호텔 검색 중 오류 발생: {e}")
                        st.exception(e)

        with b2:
            if st.button("✈️ Search Flights", use_container_width=True, type="secondary"):
                with st.spinner(f"✈️ {origin} -> {destination} 항공편 검색 중... (브라우저 뷰를 확인하세요)"):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        agent = st.session_state.travel_agent
                        departure = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
                        ret_date = (datetime.now() + timedelta(days=days+7)).strftime("%Y-%m-%d")

                        results = loop.run_until_complete(
                            agent.search_flights(origin, destination, departure, ret_date)
                        )
                        loop.close()
                        st.session_state.flight_results = results
                        st.success("항공편 검색 완료!")
                    except Exception as e:
                        st.error(f"❌ 항공편 검색 중 오류 발생: {e}")
                        st.exception(e)

# --- 📊 검색 결과 표시 ---
st.markdown("---")
st.markdown("## 📊 Search Results")

res1, res2 = st.columns(2)

with res1:
    st.markdown("#### 🏨 Hotel Results")
    if st.session_state.get('hotel_results'):
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
    if st.session_state.get('flight_results'):
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
if st.session_state.get('browser_client') and st.session_state.browser_client.screenshots:
    st.markdown("---")
    st.markdown("## 🖼️ Screenshot History")
    
    screenshots_to_show = st.session_state.browser_client.screenshots
    
    cols = st.columns(3)
    for i, screenshot in enumerate(reversed(screenshots_to_show)):
        with cols[i % 3]:
            img_data = screenshot.get('data')
            if not img_data:
                continue
                
            if img_data.startswith('data:image'):
                img_data = img_data.split(',')[1]
            
            try:
                st.image(
                    base64.b64decode(img_data),
                    caption=f"[{screenshot.get('timestamp', '')[-8:]}] {screenshot.get('url', 'N/A')}",
                    use_column_width=True
                )
            except Exception as e:
                st.warning(f"Screenshot display error: {e}")