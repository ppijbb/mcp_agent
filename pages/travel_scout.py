#!/usr/bin/env python3
"""
Travel Scout Agent - Streamlit Page

A comprehensive travel search interface using MCP Browser for incognito browsing
and real-time travel data collection without price manipulation.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srcs.travel_scout.travel_scout_agent import TravelScoutAgent

# Configure page
st.set_page_config(
    page_title="Travel Scout Agent - MCP Browser",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = TravelScoutAgent()
    st.session_state.search_results = None
    st.session_state.search_history = []

async def main():
    """Main Streamlit application"""
    
    # Header with MCP status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🧳 Travel Scout Agent")
        st.markdown("**MCP Browser를 통한 실시간 여행 검색**")
    
    with col2:
        # Get MCP status
        mcp_status = st.session_state.agent.get_mcp_status()
        status_color = '🟢' if mcp_status.get('browser_connected') else '🔴'
        
        st.metric(
            label="MCP 상태",
            value=f"{status_color} {mcp_status['status'].upper()}"
        )
    
    # MCP Status details
    with st.expander("MCP Browser 연결 상태 정보", expanded=True):
        st.info(mcp_status.get('description', '상태 정보 없음'))
        
        if not mcp_status.get('browser_connected'):
            if st.button("🔄 MCP 브라우저 연결"):
                with st.spinner("MCP Browser Use 서버에 연결 중..."):
                    connected = await st.session_state.agent.initialize_mcp()
                    if connected:
                        st.success("MCP 연결 성공!")
                    else:
                        st.error("MCP 연결 실패. 서버 로그를 확인하세요.")
                    st.rerun()
        else:
            st.success("MCP 브라우저가 성공적으로 연결되었습니다.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 검색 설정")
        
        # Search parameters
        st.subheader("📍 여행 정보")
        destination = st.text_input("목적지", value="Tokyo", help="검색할 도시명을 입력하세요")
        origin = st.text_input("출발지", value="Seoul", help="출발 도시명을 입력하세요")
        
        st.subheader("📅 날짜 설정")
        col1, col2 = st.columns(2)
        with col1:
            departure_date = st.date_input(
                "출발일",
                value=datetime.now().date() + timedelta(days=7),
                min_value=datetime.now().date()
            )
            check_in = st.date_input(
                "체크인",
                value=datetime.now().date() + timedelta(days=7),
                min_value=datetime.now().date()
            )
        
        with col2:
            return_date = st.date_input(
                "귀국일",
                value=datetime.now().date() + timedelta(days=14),
                min_value=departure_date + timedelta(days=1)
            )
            check_out = st.date_input(
                "체크아웃",
                value=datetime.now().date() + timedelta(days=10),
                min_value=check_in + timedelta(days=1)
            )
        
        # Quality criteria
        st.subheader("⚙️ 검색 기준")
        with st.expander("품질 기준 설정"):
            min_hotel_rating = st.slider("최소 호텔 평점", 3.0, 5.0, 4.0, 0.1)
            max_hotel_price = st.slider("최대 호텔 가격 (USD/박)", 50, 1000, 500, 50)
            max_flight_price = st.slider("최대 항공료 (USD)", 200, 5000, 2000, 100)
            
            st.session_state.agent.update_quality_criteria({
                'min_hotel_rating': min_hotel_rating,
                'max_hotel_price': max_hotel_price,
                'max_flight_price': max_flight_price
            })
        
        # Search button
        search_button = st.button(
            "🔍 실시간 검색 시작", 
            type="primary",
            use_container_width=True,
            help="MCP Browser를 통해 실시간으로 여행 정보를 검색합니다.",
            disabled=not st.session_state.agent.get_mcp_status().get('browser_connected')
        )
    
    # Main content area
    if search_button:
        if not destination or not origin:
            st.error("목적지와 출발지를 모두 입력해주세요.")
            return
        
        # Prepare search parameters
        search_params = {
            'destination': destination,
            'origin': origin,
            'departure_date': departure_date.strftime('%Y-%m-%d'),
            'return_date': return_date.strftime('%Y-%m-%d'),
            'check_in': check_in.strftime('%Y-%m-%d'),
            'check_out': check_out.strftime('%Y-%m-%d')
        }
        
        # Show search progress
        progress_container = st.container()
        with progress_container:
            st.info("🔍 MCP Browser Use로 검색 중...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            for i in range(5):
                progress_bar.progress((i + 1) * 20)
                status_text.text(f"검색 진행 중... {(i + 1) * 20}%")
                await asyncio.sleep(0.5)
        
        # Perform search
        try:
            with st.spinner("MCP Browser로 여행 정보를 검색하고 있습니다..."):
                search_results = await st.session_state.agent.search_travel_options(search_params)
                st.session_state.search_results = search_results
                st.session_state.search_history.append(search_results)
            
            progress_container.empty()
            
            if search_results.get('status') == 'completed':
                st.success(f"✅ 검색 완료! {len(search_results.get('hotels', []))}개 호텔, {len(search_results.get('flights', []))}개 항공편 발견")
            else:
                st.error(f"❌ 검색 실패: {search_results.get('error', 'Unknown error')}")
        
        except Exception as e:
            progress_container.empty()
            st.error(f"검색 중 오류 발생: {str(e)}")
    
    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        # Search metadata
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "검색 시간",
                f"{results['performance']['total_duration']:.1f}초"
            )
        
        with col2:
            st.metric(
                "발견된 호텔",
                results['performance']['hotels_found']
            )
        
        with col3:
            st.metric(
                "발견된 항공편",
                results['performance']['flights_found']
            )
        
        with col4:
            mcp_connected = results.get('mcp_info', {}).get('browser_connected', False)
            st.metric(
                "데이터 소스",
                "MCP 실시간" if mcp_connected else "연결 끊김"
            )
        
        # Data source breakdown
        if 'analysis' in results and 'data_sources' in results['analysis']:
            st.markdown("### 📊 데이터 소스 분석")
            data_sources = results['analysis']['data_sources']
            
            source_col1, source_col2 = st.columns(2)
            with source_col1:
                st.metric(
                    "실시간 데이터 비율", 
                    f"{data_sources.get('real_time_percentage', 0):.1f}%"
                )
            
            with source_col2:
                total_mcp = data_sources.get('mcp_hotels', 0) + data_sources.get('mcp_flights', 0)
                st.metric(
                    "MCP 검색 결과 수",
                    f"{total_mcp} 건"
                )
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏨 호텔", "✈️ 항공편", "💡 추천", "📈 분석"])
        
        with tab1:
            hotels = results.get('hotels', [])
            if hotels:
                st.markdown("### 검색된 호텔 목록")
                
                hotel_data = []
                for hotel in hotels:
                    hotel_data.append({
                        "호텔명": hotel['name'],
                        "가격": hotel['price'],
                        "평점": hotel['rating'],
                        "위치": hotel['location'],
                        "플랫폼": hotel['platform'],
                        "품질등급": hotel.get('quality_tier', 'N/A')
                    })
                
                df_hotels = pd.DataFrame(hotel_data)
                st.dataframe(df_hotels, use_container_width=True, hide_index=True)
                
                # Hotel price distribution
                hotel_prices = [h.get('price_numeric', 0) for h in hotels if h.get('price_numeric', 0) != float('inf')]
                if hotel_prices:
                    fig_hotels = px.histogram(
                        x=hotel_prices,
                        title="호텔 가격 분포",
                        labels={'x': '가격 (USD/박)', 'y': '호텔 수'}
                    )
                    st.plotly_chart(fig_hotels, use_container_width=True)
            else:
                st.info("검색된 호텔이 없습니다.")
        
        with tab2:
            flights = results.get('flights', [])
            if flights:
                st.markdown("### 검색된 항공편 목록")
                
                flight_data = []
                for flight in flights:
                    flight_data.append({
                        "항공사": flight['airline'],
                        "가격": flight['price'],
                        "소요시간": flight['duration'],
                        "출발시간": flight['departure_time'],
                        "플랫폼": flight['platform'],
                        "품질등급": flight.get('quality_tier', 'N/A')
                    })
                
                df_flights = pd.DataFrame(flight_data)
                st.dataframe(df_flights, use_container_width=True, hide_index=True)
                
                # Flight price distribution
                flight_prices = [f.get('price_numeric', 0) for f in flights if f.get('price_numeric', 0) != float('inf')]
                if flight_prices:
                    fig_flights = px.histogram(
                        x=flight_prices,
                        title="항공편 가격 분포",
                        labels={'x': '가격 (USD)', 'y': '항공편 수'}
                    )
                    st.plotly_chart(fig_flights, use_container_width=True)
            else:
                st.info("검색된 항공편이 없습니다.")
        
        with tab3:
            recommendations = results.get('recommendations', {})
            if recommendations:
                st.markdown("### 💡 추천 사항")
                
                # Best options
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'best_hotel' in recommendations:
                        hotel = recommendations['best_hotel']
                        st.success(f"""
                        **추천 호텔**
                        - {hotel['name']}
                        - 가격: {hotel['price']}
                        - 평점: {hotel['rating']}
                        - 위치: {hotel['location']}
                        """)
                
                with col2:
                    if 'best_flight' in recommendations:
                        flight = recommendations['best_flight']
                        st.success(f"""
                        **추천 항공편**
                        - {flight['airline']}
                        - 가격: {flight['price']}
                        - 소요시간: {flight['duration']}
                        - 출발: {flight['departure_time']}
                        """)
                
                # Booking strategy
                if 'booking_strategy' in recommendations:
                    st.markdown("### 📋 예약 전략")
                    for strategy in recommendations['booking_strategy']:
                        st.write(f"• {strategy}")
                
                # Total cost estimate
                if 'total_trip_cost_estimate' in recommendations:
                    cost = recommendations['total_trip_cost_estimate']
                    st.markdown("### 💰 예상 총 비용")
                    st.info(f"""
                    - 호텔: ${cost['hotel_per_night']}/박 × {cost['nights']}박 = ${cost['hotel_total']}
                    - 항공료: ${cost['flight_total']}
                    - **총합: ${cost['grand_total']}**
                    """)
            else:
                st.info("추천 정보가 없습니다.")
        
        with tab4:
            analysis = results.get('analysis', {})
            if analysis:
                st.markdown("### 📈 검색 분석")
                
                # Analysis metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'hotel_analysis' in analysis:
                        hotel_analysis = analysis['hotel_analysis']
                        st.markdown("#### 호텔 분석")
                        st.metric("평균 평점", f"{hotel_analysis.get('average_rating', 0):.1f}/5.0")
                        st.metric("평균 가격", f"${hotel_analysis.get('price_range', {}).get('average', 0):.0f}/박")
                        st.metric("품질 기준 충족", f"{hotel_analysis.get('quality_hotels_count', 0)}개")
                
                with col2:
                    if 'flight_analysis' in analysis:
                        flight_analysis = analysis['flight_analysis']
                        st.markdown("#### 항공편 분석")
                        st.metric("평균 가격", f"${flight_analysis.get('price_range', {}).get('average', 0):.0f}")
                        st.metric("항공사 수", f"{len(flight_analysis.get('airlines_found', []))}개")
                        st.metric("품질 기준 충족", f"{flight_analysis.get('quality_flights_count', 0)}개")
                
                # Data source analysis
                if 'data_sources' in analysis:
                    data_sources = analysis['data_sources']
                    st.markdown("#### 데이터 소스 분석")
                    
                    total_items = data_sources.get('mcp_hotels', 0) + data_sources.get('mcp_flights', 0)
                    if total_items > 0:
                        st.info(f"모든 데이터 ({total_items}건)는 MCP Browser를 통해 실시간으로 수집되었습니다.")
            else:
                st.info("분석 정보가 없습니다.")
    
    # Search history in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 검색 통계")
        
        stats = st.session_state.agent.get_search_stats()
        if stats:
            st.metric("총 검색 횟수", stats.get('total_searches', 0))
            st.metric("성공률", f"{stats.get('success_rate', 0):.1f}%")
            st.metric("MCP 사용률", f"{stats.get('real_time_data_percentage', 0):.1f}%")
            st.metric("평균 검색 시간", f"{stats.get('average_search_duration', 0):.1f}초")
        
        if st.session_state.search_history:
            st.markdown("### 📜 최근 검색")
            for i, search in enumerate(reversed(st.session_state.search_history[-3:])):
                with st.expander(f"검색 {len(st.session_state.search_history) - i}"):
                    mcp_status = search.get('mcp_info', {}).get('status', 'unknown')
                    mcp_icon = "🟢" if mcp_status == 'connected' else "🔴"
                    st.write(f"{mcp_icon} {search.get('search_params', {}).get('destination', 'Unknown')}")
                    st.write(f"호텔: {search.get('performance', {}).get('hotels_found', 0)}개")
                    st.write(f"항공편: {search.get('performance', {}).get('flights_found', 0)}개")
                    st.write(f"시간: {search.get('performance', {}).get('total_duration', 0):.1f}초")


# Run the Streamlit app
if __name__ == "__main__":
    asyncio.run(main()) 