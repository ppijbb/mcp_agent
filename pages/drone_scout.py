import streamlit as st
import sys
import json
from datetime import datetime
from pathlib import Path
import pydeck as pdk
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()
from configs.settings import get_reports_path

def display_results(result_data):
    """결과 데이터를 기반으로 UI를 렌더링합니다."""
    st.markdown("---")
    st.subheader("🛰️ Mission Results")

    if not result_data or not result_data.get("success"):
        st.warning("No valid mission data found in results.")
        if result_data and result_data.get("error"):
            st.error(f"Agent Error: {result_data.get('error')}")
        return

    summary = result_data.get("summary", {})
    trajectory = result_data.get("trajectory", [])

    st.markdown(f"**Mission:** `{summary.get('mission', 'N/A')}`")
    st.metric("Mission Status", summary.get("status", "UNKNOWN"))

    if trajectory:
        st.success("Flight trajectory data found! Visualizing on map...")
        
        try:
            # PyDeck을 사용하여 3D 비행 경로 시각화
            df = pd.DataFrame(trajectory, columns=['lon', 'lat', 'alt'])
            
            # 시작점과 끝점 추가
            start_point = df.iloc[[0]]
            end_point = df.iloc[[-1]]

            view_state = pdk.ViewState(
                latitude=df['lat'].mean(),
                longitude=df['lon'].mean(),
                zoom=15,
                pitch=50,
            )

            path_layer = pdk.Layer(
                'PathLayer',
                data=df,
                get_path='[lon, lat, alt]',
                get_width=5,
                get_color=[255, 0, 0, 255],  # Red
                pickable=True
            )
            
            start_layer = pdk.Layer(
                'ScatterplotLayer',
                data=start_point,
                get_position='[lon, lat, alt]',
                get_color='[0, 255, 0, 255]', # Green
                get_radius=20,
                pickable=True
            )
            
            end_layer = pdk.Layer(
                'ScatterplotLayer',
                data=end_point,
                get_position='[lon, lat, alt]',
                get_color='[0, 0, 255, 255]', # Blue
                get_radius=20,
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/satellite-streets-v11',
                initial_view_state=view_state,
                layers=[path_layer, start_layer, end_layer],
                tooltip={"text": "Altitude: {alt}m"}
            ))
        except Exception as e:
            st.error(f"Failed to render map visualization: {e}")
            st.info("This may be due to a missing Mapbox API key. Please check your Streamlit secrets.")
            st.dataframe(trajectory) # fallback to showing data
            
    else:
        st.warning("No trajectory data available for visualization.")

    with st.expander("Full Mission Log"):
        st.json(result_data)


def main():
    """드론 스카우트 에이전트 페이지 메인 함수"""
    create_agent_page(
        agent_name="Drone Scout Agent",
        page_icon="🛸",
        page_type="drone",
        title="Drone Scout Agent",
        subtitle="자연어 임무를 입력하여 자율 드론 정찰을 시작합니다."
    )

    result_placeholder = st.empty()

    with st.form("drone_scout_form"):
        st.subheader("📝 Mission Briefing")
        mission_text = st.text_area(
            "Enter mission details in natural language (Korean or English)",
            placeholder="예: 서울숲 공원 상공을 비행하며 주요 시설물의 현재 상태를 촬영하고 보고서를 작성해줘. 비행 고도는 50m로 유지해.",
            height=150
        )
        submitted = st.form_submit_button("🚀 Launch Mission", use_container_width=True)

    if submitted:
        if not mission_text:
            st.warning("Please enter mission details.")
        else:
            reports_path = Path(get_reports_path('drone_scout'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"drone_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            agent_metadata = {
                "agent_id": "drone_scout_agent",
                "agent_name": "Drone Scout Agent",
                "entry_point": "srcs.drone_scout.run_drone_scout",
                "agent_type": "mcp_agent",
                "capabilities": ["drone_mission", "aerial_survey", "autonomous_flight"],
                "description": "자연어 임무를 입력하여 자율 드론 정찰"
            }

            input_data = {
                "mission": mission_text,
                "result_json_path": str(result_json_path)
            }

            result = run_agent_via_a2a(
                placeholder=result_placeholder,
                agent_metadata=agent_metadata,
                input_data=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])
            elif result and "error" in result:
                st.error(result["error"])

    # 최신 Drone Scout 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Drone Scout 결과")
    
    latest_drone_result = result_reader.get_latest_result("drone_scout_agent", "mission_execution")
    
    if latest_drone_result:
        with st.expander("🛸 최신 드론 미션 결과", expanded=False):
            st.subheader("🤖 최근 드론 미션 실행 결과")
            
            if isinstance(latest_drone_result, dict):
                # 미션 정보 표시
                mission_text = latest_drone_result.get('mission_text', 'N/A')
                mission_status = latest_drone_result.get('mission_status', 'N/A')
                
                st.success(f"**미션 상태: {mission_status}**")
                st.info(f"**미션 내용: {mission_text}**")
                
                # 미션 결과 요약
                col1, col2, col3 = st.columns(3)
                col1.metric("비행 시간", f"{latest_drone_result.get('flight_duration', 0)}분")
                col2.metric("총 거리", f"{latest_drone_result.get('total_distance', 0):.1f}km")
                col3.metric("최고 고도", f"{latest_drone_result.get('max_altitude', 0)}m")
                
                # 궤적 데이터 표시
                trajectory = latest_drone_result.get('trajectory', [])
                if trajectory:
                    st.subheader("🗺️ 비행 궤적")
                    try:
                        df = pd.DataFrame(trajectory)
                        st.dataframe(df, use_container_width=True)
                        
                        # 지도 시각화 (간단한 버전)
                        if 'lat' in df.columns and 'lon' in df.columns:
                            st.map(df[['lat', 'lon']])
                    except Exception as e:
                        st.warning(f"궤적 데이터 시각화 실패: {e}")
                
                # 미션 로그 표시
                mission_log = latest_drone_result.get('mission_log', [])
                if mission_log:
                    st.subheader("📋 미션 로그")
                    with st.expander("상세 미션 로그", expanded=False):
                        for log_entry in mission_log:
                            st.write(f"• {log_entry}")
                
                # 메타데이터 표시
                if 'timestamp' in latest_drone_result:
                    st.caption(f"⏰ 미션 시간: {latest_drone_result['timestamp']}")
            else:
                st.json(latest_drone_result)
    else:
        st.info("💡 아직 Drone Scout Agent의 결과가 없습니다. 위에서 드론 미션을 실행해보세요.")

if __name__ == "__main__":
    main() 