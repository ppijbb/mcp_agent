import streamlit as st
import sys
import json
from datetime import datetime
from pathlib import Path
import pydeck as pdk
import pandas as pd

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
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
        subtitle="자연어 임무를 입력하여 자율 드론 정찰을 시작합니다.",
        module_path="srcs.drone_scout.run_drone_scout"
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

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.drone_scout.run_drone_scout",
                "--mission", mission_text,
                "--result-json-path", str(result_json_path)
            ]

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/drone_scout"
            )

            if result and "data" in result:
                display_results(result["data"])
            elif result and "error" in result:
                st.error(result["error"])

if __name__ == "__main__":
    main() 