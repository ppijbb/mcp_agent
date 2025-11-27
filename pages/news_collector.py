"""
📰 News Collector Agent Page

뉴스 수집 및 정리 AI
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.page_utils import create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def main():
    create_agent_page(
        agent_name="News Collector Agent",
        page_icon="📰",
        page_type="news",
        title="News Collector Agent",
        subtitle="MCP를 사용하여 국내뉴스와 국제뉴스를 수집하고 정리합니다.",
        module_path="srcs.basic_agents.news_collector_agent"
    )

    result_placeholder = st.empty()

    with st.form("news_collector_form"):
        st.subheader("📝 뉴스 수집 설정")
        
        target_date = st.date_input(
            "수집할 날짜",
            value=datetime.now().date(),
            help="수집할 뉴스의 날짜를 선택하세요"
        )
        
        news_types = st.multiselect(
            "수집할 뉴스 유형",
            options=["domestic", "international", "both"],
            default=["both"],
            help="국내뉴스, 국제뉴스, 또는 둘 다"
        )
        
        submitted = st.form_submit_button("🚀 뉴스 수집 시작", width='stretch')

    if submitted:
        reports_path = Path(get_reports_path('news_collector'))
        reports_path.mkdir(parents=True, exist_ok=True)
        result_json_path = reports_path / f"news_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        agent_metadata = {
            "agent_id": "news_collector_agent",
            "agent_name": "News Collector Agent",
            "entry_point": "srcs.common.generic_agent_runner",
            "agent_type": "mcp_agent",
            "capabilities": ["news_collection", "domestic_news", "international_news"],
            "description": "MCP를 사용하여 국내뉴스와 국제뉴스를 수집하고 정리"
        }

        input_data = {
            "module_path": "srcs.basic_agents.news_collector_agent",
            "class_name": "NewsCollectorAgent",
            "method_name": "collect_news",
            "config": {
                "target_date": target_date.strftime("%Y-%m-%d"),
                "news_types": news_types
            },
            "result_json_path": str(result_json_path)
        }

        result = run_agent_via_a2a(
            placeholder=result_placeholder,
            agent_metadata=agent_metadata,
            input_data=input_data,
            result_json_path=result_json_path,
            use_a2a=True
        )

        if result and result.get("success") and result.get("data"):
            display_results(result["data"])

    # 최신 News Collector 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 News Collector 결과")
    
    latest_news_result = result_reader.get_latest_result("news_collector_agent", "news_collection")
    
    if latest_news_result:
        with st.expander("📰 최신 뉴스 수집 결과", expanded=False):
            st.subheader("🤖 최근 뉴스 수집 결과")
            
            if isinstance(latest_news_result, dict):
                date = latest_news_result.get('date', 'N/A')
                st.success(f"**수집 날짜: {date}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("국내뉴스", len(latest_news_result.get('domestic_news', [])))
                with col2:
                    st.metric("국제뉴스", len(latest_news_result.get('international_news', [])))
                
                if latest_news_result.get('domestic_news'):
                    st.subheader("📰 국내뉴스")
                    for news in latest_news_result['domestic_news'][:5]:
                        st.write(f"• {news.get('title', 'N/A')}")
                
                if latest_news_result.get('international_news'):
                    st.subheader("🌍 국제뉴스")
                    for news in latest_news_result['international_news'][:5]:
                        st.write(f"• {news.get('title', 'N/A')}")
            else:
                st.json(latest_news_result)
    else:
        st.info("💡 아직 News Collector Agent의 결과가 없습니다. 위에서 뉴스 수집을 실행해보세요.")

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 뉴스 수집 결과")
    
    if not result_data:
        st.warning("수집 결과를 찾을 수 없습니다.")
        return
    
    date = result_data.get('date', 'N/A')
    st.success(f"**수집 날짜: {date}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("국내뉴스", len(result_data.get('domestic_news', [])))
    with col2:
        st.metric("국제뉴스", len(result_data.get('international_news', [])))
    
    if result_data.get('domestic_news'):
        st.subheader("📰 국내뉴스")
        for news in result_data['domestic_news']:
            with st.expander(news.get('title', 'N/A')):
                st.write(f"**출처**: {news.get('source', 'N/A')}")
                st.write(f"**내용**: {news.get('content', 'N/A')}")
    
    if result_data.get('international_news'):
        st.subheader("🌍 국제뉴스")
        for news in result_data['international_news']:
            with st.expander(news.get('title', 'N/A')):
                st.write(f"**출처**: {news.get('source', 'N/A')}")
                st.write(f"**내용**: {news.get('content', 'N/A')}")

if __name__ == "__main__":
    main()

