"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import json
import streamlit_process_manager as spm


# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('business_strategy')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button, create_agent_page
from srcs.common.streamlit_a2a_runner import run_agent_via_a2a
from srcs.business_strategy_agents.run_business_strategy_agents import BusinessStrategyRunner

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 분석 결과 요약")
    
    if not result_data or "summary" not in result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    summary = result_data.get("summary", {})
    results = result_data.get("results", {})

    st.metric("총 실행 시간", f"{summary.get('execution_time', 0):.2f}초")
    
    st.markdown("#### 📄 생성된 보고서 목록 및 내용")
    for agent_name, result in results.items():
        if result.get("success") and "output_file" in result:
            file_path = result['output_file']
            agent_title = agent_name.replace('_', ' ').title()
            
            with st.expander(f"📄 {agent_title} 보고서 보기", expanded=(agent_name == 'unified_strategy')):
                st.success(f"**보고서 위치**: `{file_path}`")
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        st.markdown(report_content)
                    else:
                        st.warning(f"보고서 파일({file_path})을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"보고서 파일을 읽는 중 오류 발생: {e}")
        else:
            agent_title = agent_name.replace('_', ' ').title()
            st.error(f"**{agent_title}**: 실패 - {result.get('error', '알 수 없는 오류')}")

def main():
    create_agent_page(
        agent_name="Business Strategy Agent",
        page_icon="🎯",
        page_type="business",
        title="Business Strategy Agent",
        subtitle="AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼",
        module_path="srcs.business_strategy_agents.run_business_strategy_agents"
    )

    result_placeholder = st.empty()

    with st.form("business_strategy_form"):
        st.subheader("📝 분석 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            keywords_input = st.text_input("🔍 핵심 키워드 (쉼표로 구분)", "AI, fintech, sustainability")
            business_context_input = st.text_area("🏢 비즈니스 맥락", "AI 스타트업, 핀테크 회사 등")
        with col2:
            objectives_input = st.text_input("🎯 목표 (쉼표로 구분)", "growth, expansion, efficiency")
            regions_input = st.text_input("🌍 타겟 지역 (쉼표로 구분)", "North America, Europe, Asia")

        st.subheader("⚙️ 고급 설정")
        col3, col4 = st.columns(2)
        with col3:
            time_horizon = st.selectbox("⏰ 분석 기간", ["3_months", "6_months", "12_months", "24_months"], index=2)
        with col4:
            analysis_mode = st.selectbox("🔄 분석 모드", ["unified", "individual", "both"], index=0)
        
        submitted = st.form_submit_button("🚀 비즈니스 전략 분석 시작", use_container_width=True)

    if submitted:
        if not keywords_input.strip():
            st.warning("핵심 키워드를 입력해주세요.")
        else:
            reports_path = Path(get_reports_path('business_strategy'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # runner.run_full_suite에 맞는 config 객체 생성
            config = {
                'keywords': [k.strip() for k in keywords_input.split(',')],
                'business_context': {"description": business_context_input} if business_context_input.strip() else None,
                'objectives': [o.strip() for o in objectives_input.split(',')] if objectives_input.strip() else None,
                'regions': [r.strip() for r in regions_input.split(',')] if regions_input.strip() else None,
                'time_horizon': time_horizon,
                'mode': analysis_mode
            }

            from srcs.common.agent_interface import AgentType
            
            agent_metadata = {
                "agent_id": "business_strategy_agent",
                "agent_name": "Business Strategy Agent",
                "entry_point": "srcs.business_strategy_agents.run_business_strategy_agents",
                "agent_type": AgentType.MCP_AGENT,
                "capabilities": ["market_analysis", "competitive_analysis", "strategy_planning"],
                "description": "시장, 경쟁사 분석 및 비즈니스 모델 설계"
            }

            input_data = {
                "module_path": "srcs.business_strategy_agents.run_business_strategy_agents",
                "class_name": "BusinessStrategyRunner",
                "method_name": "run_agents",
                "industry": config.get("keywords", [""])[0] if config.get("keywords") else "General",
                "company_profile": config.get("business_context", {}).get("description", "Business analysis") if config.get("business_context") else "Business analysis",
                "competitors": config.get("keywords", [])[1:] if len(config.get("keywords", [])) > 1 else [],
                "tech_trends": config.get("keywords", []),
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

    # 최신 Business Strategy Agent 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Business Strategy Agent 결과")
    
    latest_strategy_result = result_reader.get_latest_result("business_strategy_agent", "strategy_analysis")
    
    if latest_strategy_result:
        with st.expander("🎯 최신 비즈니스 전략 분석 결과", expanded=False):
            st.subheader("🤖 최근 비즈니스 전략 분석 결과")
            
            if isinstance(latest_strategy_result, dict):
                # 전략 정보 표시
                keywords = latest_strategy_result.get('keywords', [])
                time_horizon = latest_strategy_result.get('time_horizon', 'N/A')
                
                st.success(f"**핵심 키워드: {', '.join(keywords)}**")
                st.info(f"**분석 기간: {time_horizon}**")
                
                # 분석 결과 요약
                col1, col2, col3 = st.columns(3)
                col1.metric("실행 시간", f"{latest_strategy_result.get('execution_time', 0):.2f}초")
                col2.metric("생성된 보고서", len(latest_strategy_result.get('results', {})))
                col3.metric("분석 상태", "완료" if latest_strategy_result.get('success', False) else "실패")
                
                # 생성된 보고서 표시
                results = latest_strategy_result.get('results', {})
                if results:
                    st.subheader("📄 생성된 보고서")
                    for agent_name, result in results.items():
                        if result.get('success'):
                            agent_title = agent_name.replace('_', ' ').title()
                            st.write(f"✅ **{agent_title}**: 성공")
                            
                            # 보고서 내용 표시
                            if 'output_file' in result:
                                file_path = result['output_file']
                                st.info(f"파일 위치: {file_path}")
                        else:
                            agent_title = agent_name.replace('_', ' ').title()
                            st.write(f"❌ **{agent_title}**: 실패 - {result.get('error', '알 수 없는 오류')}")
                
                # 메타데이터 표시
                if 'timestamp' in latest_strategy_result:
                    st.caption(f"⏰ 분석 시간: {latest_strategy_result['timestamp']}")
            else:
                st.json(latest_strategy_result)
    else:
        st.info("💡 아직 Business Strategy Agent의 결과가 없습니다. 위에서 비즈니스 전략 분석을 실행해보세요.")

if __name__ == "__main__":
    main() 