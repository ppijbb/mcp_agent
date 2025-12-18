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
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

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
    
    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    # run_agents가 반환하는 구조 처리
    # 구조 1: summary와 results가 있는 경우 (기존 구조)
    if "summary" in result_data:
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
    
    # 구조 2: run_agents가 반환하는 직접 구조 (data_scout_output, trend_analyzer_output)
    elif "data_scout_output" in result_data or "trend_analyzer_output" in result_data:
        st.success("✅ Business Strategy 분석이 완료되었습니다!")
        
        # Business Data Scout 결과
        if "data_scout_output" in result_data:
            scout_result = result_data["data_scout_output"]
            if scout_result.get("success"):
                report_path = scout_result.get("report_path")
                report_data = scout_result.get("data", "")
                
                with st.expander("📊 Business Data Scout 보고서", expanded=True):
                    st.success(f"✅ Business Data Scout 완료")
                    if report_path:
                        st.info(f"**보고서 위치**: `{report_path}`")
                        # 파일이 존재하면 파일에서 읽기, 없으면 data 필드 사용
                        if os.path.exists(report_path):
                            try:
                                with open(report_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                st.markdown(file_content)
                            except Exception as e:
                                st.warning(f"파일 읽기 실패, 저장된 데이터 표시: {e}")
                                if report_data:
                                    st.markdown(report_data)
                        elif report_data:
                            st.markdown(report_data)
                        else:
                            st.warning("보고서 내용을 찾을 수 없습니다.")
        
        # Trend Analyzer 결과
        if "trend_analyzer_output" in result_data:
            trend_result = result_data["trend_analyzer_output"]
            if trend_result.get("success"):
                report_path = trend_result.get("report_path")
                report_data = trend_result.get("data", "")
                
                with st.expander("📈 Trend Analyzer 보고서", expanded=True):
                    st.success(f"✅ Trend Analyzer 완료")
                    if report_path:
                        st.info(f"**보고서 위치**: `{report_path}`")
                        # 파일이 존재하면 파일에서 읽기, 없으면 data 필드 사용
                        if os.path.exists(report_path):
                            try:
                                with open(report_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                # 마크다운 코드 블록 제거 (이미 마크다운 형식인 경우)
                                if file_content.startswith("```markdown"):
                                    file_content = file_content.replace("```markdown", "").replace("```", "").strip()
                                st.markdown(file_content)
                            except Exception as e:
                                st.warning(f"파일 읽기 실패, 저장된 데이터 표시: {e}")
                                if report_data:
                                    # 마크다운 코드 블록 제거
                                    if report_data.startswith("```markdown"):
                                        report_data = report_data.replace("```markdown", "").replace("```", "").strip()
                                    st.markdown(report_data)
                        elif report_data:
                            # 마크다운 코드 블록 제거
                            if report_data.startswith("```markdown"):
                                report_data = report_data.replace("```markdown", "").replace("```", "").strip()
                            st.markdown(report_data)
                        else:
                            st.warning("보고서 내용을 찾을 수 없습니다.")
        
        # 최종 요약 JSON 파일은 표시하지 않음 (실제 결과만 표시)
    
    # 구조 3: 알 수 없는 구조
    else:
        st.warning("분석 결과 구조를 인식할 수 없습니다.")
        # JSON 출력 제거 - 실제 결과만 표시
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if key not in ["success", "error"]:
                    st.write(f"**{key}**: {value}")

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

            # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="business_strategy_agent",
                agent_name="Business Strategy Agent",
                entry_point="srcs.business_strategy_agents.run_business_strategy_agents",
                agent_type=AgentType.MCP_AGENT,
                capabilities=["market_analysis", "competitive_analysis", "strategy_planning"],
                description="시장, 경쟁사 분석 및 비즈니스 모델 설계",
                input_params={
                    "industry": config.get("keywords", [""])[0] if config.get("keywords") else "General",
                    "company_profile": config.get("business_context", {}).get("description", "Business analysis") if config.get("business_context") else "Business analysis",
                    "competitors": config.get("keywords", [])[1:] if len(config.get("keywords", [])) > 1 else [],
                    "tech_trends": config.get("keywords", []),
                    "result_json_path": str(result_json_path)
                },
                class_name="BusinessStrategyRunner",
                method_name="run_agents",
                result_json_path=result_json_path,
                use_a2a=True
            )

            if result and "data" in result:
                display_results(result["data"])

    # 최신 Business Strategy Agent 결과 확인
    st.markdown("---")
    st.markdown("## 📊 최신 Business Strategy Agent 결과")
    
    # 최신 리포트 파일 직접 찾기 (REPORTS_PATH 사용)
    reports_dir = Path(REPORTS_PATH)
    latest_json = None
    latest_time = None
    
    if reports_dir.exists():
        # strategy_report_*.json 패턴으로 검색
        for json_file in reports_dir.glob("strategy_report_*.json"):
            file_time = json_file.stat().st_mtime
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_json = json_file
    
    # JSON 파일에서 결과 읽기
    latest_strategy_result = None
    if latest_json:
        try:
            with open(latest_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # BusinessStrategyRunner 구조: {"results": {...}}
                # 또는 직접 결과 구조인 경우 처리
                if "results" in json_data:
                    latest_strategy_result = json_data["results"]
                else:
                    latest_strategy_result = json_data
        except Exception as e:
            st.warning(f"최신 결과 파일 읽기 실패: {e}")
    
    # result_reader도 시도
    if not latest_strategy_result:
        latest_strategy_result = result_reader.get_latest_result("business_strategy_agent", "strategy_analysis")
    
    if latest_strategy_result:
        with st.expander("🎯 최신 비즈니스 전략 분석 결과", expanded=False):
            st.subheader("🤖 최근 비즈니스 전략 분석 결과")
            
            if isinstance(latest_strategy_result, dict):
                # run_agents가 반환하는 구조 처리
                # 키워드 구성 (industry + competitors + tech_trends)
                industry = latest_strategy_result.get('industry', '')
                competitors = latest_strategy_result.get('competitors', [])
                tech_trends = latest_strategy_result.get('tech_trends', [])
                keywords = [industry] + competitors + tech_trends
                keywords = [k for k in keywords if k]  # 빈 값 제거
                keywords = list(set(keywords))  # 중복 제거
                
                company_profile = latest_strategy_result.get('company_profile', 'N/A')
                
                if keywords:
                    st.success(f"**핵심 키워드:** {', '.join(keywords)}")
                else:
                    st.info("**핵심 키워드:** 정보 없음")
                
                st.info(f"**비즈니스 맥락:** {company_profile}")
                
                # 분석 결과 요약
                col1, col2, col3 = st.columns(3)
                
                # 성공한 리포트 개수 계산
                success_count = 0
                if latest_strategy_result.get('data_scout_output', {}).get('success'):
                    success_count += 1
                if latest_strategy_result.get('trend_analyzer_output', {}).get('success'):
                    success_count += 1
                
                col1.metric("생성된 보고서", f"{success_count}/2")
                col2.metric("분석 상태", "완료" if success_count == 2 else "부분 완료")
                
                # 실행 시간은 JSON 파일의 execution_timestamp에서 계산
                if latest_json:
                    try:
                        with open(latest_json, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            timestamp_str = json_data.get('execution_timestamp', '')
                            if timestamp_str:
                                exec_time = datetime.fromisoformat(timestamp_str)
                                time_str = exec_time.strftime('%Y-%m-%d %H:%M:%S')
                                col3.metric("실행 시간", time_str)
                            else:
                                col3.metric("실행 시간", "N/A")
                    except:
                        col3.metric("실행 시간", "N/A")
                else:
                    col3.metric("실행 시간", "N/A")
                
                # 생성된 보고서 표시
                st.subheader("📄 생성된 보고서")
                
                # Business Data Scout 보고서
                if 'data_scout_output' in latest_strategy_result:
                    scout_result = latest_strategy_result['data_scout_output']
                    if scout_result.get('success'):
                        with st.expander("📊 Business Data Scout 보고서", expanded=False):
                            st.success("✅ Business Data Scout 완료")
                            report_path = scout_result.get('report_path')
                            if report_path:
                                st.info(f"**보고서 위치**: `{report_path}`")
                                if os.path.exists(report_path):
                                    try:
                                        with open(report_path, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        st.markdown(content)
                                    except Exception as e:
                                        st.warning(f"파일 읽기 실패: {e}")
                                        if scout_result.get('data'):
                                            st.markdown(scout_result['data'])
                    else:
                        st.error(f"❌ **Business Data Scout**: 실패 - {scout_result.get('error', '알 수 없는 오류')}")
                
                # Trend Analyzer 보고서
                if 'trend_analyzer_output' in latest_strategy_result:
                    trend_result = latest_strategy_result['trend_analyzer_output']
                    if trend_result.get('success'):
                        with st.expander("📈 Trend Analyzer 보고서", expanded=False):
                            st.success("✅ Trend Analyzer 완료")
                            report_path = trend_result.get('report_path')
                            if report_path:
                                st.info(f"**보고서 위치**: `{report_path}`")
                                if os.path.exists(report_path):
                                    try:
                                        with open(report_path, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        # 마크다운 코드 블록 제거
                                        if content.startswith("```markdown"):
                                            content = content.replace("```markdown", "").replace("```", "").strip()
                                        st.markdown(content)
                                    except Exception as e:
                                        st.warning(f"파일 읽기 실패: {e}")
                                        if trend_result.get('data'):
                                            data = trend_result['data']
                                            if data.startswith("```markdown"):
                                                data = data.replace("```markdown", "").replace("```", "").strip()
                                            st.markdown(data)
                    else:
                        st.error(f"❌ **Trend Analyzer**: 실패 - {trend_result.get('error', '알 수 없는 오류')}")
                
                # 메타데이터 표시
                if latest_json:
                    try:
                        with open(latest_json, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                            timestamp_str = json_data.get('execution_timestamp', '')
                            if timestamp_str:
                                exec_time = datetime.fromisoformat(timestamp_str)
                                time_str = exec_time.strftime('%Y-%m-%d %H:%M:%S')
                                st.caption(f"⏰ 분석 시간: {time_str}")
                    except:
                        pass
    else:
        st.info("💡 아직 Business Strategy Agent의 결과가 없습니다. 위에서 비즈니스 전략 분석을 실행해보세요.")

if __name__ == "__main__":
    main() 