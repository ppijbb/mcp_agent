import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime
import streamlit_process_manager as spm
from srcs.common.ui_utils import run_agent_process

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.urban_hive.urban_hive_agent import UrbanDataCategory
from srcs.common.page_utils import setup_page, render_home_button
from srcs.common.styles import get_common_styles, get_page_header
from configs.settings import get_reports_path

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

def format_urban_hive_output(result: dict) -> str:
    """Formats the result dictionary from the agent into a Markdown string."""
    if not result.get('critical_issues') or "분석 실패" in result['critical_issues'][0]:
        return f"## 🚨 분석 실패\n\n**오류**: {result.get('critical_issues', ['알 수 없는 오류'])[0]}"

    display_location = result.get('affected_areas', ["지정되지 않은 위치"])[0]

    md_lines = [
        f"## 🏙️ 도시 데이터 분석 결과: {display_location}",
        f"**분석 카테고리**: {result.get('data_category', 'N/A')}",
        f"**분석 시간**: {result.get('analysis_timestamp', 'N/A')}",
        f"**위협 수준**: {result.get('threat_level', 'N/A')}",
        f"**도시 건강 점수**: {result.get('overall_score', 0)}/100",
    ]
    
    key_metrics = result.get('key_metrics', {})
    if key_metrics:
        md_lines.append("\n### 📊 주요 지표:")
        for key, value in key_metrics.items():
            md_lines.append(f"- **{key.replace('_', ' ').title()}**: {value if value is not None else '데이터 없음'}")
    
    critical_issues = result.get('critical_issues', [])
    if critical_issues:
        md_lines.append("\n### ⚠️ 주요 문제점:")
        for issue in critical_issues: md_lines.append(f"- {issue}")
    
    recommendations = result.get('recommendations', [])
    if recommendations:
        md_lines.append("\n### 💡 추천 사항:")
        for rec in recommendations: md_lines.append(f"- {rec}")

    predicted_trends = result.get('predicted_trends', [])
    if predicted_trends:
        md_lines.append("\n### 📈 예측 동향:")
        for trend in predicted_trends: md_lines.append(f"- {trend}")
        
    return "\n".join(md_lines)


def main():
    setup_page("🏙️ Urban Hive Agent", "🏙️")
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    header_html = get_page_header(
        "urban",
        "🏙️ Urban Hive Agent",
        "AI 기반 도시 데이터 분석 플랫폼. 교통, 안전, 환경, 부동산 등 다양한 도시 문제를 심층 분석합니다."
    )
    st.markdown(header_html, unsafe_allow_html=True)
    render_home_button()

    st.markdown("---")

    main_content_col, agent_info_col = st.columns([2, 1])

    with main_content_col:
        st.subheader("💬 도시 데이터 분석 요청")
        st.markdown(
            "아래 채팅창에 분석하고 싶은 도시 문제에 대해 질문해주세요. "
            "예를 들어, 특정 지역의 부동산 동향, 교통 상황, 환경 문제 등을 물어볼 수 있습니다."
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "안녕하세요! 저는 Urban Hive 에이전트입니다. 🏙️\n"
                        "분석하고 싶은 도시, 주제, 기간 등을 알려주시면 관련 데이터를 분석해 드릴게요.\n"
                        "예: '서울 성동구의 최근 3개월간 부동산 시장 동향과 전망을 알려줘.'"
                    ),
                }
            ]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("예: '서울 강남구 아파트의 최근 3개월 시세와 시장 동향을 알려줘'"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                reports_path = Path(get_reports_path('urban_hive'))
                reports_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_json_path = reports_path / f"urban_hive_result_{timestamp}.json"
                
                py_executable = sys.executable
                command = [
                    py_executable, "-u", "-m", "srcs.urban_hive.run_urban_hive_agent",
                    "--prompt", prompt,
                    "--result-json-path", str(result_json_path)
                ]
                
                process_key = f"urban_hive_{timestamp}"

                placeholder = st.empty()
                result = run_agent_process(
                    placeholder=placeholder,
                    command=command,
                    process_key_prefix="logs/urban_hive",
                    log_expander_title="실시간 실행 로그"
                )
                
                if result:
                    response_md = format_urban_hive_output(result)
                    st.markdown(response_md)
                    st.session_state["messages"].append({"role": "assistant", "content": response_md})
                else:
                    error_msg = "❌ 에이전트 실행에 실패했습니다."
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})

    with agent_info_col:
        st.markdown("### ✨ Urban Hive 특징")
        st.markdown("- **실시간 데이터 기반 분석**: 최신 도시 현황 반영")
        st.markdown("- **다각적 인사이트 제공**: 교통, 안전, 환경, 부동산 등 종합 분석")
        st.markdown("- **예측 모델링**: 미래 도시 변화 예측 및 선제적 대응 방안 제시")
        st.markdown("- **실행 가능한 솔루션**: 데이터 기반 정책 제언 및 실행 계획 수립 지원")

    st.markdown("---")
    with st.expander("💡 Urban Hive Agent 정보 더보기", expanded=False):
        st.markdown("## 💡 Urban Hive Agent란?")
        st.markdown(
            "Urban Hive는 복잡한 도시 데이터를 분석하여 시민 생활 개선과 지속 가능한 도시 발전에 필요한 "
            "통찰력을 제공하는 AI 에이전트입니다. 자연어 질문을 통해 특정 지역의 다양한 도시 문제에 대한 "
            "심층 분석 보고서를 받아보세요."
        )
        st.markdown("---")
        st.markdown("###  주요 분석 카테고리")
        cat_cols = st.columns(2)
        for i, category in enumerate(UrbanDataCategory):
            with cat_cols[i % 2]:
                st.markdown(f"- {category.value}")
        
        st.markdown("---")
        st.markdown("### 🚀 활용 예시")
        st.markdown("- '서울 강남구의 최근 1개월간 교통 혼잡도와 해결 방안은?'")
        st.markdown("- '부산 해운대구의 여름철 관광객 안전 문제와 대응 전략은?'")
        st.markdown("- '인천 송도 국제도시의 미세먼지 현황과 환경 개선 방안을 알려줘.'")
        st.markdown("- '대전 유성구의 신규 아파트 단지 주변 상권 활성화 가능성은?'")

if __name__ == "__main__":
    main()

# 최신 Urban Hive Agent 결과 확인
st.markdown("---")
st.markdown("## 📊 최신 Urban Hive Agent 결과")

latest_urban_result = result_reader.get_latest_result("urban_hive_agent", "urban_analysis")

if latest_urban_result:
    with st.expander("🏙️ 최신 도시 데이터 분석 결과", expanded=False):
        st.subheader("🤖 최근 도시 데이터 분석 결과")
        
        if isinstance(latest_urban_result, dict):
            # 도시 정보 표시
            affected_areas = latest_urban_result.get('affected_areas', ['N/A'])
            data_category = latest_urban_result.get('data_category', 'N/A')
            
            st.success(f"**분석 지역: {', '.join(affected_areas)}**")
            st.info(f"**분석 카테고리: {data_category}**")
            
            # 분석 결과 요약
            col1, col2, col3 = st.columns(3)
            col1.metric("도시 건강 점수", f"{latest_urban_result.get('overall_score', 0)}/100")
            col2.metric("위협 수준", latest_urban_result.get('threat_level', 'N/A'))
            col3.metric("분석 상태", "완료" if latest_urban_result.get('success', False) else "실패")
            
            # 주요 지표 표시
            key_metrics = latest_urban_result.get('key_metrics', {})
            if key_metrics:
                st.subheader("📊 주요 지표")
                for key, value in key_metrics.items():
                    st.write(f"• **{key.replace('_', ' ').title()}**: {value if value is not None else '데이터 없음'}")
            
            # 주요 문제점 표시
            critical_issues = latest_urban_result.get('critical_issues', [])
            if critical_issues:
                st.subheader("⚠️ 주요 문제점")
                for issue in critical_issues:
                    st.write(f"• {issue}")
            
            # 추천 사항 표시
            recommendations = latest_urban_result.get('recommendations', [])
            if recommendations:
                st.subheader("💡 추천 사항")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # 예측 동향 표시
            predicted_trends = latest_urban_result.get('predicted_trends', [])
            if predicted_trends:
                st.subheader("📈 예측 동향")
                for trend in predicted_trends:
                    st.write(f"• {trend}")
            
            # 메타데이터 표시
            if 'analysis_timestamp' in latest_urban_result:
                st.caption(f"⏰ 분석 시간: {latest_urban_result['analysis_timestamp']}")
        else:
            st.json(latest_urban_result)
else:
    st.info("💡 아직 Urban Hive Agent의 결과가 없습니다. 위에서 도시 데이터 분석을 실행해보세요.")
