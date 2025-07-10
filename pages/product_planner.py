import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime
import os

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

# Product Planner Agent는 자체 환경변수 로더를 사용
from srcs.product_planner_agent.utils import env_settings as env

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 제품 기획 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    final_report = result_data.get('final_report', {})
    if not final_report:
        st.info("최종 보고서가 생성되지 않았습니다. 상세 로그를 확인해주세요.")
    else:
        st.success("✅ 최종 보고서가 성공적으로 생성되었습니다.")
        
        # 파일 경로가 있다면 링크 제공
        if 'file_path' in final_report:
            st.markdown(f"**보고서 위치**: `{final_report['file_path']}`")
        
        # 보고서 내용 표시
        with st.expander("📄 최종 보고서 내용 보기", expanded=True):
            st.markdown(final_report.get('content', '내용 없음'))

    with st.expander("상세 분석 결과 보기 (JSON)"):
        st.json(result_data)


def main():
    create_agent_page(
        agent_name="Product Planner Agent",
        page_icon="🚀",
        page_type="product",
        title="Product Planner Agent",
        subtitle="Figma 디자인을 분석하여 시장 조사, 전략, 실행 계획까지 한번에 수립합니다.",
        module_path="srcs.product_planner_agent.run_product_planner"
    )
    result_placeholder = st.empty()

    # Figma API 키 확인
    figma_api_key = env.get("FIGMA_API_KEY")
    if not figma_api_key:
        st.error("FIGMA_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.info("Product Planner Agent는 Figma API 키가 있어야 실행할 수 있습니다.")
        st.stop()

    with st.form("product_planner_form"):
        st.subheader("📝 제품 기획 정보 입력")
        product_concept = st.text_area(
            "제품 컨셉",
            placeholder="예: AI 기반의 개인화된 뉴스 추천 서비스",
            help="제품의 핵심 아이디어나 목표를 설명해주세요."
        )
        user_persona = st.text_area(
            "사용자 페르소나",
            placeholder="예: 기술에 정통하고, 바쁜 일상 속에서 자신에게 맞는 뉴스를 빠르게 소비하고 싶어하는 30대 전문가",
            help="이 제품을 사용할 타겟 사용자에 대해 설명해주세요."
        )
        figma_url = st.text_input(
            "분석할 Figma URL (선택 사항)",
            placeholder="https://www.figma.com/file/FILE_ID/...?node-id=NODE_ID"
        )
        submitted = st.form_submit_button("🚀 제품 기획 시작", use_container_width=True)

    if submitted:
        if not product_concept or not user_persona:
            st.warning("제품 컨셉과 사용자 페르소나를 반드시 입력해야 합니다.")
        else:
            reports_path = Path(get_reports_path('product_planner'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"planner_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.product_planner_agent.run_product_planner",
                "--product-concept", product_concept,
                "--user-persona", user_persona,
                "--result-json-path", str(result_json_path)
            ]
            if figma_url:
                command.extend(["--figma-url", figma_url])

            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/product_planner"
            )

            if result and "data" in result:
                display_results(result["data"])

if __name__ == "__main__":
    main() 