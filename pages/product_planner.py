"""
🚀 Product Planner Agent Page

Figma 디자인 분석과 프로덕트 기획을 위한 AI 어시스턴트
표준 A2A 패턴 적용
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_a2a_page_template import create_standard_a2a_page
from srcs.common.agent_interface import AgentType

def main():
    # 표준화된 A2A Page 생성
    create_standard_a2a_page(
        agent_id="product_planner_agent",
        agent_name="Product Planner Agent",
        page_icon="🚀",
        page_type="product",
        title="Product Planner Agent",
        subtitle="Figma 디자인을 분석하여 시장 조사, 전략, 실행 계획까지 한번에 수립합니다.",
        entry_point="srcs.product_planner_agent.run_product_planner",
        agent_type=AgentType.MCP_AGENT,
        capabilities=["market_analysis", "product_planning", "figma_analysis", "strategy_planning"],
        description="Figma 디자인 분석, 프로덕트 기획, 시장 조사",
        form_fields=[
            {
                "type": "text_area",
                "key": "product_concept",
                "label": "제품 컨셉",
                "default": "모바일 앱 제품을 기획해주세요.",
                "height": 100,
                "help": "기획하고자 하는 제품의 핵심 아이디어를 입력하세요",
                "required": True
            },
            {
                "type": "text_area",
                "key": "user_persona",
                "label": "타켓 사용자 페르소나",
                "default": "일반 사용자",
                "height": 100,
                "help": "핵심 타겟 사용자에 대한 설명을 입력하세요",
                "required": True
            },
            {
                "type": "text_input",
                "key": "figma_url",
                "label": "Figma URL (선택)",
                "default": "",
                "help": "분석할 Figma 파일의 URL을 입력하세요"
            }
        ],
        display_results_func=display_results,
        result_category="product_planning"
    )

def display_results(result_data):
    """결과 표시"""
    st.markdown("---")
    st.subheader("📊 제품 기획 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    # final_report 처리
    final_report = result_data.get('final_report', {})
    if not final_report:
        # result_data 자체가 리포트일 가능성 확인
        if 'content' in result_data:
            final_report = result_data
        else:
            st.info("상세 분석 결과를 확인 중입니다...")
            st.json(result_data)
            return

    st.success("✅ 제품 기획 분석이 완료되었습니다.")
    
    # 제품 정보 (있는 경우)
    if 'product_name' in result_data:
        st.info(f"**제품명**: {result_data['product_name']}")

    # 보고서 내용 표시
    with st.expander("📄 최종 보고서 내용 보기", expanded=True):
        st.markdown(final_report.get('content', '내용 없음'))
    
    # 상세 데이터
    with st.expander("🔍 상세 데이터 보기"):
        st.json(result_data)

if __name__ == "__main__":
    main()
 