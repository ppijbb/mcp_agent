"""
Business Strategy Agents Streamlit Interface
============================================
Streamlit 인터페이스용 Business Strategy Agent 래퍼
"""

import streamlit as st
import asyncio
from typing import List, Dict, Any
from datetime import datetime

from srcs.business_strategy_agents.run_business_strategy_agents import BusinessStrategyRunner


def main():
    """Streamlit용 Business Strategy Agent 메인 함수"""

    st.header("🎯 Business Strategy MCPAgent Suite")
    st.markdown("AI 기반 종합 비즈니스 전략 수립 및 시장 분석 플랫폼")

    # 입력 폼 생성
    with st.form("business_strategy_form"):
        st.subheader("📝 분석 설정")

        # 기본 설정
        col1, col2 = st.columns(2)

        with col1:
            keywords_input = st.text_input(
                "🔍 핵심 키워드 (쉼표로 구분)",
                placeholder="예: AI, fintech, sustainability",
                help="분석하고자 하는 핵심 키워드들을 입력하세요"
            )

            business_context = st.text_area(
                "🏢 비즈니스 맥락",
                placeholder="예: AI 스타트업, 핀테크 회사 등",
                help="비즈니스 상황이나 배경을 설명해주세요"
            )

        with col2:
            objectives_input = st.text_input(
                "🎯 목표 (쉼표로 구분)",
                placeholder="예: growth, expansion, efficiency",
                help="달성하고자 하는 비즈니스 목표들을 입력하세요"
            )

            regions_input = st.text_input(
                "🌍 타겟 지역 (쉼표로 구분)",
                placeholder="예: North America, Europe, Asia",
                help="분석 대상 지역을 입력하세요"
            )

        # 고급 설정
        st.subheader("⚙️ 고급 설정")

        col3, col4 = st.columns(2)

        with col3:
            time_horizon = st.selectbox(
                "⏰ 분석 기간",
                ["3_months", "6_months", "12_months", "24_months"],
                index=2,
                help="분석 및 전략 수립 기간을 선택하세요"
            )

        with col4:
            analysis_mode = st.selectbox(
                "🔄 분석 모드",
                ["unified", "individual", "both"],
                index=0,
                help="unified: 통합분석(권장), individual: 개별분석, both: 전체분석"
            )

        # 실행 버튼
        submitted = st.form_submit_button("🚀 비즈니스 전략 분석 시작", use_container_width=True)

    # 폼 제출 처리
    if submitted:
        # 입력 유효성 검사
        if not keywords_input.strip():
            st.error("❌ 핵심 키워드를 입력해주세요!")
            return

        # 키워드 파싱
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        objectives = [o.strip() for o in objectives_input.split(',') if o.strip()] if objectives_input.strip() else None
        regions = [r.strip() for r in regions_input.split(',') if r.strip()] if regions_input.strip() else None

        business_ctx = {"description": business_context} if business_context.strip() else None

        # 분석 실행
        st.info("🔄 Business Strategy MCPAgent 실행 중...")

        with st.spinner("AI 에이전트들이 비즈니스 전략을 분석하고 있습니다..."):
            try:
                # 비동기 함수를 동기적으로 실행
                results = asyncio.run(run_business_strategy_analysis(
                    keywords=keywords,
                    business_context=business_ctx,
                    objectives=objectives,
                    regions=regions,
                    time_horizon=time_horizon,
                    mode=analysis_mode
                ))

                # 결과 표시
                display_results(results)

            except Exception as e:
                st.error(f"❌ 분석 실행 중 오류가 발생했습니다: {e}")
                st.error("Business Strategy Agent 구성을 확인해주세요.")


async def run_business_strategy_analysis(
    keywords: List[str],
    business_context: Dict[str, Any] = None,
    objectives: List[str] = None,
    regions: List[str] = None,
    time_horizon: str = "12_months",
    mode: str = "unified"
) -> Dict[str, Any]:
    """비동기 Business Strategy 분석 실행"""

    output_dir = "business_strategy_reports"
    runner = BusinessStrategyRunner(output_dir=output_dir)

    # 전체 분석 실행
    results = await runner.run_full_suite(
        keywords=keywords,
        business_context=business_context,
        objectives=objectives,
        regions=regions,
        time_horizon=time_horizon,
        mode=mode
    )

    # 실행 보고서 저장
    report_file = runner.save_execution_report()

    return {
        "success": True,
        "results": results,
        "report_file": report_file,
        "runner": runner
    }


def display_results(analysis_results: Dict[str, Any]):
    """분석 결과 표시"""

    if not analysis_results.get("success"):
        st.error("❌ 분석 실행에 실패했습니다.")
        return

    results = analysis_results["results"]
    summary = results.get("summary", {})

    # 성공 메시지
    st.success("✅ Business Strategy 분석이 완료되었습니다!")

    # 실행 요약
    st.subheader("📊 실행 요약")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "성공한 에이전트",
            f"{summary.get('successful_agents', 0)}/{summary.get('total_agents', 0)}"
        )

    with col2:
        st.metric(
            "실행 시간",
            f"{summary.get('execution_time', 0):.2f}초"
        )

    with col3:
        st.metric(
            "생성된 보고서",
            len([r for r in results.get("results", {}).values() if r.get("success")])
        )

    with col4:
        if summary.get('successful_agents') == summary.get('total_agents'):
            st.success("완료")
        else:
            st.warning("부분 완료")

    # 개별 에이전트 결과
    st.subheader("🤖 에이전트별 결과")

    agent_results = results.get("results", {})

    for agent_name, result in agent_results.items():
        with st.expander(f"📈 {agent_name.replace('_', ' ').title()}"):
            if result.get("success"):
                st.success(f"✅ 성공적으로 완료되었습니다")

                if "output_file" in result:
                    st.info(f"📄 보고서: `{result['output_file']}`")

                if "analysis_summary" in result:
                    st.markdown("**분석 요약:**")
                    st.markdown(result["analysis_summary"])

            else:
                st.error(f"❌ 실행 실패: {result.get('error', '알 수 없는 오류')}")

    # 실행 보고서 다운로드
    if "report_file" in analysis_results:
        st.subheader("📥 보고서 다운로드")

        try:
            with open(analysis_results["report_file"], 'r') as f:
                report_data = f.read()

            st.download_button(
                label="📄 실행 보고서 다운로드 (JSON)",
                data=report_data,
                file_name=f"business_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.warning(f"보고서 다운로드 준비 중 오류: {e}")

    # 출력 디렉토리 정보
    st.info(f"💼 모든 보고서는 `{summary.get('output_directory', 'business_strategy_reports')}` 디렉토리에 저장됩니다.")


if __name__ == "__main__":
    main()
