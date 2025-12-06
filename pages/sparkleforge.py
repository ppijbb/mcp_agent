"""
SparkleForge Multi-Agent Research System Page

혁신적인 다중 에이전트 연구 시스템인 SparkleForge를 A2A로 호출하는 페이지
"""

import streamlit as st
import asyncio
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType
from configs.settings import get_reports_path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="SparkleForge - Multi-Agent Research System",
    page_icon="✨",
    layout="wide"
)

# 페이지 헤더
st.title("✨ SparkleForge")
st.markdown("*혁신적인 다중 에이전트 연구 시스템*")
st.markdown("*아이디어가 반짝이고 단련되는 곳* ⚒️✨")

st.markdown("---")

# 사이드바 설명
with st.sidebar:
    st.header("🔍 SparkleForge 소개")

    st.markdown("""
    **SparkleForge**는 혁신적인 다중 에이전트 연구 시스템입니다.

    ### 🚀 핵심 기능
    - **5+ 전문 AI 장인**들이 협업
    - **실시간 반짝임** 관찰 가능
    - **창의적 합성**으로 새로운 아이디어 생성
    - **출처 검증** 및 신뢰도 점수
    - **연구 기억**으로 지속적 개선

    ### 🎯 사용 방법
    1. 연구 주제를 입력하세요
    2. 스트리밍 모드 선택 (선택사항)
    3. '연구 시작' 버튼을 클릭하세요
    4. 실시간으로 연구 과정을 확인하세요
    """)

    st.markdown("---")

    # 기술 스택 정보
    with st.expander("🛠️ 기술 스택"):
        st.markdown("""
        - **프레임워크**: LangGraph 기반 다중 에이전트 시스템
        - **AI 모델**: Gemini 2.5 Flash Lite, OpenRouter
        - **통신**: A2A (Agent-to-Agent) 프로토콜
        - **검증**: MCP (Model Context Protocol)
        - **메모리**: 공유 메모리 및 세션 관리
        """)

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔬 연구 요청")

    # 연구 주제 입력
    research_topic = st.text_area(
        "연구 주제 입력",
        placeholder="예: 인공지능의 미래 전망, 블록체인 기술 동향, 지속 가능한 에너지 솔루션...",
        height=100,
        help="구체적이고 명확한 연구 주제를 입력하세요."
    )

    # 추가 옵션들
    col_a, col_b = st.columns(2)

    with col_a:
        streaming_mode = st.checkbox(
            "스트리밍 모드",
            value=False,
            help="연구 과정을 실시간으로 확인할 수 있습니다."
        )

    with col_b:
        save_results = st.checkbox(
            "결과 저장",
            value=True,
            help="연구 결과를 파일로 저장합니다."
        )

    # 실행 버튼
    execute_button = st.button(
        "🚀 연구 시작",
        type="primary",
        use_container_width=True,
        disabled=not research_topic.strip()
    )

with col2:
    st.subheader("📊 연구 설정")

    # 연구 범위 설정
    research_depth = st.selectbox(
        "연구 깊이",
        options=["기본", "상세", "종합"],
        index=1,
        help="연구의 깊이와 범위를 설정합니다."
    )

    # 출력 형식
    output_format = st.selectbox(
        "출력 형식",
        options=["마크다운", "JSON", "HTML"],
        index=0,
        help="연구 결과의 출력 형식을 선택합니다."
    )

    # 추가 설정
    with st.expander("⚙️ 고급 설정"):
        max_sources = st.slider(
            "최대 출처 수",
            min_value=5,
            max_value=50,
            value=20,
            help="분석할 최대 출처 수를 설정합니다."
        )

        include_images = st.checkbox(
            "이미지 포함",
            value=True,
            help="관련 이미지들을 결과에 포함합니다."
        )

# 결과 표시 영역
result_placeholder = st.empty()

# 실행 로직
if execute_button and research_topic.strip():
    with st.spinner("🔍 SparkleForge가 연구를 시작합니다..."):
        try:
            # 입력 데이터 구성
            input_params = {
                "request": research_topic.strip(),
                "streaming": streaming_mode,
                "depth": research_depth,
                "format": output_format,
                "max_sources": max_sources,
                "include_images": include_images,
                "save_results": save_results
            }

            # 결과 파일 경로 설정
            result_json_path = None
            if save_results:
                reports_path = Path(get_reports_path('sparkleforge'))
                reports_path.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_json_path = reports_path / f"research_{timestamp}.json"

            # A2A를 통해 SparkleForge 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                agent_id="sparkleforge_agent",
                agent_name="SparkleForge Multi-Agent Research System",
                entry_point="sparkleforge.common.sparkleforge_entry_point.run_sparkleforge_agent",
                agent_type=AgentType.MCP_AGENT,
                capabilities=[
                    "research",
                    "multi_agent_collaboration",
                    "source_validation",
                    "creative_synthesis",
                    "domain_exploration"
                ],
                description="혁신적인 다중 에이전트 연구 시스템",
                input_params=input_params,
                result_json_path=result_json_path,
                use_a2a=True
            )

            # 결과 표시
            display_results(result)

        except Exception as e:
            st.error(f"연구 실행 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"SparkleForge execution error: {e}", exc_info=True)

def display_results(result: Dict[str, Any]):
    """
    연구 결과를 표시하는 함수

    Args:
        result: A2A 실행 결과
    """
    if not result:
        st.warning("결과가 없습니다.")
        return

    # 성공/실패 상태 확인
    success = result.get("success", False)

    if success:
        st.success("🎉 연구가 성공적으로 완료되었습니다!")

        # 결과 데이터 추출
        result_data = result.get("data", {})
        if isinstance(result_data, dict) and "result" in result_data:
            sparkleforge_result = result_data["result"]
        else:
            sparkleforge_result = result_data

        # 결과 표시
        if isinstance(sparkleforge_result, dict):
            # 구조화된 결과인 경우

            # 요약 표시
            if "summary" in sparkleforge_result:
                st.subheader("📋 연구 요약")
                st.info(sparkleforge_result["summary"])

            # 주요 결과 표시
            if "key_findings" in sparkleforge_result:
                st.subheader("🔑 주요 발견")
                findings = sparkleforge_result["key_findings"]
                if isinstance(findings, list):
                    for i, finding in enumerate(findings, 1):
                        st.markdown(f"**{i}.** {finding}")
                else:
                    st.write(findings)

            # 출처 표시
            if "sources" in sparkleforge_result:
                st.subheader("📚 참고 출처")
                sources = sparkleforge_result["sources"]
                if isinstance(sources, list):
                    for source in sources:
                        if isinstance(source, dict):
                            st.markdown(f"- **{source.get('title', 'N/A')}** ({source.get('url', 'N/A')})")
                        else:
                            st.markdown(f"- {source}")
                else:
                    st.write(sources)

            # 상세 결과 (접을 수 있게)
            with st.expander("📄 상세 결과 보기"):
                st.json(sparkleforge_result)

        elif isinstance(sparkleforge_result, str):
            # 텍스트 결과인 경우
            st.subheader("📄 연구 결과")
            st.markdown(sparkleforge_result)

        else:
            # 기타 형태의 결과
            st.subheader("📄 연구 결과")
            st.write(sparkleforge_result)

        # 메타데이터 표시
        with st.expander("ℹ️ 실행 정보"):
            st.markdown(f"**Agent**: {result.get('agent', 'sparkleforge')}")
            st.markdown(f"**실행 시간**: {result.get('execution_time', 'N/A')}")
            st.markdown(f"**타임스탬프**: {result.get('timestamp', 'N/A')}")

    else:
        # 실패한 경우
        st.error("❌ 연구 실행에 실패했습니다.")

        error_msg = result.get("error", "알 수 없는 오류")
        st.error(f"오류 내용: {error_msg}")

        # 상세 오류 정보 (개발자용)
        if st.checkbox("개발자용 상세 오류 정보"):
            st.code(str(result), language="json")

# 푸터
st.markdown("---")
st.markdown("*SparkleForge - Where Ideas Sparkle and Get Forged* ⚒️✨")
