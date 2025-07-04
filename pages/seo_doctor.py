"""
🏥 SEO Doctor Page

사이트 응급처치 + 경쟁사 스파이 AI
"""

import streamlit as st
from pathlib import Path
import sys
import json
from datetime import datetime
import os
import pandas as pd

from srcs.common.page_utils import create_agent_page
from srcs.common.ui_utils import run_agent_process
from configs.settings import get_reports_path

def display_results(result_data):
    st.markdown("---")
    st.subheader("📊 SEO 분석 결과")

    if not result_data:
        st.warning("분석 결과를 찾을 수 없습니다.")
        return

    emergency_level = result_data.get('emergency_level', 'N/A')
    st.metric("진단 수준", emergency_level)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("종합 점수", f"{result_data.get('overall_score', 0):.0f}")
    c2.metric("성능", f"{result_data.get('performance_score', 0):.0f}")
    c3.metric("SEO", f"{result_data.get('seo_score', 0):.0f}")
    c4.metric("접근성", f"{result_data.get('accessibility_score', 0):.0f}")
    
    with st.expander("세부 진단 내용 보기", expanded=True):
        st.markdown("#### 주요 웹 지표 (Core Web Vitals)")
        st.json(result_data.get('core_web_vitals', {}))
        
        st.markdown("#### 🚨 치명적인 문제")
        st.table(pd.DataFrame(result_data.get('critical_issues', []), columns=["문제점"]))
        
        st.markdown("#### ⚡️ 빠른 수정 제안")
        st.table(pd.DataFrame(result_data.get('quick_fixes', []), columns=["수정 제안"]))
    
    with st.expander("종합 개선 권장 사항"):
        st.table(pd.DataFrame(result_data.get('recommendations', []), columns=["권장 사항"]))

    with st.expander("경쟁사 분석"):
        st.table(pd.DataFrame(result_data.get('competitor_analysis', [])))


def main():
    create_agent_page(
        agent_name="SEO Doctor Agent",
        page_icon="🏥",
        page_type="seo",
        title="SEO Doctor Agent",
        subtitle="웹사이트를 정밀 진단하고 검색 엔진 최적화(SEO)를 위한 처방을 내립니다.",
        module_path="srcs.seo_doctor.run_seo_doctor"
    )
    result_placeholder = st.empty()

    with st.form("seo_doctor_form"):
        st.subheader("📝 분석할 웹사이트 정보 입력")
        
        url = st.text_input("분석할 웹사이트 URL", placeholder="https://example.com")
        
        include_competitors = st.checkbox("경쟁사 분석 포함", value=True)
        
        competitor_urls_text = st.text_area(
            "경쟁사 URL (쉼표로 구분)",
            placeholder="https://competitor1.com, https://competitor2.com",
            disabled=not include_competitors
        )
        
        submitted = st.form_submit_button("🚀 SEO 진단 시작", use_container_width=True)

    if submitted:
        if not url or "http" not in url:
            st.warning("유효한 URL을 입력해주세요. (http:// 또는 https:// 포함)")
        else:
            reports_path = Path(get_reports_path('seo_doctor'))
            reports_path.mkdir(parents=True, exist_ok=True)
            result_json_path = reports_path / f"seo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            py_executable = sys.executable
            command = [
                py_executable, "-m", "srcs.seo_doctor.run_seo_doctor",
                "--url", url,
                "--result-json-path", str(result_json_path)
            ]
            if include_competitors:
                command.append("--include-competitors")
                if competitor_urls_text.strip():
                    competitor_urls = [u.strip() for u in competitor_urls_text.split(',')]
                    command.append("--competitor-urls")
                    command.extend(competitor_urls)
            
            result = run_agent_process(
                placeholder=result_placeholder,
                command=command,
                process_key_prefix="logs/seo_doctor"
            )

            if result and "data" in result:
                display_results(result["data"])

if __name__ == "__main__":
    main()