"""
📝 RAG Agent Page

문서 기반 질의응답 및 지식 관리 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

def main():
    """RAG Agent 메인 페이지"""
    
    features = [
        "**문서 기반 QA**: 업로드한 문서를 기반으로 정확한 답변",
        "**지식 베이스 구축**: 자동 문서 인덱싱 및 검색",
        "**다양한 파일 지원**: PDF, DOCX, TXT, HTML 등",
        "**의미적 검색**: 키워드가 아닌 의미 기반 검색",
        "**출처 추적**: 답변의 근거 문서 및 페이지 제공"
    ]
    
    special_features = [
        "**실시간 학습**: 새로운 문서 자동 반영",
        "**다중 언어**: 한국어, 영어 등 다국어 지원",
        "**개인화**: 사용자별 맞춤 지식 베이스",
        "**버전 관리**: 문서 변경 이력 추적",
        "**API 연동**: 외부 시스템과의 연계"
    ]
    
    use_cases = [
        "기업 내부 문서 검색 시스템",
        "고객 지원 챗봇",
        "연구 논문 분석 도구",
        "정책 및 규정 문의 시스템"
    ]
    
    create_agent_page(
        agent_name="RAG Agent",
        page_icon="📝",
        page_type="data",
        title="📝 RAG Agent",
        subtitle="문서 기반 질의응답 및 지식 관리 시스템",
        module_path="srcs.basic_agents.rag_agent",
        features=features,
        special_features=special_features,
        use_cases=use_cases
    )

if __name__ == "__main__":
    main() 