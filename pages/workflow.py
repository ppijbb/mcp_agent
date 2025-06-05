"""
🔄 Workflow Orchestrator Page

복잡한 워크플로우 자동화 및 다중 에이전트 협업
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
    """Workflow Orchestrator 메인 페이지"""
    
    features = [
        "**다중 에이전트 협업**: 여러 AI 에이전트 동시 운영",
        "**워크플로우 자동화**: 복잡한 비즈니스 프로세스 자동화",
        "**실시간 모니터링**: 작업 진행 상황 추적 및 알림",
        "**동적 스케줄링**: 우선순위 기반 작업 배정",
        "**오류 복구**: 자동 재시도 및 대안 경로 실행"
    ]
    
    special_features = [
        "**적응형 워크플로우**: 실행 결과 기반 자동 최적화",
        "**병렬 처리**: 독립적 작업의 동시 실행",
        "**조건부 분기**: 상황별 다른 경로 실행",
        "**리소스 관리**: 자동 부하 분산 및 리소스 할당",
        "**감사 추적**: 모든 실행 과정 기록 및 분석"
    ]
    
    use_cases = [
        "대규모 데이터 처리 파이프라인",
        "고객 서비스 자동화 시스템",
        "콘텐츠 생성 및 배포 워크플로우",
        "비즈니스 프로세스 최적화"
    ]
    
    create_agent_page(
        agent_name="Workflow Orchestrator",
        page_icon="🔄",
        page_type="business",
        title="🔄 Workflow Orchestrator",
        subtitle="복잡한 워크플로우 자동화 및 다중 에이전트 협업 시스템",
        module_path="srcs.basic_agents.workflow_orchestration",
        features=features,
        special_features=special_features,
        use_cases=use_cases
    )

if __name__ == "__main__":
    main() 