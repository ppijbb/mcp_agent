"""
State Management for LangGraph Compliance Chain

LangGraph StateGraph에서 사용할 상태 정의
"""

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime


class ComplianceState(TypedDict):
    """
    의료기기 규제 컴플라이언스 테스트 워크플로우 상태
    
    LangGraph StateGraph에서 사용하는 상태 구조
    """
    # 입력 정보
    device_info: Dict[str, Any]
    
    # 규제 프레임워크 분석 결과
    regulatory_frameworks: List[str]
    framework_requirements: Dict[str, Any]
    
    # 테스트 케이스
    test_cases: List[Dict[str, Any]]
    
    # 테스트 실행 결과
    test_results: List[Dict[str, Any]]
    
    # 규제 준수 검증 결과
    compliance_status: str  # "COMPLIANT", "NON_COMPLIANT", "PARTIAL"
    compliance_score: float  # 0.0 ~ 1.0
    risk_assessment: Dict[str, Any]
    
    # 리포트
    report: Optional[str]
    report_path: Optional[str]
    
    # 메타데이터
    timestamp: str
    workflow_stage: str  # "framework_analysis", "test_generation", "test_execution", "validation", "report_generation"
    errors: List[str]
    warnings: List[str]

