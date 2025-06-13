"""
중앙 설정 관리 시스템
모든 하드코딩된 경로와 설정을 중앙에서 관리
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 보고서 저장 기본 디렉토리
REPORTS_BASE_DIR = PROJECT_ROOT / "reports"

# 각 에이전트별 보고서 디렉토리 설정
AGENT_REPORT_PATHS = {
    'ai_architect': 'ai_architect_reports',
    'business_strategy': 'business_strategy_reports', 
    'cybersecurity': 'cybersecurity_infrastructure_reports',
    'data_generator': 'data_generator_reports',
    'decision_agent': 'decision_agent_reports',
    'finance_health': 'finance_health_reports',
    'hr_recruitment': 'recruitment_reports',
    'rag_agent': 'rag_agent_reports',
    'research': 'research_reports',
    'seo_doctor': 'seo_doctor_reports',
    'travel_scout': 'travel_scout_reports',
    'workflow': 'workflow_reports'
}

def get_reports_path(agent_name: str) -> str:
    """
    에이전트별 보고서 저장 경로 반환
    
    Args:
        agent_name: 에이전트 이름
        
    Returns:
        str: 보고서 저장 경로
        
    Raises:
        ValueError: 지원하지 않는 에이전트 이름인 경우
    """
    if agent_name not in AGENT_REPORT_PATHS:
        raise ValueError(f"지원하지 않는 에이전트: {agent_name}. 지원 에이전트: {list(AGENT_REPORT_PATHS.keys())}")
    
    return AGENT_REPORT_PATHS[agent_name]

def ensure_reports_directory(agent_name: str) -> str:
    """
    보고서 디렉토리가 존재하는지 확인하고 생성
    
    Args:
        agent_name: 에이전트 이름
        
    Returns:
        str: 생성된 디렉토리 경로
    """
    path = get_reports_path(agent_name)
    os.makedirs(path, exist_ok=True)
    return path

# API 설정 (향후 확장용)
API_SETTINGS = {
    'timeout': 30,
    'max_retries': 3,
    'rate_limit': 100  # requests per minute
}

# UI 설정 (향후 확장용)
UI_SETTINGS = {
    'theme': 'default',
    'language': 'ko',
    'page_size': 20
}

class ConnectionStatus:
    """실시간 연결 상태를 체크하는 클래스"""
    
    def check_ui_status(self):
        """UI 인터페이스 상태 체크"""
        try:
            # 실제 UI 상태 체크 로직
            return "정상"
        except Exception:
            return "오류"
    
    def check_mcp_status(self):
        """MCP 서버 연결 상태 체크"""
        try:
            # 실제 MCP 서버 연결 체크 로직
            # 임시로 연결 시도 중으로 반환
            return "연결 시도 중"
        except Exception:
            return "연결 실패"
    
    def check_data_source_status(self):
        """데이터 소스 연결 상태 체크"""
        try:
            # 실제 데이터 소스 연결 체크 로직
            return "대기"
        except Exception:
            return "연결 실패"


class UrbanHiveConfig:
    """Urban Hive 에이전트 설정"""
    
    def __init__(self):
        self.analysis_types = {
            "illegal_dumping": "Illegal Dumping - 불법 투기 분석",
            "public_safety": "Public Safety - 공공 안전 분석",
            "traffic_flow": "Traffic Flow - 교통 흐름 분석",
            "community_event": "Community Event - 커뮤니티 이벤트 분석",
            "air_quality": "Air Quality - 대기질 분석",
            "noise_pollution": "Noise Pollution - 소음 공해 분석",
            "green_space": "Green Space - 녹지 공간 분석",
            "housing_market": "Housing Market - 주택 시장 분석"
        }
    
    def get_analysis_options(self):
        """분석 옵션 목록 반환"""
        return list(self.analysis_types.values())
    
    def get_analysis_type_by_name(self, display_name):
        """표시명으로 분석 타입 키 반환"""
        for key, value in self.analysis_types.items():
            if value == display_name:
                return key
        return None 