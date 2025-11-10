"""
State Management for LangGraph Home Chain

LangGraph StateGraph에서 사용할 상태 정의
"""

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime


class HomeState(TypedDict):
    """
    스마트 홈 매니저 워크플로우 상태
    
    LangGraph StateGraph에서 사용하는 상태 구조
    """
    # 사용자 및 홈 정보
    user_id: str
    home_id: str
    
    # 기기 정보
    devices: List[Dict[str, Any]]
    device_status: Dict[str, Any]
    
    # 에너지 정보
    energy_usage: Dict[str, Any]
    energy_optimization: Dict[str, Any]
    
    # 보안 정보
    security_status: Dict[str, Any]
    security_alerts: List[Dict[str, Any]]
    
    # 유지보수 정보
    maintenance_schedule: List[Dict[str, Any]]
    
    # 자동화 정보
    automation_scenarios: List[Dict[str, Any]]
    
    # 메타데이터
    timestamp: str
    workflow_stage: str  # "device_management", "energy_optimization", "security_monitoring", "maintenance_alerts", "automation_scenarios", "final"
    errors: List[str]
    warnings: List[str]

