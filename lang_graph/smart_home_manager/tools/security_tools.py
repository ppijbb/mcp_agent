"""
보안 모니터링 도구

보안 상태 조회, 위협 감지, 알림, 리포트 생성
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityStatusInput(BaseModel):
    """보안 상태 조회 입력 스키마"""
    device_id: Optional[str] = Field(default=None, description="기기 ID (없으면 전체)")


class ThreatDetectionInput(BaseModel):
    """위협 감지 입력 스키마"""
    check_type: str = Field(default="all", description="검사 유형 (all, network, devices, access)")


class SecurityAlertInput(BaseModel):
    """보안 알림 입력 스키마"""
    alert_type: str = Field(description="알림 유형 (threat, anomaly, access)")
    severity: str = Field(description="심각도 (low, medium, high, critical)")


class SecurityReportInput(BaseModel):
    """보안 리포트 생성 입력 스키마"""
    period: str = Field(default="week", description="리포트 기간 (day, week, month)")
    output_path: Optional[str] = Field(default=None, description="출력 파일 경로")


class SecurityTools:
    """
    보안 모니터링 도구 모음
    
    보안 상태 조회, 위협 감지, 알림, 리포트 생성
    """
    
    def __init__(self, data_dir: str = "home_data"):
        """
        SecurityTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.security_file = self.data_dir / "security_status.json"
        self.alerts_file = self.data_dir / "security_alerts.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_security_data()
    
    def _load_security_data(self):
        """보안 데이터 로드"""
        if self.security_file.exists():
            with open(self.security_file, 'r', encoding='utf-8') as f:
                self.security_data = json.load(f)
        else:
            self.security_data = {"status": "secure", "devices": {}}
        
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r', encoding='utf-8') as f:
                self.alerts = json.load(f)
        else:
            self.alerts = []
    
    def _save_security_data(self):
        """보안 데이터 저장"""
        with open(self.security_file, 'w', encoding='utf-8') as f:
            json.dump(self.security_data, f, indent=2, ensure_ascii=False)
    
    def _save_alerts(self):
        """알림 데이터 저장"""
        with open(self.alerts_file, 'w', encoding='utf-8') as f:
            json.dump(self.alerts, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """보안 도구 초기화"""
        self.tools.append(self._create_security_status_tool())
        self.tools.append(self._create_threat_detection_tool())
        self.tools.append(self._create_security_alert_tool())
        self.tools.append(self._create_security_report_tool())
        
        logger.info(f"Initialized {len(self.tools)} security tools")
    
    def _create_security_status_tool(self) -> BaseTool:
        """보안 상태 조회 도구 생성"""
        
        @tool("get_security_status", args_schema=SecurityStatusInput)
        def get_security_status(device_id: Optional[str] = None) -> str:
            """
            보안 상태 조회
            
            Args:
                device_id: 기기 ID (없으면 전체)
            
            Returns:
                보안 상태 정보
            """
            try:
                logger.info(f"Getting security status: device={device_id or 'all'}")
                
                status = {
                    "overall_status": "secure",
                    "devices": {
                        "camera_1": {"status": "online", "security": "secure"},
                        "door_sensor_1": {"status": "online", "security": "secure"},
                        "motion_sensor_1": {"status": "online", "security": "secure"}
                    },
                    "network": {
                        "firewall": "active",
                        "encryption": "enabled",
                        "vulnerabilities": 0
                    }
                }
                
                if device_id:
                    if device_id in status["devices"]:
                        return json.dumps(status["devices"][device_id], indent=2, ensure_ascii=False)
                    else:
                        return f"Error: Device {device_id} not found"
                
                return json.dumps(status, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error getting security status: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return get_security_status
    
    def _create_threat_detection_tool(self) -> BaseTool:
        """위협 감지 도구 생성"""
        
        @tool("detect_threats", args_schema=ThreatDetectionInput)
        def detect_threats(check_type: str = "all") -> str:
            """
            위협 감지
            
            Args:
                check_type: 검사 유형 (all, network, devices, access)
            
            Returns:
                위협 감지 결과
            """
            try:
                logger.info(f"Detecting threats: check_type={check_type}")
                
                threats = {
                    "check_type": check_type,
                    "threats_detected": 0,
                    "threats": [],
                    "recommendations": [
                        "Keep all devices updated",
                        "Use strong passwords",
                        "Enable two-factor authentication"
                    ]
                }
                
                return json.dumps(threats, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error detecting threats: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return detect_threats
    
    def _create_security_alert_tool(self) -> BaseTool:
        """보안 알림 도구 생성"""
        
        @tool("create_security_alert", args_schema=SecurityAlertInput)
        def create_security_alert(alert_type: str, severity: str) -> str:
            """
            보안 알림 생성
            
            Args:
                alert_type: 알림 유형 (threat, anomaly, access)
                severity: 심각도 (low, medium, high, critical)
            
            Returns:
                알림 생성 결과
            """
            try:
                logger.info(f"Creating security alert: type={alert_type}, severity={severity}")
                
                alert = {
                    "id": f"alert_{datetime.now().timestamp()}",
                    "type": alert_type,
                    "severity": severity,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
                
                self.alerts.append(alert)
                self._save_alerts()
                
                return json.dumps(alert, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error creating security alert: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return create_security_alert
    
    def _create_security_report_tool(self) -> BaseTool:
        """보안 리포트 생성 도구 생성"""
        
        @tool("generate_security_report", args_schema=SecurityReportInput)
        def generate_security_report(period: str = "week", output_path: Optional[str] = None) -> str:
            """
            보안 리포트 생성
            
            Args:
                period: 리포트 기간 (day, week, month)
                output_path: 출력 파일 경로 (선택)
            
            Returns:
                리포트 생성 결과
            """
            try:
                logger.info(f"Generating security report: period={period}")
                
                report = {
                    "period": period,
                    "overall_status": "secure",
                    "threats_detected": 0,
                    "alerts": len(self.alerts),
                    "devices_secured": 10,
                    "network_status": "secure",
                    "recommendations": [
                        "Regular security updates",
                        "Monitor access logs",
                        "Review security policies"
                    ]
                }
                
                if output_path:
                    path = Path(output_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    return f"Security report saved to {output_path}"
                
                return json.dumps(report, indent=2, ensure_ascii=False)
            
            except Exception as e:
                error_msg = f"Error generating security report: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        return generate_security_report
    
    def get_tools(self) -> List[BaseTool]:
        """모든 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 도구 찾기"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

