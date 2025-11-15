"""
반려동물 건강 관리 도구

건강 모니터링, 이상 행동 감지, 건강 데이터 분석
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthDataInput(BaseModel):
    """건강 데이터 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    metric_type: str = Field(description="메트릭 타입 (weight, temperature, activity_level, etc.)")
    value: float = Field(description="값")
    unit: Optional[str] = Field(default=None, description="단위")


class AnomalyDetectionInput(BaseModel):
    """이상 행동 감지 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    behavior_data: Dict[str, Any] = Field(description="행동 데이터")


class HealthAnalysisInput(BaseModel):
    """건강 분석 입력 스키마"""
    pet_id: str = Field(description="반려동물 ID")
    period_days: Optional[int] = Field(default=30, description="분석 기간 (일)")


class HealthTools:
    """
    반려동물 건강 관리 도구 모음
    
    건강 모니터링, 이상 행동 감지, 건강 데이터 분석
    """
    
    def __init__(self, data_dir: str = "petcare_data"):
        """
        HealthTools 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.health_data_file = self.data_dir / "health_data.json"
        self.anomalies_file = self.data_dir / "anomalies.json"
        self.tools: List[BaseTool] = []
        self._initialize_tools()
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        if self.health_data_file.exists():
            with open(self.health_data_file, 'r', encoding='utf-8') as f:
                self.health_data = json.load(f)
        else:
            self.health_data = {}
        
        if self.anomalies_file.exists():
            with open(self.anomalies_file, 'r', encoding='utf-8') as f:
                self.anomalies = json.load(f)
        else:
            self.anomalies = {}
    
    def _save_health_data(self):
        """건강 데이터 저장"""
        with open(self.health_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.health_data, f, indent=2, ensure_ascii=False)
    
    def _save_anomalies(self):
        """이상 행동 저장"""
        with open(self.anomalies_file, 'w', encoding='utf-8') as f:
            json.dump(self.anomalies, f, indent=2, ensure_ascii=False)
    
    def _initialize_tools(self):
        """건강 도구 초기화"""
        self.tools.append(self._create_record_health_data_tool())
        self.tools.append(self._create_detect_anomaly_tool())
        self.tools.append(self._create_analyze_health_tool())
        self.tools.append(self._create_get_health_summary_tool())
        logger.info(f"Initialized {len(self.tools)} health tools")
    
    def _create_record_health_data_tool(self) -> BaseTool:
        @tool("health_record_data", args_schema=HealthDataInput)
        def record_health_data(pet_id: str, metric_type: str, value: float, unit: Optional[str] = None) -> str:
            """
            반려동물 건강 데이터를 기록합니다.
            Args:
                pet_id: 반려동물 ID
                metric_type: 메트릭 타입
                value: 값
                unit: 단위
            Returns:
                기록 결과 메시지
            """
            logger.info(f"Recording health data for pet '{pet_id}': {metric_type}={value} {unit or ''}")
            if pet_id not in self.health_data:
                self.health_data[pet_id] = []
            
            health_record = {
                "pet_id": pet_id,
                "metric_type": metric_type,
                "value": value,
                "unit": unit,
                "timestamp": datetime.now().isoformat(),
            }
            self.health_data[pet_id].append(health_record)
            self._save_health_data()
            return f"Health data recorded for pet '{pet_id}': {metric_type}={value} {unit or ''}"
        return record_health_data
    
    def _create_detect_anomaly_tool(self) -> BaseTool:
        @tool("health_detect_anomaly", args_schema=AnomalyDetectionInput)
        def detect_anomaly(pet_id: str, behavior_data: Dict[str, Any]) -> str:
            """
            반려동물의 이상 행동을 감지합니다.
            Args:
                pet_id: 반려동물 ID
                behavior_data: 행동 데이터
            Returns:
                이상 행동 감지 결과 (JSON 문자열)
            """
            logger.info(f"Detecting anomalies for pet '{pet_id}'")
            # 실제 구현에서는 ML 모델 또는 규칙 기반 이상 감지
            # 여기서는 기본적인 패턴 매칭
            anomaly_detected = False
            anomaly_type = None
            severity = "low"
            
            # 예시: 활동량이 평균보다 50% 이상 낮으면 이상
            if "activity_level" in behavior_data:
                activity = behavior_data.get("activity_level", 0)
                if activity < 0.5:  # 임계값
                    anomaly_detected = True
                    anomaly_type = "low_activity"
                    severity = "medium"
            
            result = {
                "pet_id": pet_id,
                "anomaly_detected": anomaly_detected,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "timestamp": datetime.now().isoformat(),
                "behavior_data": behavior_data,
            }
            
            if anomaly_detected:
                if pet_id not in self.anomalies:
                    self.anomalies[pet_id] = []
                self.anomalies[pet_id].append(result)
                self._save_anomalies()
            
            return json.dumps(result, ensure_ascii=False, indent=2)
        return detect_anomaly
    
    def _create_analyze_health_tool(self) -> BaseTool:
        @tool("health_analyze", args_schema=HealthAnalysisInput)
        def analyze_health(pet_id: str, period_days: Optional[int] = 30) -> str:
            """
            반려동물 건강 데이터를 분석합니다.
            Args:
                pet_id: 반려동물 ID
                period_days: 분석 기간 (일)
            Returns:
                건강 분석 결과 (JSON 문자열)
            """
            logger.info(f"Analyzing health for pet '{pet_id}' over {period_days} days")
            if pet_id not in self.health_data:
                return f"Error: No health data found for pet '{pet_id}'"
            
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_data = [
                record for record in self.health_data[pet_id]
                if datetime.fromisoformat(record["timestamp"]) >= cutoff_date
            ]
            
            if not recent_data:
                return f"Error: No health data found for pet '{pet_id}' in the last {period_days} days"
            
            # 기본 통계 계산
            metrics = {}
            for record in recent_data:
                metric_type = record["metric_type"]
                if metric_type not in metrics:
                    metrics[metric_type] = []
                metrics[metric_type].append(record["value"])
            
            analysis = {
                "pet_id": pet_id,
                "period_days": period_days,
                "metrics": {
                    metric_type: {
                        "count": len(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }
                    for metric_type, values in metrics.items()
                },
                "timestamp": datetime.now().isoformat(),
            }
            
            return json.dumps(analysis, ensure_ascii=False, indent=2)
        return analyze_health
    
    def _create_get_health_summary_tool(self) -> BaseTool:
        @tool("health_get_summary", args_schema=HealthAnalysisInput)
        def get_health_summary(pet_id: str, period_days: Optional[int] = 30) -> str:
            """
            반려동물 건강 요약을 조회합니다.
            Args:
                pet_id: 반려동물 ID
                period_days: 기간 (일)
            Returns:
                건강 요약 (JSON 문자열)
            """
            logger.info(f"Getting health summary for pet '{pet_id}'")
            summary = {
                "pet_id": pet_id,
                "overall_status": "healthy",
                "recent_anomalies": len(self.anomalies.get(pet_id, [])) if pet_id in self.anomalies else 0,
                "last_check": datetime.now().isoformat(),
            }
            return json.dumps(summary, ensure_ascii=False, indent=2)
        return get_health_summary
    
    def get_tools(self) -> List[BaseTool]:
        """모든 건강 도구 반환"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """이름으로 건강 도구 찾기"""
        for tool_item in self.tools:
            if tool_item.name == name:
                return tool_item
        return None

