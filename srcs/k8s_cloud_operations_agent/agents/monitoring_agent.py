"""
모니터링 Agent
==============

시스템 상태 및 성능을 모니터링하고 예측적 분석을 수행하는 Agent
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class MetricData:
    """메트릭 데이터"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str]

@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    condition: str
    threshold: float
    duration: str
    severity: str  # info, warning, critical
    enabled: bool = True

@dataclass
class Alert:
    """알림"""
    id: str
    rule_name: str
    message: str
    severity: str
    timestamp: datetime
    status: str  # firing, resolved
    labels: Dict[str, str]

@dataclass
class PerformanceAnalysis:
    """성능 분석 결과"""
    resource_usage: Dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]
    predicted_scaling_needs: Dict[str, Any]
    health_score: float

class MonitoringAgent:
    """mcp_agent 기반 모니터링 Agent"""
    
    def __init__(self, output_dir: str = "monitoring_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # mcp_agent App 초기화 (설정 파일 없이 동적 생성)
        self.app = MCPApp(
            name="monitoring_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="monitoring_agent",
            instruction="시스템 상태 및 성능을 모니터링하고 예측적 분석을 수행하는 Agent입니다.",
            server_names=["monitoring-mcp", "k8s-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        # 모니터링 상태
        self.active_alerts: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        self.metric_history: Dict[str, List[MetricData]] = {}
        
        # 기본 알림 규칙 설정
        self._setup_default_alert_rules()
        
    def _setup_default_alert_rules(self):
        """기본 알림 규칙 설정"""
        self.alert_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_usage > 80%",
                threshold=80.0,
                duration="5m",
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage > 85%",
                threshold=85.0,
                duration="5m",
                severity="warning"
            ),
            AlertRule(
                name="pod_restart_frequency",
                condition="pod_restarts > 5",
                threshold=5.0,
                duration="10m",
                severity="critical"
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 5%",
                threshold=5.0,
                duration="2m",
                severity="critical"
            ),
            AlertRule(
                name="disk_usage_high",
                condition="disk_usage > 90%",
                threshold=90.0,
                duration="5m",
                severity="warning"
            )
        ]
    
    async def start_monitoring(self, namespace: str = "default"):
        """모니터링 시작"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info(f"Starting monitoring for namespace: {namespace}")
                
                # 1. 초기 시스템 상태 확인
                await self._check_initial_system_state(namespace, context)
                
                # 2. 메트릭 수집 시작
                await self._start_metric_collection(namespace, context)
                
                # 3. 알림 모니터링 시작
                await self._start_alert_monitoring(context)
                
                # 4. 예측적 분석 시작
                await self._start_predictive_analysis(context)
                
                logger.info("Monitoring started successfully")
                
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise
    
    async def _check_initial_system_state(self, namespace: str, context):
        """초기 시스템 상태 확인"""
        logger = context.logger
        
        # 클러스터 상태 확인
        cluster_health = await self._get_cluster_health(context)
        logger.info(f"Cluster health: {cluster_health}")
        
        # 노드 상태 확인
        nodes = await self._get_node_status(context)
        logger.info(f"Node count: {len(nodes)}")
        
        # 네임스페이스 리소스 확인
        namespace_resources = await self._get_namespace_resources(namespace, context)
        logger.info(f"Namespace resources: {namespace_resources}")
        
        # 초기 메트릭 수집
        initial_metrics = await self._collect_initial_metrics(namespace, context)
        self.metric_history["initial"] = initial_metrics
        
        logger.info("Initial system state check completed")
    
    async def _start_metric_collection(self, namespace: str, context):
        """메트릭 수집 시작"""
        logger = context.logger
        
        # 주기적 메트릭 수집 태스크 시작
        asyncio.create_task(self._periodic_metric_collection(namespace, context))
        
        logger.info("Metric collection started")
    
    async def _periodic_metric_collection(self, namespace: str, context):
        """주기적 메트릭 수집"""
        logger = context.logger
        
        while True:
            try:
                # CPU 메트릭 수집
                cpu_metrics = await self._collect_cpu_metrics(namespace, context)
                self._store_metrics("cpu", cpu_metrics)
                
                # 메모리 메트릭 수집
                memory_metrics = await self._collect_memory_metrics(namespace, context)
                self._store_metrics("memory", memory_metrics)
                
                # 네트워크 메트릭 수집
                network_metrics = await self._collect_network_metrics(namespace, context)
                self._store_metrics("network", network_metrics)
                
                # 디스크 메트릭 수집
                disk_metrics = await self._collect_disk_metrics(namespace, context)
                self._store_metrics("disk", disk_metrics)
                
                # 애플리케이션 메트릭 수집
                app_metrics = await self._collect_application_metrics(namespace, context)
                self._store_metrics("application", app_metrics)
                
                logger.debug("Periodic metric collection completed")
                
                # 30초 대기
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(60)  # 에러 시 1분 대기
    
    async def _start_alert_monitoring(self, context):
        """알림 모니터링 시작"""
        logger = context.logger
        
        # 주기적 알림 체크 태스크 시작
        asyncio.create_task(self._periodic_alert_check(context))
        
        logger.info("Alert monitoring started")
    
    async def _periodic_alert_check(self, context):
        """주기적 알림 체크"""
        logger = context.logger
        
        while True:
            try:
                # 각 알림 규칙에 대해 체크
                for rule in self.alert_rules:
                    if rule.enabled:
                        await self._check_alert_rule(rule, context)
                
                # 해결된 알림 처리
                await self._process_resolved_alerts(context)
                
                logger.debug("Periodic alert check completed")
                
                # 10초 대기
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Alert check error: {e}")
                await asyncio.sleep(30)
    
    async def _start_predictive_analysis(self, context):
        """예측적 분석 시작"""
        logger = context.logger
        
        # 주기적 예측 분석 태스크 시작
        asyncio.create_task(self._periodic_predictive_analysis(context))
        
        logger.info("Predictive analysis started")
    
    async def _periodic_predictive_analysis(self, context):
        """주기적 예측 분석"""
        logger = context.logger
        
        while True:
            try:
                # 성능 분석 수행
                analysis = await self._perform_performance_analysis(context)
                
                # 스케일링 예측
                scaling_prediction = await self._predict_scaling_needs(context)
                
                # 분석 결과 저장
                await self._save_analysis_results(analysis, scaling_prediction)
                
                logger.debug("Predictive analysis completed")
                
                # 5분 대기
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Predictive analysis error: {e}")
                await asyncio.sleep(600)  # 에러 시 10분 대기
    
    async def _check_alert_rule(self, rule: AlertRule, context):
        """알림 규칙 체크"""
        try:
            # 메트릭 값 조회
            metric_value = await self._get_metric_value(rule.name, context)
            
            # 임계값 체크
            if metric_value > rule.threshold:
                # 새로운 알림 생성
                alert = Alert(
                    id=f"{rule.name}_{int(datetime.now().timestamp())}",
                    rule_name=rule.name,
                    message=f"{rule.name}: {metric_value} > {rule.threshold}",
                    severity=rule.severity,
                    timestamp=datetime.now(),
                    status="firing",
                    labels={"rule": rule.name, "threshold": str(rule.threshold)}
                )
                
                # 중복 알림 체크
                if not self._is_duplicate_alert(alert):
                    self.active_alerts.append(alert)
                    await self._send_alert_notification(alert, context)
                    
        except Exception as e:
            context.logger.error(f"Alert rule check failed for {rule.name}: {e}")
    
    async def _process_resolved_alerts(self, context):
        """해결된 알림 처리"""
        current_time = datetime.now()
        resolved_alerts = []
        
        for alert in self.active_alerts:
            if alert.status == "firing":
                # 알림이 해결되었는지 체크
                metric_value = await self._get_metric_value(alert.rule_name, context)
                rule = self._get_alert_rule(alert.rule_name)
                
                if rule and metric_value <= rule.threshold:
                    alert.status = "resolved"
                    resolved_alerts.append(alert)
                    await self._send_resolution_notification(alert, context)
        
        # 해결된 알림을 활성 알림 목록에서 제거
        self.active_alerts = [alert for alert in self.active_alerts if alert.status == "firing"]
    
    async def _perform_performance_analysis(self, context) -> PerformanceAnalysis:
        """성능 분석 수행"""
        logger = context.logger
        
        # 리소스 사용량 분석
        resource_usage = await self._analyze_resource_usage(context)
        
        # 병목 지점 분석
        bottlenecks = await self._identify_bottlenecks(context)
        
        # 최적화 권장사항 생성
        recommendations = await self._generate_recommendations(resource_usage, bottlenecks, context)
        
        # 헬스 스코어 계산
        health_score = await self._calculate_health_score(resource_usage, context)
        
        analysis = PerformanceAnalysis(
            resource_usage=resource_usage,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            predicted_scaling_needs={},  # 별도 메서드에서 설정
            health_score=health_score
        )
        
        logger.info(f"Performance analysis completed. Health score: {health_score}")
        return analysis
    
    async def _predict_scaling_needs(self, context) -> Dict[str, Any]:
        """스케일링 요구사항 예측"""
        logger = context.logger
        
        # CPU 사용량 트렌드 분석
        cpu_trend = await self._analyze_cpu_trend(context)
        
        # 메모리 사용량 트렌드 분석
        memory_trend = await self._analyze_memory_trend(context)
        
        # 트래픽 패턴 분석
        traffic_pattern = await self._analyze_traffic_pattern(context)
        
        # 예측 모델 적용
        scaling_prediction = {
            "cpu_scaling_needed": cpu_trend.get("trend") == "increasing",
            "memory_scaling_needed": memory_trend.get("trend") == "increasing",
            "predicted_cpu_usage_1h": cpu_trend.get("prediction_1h", 0),
            "predicted_memory_usage_1h": memory_trend.get("prediction_1h", 0),
            "recommended_replicas": await self._calculate_recommended_replicas(context),
            "confidence": 0.85
        }
        
        logger.info(f"Scaling prediction: {scaling_prediction}")
        return scaling_prediction
    
    def _store_metrics(self, metric_type: str, metrics: List[MetricData]):
        """메트릭 저장"""
        if metric_type not in self.metric_history:
            self.metric_history[metric_type] = []
        
        self.metric_history[metric_type].extend(metrics)
        
        # 메트릭 히스토리 크기 제한 (최근 1000개만 유지)
        if len(self.metric_history[metric_type]) > 1000:
            self.metric_history[metric_type] = self.metric_history[metric_type][-1000:]
    
    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """중복 알림 체크"""
        for active_alert in self.active_alerts:
            if (active_alert.rule_name == alert.rule_name and 
                active_alert.status == "firing"):
                return True
        return False
    
    def _get_alert_rule(self, rule_name: str) -> Optional[AlertRule]:
        """알림 규칙 조회"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                return rule
        return None
    
    async def _save_analysis_results(self, analysis: PerformanceAnalysis, scaling_prediction: Dict[str, Any]):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"performance_analysis_{timestamp}.json")
        
        report_data = {
            "timestamp": timestamp,
            "performance_analysis": asdict(analysis),
            "scaling_prediction": scaling_prediction,
            "active_alerts": [asdict(alert) for alert in self.active_alerts]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Performance analysis report saved to: {report_file}")
    
    # 메트릭 수집 메서드들 (시뮬레이션)
    async def _get_cluster_health(self, context) -> Dict[str, Any]:
        """클러스터 헬스 조회"""
        return {"status": "healthy", "nodes": 5, "pods": 150}
    
    async def _get_node_status(self, context) -> List[Dict[str, Any]]:
        """노드 상태 조회"""
        return [
            {"name": "node-1", "status": "Ready", "cpu": "45%", "memory": "60%"},
            {"name": "node-2", "status": "Ready", "cpu": "52%", "memory": "65%"},
            {"name": "node-3", "status": "Ready", "cpu": "38%", "memory": "55%"}
        ]
    
    async def _get_namespace_resources(self, namespace: str, context) -> Dict[str, Any]:
        """네임스페이스 리소스 조회"""
        return {
            "pods": 25,
            "services": 8,
            "deployments": 12,
            "configmaps": 15,
            "secrets": 10
        }
    
    async def _collect_initial_metrics(self, namespace: str, context) -> List[MetricData]:
        """초기 메트릭 수집"""
        return [
            MetricData("cluster_cpu_usage", 45.2, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("cluster_memory_usage", 62.8, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("pod_count", 25, "count", datetime.now(), {"namespace": namespace})
        ]
    
    async def _collect_cpu_metrics(self, namespace: str, context) -> List[MetricData]:
        """CPU 메트릭 수집"""
        return [
            MetricData("cpu_usage", 45.2, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("cpu_requests", 80.5, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("cpu_limits", 65.3, "percent", datetime.now(), {"namespace": namespace})
        ]
    
    async def _collect_memory_metrics(self, namespace: str, context) -> List[MetricData]:
        """메모리 메트릭 수집"""
        return [
            MetricData("memory_usage", 62.8, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("memory_requests", 85.2, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("memory_limits", 70.1, "percent", datetime.now(), {"namespace": namespace})
        ]
    
    async def _collect_network_metrics(self, namespace: str, context) -> List[MetricData]:
        """네트워크 메트릭 수집"""
        return [
            MetricData("network_rx_bytes", 1024.5, "bytes", datetime.now(), {"namespace": namespace}),
            MetricData("network_tx_bytes", 2048.3, "bytes", datetime.now(), {"namespace": namespace}),
            MetricData("network_errors", 0.1, "count", datetime.now(), {"namespace": namespace})
        ]
    
    async def _collect_disk_metrics(self, namespace: str, context) -> List[MetricData]:
        """디스크 메트릭 수집"""
        return [
            MetricData("disk_usage", 75.2, "percent", datetime.now(), {"namespace": namespace}),
            MetricData("disk_read_bytes", 512.8, "bytes", datetime.now(), {"namespace": namespace}),
            MetricData("disk_write_bytes", 1024.6, "bytes", datetime.now(), {"namespace": namespace})
        ]
    
    async def _collect_application_metrics(self, namespace: str, context) -> List[MetricData]:
        """애플리케이션 메트릭 수집"""
        return [
            MetricData("http_requests_total", 1250, "count", datetime.now(), {"namespace": namespace}),
            MetricData("http_request_duration", 150.5, "milliseconds", datetime.now(), {"namespace": namespace}),
            MetricData("error_rate", 2.1, "percent", datetime.now(), {"namespace": namespace})
        ]
    
    async def _get_metric_value(self, metric_name: str, context) -> float:
        """메트릭 값 조회"""
        # 실제로는 Prometheus API 호출
        metric_values = {
            "high_cpu_usage": 45.2,
            "high_memory_usage": 62.8,
            "pod_restart_frequency": 2.0,
            "high_error_rate": 2.1,
            "disk_usage_high": 75.2
        }
        return metric_values.get(metric_name, 0.0)
    
    async def _send_alert_notification(self, alert: Alert, context):
        """알림 전송"""
        logger = context.logger
        logger.warning(f"ALERT: {alert.message} (Severity: {alert.severity})")
        
        # 실제로는 Slack, 이메일, PagerDuty 등으로 전송
        print(f"🚨 ALERT: {alert.message}")
    
    async def _send_resolution_notification(self, alert: Alert, context):
        """해결 알림 전송"""
        logger = context.logger
        logger.info(f"RESOLVED: {alert.rule_name}")
        
        # 실제로는 Slack, 이메일 등으로 전송
        print(f"✅ RESOLVED: {alert.rule_name}")
    
    async def _analyze_resource_usage(self, context) -> Dict[str, float]:
        """리소스 사용량 분석"""
        return {
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "disk_usage": 75.2,
            "network_usage": 35.6
        }
    
    async def _identify_bottlenecks(self, context) -> List[str]:
        """병목 지점 식별"""
        bottlenecks = []
        
        # CPU 병목 체크
        if await self._get_metric_value("high_cpu_usage", context) > 80:
            bottlenecks.append("High CPU usage detected")
        
        # 메모리 병목 체크
        if await self._get_metric_value("high_memory_usage", context) > 85:
            bottlenecks.append("High memory usage detected")
        
        # 디스크 병목 체크
        if await self._get_metric_value("disk_usage_high", context) > 90:
            bottlenecks.append("High disk usage detected")
        
        return bottlenecks
    
    async def _generate_recommendations(self, resource_usage: Dict[str, float], bottlenecks: List[str], context) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        if resource_usage["cpu_usage"] > 70:
            recommendations.append("Consider scaling up CPU resources or adding more replicas")
        
        if resource_usage["memory_usage"] > 75:
            recommendations.append("Consider increasing memory limits or optimizing memory usage")
        
        if resource_usage["disk_usage"] > 80:
            recommendations.append("Consider cleaning up unused data or expanding storage")
        
        if "High CPU usage detected" in bottlenecks:
            recommendations.append("Implement horizontal pod autoscaling for CPU-based scaling")
        
        return recommendations
    
    async def _calculate_health_score(self, resource_usage: Dict[str, float], context) -> float:
        """헬스 스코어 계산"""
        # 각 리소스 사용률을 기반으로 점수 계산
        cpu_score = max(0, 100 - resource_usage["cpu_usage"])
        memory_score = max(0, 100 - resource_usage["memory_usage"])
        disk_score = max(0, 100 - resource_usage["disk_usage"])
        
        # 가중 평균 계산
        health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        
        return round(health_score, 2)
    
    async def _analyze_cpu_trend(self, context) -> Dict[str, Any]:
        """CPU 트렌드 분석"""
        return {
            "trend": "stable",
            "prediction_1h": 48.5,
            "prediction_6h": 52.3,
            "confidence": 0.85
        }
    
    async def _analyze_memory_trend(self, context) -> Dict[str, Any]:
        """메모리 트렌드 분석"""
        return {
            "trend": "increasing",
            "prediction_1h": 68.2,
            "prediction_6h": 75.8,
            "confidence": 0.78
        }
    
    async def _analyze_traffic_pattern(self, context) -> Dict[str, Any]:
        """트래픽 패턴 분석"""
        return {
            "pattern": "normal",
            "peak_hours": ["09:00", "14:00", "18:00"],
            "traffic_spike": False,
            "anomaly_detected": False
        }
    
    async def _calculate_recommended_replicas(self, context) -> int:
        """권장 레플리카 수 계산"""
        current_cpu = await self._get_metric_value("high_cpu_usage", context)
        current_memory = await self._get_metric_value("high_memory_usage", context)
        
        # 간단한 로직: 리소스 사용률이 높으면 레플리카 증가
        if current_cpu > 80 or current_memory > 85:
            return 5
        elif current_cpu > 60 or current_memory > 70:
            return 4
        else:
            return 3
    
    async def get_active_alerts(self) -> List[Alert]:
        """활성 알림 조회"""
        return self.active_alerts
    
    async def get_metric_history(self, metric_type: str) -> List[MetricData]:
        """메트릭 히스토리 조회"""
        return self.metric_history.get(metric_type, [])
    
    async def add_alert_rule(self, rule: AlertRule):
        """알림 규칙 추가"""
        self.alert_rules.append(rule)
    
    async def remove_alert_rule(self, rule_name: str):
        """알림 규칙 제거"""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]

# 사용 예시
async def main():
    """사용 예시"""
    agent = MonitoringAgent()
    
    # 모니터링 시작
    await agent.start_monitoring("default")
    
    # 30초 동안 모니터링 실행
    await asyncio.sleep(30)
    
    # 활성 알림 조회
    alerts = await agent.get_active_alerts()
    print(f"Active alerts: {len(alerts)}")
    
    # 메트릭 히스토리 조회
    cpu_metrics = await agent.get_metric_history("cpu")
    print(f"CPU metrics collected: {len(cpu_metrics)}")

if __name__ == "__main__":
    asyncio.run(main()) 