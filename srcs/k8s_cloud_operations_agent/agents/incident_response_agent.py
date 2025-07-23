"""
장애 대응 Agent
==============

장애 감지 및 자동 복구를 담당하는 Agent
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class Incident:
    """장애 사고"""
    id: str
    type: str  # pod_failure, service_unavailable, resource_exhaustion, etc.
    severity: str  # low, medium, high, critical
    status: str  # open, investigating, resolved, closed
    description: str
    affected_resources: List[str]
    timestamp: datetime
    resolution_time: Optional[datetime] = None
    auto_resolved: bool = False

@dataclass
class RecoveryAction:
    """복구 액션"""
    name: str
    type: str  # restart, scale, rollback, etc.
    target_resource: str
    parameters: Dict[str, Any]
    success: bool = False
    execution_time: Optional[datetime] = None

@dataclass
class IncidentReport:
    """장애 보고서"""
    incident_id: str
    summary: str
    root_cause: str
    impact_assessment: str
    recovery_actions: List[RecoveryAction]
    lessons_learned: List[str]
    prevention_measures: List[str]

class IncidentResponseAgent:
    """mcp_agent 기반 장애 대응 Agent"""
    
    def __init__(self, output_dir: str = "incident_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="incident_response",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="incident_response",
            instruction="장애 감지 및 자동 복구를 담당하는 전문 Agent입니다.",
            server_names=["monitoring-mcp", "k8s-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        # 장애 상태
        self.active_incidents: List[Incident] = []
        self.incident_history: List[Incident] = []
        self.recovery_actions: List[RecoveryAction] = []
        
        # 장애 패턴 및 임계값
        self.failure_patterns = self._setup_failure_patterns()
        self.auto_recovery_enabled = True
        
    def _setup_failure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """장애 패턴 설정"""
        return {
            "pod_failure": {
                "detection": "pod_status == 'Failed'",
                "severity": "medium",
                "auto_recovery": True,
                "recovery_action": "restart_pod"
            },
            "service_unavailable": {
                "detection": "service_endpoints == 0",
                "severity": "high",
                "auto_recovery": True,
                "recovery_action": "restart_service"
            },
            "resource_exhaustion": {
                "detection": "cpu_usage > 95% OR memory_usage > 95%",
                "severity": "high",
                "auto_recovery": True,
                "recovery_action": "scale_up"
            },
            "network_connectivity": {
                "detection": "network_errors > threshold",
                "severity": "medium",
                "auto_recovery": True,
                "recovery_action": "restart_network_policy"
            },
            "database_connection": {
                "detection": "db_connection_failures > 0",
                "severity": "critical",
                "auto_recovery": True,
                "recovery_action": "restart_database"
            }
        }
    
    async def start_incident_monitoring(self):
        """장애 모니터링 시작"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("Starting incident monitoring")
                
                # 1. 시스템 상태 초기 확인
                await self._check_initial_system_state(context)
                
                # 2. 실시간 장애 감지 시작
                await self._start_real_time_failure_detection(context)
                
                # 3. 예측적 장애 방지 시작
                await self._start_predictive_failure_prevention(context)
                
                logger.info("Incident monitoring started successfully")
                
        except Exception as e:
            logger.error(f"Failed to start incident monitoring: {e}")
            raise
    
    async def _check_initial_system_state(self, context):
        """시스템 상태 초기 확인"""
        logger = context.logger
        
        # 클러스터 상태 확인
        cluster_health = await self._get_cluster_health(context)
        logger.info(f"Initial cluster health: {cluster_health}")
        
        # 노드 상태 확인
        node_health = await self._get_node_health(context)
        logger.info(f"Node health status: {node_health}")
        
        # 서비스 상태 확인
        service_health = await self._get_service_health(context)
        logger.info(f"Service health status: {service_health}")
        
        # 기존 장애 확인
        existing_incidents = await self._check_existing_incidents(context)
        if existing_incidents:
            logger.warning(f"Found {len(existing_incidents)} existing incidents")
    
    async def _start_real_time_failure_detection(self, context):
        """실시간 장애 감지 시작"""
        logger = context.logger
        
        # 주기적 장애 감지 태스크 시작
        asyncio.create_task(self._periodic_failure_detection(context))
        
        logger.info("Real-time failure detection started")
    
    async def _periodic_failure_detection(self, context):
        """주기적 장애 감지"""
        logger = context.logger
        
        while True:
            try:
                # 1. Pod 장애 감지
                pod_failures = await self._detect_pod_failures(context)
                for failure in pod_failures:
                    await self._handle_incident(failure, context)
                
                # 2. 서비스 장애 감지
                service_failures = await self._detect_service_failures(context)
                for failure in service_failures:
                    await self._handle_incident(failure, context)
                
                # 3. 리소스 고갈 감지
                resource_failures = await self._detect_resource_exhaustion(context)
                for failure in resource_failures:
                    await self._handle_incident(failure, context)
                
                # 4. 네트워크 장애 감지
                network_failures = await self._detect_network_issues(context)
                for failure in network_failures:
                    await self._handle_incident(failure, context)
                
                # 5. 데이터베이스 장애 감지
                db_failures = await self._detect_database_issues(context)
                for failure in db_failures:
                    await self._handle_incident(failure, context)
                
                logger.debug("Failure detection cycle completed")
                
                # 30초 대기
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Failure detection error: {e}")
                await asyncio.sleep(60)
    
    async def _start_predictive_failure_prevention(self, context):
        """예측적 장애 방지 시작"""
        logger = context.logger
        
        # 주기적 예측 분석 태스크 시작
        asyncio.create_task(self._periodic_predictive_analysis(context))
        
        logger.info("Predictive failure prevention started")
    
    async def _periodic_predictive_analysis(self, context):
        """주기적 예측 분석"""
        logger = context.logger
        
        while True:
            try:
                # 1. 리소스 사용량 트렌드 분석
                resource_trends = await self._analyze_resource_trends(context)
                
                # 2. 성능 저하 패턴 감지
                performance_degradation = await self._detect_performance_degradation(context)
                
                # 3. 예측적 스케일링
                if resource_trends.get("trend") == "increasing":
                    await self._proactive_scaling(context)
                
                # 4. 예방적 조치
                if performance_degradation:
                    await self._take_preventive_actions(performance_degradation, context)
                
                logger.debug("Predictive analysis cycle completed")
                
                # 5분 대기
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Predictive analysis error: {e}")
                await asyncio.sleep(600)
    
    async def _handle_incident(self, incident_data: Dict[str, Any], context):
        """장애 처리"""
        logger = context.logger
        
        # 장애 생성
        incident = Incident(
            id=f"incident_{int(datetime.now().timestamp())}",
            type=incident_data["type"],
            severity=incident_data["severity"],
            status="open",
            description=incident_data["description"],
            affected_resources=incident_data["affected_resources"],
            timestamp=datetime.now()
        )
        
        self.active_incidents.append(incident)
        logger.warning(f"New incident detected: {incident.id} - {incident.description}")
        
        # 자동 복구 시도
        if self.auto_recovery_enabled and incident_data.get("auto_recovery", False):
            await self._attempt_auto_recovery(incident, context)
        else:
            # 수동 개입 필요 알림
            await self._send_manual_intervention_alert(incident, context)
    
    async def _attempt_auto_recovery(self, incident: Incident, context):
        """자동 복구 시도"""
        logger = context.logger
        
        logger.info(f"Attempting auto-recovery for incident: {incident.id}")
        
        # 장애 타입별 복구 액션 실행
        recovery_action = None
        
        if incident.type == "pod_failure":
            recovery_action = await self._restart_pod(incident.affected_resources[0], context)
        elif incident.type == "service_unavailable":
            recovery_action = await self._restart_service(incident.affected_resources[0], context)
        elif incident.type == "resource_exhaustion":
            recovery_action = await self._scale_up_resources(incident.affected_resources[0], context)
        elif incident.type == "network_connectivity":
            recovery_action = await self._restart_network_policy(incident.affected_resources[0], context)
        elif incident.type == "database_connection":
            recovery_action = await self._restart_database(incident.affected_resources[0], context)
        
        if recovery_action and recovery_action.success:
            # 복구 성공
            incident.status = "resolved"
            incident.auto_resolved = True
            incident.resolution_time = datetime.now()
            
            # 활성 장애 목록에서 제거
            self.active_incidents.remove(incident)
            self.incident_history.append(incident)
            
            logger.info(f"Auto-recovery successful for incident: {incident.id}")
            await self._send_recovery_success_notification(incident, context)
        else:
            # 복구 실패
            incident.status = "investigating"
            logger.error(f"Auto-recovery failed for incident: {incident.id}")
            await self._send_recovery_failure_alert(incident, context)
    
    async def _restart_pod(self, pod_name: str, context) -> RecoveryAction:
        """Pod 재시작"""
        logger = context.logger
        
        try:
            # Pod 삭제 (Kubernetes가 자동으로 재생성)
            await self._delete_pod(pod_name, context)
            
            # 재시작 확인
            await asyncio.sleep(10)
            pod_status = await self._get_pod_status(pod_name, context)
            
            success = pod_status.get("status") == "Running"
            
            action = RecoveryAction(
                name="restart_pod",
                type="restart",
                target_resource=pod_name,
                parameters={"action": "delete_and_recreate"},
                success=success,
                execution_time=datetime.now()
            )
            
            self.recovery_actions.append(action)
            logger.info(f"Pod restart {'successful' if success else 'failed'}: {pod_name}")
            
            return action
            
        except Exception as e:
            logger.error(f"Pod restart failed: {e}")
            return RecoveryAction(
                name="restart_pod",
                type="restart",
                target_resource=pod_name,
                parameters={"action": "delete_and_recreate"},
                success=False,
                execution_time=datetime.now()
            )
    
    async def _restart_service(self, service_name: str, context) -> RecoveryAction:
        """서비스 재시작"""
        logger = context.logger
        
        try:
            # 서비스 재시작
            await self._restart_kubernetes_service(service_name, context)
            
            # 서비스 상태 확인
            await asyncio.sleep(15)
            service_status = await self._get_service_status(service_name, context)
            
            success = service_status.get("endpoints") > 0
            
            action = RecoveryAction(
                name="restart_service",
                type="restart",
                target_resource=service_name,
                parameters={"action": "service_restart"},
                success=success,
                execution_time=datetime.now()
            )
            
            self.recovery_actions.append(action)
            logger.info(f"Service restart {'successful' if success else 'failed'}: {service_name}")
            
            return action
            
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            return RecoveryAction(
                name="restart_service",
                type="restart",
                target_resource=service_name,
                parameters={"action": "service_restart"},
                success=False,
                execution_time=datetime.now()
            )
    
    async def _scale_up_resources(self, resource_name: str, context) -> RecoveryAction:
        """리소스 스케일 업"""
        logger = context.logger
        
        try:
            # 현재 레플리카 수 확인
            current_replicas = await self._get_current_replicas(resource_name, context)
            
            # 레플리카 수 증가
            new_replicas = min(current_replicas + 2, 10)  # 최대 10개로 제한
            await self._scale_deployment(resource_name, new_replicas, context)
            
            # 스케일링 확인
            await asyncio.sleep(30)
            scaled_status = await self._get_deployment_status(resource_name, context)
            
            success = scaled_status.get("ready_replicas") == new_replicas
            
            action = RecoveryAction(
                name="scale_up_resources",
                type="scale",
                target_resource=resource_name,
                parameters={"old_replicas": current_replicas, "new_replicas": new_replicas},
                success=success,
                execution_time=datetime.now()
            )
            
            self.recovery_actions.append(action)
            logger.info(f"Resource scaling {'successful' if success else 'failed'}: {resource_name}")
            
            return action
            
        except Exception as e:
            logger.error(f"Resource scaling failed: {e}")
            return RecoveryAction(
                name="scale_up_resources",
                type="scale",
                target_resource=resource_name,
                parameters={},
                success=False,
                execution_time=datetime.now()
            )
    
    async def _proactive_scaling(self, context):
        """예측적 스케일링"""
        logger = context.logger
        
        # 리소스 사용량이 증가 추세일 때 사전 스케일링
        deployments = await self._get_deployments(context)
        
        for deployment in deployments:
            if deployment.get("cpu_trend") == "increasing" and deployment.get("cpu_usage") > 70:
                await self._scale_up_resources(deployment["name"], context)
                logger.info(f"Proactive scaling for {deployment['name']}")
    
    async def _take_preventive_actions(self, degradation_data: Dict[str, Any], context):
        """예방적 조치"""
        logger = context.logger
        
        # 성능 저하 패턴에 따른 예방적 조치
        if degradation_data.get("type") == "memory_leak":
            await self._restart_pod(degradation_data["pod_name"], context)
        elif degradation_data.get("type") == "high_latency":
            await self._scale_up_resources(degradation_data["service_name"], context)
        
        logger.info("Preventive actions taken")
    
    # 시뮬레이션 메서드들
    async def _get_cluster_health(self, context) -> Dict[str, Any]:
        """클러스터 헬스 조회"""
        return {"status": "healthy", "nodes": 5, "pods": 150}
    
    async def _get_node_health(self, context) -> Dict[str, Any]:
        """노드 헬스 조회"""
        return {"healthy_nodes": 5, "total_nodes": 5}
    
    async def _get_service_health(self, context) -> Dict[str, Any]:
        """서비스 헬스 조회"""
        return {"healthy_services": 25, "total_services": 25}
    
    async def _check_existing_incidents(self, context) -> List[Dict[str, Any]]:
        """기존 장애 확인"""
        return []  # 시뮬레이션
    
    async def _detect_pod_failures(self, context) -> List[Dict[str, Any]]:
        """Pod 장애 감지"""
        # 시뮬레이션: 가끔 Pod 장애 발생
        import random
        if random.random() < 0.1:  # 10% 확률로 장애 발생
            return [{
                "type": "pod_failure",
                "severity": "medium",
                "description": "Pod app-pod-1 failed",
                "affected_resources": ["app-pod-1"],
                "auto_recovery": True
            }]
        return []
    
    async def _detect_service_failures(self, context) -> List[Dict[str, Any]]:
        """서비스 장애 감지"""
        return []  # 시뮬레이션
    
    async def _detect_resource_exhaustion(self, context) -> List[Dict[str, Any]]:
        """리소스 고갈 감지"""
        return []  # 시뮬레이션
    
    async def _detect_network_issues(self, context) -> List[Dict[str, Any]]:
        """네트워크 장애 감지"""
        return []  # 시뮬레이션
    
    async def _detect_database_issues(self, context) -> List[Dict[str, Any]]:
        """데이터베이스 장애 감지"""
        return []  # 시뮬레이션
    
    async def _analyze_resource_trends(self, context) -> Dict[str, Any]:
        """리소스 트렌드 분석"""
        return {"trend": "stable", "cpu_usage": 45, "memory_usage": 62}
    
    async def _detect_performance_degradation(self, context) -> Optional[Dict[str, Any]]:
        """성능 저하 감지"""
        return None  # 시뮬레이션
    
    async def _delete_pod(self, pod_name: str, context):
        """Pod 삭제"""
        print(f"Deleting pod: {pod_name}")
    
    async def _get_pod_status(self, pod_name: str, context) -> Dict[str, Any]:
        """Pod 상태 조회"""
        return {"status": "Running", "ready": True}
    
    async def _restart_kubernetes_service(self, service_name: str, context):
        """Kubernetes 서비스 재시작"""
        print(f"Restarting service: {service_name}")
    
    async def _get_service_status(self, service_name: str, context) -> Dict[str, Any]:
        """서비스 상태 조회"""
        return {"endpoints": 3, "status": "healthy"}
    
    async def _get_current_replicas(self, deployment_name: str, context) -> int:
        """현재 레플리카 수 조회"""
        return 3
    
    async def _scale_deployment(self, deployment_name: str, replicas: int, context):
        """Deployment 스케일링"""
        print(f"Scaling deployment {deployment_name} to {replicas} replicas")
    
    async def _get_deployment_status(self, deployment_name: str, context) -> Dict[str, Any]:
        """Deployment 상태 조회"""
        return {"ready_replicas": 5, "total_replicas": 5}
    
    async def _get_deployments(self, context) -> List[Dict[str, Any]]:
        """Deployment 목록 조회"""
        return [
            {"name": "app-deployment", "cpu_usage": 45, "cpu_trend": "stable"},
            {"name": "api-deployment", "cpu_usage": 75, "cpu_trend": "increasing"}
        ]
    
    async def _send_manual_intervention_alert(self, incident: Incident, context):
        """수동 개입 알림 전송"""
        logger = context.logger
        logger.warning(f"Manual intervention required for incident: {incident.id}")
        print(f"🚨 MANUAL INTERVENTION: {incident.description}")
    
    async def _send_recovery_success_notification(self, incident: Incident, context):
        """복구 성공 알림 전송"""
        logger = context.logger
        logger.info(f"Recovery successful for incident: {incident.id}")
        print(f"✅ RECOVERY SUCCESS: {incident.description}")
    
    async def _send_recovery_failure_alert(self, incident: Incident, context):
        """복구 실패 알림 전송"""
        logger = context.logger
        logger.error(f"Recovery failed for incident: {incident.id}")
        print(f"❌ RECOVERY FAILED: {incident.description}")
    
    async def get_active_incidents(self) -> List[Incident]:
        """활성 장애 조회"""
        return self.active_incidents
    
    async def get_incident_history(self) -> List[Incident]:
        """장애 히스토리 조회"""
        return self.incident_history
    
    async def get_recovery_actions(self) -> List[RecoveryAction]:
        """복구 액션 조회"""
        return self.recovery_actions
    
    async def generate_incident_report(self, incident_id: str):
        """장애 보고서 생성"""
        # 장애 찾기
        incident = None
        for inc in self.incident_history:
            if inc.id == incident_id:
                incident = inc
                break
        
        if not incident:
            print(f"Incident {incident_id} not found")
            return
        
        # 복구 액션 찾기
        actions = [action for action in self.recovery_actions if action.target_resource in incident.affected_resources]
        
        # 보고서 생성
        report = IncidentReport(
            incident_id=incident.id,
            summary=f"Incident {incident.type} affecting {len(incident.affected_resources)} resources",
            root_cause="Resource exhaustion due to high load",
            impact_assessment=f"Service unavailable for {incident.resolution_time - incident.timestamp if incident.resolution_time else 'ongoing'}",
            recovery_actions=actions,
            lessons_learned=[
                "Implement better resource monitoring",
                "Add auto-scaling policies",
                "Improve error handling"
            ],
            prevention_measures=[
                "Set up proactive scaling",
                "Implement circuit breakers",
                "Add health checks"
            ]
        )
        
        # 보고서 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"incident_report_{incident_id}_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        
        print(f"Incident report saved to: {report_file}")

# 사용 예시
async def main():
    """사용 예시"""
    agent = IncidentResponseAgent()
    
    # 장애 모니터링 시작
    await agent.start_incident_monitoring()
    
    # 30초 대기
    await asyncio.sleep(30)
    
    # 활성 장애 조회
    active_incidents = await agent.get_active_incidents()
    print(f"Active incidents: {len(active_incidents)}")
    
    # 장애 히스토리 조회
    history = await agent.get_incident_history()
    print(f"Incident history: {len(history)}")

if __name__ == "__main__":
    asyncio.run(main()) 