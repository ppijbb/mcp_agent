"""
배포 관리 Agent
===============

Kubernetes 애플리케이션 배포 및 업데이트를 관리하는 Agent
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class DeploymentConfig:
    """배포 설정"""
    name: str
    image: str
    replicas: int = 1
    namespace: str = "default"
    strategy: str = "RollingUpdate"
    health_check: bool = True
    environment: str = "prod"
    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    cpu_limit: str = "200m"
    memory_limit: str = "256Mi"

@dataclass
class DeploymentResult:
    """배포 결과"""
    deployment_id: str
    status: str
    blue_deployment: Optional[str] = None
    green_deployment: Optional[str] = None
    current: str = "green"
    timestamp: datetime = None
    health_status: str = "unknown"
    rollback_available: bool = False

class DeploymentManagementAgent:
    """mcp_agent 기반 배포 관리 Agent"""
    
    def __init__(self, output_dir: str = "deployment_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # mcp_agent App 초기화 (설정 파일 없이 동적 생성)
        self.app = MCPApp(
            name="deployment_management",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="deployment_manager",
            instruction="Kubernetes 애플리케이션 배포 및 업데이트를 관리하는 전문 Agent입니다.",
            server_names=["k8s-mcp", "monitoring-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        # 배포 히스토리
        self.deployment_history: List[DeploymentResult] = []
        self.active_deployments: Dict[str, DeploymentConfig] = {}
        
        # 배포 상태 추적
        self.deployment_status: Dict[str, str] = {}
        
    async def deploy_application(self, config: DeploymentConfig) -> DeploymentResult:
        """애플리케이션 배포 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info(f"Starting deployment for {config.name}")
                
                # 1. 배포 전 검증
                await self._validate_deployment(config, context)
                
                # 2. Blue-Green 배포 실행
                result = await self._execute_blue_green_deployment(config, context)
                
                # 3. 배포 후 검증
                await self._validate_deployment_health(result, context)
                
                # 4. 트래픽 전환
                await self._switch_traffic(result, context)
                
                # 5. 결과 저장
                self.deployment_history.append(result)
                self.active_deployments[result.deployment_id] = config
                self.deployment_status[result.deployment_id] = "completed"
                
                # 6. 배포 보고서 생성
                await self._generate_deployment_report(result, config)
                
                logger.info(f"Deployment completed successfully: {result.deployment_id}")
                return result
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            # 롤백 실행
            await self._rollback_deployment(config)
            raise
    
    async def _validate_deployment(self, config: DeploymentConfig, context) -> None:
        """배포 전 검증"""
        logger = context.logger
        
        # 시스템 리소스 확인
        system_health = await self._check_system_health(context)
        
        if not system_health.get("healthy", False):
            raise Exception("System health check failed")
        
        # 이미지 존재 확인
        image_check = await self._check_image_exists(config.image, context)
        
        if not image_check.get("exists", False):
            raise Exception(f"Image {config.image} not found")
        
        # 네임스페이스 확인
        namespace_check = await self._check_namespace_exists(config.namespace, context)
        
        if not namespace_check.get("exists", False):
            await self._create_namespace(config.namespace, context)
        
        logger.info(f"Pre-deployment validation passed for {config.name}")
    
    async def _execute_blue_green_deployment(self, config: DeploymentConfig, context) -> DeploymentResult:
        """Blue-Green 배포 실행"""
        logger = context.logger
        deployment_id = f"{config.name}-{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting Blue-Green deployment: {deployment_id}")
        
        # Blue 환경 생성 (0 replicas)
        blue_deployment = await self._create_deployment(
            f"{config.name}-blue",
            config,
            replicas=0,
            context=context
        )
        
        # Green 환경 생성 (실제 replicas)
        green_deployment = await self._create_deployment(
            f"{config.name}-green",
            config,
            replicas=config.replicas,
            context=context
        )
        
        # Green 환경 스케일 업
        await self._scale_deployment(
            f"{config.name}-green",
            config.replicas,
            config.namespace,
            context
        )
        
        # 헬스 체크 대기
        await self._wait_for_deployment_ready(f"{config.name}-green", context)
        
        # Service 생성
        await self._create_service(f"{config.name}-service", config, context)
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            status="completed",
            blue_deployment=f"{config.name}-blue",
            green_deployment=f"{config.name}-green",
            current="green",
            timestamp=datetime.now(),
            health_status="healthy",
            rollback_available=True
        )
        
        logger.info(f"Blue-Green deployment completed: {deployment_id}")
        return result
    
    async def _wait_for_deployment_ready(self, deployment_name: str, context, timeout: int = 300) -> None:
        """배포 준비 완료 대기"""
        logger = context.logger
        start_time = datetime.now()
        
        logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        while (datetime.now() - start_time).seconds < timeout:
            deployment_status = await self._get_deployment_status(deployment_name, context)
            
            if deployment_status.get("ready", False):
                logger.info(f"Deployment {deployment_name} is ready")
                return
            
            await asyncio.sleep(5)
        
        raise Exception(f"Deployment {deployment_name} not ready within {timeout} seconds")
    
    async def _validate_deployment_health(self, result: DeploymentResult, context) -> None:
        """배포 후 헬스 검증"""
        logger = context.logger
        deployment_name = result.green_deployment if result.current == "green" else result.blue_deployment
        
        logger.info(f"Validating deployment health for {deployment_name}")
        
        # Pod 상태 확인
        pods = await self._list_pods(deployment_name, context)
        
        ready_pods = [pod for pod in pods if pod.get("ready", False)]
        
        if len(ready_pods) < len(pods):
            raise Exception(f"Health check failed: {len(ready_pods)}/{len(pods)} pods ready")
        
        # 메트릭 기반 헬스 체크
        if len(ready_pods) > 0:
            metrics = await self._get_pod_metrics(ready_pods[0]["name"], context)
            
            cpu_usage = metrics.get("cpu", {}).get("usage", 0)
            memory_usage = metrics.get("memory", {}).get("usage", 0)
            
            if cpu_usage > 0.8 or memory_usage > 0.9:
                raise Exception("Deployment metrics indicate poor health")
        
        # 헬스 체크 엔드포인트 확인
        health_check = await self._check_health_endpoint(deployment_name, context)
        
        if not health_check.get("healthy", False):
            raise Exception("Health check endpoint failed")
        
        result.health_status = "healthy"
        logger.info(f"Deployment health validation passed for {deployment_name}")
    
    async def _switch_traffic(self, result: DeploymentResult, context) -> None:
        """트래픽 전환"""
        logger = context.logger
        service_name = f"{result.deployment_id}-service"
        target_deployment = result.green_deployment if result.current == "green" else result.blue_deployment
        
        logger.info(f"Switching traffic to {target_deployment}")
        
        # Service 업데이트
        await self._update_service(service_name, target_deployment, context)
        
        # Ingress 업데이트 (있는 경우)
        await self._update_ingress(result.deployment_id, target_deployment, context)
        
        logger.info(f"Traffic switched successfully to {target_deployment}")
    
    async def _rollback_deployment(self, config: DeploymentConfig) -> None:
        """배포 롤백"""
        logger = self.app.logger
        logger.info(f"Rolling back deployment for {config.name}")
        
        # 이전 버전으로 롤백
        previous_deployment = self._find_previous_deployment(config.name)
        
        if previous_deployment:
            await self.deploy_application(previous_deployment)
        else:
            logger.warning(f"No previous deployment found for {config.name}")
    
    def _find_previous_deployment(self, app_name: str) -> Optional[DeploymentConfig]:
        """이전 배포 설정 찾기"""
        for result in reversed(self.deployment_history):
            if result.deployment_id.startswith(app_name):
                return self.active_deployments.get(result.deployment_id)
        return None
    
    async def _generate_deployment_report(self, result: DeploymentResult, config: DeploymentConfig):
        """배포 보고서 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"deployment_report_{result.deployment_id}_{timestamp}.json")
        
        report_data = {
            "deployment_id": result.deployment_id,
            "application_name": config.name,
            "deployment_config": asdict(config),
            "deployment_result": asdict(result),
            "timestamp": timestamp,
            "status": "success"
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Deployment report saved to: {report_file}")
    
    # Kubernetes API 호출 메서드들 (시뮬레이션)
    async def _check_system_health(self, context) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        # 실제로는 Kubernetes API 호출
        return {"healthy": True, "cpu_usage": "45%", "memory_usage": "60%"}
    
    async def _check_image_exists(self, image: str, context) -> Dict[str, Any]:
        """이미지 존재 확인"""
        # 실제로는 이미지 레지스트리 API 호출
        return {"exists": True, "image": image, "size": "256MB"}
    
    async def _check_namespace_exists(self, namespace: str, context) -> Dict[str, Any]:
        """네임스페이스 존재 확인"""
        # 실제로는 Kubernetes API 호출
        return {"exists": True, "namespace": namespace}
    
    async def _create_namespace(self, namespace: str, context):
        """네임스페이스 생성"""
        # 실제로는 Kubernetes API 호출
        print(f"Creating namespace: {namespace}")
    
    async def _create_deployment(self, name: str, config: DeploymentConfig, replicas: int, context) -> Dict[str, Any]:
        """Deployment 생성"""
        # 실제로는 Kubernetes API 호출
        deployment_data = {
            "name": name,
            "namespace": config.namespace,
            "replicas": replicas,
            "image": config.image,
            "status": "created"
        }
        
        print(f"Created deployment: {name} with {replicas} replicas")
        return deployment_data
    
    async def _scale_deployment(self, name: str, replicas: int, namespace: str, context):
        """Deployment 스케일링"""
        # 실제로는 Kubernetes API 호출
        print(f"Scaling deployment {name} to {replicas} replicas")
    
    async def _get_deployment_status(self, name: str, context) -> Dict[str, Any]:
        """Deployment 상태 조회"""
        # 실제로는 Kubernetes API 호출
        return {"ready": True, "replicas": 3, "available": 3}
    
    async def _list_pods(self, deployment_name: str, context) -> List[Dict[str, Any]]:
        """Pod 목록 조회"""
        # 실제로는 Kubernetes API 호출
        return [
            {"name": f"{deployment_name}-pod-1", "ready": True, "status": "Running"},
            {"name": f"{deployment_name}-pod-2", "ready": True, "status": "Running"},
            {"name": f"{deployment_name}-pod-3", "ready": True, "status": "Running"}
        ]
    
    async def _get_pod_metrics(self, pod_name: str, context) -> Dict[str, Any]:
        """Pod 메트릭 조회"""
        # 실제로는 Prometheus API 호출
        return {
            "cpu": {"usage": 0.3, "limit": 1.0},
            "memory": {"usage": 0.5, "limit": 1.0}
        }
    
    async def _check_health_endpoint(self, deployment_name: str, context) -> Dict[str, Any]:
        """헬스 체크 엔드포인트 확인"""
        # 실제로는 HTTP 요청
        return {"healthy": True, "response_time": "50ms"}
    
    async def _create_service(self, name: str, config: DeploymentConfig, context):
        """Service 생성"""
        # 실제로는 Kubernetes API 호출
        print(f"Created service: {name}")
    
    async def _update_service(self, service_name: str, target_deployment: str, context):
        """Service 업데이트"""
        # 실제로는 Kubernetes API 호출
        print(f"Updated service {service_name} to target {target_deployment}")
    
    async def _update_ingress(self, deployment_id: str, target_deployment: str, context):
        """Ingress 업데이트"""
        # 실제로는 Kubernetes API 호출
        print(f"Updated ingress for {deployment_id} to target {target_deployment}")
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """배포 상태 조회"""
        for result in self.deployment_history:
            if result.deployment_id == deployment_id:
                return result
        return None
    
    async def list_deployments(self) -> List[DeploymentResult]:
        """배포 목록 조회"""
        return self.deployment_history
    
    async def rollback_deployment(self, deployment_id: str) -> bool:
        """배포 롤백"""
        try:
            deployment = await self.get_deployment_status(deployment_id)
            if deployment and deployment.rollback_available:
                config = self.active_deployments.get(deployment_id)
                if config:
                    await self._rollback_deployment(config)
                    return True
            return False
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False

# 사용 예시
async def main():
    """사용 예시"""
    agent = DeploymentManagementAgent()
    
    # 배포 설정
    config = DeploymentConfig(
        name="my-web-app",
        image="nginx:latest",
        replicas=3,
        namespace="default",
        strategy="RollingUpdate",
        environment="prod",
        cpu_request="100m",
        memory_request="128Mi"
    )
    
    # 배포 실행
    result = await agent.deploy_application(config)
    
    print(f"Deployment completed: {result.deployment_id}")
    print(f"Status: {result.status}")
    print(f"Health: {result.health_status}")

if __name__ == "__main__":
    asyncio.run(main()) 