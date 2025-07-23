"""
동적 설정 생성 Agent
====================

워크로드 요구사항을 분석하여 최적화된 Kubernetes 설정을 실시간으로 생성하는 Agent
"""

import asyncio
import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# OpenAI Agents SDK 사용
from agents import Agent
from agents.mcp import MCPServerStdio, create_static_tool_filter

@dataclass
class WorkloadRequirements:
    """워크로드 요구사항"""
    name: str
    type: str  # web, api, batch, ml, etc.
    cpu_request: str
    memory_request: str
    replicas: int
    environment: str  # dev, staging, prod
    security_level: str  # low, medium, high
    scaling_requirements: Dict[str, Any]
    image: str = ""
    port: int = 80
    namespace: str = "default"

@dataclass
class GeneratedConfig:
    """생성된 설정"""
    deployment_yaml: str
    service_yaml: str
    ingress_yaml: str
    configmap_yaml: str
    secret_yaml: str
    network_policy_yaml: str
    monitoring_config: Dict[str, Any]
    security_policies: Dict[str, Any]
    hpa_yaml: str = ""
    pdb_yaml: str = ""

class DynamicConfigGenerator:
    """동적 Kubernetes 설정 생성기 - OpenAI Agents SDK MCP 사용"""
    
    def __init__(self, output_dir: str = "generated_configs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # OpenAI Agents SDK Agent 설정
        self.agent = Agent(
            name="dynamic_config_generator",
            instructions="워크로드 요구사항을 분석하여 최적화된 Kubernetes 설정을 생성하는 전문 Agent입니다.",
            mcp_servers=[
                # Kubernetes MCP 서버
                MCPServerStdio(
                    params={
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-kubernetes"],
                    },
                    tool_filter=create_static_tool_filter(
                        allowed_tool_names=["list_pods", "get_deployment", "create_namespace"]
                    )
                ),
                # Filesystem MCP 서버 (설정 파일 저장용)
                MCPServerStdio(
                    params={
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", self.output_dir],
                    },
                    tool_filter=create_static_tool_filter(
                        allowed_tool_names=["read_file", "write_file", "list_files"]
                    )
                )
            ]
        )
        
        # 설정 템플릿 캐시
        self.config_templates = self._load_config_templates()
        
        # 생성 히스토리
        self.generation_history: List[Dict[str, Any]] = []
        
    async def generate_k8s_config(self, requirements: WorkloadRequirements) -> GeneratedConfig:
        """Kubernetes 설정 동적 생성"""
        try:
            # Agent 실행
            result = await self.agent.run(
                f"""
                워크로드 요구사항을 분석하여 최적화된 Kubernetes 설정을 생성하세요:
                
                워크로드: {requirements.name}
                타입: {requirements.type}
                CPU 요청: {requirements.cpu_request}
                메모리 요청: {requirements.memory_request}
                레플리카: {requirements.replicas}
                환경: {requirements.environment}
                보안 레벨: {requirements.security_level}
                스케일링 요구사항: {requirements.scaling_requirements}
                
                다음을 생성해주세요:
                1. Deployment YAML
                2. Service YAML
                3. Ingress YAML (필요시)
                4. ConfigMap YAML
                5. Secret YAML
                6. NetworkPolicy YAML
                7. HPA YAML (필요시)
                8. PDB YAML (필요시)
                9. 모니터링 설정
                10. 보안 정책
                
                모든 파일을 {self.output_dir} 디렉토리에 저장하고 설정을 반환하세요.
                """
            )
            
            # 결과에서 설정 추출
            config = self._extract_config_from_result(result, requirements)
            
            # 생성 히스토리 저장
            self.generation_history.append({
                "timestamp": datetime.now().isoformat(),
                "workload": asdict(requirements),
                "config": asdict(config)
            })
            
            print(f"Configuration generated successfully for {requirements.name}")
            return config
                
        except Exception as e:
            print(f"Config generation failed: {e}")
            raise
    
    def _extract_config_from_result(self, result, requirements: WorkloadRequirements) -> GeneratedConfig:
        """Agent 결과에서 설정 추출"""
        # 실제로는 Agent의 응답을 파싱하여 설정 추출
        # 여기서는 시뮬레이션
        
        deployment_yaml = self._get_web_deployment_template().format(
            name=requirements.name,
            namespace=requirements.namespace,
            image=requirements.image or f"{requirements.name}:latest",
            port=requirements.port,
            replicas=requirements.replicas,
            cpu_request=requirements.cpu_request,
            memory_request=requirements.memory_request,
            cpu_limit=requirements.cpu_request,
            memory_limit=requirements.memory_request
        )
        
        service_yaml = self._get_web_service_template().format(
            name=requirements.name,
            namespace=requirements.namespace,
            port=requirements.port
        )
        
        config = GeneratedConfig(
            deployment_yaml=deployment_yaml,
            service_yaml=service_yaml,
            ingress_yaml=self._get_web_ingress_template().format(
                name=requirements.name,
                namespace=requirements.namespace,
                host=f"{requirements.name}.example.com"
            ),
            configmap_yaml=self._generate_configmap_yaml(requirements),
            secret_yaml=self._generate_secret_yaml(requirements),
            network_policy_yaml=self._generate_network_policy_yaml(requirements.name, {}),
            monitoring_config=self._generate_monitoring_config(requirements),
            security_policies=self._generate_security_policies(requirements),
            hpa_yaml=self._generate_hpa_yaml(requirements),
            pdb_yaml=self._generate_pdb_yaml(requirements)
        )
        
        return config
    
    def _generate_monitoring_config(self, requirements: WorkloadRequirements) -> Dict[str, Any]:
        """모니터링 설정 생성"""
        return {
            "prometheus_metrics": [
                "http_requests_total",
                "http_request_duration_seconds",
                "container_cpu_usage_seconds_total",
                "container_memory_usage_bytes"
            ],
            "grafana_dashboard": {
                "title": f"{requirements.name} Dashboard",
                "panels": [
                    "CPU Usage",
                    "Memory Usage",
                    "Request Rate",
                    "Error Rate"
                ]
            },
            "alert_rules": [
                {
                    "name": f"{requirements.name}_high_cpu",
                    "condition": "cpu_usage > 80%",
                    "duration": "5m"
                }
            ]
        }
    
    def _generate_security_policies(self, requirements: WorkloadRequirements) -> Dict[str, Any]:
        """보안 정책 생성"""
        return {
            "pod_security_standards": "restricted" if requirements.security_level == "high" else "baseline",
            "network_policies": "strict" if requirements.security_level == "high" else "basic",
            "rbac": "strict" if requirements.security_level == "high" else "standard"
        }
    
    def _generate_hpa_yaml(self, requirements: WorkloadRequirements) -> str:
        """HPA YAML 생성"""
        if requirements.type in ["web", "api"]:
            return f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {requirements.name}-hpa
  namespace: {requirements.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {requirements.name}
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
        return ""
    
    def _generate_pdb_yaml(self, requirements: WorkloadRequirements) -> str:
        """PDB YAML 생성"""
        if requirements.environment == "prod":
            return f"""
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {requirements.name}-pdb
  namespace: {requirements.namespace}
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: {requirements.name}
"""
        return ""
    
    def _load_config_templates(self) -> Dict[str, Any]:
        """설정 템플릿 로드"""
        return {
            "web_app": {
                "deployment_template": self._get_web_deployment_template(),
                "service_template": self._get_web_service_template(),
                "ingress_template": self._get_web_ingress_template()
            }
        }
    
    def _get_web_deployment_template(self) -> str:
        """웹 애플리케이션 Deployment 템플릿"""
        return """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: {name}
    type: web
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            cpu: {cpu_request}
            memory: {memory_request}
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
        """
    
    def _get_web_service_template(self) -> str:
        """웹 애플리케이션 Service 템플릿"""
        return """
apiVersion: v1
kind: Service
metadata:
  name: {name}-service
  namespace: {namespace}
spec:
  selector:
    app: {name}
  ports:
  - protocol: TCP
    port: 80
    targetPort: {port}
  type: ClusterIP
        """
    
    def _get_web_ingress_template(self) -> str:
        """웹 애플리케이션 Ingress 템플릿"""
        return """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}-ingress
  namespace: {namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {name}-service
            port:
              number: 80
        """
    
    def _generate_configmap_yaml(self, requirements: WorkloadRequirements) -> str:
        """ConfigMap YAML 생성"""
        return f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {requirements.name}-config
  namespace: {requirements.namespace}
data:
  ENVIRONMENT: {requirements.environment}
  LOG_LEVEL: INFO
  API_VERSION: v1
"""
    
    def _generate_secret_yaml(self, requirements: WorkloadRequirements) -> str:
        """Secret YAML 생성"""
        return f"""
apiVersion: v1
kind: Secret
metadata:
  name: {requirements.name}-secret
  namespace: {requirements.namespace}
type: Opaque
data:
  # Base64 encoded values would go here
  api-key: ""
  database-url: ""
"""
    
    def _generate_network_policy_yaml(self, name: str, policy: Dict[str, Any]) -> str:
        """네트워크 정책 YAML 생성"""
        return f"""
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {name}-network-policy
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: {name}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
  egress:
  - ports:
    - protocol: TCP
      port: 443
    - protocol: UDP
      port: 53
"""

# 사용 예시
async def main():
    """사용 예시"""
    generator = DynamicConfigGenerator()
    
    # 워크로드 요구사항 정의
    requirements = WorkloadRequirements(
        name="my-web-app",
        type="web",
        cpu_request="100m",
        memory_request="128Mi",
        replicas=3,
        environment="prod",
        security_level="high",
        scaling_requirements={"min": 2, "max": 10},
        image="nginx:latest",
        port=80
    )
    
    # 설정 생성
    config = await generator.generate_k8s_config(requirements)
    
    print("Generated Kubernetes configuration:")
    print(f"Deployment: {config.deployment_yaml[:200]}...")
    print(f"Service: {config.service_yaml[:200]}...")
    print(f"Monitoring config: {config.monitoring_config}")

if __name__ == "__main__":
    asyncio.run(main()) 