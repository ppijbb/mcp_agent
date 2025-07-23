"""
K8s Cloud Operations Agent - 메인 실행 파일
===========================================

Python MCP 기반 Kubernetes 및 클라우드 운영 관리 시스템
(OpenAI Agents SDK + Pydantic AI)
"""

import asyncio
import argparse
import sys
from typing import List, Optional

from agents.dynamic_config_agent import DynamicConfigGenerator, WorkloadRequirements
from agents.deployment_agent import DeploymentManagementAgent, DeploymentConfig
from agents.monitoring_agent import MonitoringAgent
from agents.security_agent import SecurityAgent
from agents.cost_optimizer_agent import CostOptimizerAgent
from agents.incident_response_agent import IncidentResponseAgent

class K8sCloudOperationsOrchestrator:
    """K8s Cloud Operations 오케스트레이터"""
    
    def __init__(self):
        self.agents = {
            "config": DynamicConfigGenerator(),
            "deployment": DeploymentManagementAgent(),
            "monitoring": MonitoringAgent(),
            "security": SecurityAgent(),
            "cost": CostOptimizerAgent(),
            "incident": IncidentResponseAgent()
        }
        
    async def run_dynamic_config_generation(self, workload_name: str, workload_type: str):
        """동적 설정 생성 실행"""
        print(f"🚀 Generating dynamic configuration for {workload_name}")
        
        # 워크로드 요구사항 정의
        requirements = WorkloadRequirements(
            name=workload_name,
            type=workload_type,
            cpu_request="100m",
            memory_request="128Mi",
            replicas=3,
            environment="prod",
            security_level="high",
            scaling_requirements={"min": 2, "max": 10},
            image=f"{workload_name}:latest",
            port=80
        )
        
        # 설정 생성
        config = await self.agents["config"].generate_k8s_config(requirements)
        
        print(f"✅ Configuration generated successfully!")
        print(f"📁 Files saved to: generated_configs/")
        print(f"🔧 Deployment YAML: {len(config.deployment_yaml)} characters")
        print(f"🌐 Service YAML: {len(config.service_yaml)} characters")
        
        return config
    
    async def run_deployment(self, app_name: str, image: str):
        """배포 실행"""
        print(f"🚀 Deploying {app_name}")
        
        # 배포 설정
        config = DeploymentConfig(
            name=app_name,
            image=image,
            replicas=3,
            namespace="default",
            strategy="RollingUpdate",
            environment="prod",
            cpu_request="100m",
            memory_request="128Mi"
        )
        
        # 배포 실행
        result = await self.agents["deployment"].deploy_application(config)
        
        print(f"✅ Deployment completed!")
        print(f"🆔 Deployment ID: {result.deployment_id}")
        print(f"📊 Status: {result.status}")
        print(f"💚 Health: {result.health_status}")
        
        return result
    
    async def start_monitoring(self, namespace: str = "default"):
        """모니터링 시작"""
        print(f"📊 Starting monitoring for namespace: {namespace}")
        
        await self.agents["monitoring"].start_monitoring(namespace)
        
        print("✅ Monitoring started successfully!")
        print("📈 Real-time metrics collection active")
        print("🚨 Alert monitoring active")
        print("🔮 Predictive analysis active")
    
    async def start_security_monitoring(self, namespace: str = "default"):
        """보안 모니터링 시작"""
        print(f"🔒 Starting security monitoring for namespace: {namespace}")
        
        await self.agents["security"].start_security_monitoring(namespace)
        
        print("✅ Security monitoring started successfully!")
        print("🛡️ Security scanning active")
        print("🔍 Vulnerability assessment active")
        print("📋 Compliance checking active")
    
    async def start_cost_monitoring(self):
        """비용 모니터링 시작"""
        print("💰 Starting cost monitoring")
        
        await self.agents["cost"].start_cost_monitoring()
        
        print("✅ Cost monitoring started successfully!")
        print("📊 Cost analysis active")
        print("💡 Optimization recommendations active")
        print("🚨 Budget alerts active")
    
    async def start_incident_monitoring(self):
        """장애 모니터링 시작"""
        print("🚨 Starting incident monitoring")
        
        await self.agents["incident"].start_incident_monitoring()
        
        print("✅ Incident monitoring started successfully!")
        print("🔍 Failure detection active")
        print("🔄 Auto-recovery active")
        print("🔮 Predictive prevention active")
    
    async def run_full_operations(self, workload_name: str, workload_type: str, image: str):
        """전체 운영 프로세스 실행"""
        print("🎯 Starting full K8s Cloud Operations")
        print("=" * 50)
        
        try:
            # 1. 동적 설정 생성
            config = await self.run_dynamic_config_generation(workload_name, workload_type)
            
            # 2. 배포 실행
            deployment = await self.run_deployment(workload_name, image)
            
            # 3. 모든 모니터링 시작
            await asyncio.gather(
                self.start_monitoring(),
                self.start_security_monitoring(),
                self.start_cost_monitoring(),
                self.start_incident_monitoring()
            )
            
            print("=" * 50)
            print("🎉 Full operations started successfully!")
            print("📊 All monitoring systems are active")
            print("🔧 Dynamic configuration generated")
            print("🚀 Application deployed")
            
            # 30초 동안 상태 모니터링
            print("\n📈 Monitoring system status for 30 seconds...")
            await asyncio.sleep(30)
            
            # 상태 리포트
            await self.generate_status_report()
            
        except Exception as e:
            print(f"❌ Error during operations: {e}")
            raise
    
    async def generate_status_report(self):
        """상태 리포트 생성"""
        print("\n📋 Generating Status Report")
        print("-" * 30)
        
        # 모니터링 상태
        alerts = await self.agents["monitoring"].get_active_alerts()
        print(f"🚨 Active Alerts: {len(alerts)}")
        
        # 보안 상태
        violations = await self.agents["security"].get_security_violations()
        print(f"🔒 Security Violations: {len(violations)}")
        
        # 비용 상태
        recommendations = await self.agents["cost"].get_optimization_recommendations()
        print(f"💰 Cost Optimization Opportunities: {len(recommendations)}")
        
        # 장애 상태
        incidents = await self.agents["incident"].get_active_incidents()
        print(f"🚨 Active Incidents: {len(incidents)}")
        
        print("-" * 30)
        print("✅ Status report completed")

async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="K8s Cloud Operations Agent")
    parser.add_argument("--mode", choices=["config", "deploy", "monitor", "security", "cost", "incident", "full"], 
                       default="full", help="Operation mode")
    parser.add_argument("--workload", default="my-app", help="Workload name")
    parser.add_argument("--type", default="web", help="Workload type (web, api, batch)")
    parser.add_argument("--image", default="nginx:latest", help="Container image")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    
    args = parser.parse_args()
    
    orchestrator = K8sCloudOperationsOrchestrator()
    
    try:
        if args.mode == "config":
            await orchestrator.run_dynamic_config_generation(args.workload, args.type)
        
        elif args.mode == "deploy":
            await orchestrator.run_deployment(args.workload, args.image)
        
        elif args.mode == "monitor":
            await orchestrator.start_monitoring(args.namespace)
            # 계속 실행
            await asyncio.sleep(3600)  # 1시간
        
        elif args.mode == "security":
            await orchestrator.start_security_monitoring(args.namespace)
            # 계속 실행
            await asyncio.sleep(3600)  # 1시간
        
        elif args.mode == "cost":
            await orchestrator.start_cost_monitoring()
            # 계속 실행
            await asyncio.sleep(3600)  # 1시간
        
        elif args.mode == "incident":
            await orchestrator.start_incident_monitoring()
            # 계속 실행
            await asyncio.sleep(3600)  # 1시간
        
        elif args.mode == "full":
            await orchestrator.run_full_operations(args.workload, args.type, args.image)
        
    except KeyboardInterrupt:
        print("\n🛑 Operations interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 