#!/usr/bin/env python3
"""
DevOps Productivity MCP Agent
================================
Production-level DevOps assistant with MCP server integrations:
- AWS management via official MCP servers
- GitHub operations via MCP
- Prometheus monitoring via MCP
- Kubernetes cluster management
- Multi-cloud support (GCP, Azure)
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

# MCP Agent imports
from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from srcs.common.llm.fallback_llm import create_fallback_orchestrator_llm_factory
from mcp_agent.workflows.llm.augmented_llm import RequestParams


class DevOpsProductivityAgent(BaseAgent):
    """Production DevOps Assistant with MCP server integrations"""

    def __init__(self, output_dir: str = "devops_reports"):
        super().__init__(
            name="devops_productivity_agent",
            instruction="전문 DevOps 엔지니어. AWS 리소스 관리, GitHub CI/CD, Kubernetes, 인프라 모니터링을 수행합니다. MCP 서버를 통해 다양한 클라우드 리소스와 도구들을 자동으로 조정합니다.",
            server_names=["aws-kb", "github", "prometheus", "kubernetes", "gcp-admin", "azure-admin"]
        )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Define agent capabilities
        self.capabilities = {
            "aws_management": "AWS EC2, S3, Lambda, CloudFormation 관리",
            "github_operations": "GitHub 리포지토리, PR, 이슈, CI/CD 파이프라인 관리",
            "kubernetes_ops": "Kubernetes 클러스터 및 워크로드 관리",
            "infrastructure_monitoring": "Prometheus 메트릭 기반 인프라 모니터링",
            "multi_cloud_coordination": "AWS, GCP, Azure 간 리소스 조정"
        }

    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized DevOps agents according to mcp_agent standards"""
        return {
            "aws_manager": Agent(
                name="aws_manager",
                instruction="AWS EC2, S3, Lambda, CloudFormation 관리 전문가. AWS 리소스 상태 확인, 생성, 삭제, 모니터링을 담당합니다.",
                server_names=["aws-kb"]
            ),
            "github_ops": Agent(
                name="github_ops",
                instruction="GitHub 리포지토리, PR, 이슈, CI/CD 파이프라인 관리 전문가. GitHub Actions 워크플로우 모니터링과 리포지토리 분석을 담당합니다.",
                server_names=["github"]
            ),
            "prometheus_monitor": Agent(
                name="prometheus_monitor",
                instruction="Prometheus 메트릭 기반 인프라 모니터링 전문가. 시스템 메트릭 수집, 분석, 알림 관리를 담당합니다.",
                server_names=["prometheus"]
            ),
            "k8s_ops": Agent(
                name="k8s_ops",
                instruction="Kubernetes 클러스터 및 워크로드 관리 전문가. Pod, Service, Deployment, ConfigMap 관리를 담당합니다.",
                server_names=["kubernetes"]
            ),
            "cloud_coordinator": Agent(
                name="cloud_coordinator",
                instruction="멀티클라우드 리소스 조정 전문가. AWS, GCP, Azure 간 리소스 비교, 마이그레이션, 통합 관리를 담당합니다.",
                server_names=["gcp-admin", "azure-admin"]
            )
        }

    async def run_workflow(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        mcp_agent 표준에 따른 워크플로우 실행
        """
        try:
            async with self.app.run() as devops_app:
                app_context = devops_app.context
                logger = devops_app.logger

                logger.info(f"Processing DevOps request: {request}")

                # 서버 설정
                if "filesystem" in app_context.config.mcp.servers:
                    app_context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                    logger.info("Filesystem server configured")

                # 전문 Agent 생성
                agents = self._create_agents()
                logger.info(f"Created {len(agents)} specialized agents: {list(agents.keys())}")

                # Orchestrator 생성
                orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(
                    primary_model="gemini-2.5-flash-lite",
                    logger_instance=logger
                )
                orchestrator = Orchestrator(
                    llm_factory=orchestrator_llm_factory,
                    available_agents=list(agents.values()),
                    plan_type="full"
                )

                # 실행
                result = await orchestrator.generate_str(
                    message=request,
                    request_params=RequestParams(
                        model="gemini-2.5-flash-latest",
                        temperature=0.1
                    )
                )

                # 결과 저장
                output_file = os.path.join(
                    self.output_dir,
                    f"devops_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                )

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)

                logger.info(f"Result saved to: {output_file}")

                return {
                    "status": "success",
                    "request": request,
                    "result": result,
                    "output_file": output_file,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Workflow failed: {str(e)}")
            return {
                "status": "error",
                "request": request,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test the DevOps assistant with mcp_agent standard integration"""
    agent = DevOpsProductivityAgent()

    # Test with sample requests
    test_requests = [
        "AWS EC2 인스턴스 상태를 확인해주세요",
        "GitHub microsoft 조직의 리포지토리를 분석해주세요",
        "Kubernetes 클러스터의 리소스 사용률을 조회해주세요",
        "Prometheus 메트릭을 통해 인프라 상태를 모니터링해주세요"
    ]

    print("🚀 DevOps Productivity Agent - mcp_agent 표준 테스트")
    print("=" * 60)

    for i, request in enumerate(test_requests, 1):
        print(f"\n[{i}/{len(test_requests)}] 🔄 Processing: {request}")
        try:
            result = await agent.run_workflow(request)
            print(f"✅ Status: {result['status']}")
            if result['status'] == 'success':
                print(f"📁 Output saved to: {result['output_file']}")
                print(f"📄 Result preview: {result['result'][:200]}...")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(main())
