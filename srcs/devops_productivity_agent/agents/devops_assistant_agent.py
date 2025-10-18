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
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# MCP Agent imports
from srcs.core.agent.base import BaseAgent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.google_augmented_llm import GoogleAugmentedLLM
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
        
        # Initialize Google LLM with latest Gemini model
        self.model_name = "gemini-2.5-flash-latest"
        self.llm = GoogleAugmentedLLM(
            model_name=self.model_name,
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        
        # Define agent capabilities
        self.capabilities = {
            "aws_management": "AWS EC2, S3, Lambda, CloudFormation 관리",
            "github_operations": "GitHub 리포지토리, PR, 이슈, CI/CD 파이프라인 관리",
            "kubernetes_ops": "Kubernetes 클러스터 및 워크로드 관리",
            "infrastructure_monitoring": "Prometheus 메트릭 기반 인프라 모니터링",
            "multi_cloud_coordination": "AWS, GCP, Azure 간 리소스 조정"
        }
    
    async def run_workflow(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main workflow using Orchestrator for automatic tool selection and execution.
        The LLM will automatically choose appropriate MCP tools based on the request.
        """
        try:
            self.logger.info(f"Processing DevOps request: {request}")
            
            # Create orchestrator for automatic tool selection
            orchestrator = self.get_orchestrator([])
            
            # Prepare context for the orchestrator
            workflow_context = {
                "request": request,
                "capabilities": self.capabilities,
                "timestamp": datetime.now().isoformat(),
                **(context or {})
            }
            
            # Let the orchestrator handle the request using available MCP tools
            result = await orchestrator.execute(request, workflow_context)
            
            # Save result to output directory
            output_file = os.path.join(
                self.output_dir, 
                f"devops_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"DevOps workflow completed. Result saved to: {output_file}")
            
            return {
                "status": "success",
                "request": request,
                "result": result,
                "output_file": output_file,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DevOps workflow failed: {str(e)}")
            return {
                "status": "error",
                "request": request,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """Test the DevOps assistant with MCP integration"""
    agent = DevOpsProductivityAgent()
    
    # Test with sample requests
    test_requests = [
        "AWS EC2 인스턴스 상태를 확인해주세요",
        "GitHub microsoft 조직의 리포지토리를 분석해주세요",
        "Kubernetes 클러스터의 리소스 사용률을 조회해주세요",
        "Prometheus 메트릭을 통해 인프라 상태를 모니터링해주세요"
    ]
    
    for request in test_requests:
        print(f"\n🔄 Processing: {request}")
        try:
            result = await agent.run_workflow(request)
            print(f"✅ Status: {result['status']}")
            if result['status'] == 'success':
                print(f"📁 Output saved to: {result['output_file']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Exception: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 