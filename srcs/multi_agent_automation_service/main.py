"""
Multi-Agent Automation Service - Main Entry Point

실제 mcp_agent 라이브러리를 사용한 Multi-Agent 자동화 서비스의 메인 진입점입니다.
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

from .orchestrator import MultiAgentOrchestrator
from .agents.code_review_agent import CodeReviewAgent
from .agents.documentation_agent import DocumentationAgent
from .agents.performance_agent import PerformanceAgent
from .agents.security_agent import SecurityAgent
from .agents.kubernetes_agent import KubernetesAgent  # 🆕 K8s Agent 추가
from .gemini_executor import GeminiCLIExecutor


class MultiAgentAutomationService:
    """Multi-Agent 자동화 서비스 메인 클래스"""
    
    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator()
        self.code_review_agent = CodeReviewAgent()
        self.documentation_agent = DocumentationAgent()
        self.performance_agent = PerformanceAgent()
        self.security_agent = SecurityAgent()
        self.kubernetes_agent = KubernetesAgent()  # 🆕 K8s Agent 추가
        self.gemini_executor = GeminiCLIExecutor()
    
    async def run_full_automation(self, target_path: str = "srcs") -> Dict[str, Any]:
        """전체 자동화 실행"""
        print("🚀 Starting Full Automation Workflow...")
        print(f"Target Path: {target_path}")
        print("=" * 60)
        
        result = await self.orchestrator.run_full_automation(target_path)
        
        print(f"\n✅ Full Automation Completed!")
        print(f"Success: {result.success}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Gemini Commands Executed: {len(result.gemini_commands)}")
        
        return {
            "success": result.success,
            "duration": result.total_duration,
            "gemini_commands_count": len(result.gemini_commands),
            "agent_results": result.agent_results
        }
    
    async def run_kubernetes_workflow(self, app_name: str = "myapp", 
                                    config_path: str = "k8s/") -> Dict[str, Any]:
        """Kubernetes 워크플로우 실행 🆕"""
        print("🐳 Starting Kubernetes Workflow...")
        print(f"Application: {app_name}")
        print(f"Config Path: {config_path}")
        print("=" * 60)
        
        result = await self.orchestrator.run_kubernetes_workflow(app_name, config_path)
        
        print(f"\n✅ Kubernetes Workflow Completed!")
        print(f"Success: {result.success}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"K8s Commands Executed: {len(result.gemini_commands)}")
        
        # K8s 특화 결과 출력
        if "kubernetes_deploy" in result.agent_results:
            k8s_deploy = result.agent_results["kubernetes_deploy"]
            print(f"Deployment Status: {k8s_deploy.status}")
            print(f"Target: {k8s_deploy.target}")
        
        if "kubernetes_monitor" in result.agent_results:
            k8s_monitor = result.agent_results["kubernetes_monitor"]
            print(f"Monitoring Status: {k8s_monitor.status}")
        
        return {
            "success": result.success,
            "duration": result.total_duration,
            "gemini_commands_count": len(result.gemini_commands),
            "kubernetes_results": result.agent_results
        }
    
    async def run_code_review_workflow(self, target_path: str = "srcs") -> Dict[str, Any]:
        """코드 리뷰 워크플로우 실행"""
        print("🔍 Starting Code Review Workflow...")
        print(f"Target Path: {target_path}")
        print("=" * 60)
        
        result = await self.orchestrator.run_code_review_workflow(target_path)
        
        print(f"\n✅ Code Review Completed!")
        print(f"Success: {result.success}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Review Commands Executed: {len(result.gemini_commands)}")
        
        return {
            "success": result.success,
            "duration": result.total_duration,
            "gemini_commands_count": len(result.gemini_commands),
            "review_results": result.agent_results
        }
    
    async def run_deployment_workflow(self, target_path: str = "srcs") -> Dict[str, Any]:
        """배포 워크플로우 실행"""
        print("🚀 Starting Deployment Workflow...")
        print(f"Target Path: {target_path}")
        print("=" * 60)
        
        result = await self.orchestrator.run_deployment_workflow(target_path)
        
        print(f"\n✅ Deployment Workflow Completed!")
        print(f"Success: {result.success}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Deployment Commands Executed: {len(result.gemini_commands)}")
        
        return {
            "success": result.success,
            "duration": result.total_duration,
            "gemini_commands_count": len(result.gemini_commands),
            "deployment_results": result.agent_results
        }
    
    async def run_individual_agent(self, agent_name: str, target_path: str = "srcs") -> Dict[str, Any]:
        """개별 Agent 실행"""
        print(f"🤖 Running Individual Agent: {agent_name}")
        print(f"Target Path: {target_path}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            if agent_name == "code_review":
                result = await self.code_review_agent.review_code(target_path)
            elif agent_name == "documentation":
                result = await self.documentation_agent.update_documentation(target_path)
            elif agent_name == "performance":
                result = await self.performance_agent.analyze_performance(target_path)
            elif agent_name == "security":
                result = await self.security_agent.security_scan(target_path)
            elif agent_name == "kubernetes":  # 🆕 K8s Agent 추가
                result = await self.kubernetes_agent.monitor_cluster()
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            # Gemini CLI 명령어 실행
            execution_results = []
            if hasattr(result, 'gemini_commands') and result.gemini_commands:
                execution_results = await self.gemini_executor.execute_batch_commands(
                    result.gemini_commands
                )
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ {agent_name.title()} Agent Completed!")
            print(f"Duration: {total_duration:.2f}s")
            print(f"Commands Executed: {len(result.gemini_commands) if hasattr(result, 'gemini_commands') else 0}")
            
            return {
                "success": True,
                "duration": total_duration,
                "agent_name": agent_name,
                "result": result,
                "execution_results": execution_results
            }
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            print(f"\n❌ {agent_name.title()} Agent Failed: {e}")
            
            return {
                "success": False,
                "duration": total_duration,
                "agent_name": agent_name,
                "error": str(e)
            }
    
    def show_summary(self):
        """전체 요약 정보 표시"""
        print("📊 Multi-Agent Automation Service Summary")
        print("=" * 60)
        
        # Orchestrator 요약
        orchestrator_summary = self.orchestrator.get_orchestration_summary()
        print(f"Total Orchestrations: {orchestrator_summary.get('total_orchestrations', 0)}")
        print(f"Success Rate: {orchestrator_summary.get('success_rate', 0):.2%}")
        print(f"Average Duration: {orchestrator_summary.get('average_duration', 0):.2f}s")
        
        # Agent 요약
        agent_summaries = self.orchestrator.get_agent_summaries()
        print("\n🤖 Agent Summaries:")
        
        for agent_name, summary in agent_summaries.items():
            if isinstance(summary, dict) and "message" not in summary:
                if "total_operations" in summary:
                    print(f"  {agent_name}: {summary.get('total_operations', 0)} operations, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
                elif "total_reviews" in summary:
                    print(f"  {agent_name}: {summary.get('total_reviews', 0)} reviews, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
                elif "total_documentations" in summary:
                    print(f"  {agent_name}: {summary.get('total_documentations', 0)} documentations, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
                elif "total_analyses" in summary:
                    print(f"  {agent_name}: {summary.get('total_analyses', 0)} analyses, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
                elif "total_scans" in summary:
                    print(f"  {agent_name}: {summary.get('total_scans', 0)} scans, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
                elif "total_executions" in summary:
                    print(f"  {agent_name}: {summary.get('total_executions', 0)} executions, "
                          f"{summary.get('success_rate', 0):.2%} success rate")
            else:
                print(f"  {agent_name}: {summary.get('message', 'No data')}")
        
        # 최근 워크플로우
        if orchestrator_summary.get('recent_orchestrations'):
            print("\n🕒 Recent Workflows:")
            for workflow in orchestrator_summary['recent_orchestrations']:
                status = "✅" if workflow['success'] else "❌"
                print(f"  {status} {workflow['workflow_type']}: "
                      f"{workflow['duration']:.2f}s, {workflow['gemini_commands_count']} commands")


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Multi-Agent Automation Service")
    parser.add_argument("--workflow", "-w", 
                       choices=["full", "kubernetes", "code_review", "deployment"],
                       default="full",
                       help="Workflow type to run")
    parser.add_argument("--agent", "-a",
                       choices=["code_review", "documentation", "performance", "security", "kubernetes"],
                       help="Individual agent to run")
    parser.add_argument("--target", "-t", default="srcs",
                       help="Target path for analysis")
    parser.add_argument("--app-name", default="myapp",
                       help="Application name for Kubernetes workflow")
    parser.add_argument("--config-path", default="k8s/",
                       help="Config path for Kubernetes workflow")
    parser.add_argument("--summary", "-s", action="store_true",
                       help="Show summary information")
    
    args = parser.parse_args()
    
    service = MultiAgentAutomationService()
    
    try:
        if args.summary:
            service.show_summary()
            return
        
        if args.agent:
            # 개별 Agent 실행
            result = await service.run_individual_agent(args.agent, args.target)
        elif args.workflow == "kubernetes":
            # Kubernetes 워크플로우 실행
            result = await service.run_kubernetes_workflow(args.app_name, args.config_path)
        elif args.workflow == "code_review":
            # 코드 리뷰 워크플로우 실행
            result = await service.run_code_review_workflow(args.target)
        elif args.workflow == "deployment":
            # 배포 워크플로우 실행
            result = await service.run_deployment_workflow(args.target)
        else:
            # 전체 자동화 실행
            result = await service.run_full_automation(args.target)
        
        # 최종 요약
        print("\n" + "=" * 60)
        service.show_summary()
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 