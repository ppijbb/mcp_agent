"""
Multi-Agent Orchestrator

실제 mcp_agent 라이브러리를 사용한 Multi-Agent 조율 시스템입니다.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating
)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from srcs.common.utils import setup_agent_app, save_report

from agents.code_review_agent import CodeReviewAgent
from agents.documentation_agent import DocumentationAgent
from agents.performance_agent import PerformanceAgent
from agents.security_agent import SecurityAgent
from agents.kubernetes_agent import KubernetesAgent
from .external_mcp import configure_external_servers
from gemini_executor import GeminiCLIExecutor


@dataclass
class OrchestrationResult:
    """조율 결과"""
    workflow_type: str
    agent_results: Dict[str, Any]
    gemini_commands: List[str]
    execution_results: List[Any]
    total_duration: float
    success: bool
    timestamp: datetime


class MultiAgentOrchestrator:
    """Multi-Agent Orchestrator - 실제 mcp_agent 표준 사용"""
    
    def __init__(self):
        self.app = setup_agent_app("multi_agent_orchestrator")
        self.code_review_agent = CodeReviewAgent()
        self.documentation_agent = DocumentationAgent()
        self.performance_agent = PerformanceAgent()
        self.security_agent = SecurityAgent()
        self.kubernetes_agent = KubernetesAgent()  # 🆕 K8s Agent 추가
        self.gemini_executor = GeminiCLIExecutor()
        
        # 메인 Orchestrator Agent
        self.orchestrator = Orchestrator(
            llm_factory=GoogleAugmentedLLM,
            name="automation_orchestrator",
            server_names=["filesystem", "playwright", "fetch", "kubernetes"],  # K8s 서버 추가
        )
        self.orchestration_history: List[OrchestrationResult] = []
    
    async def run_full_automation(self, target_path: str = "srcs") -> OrchestrationResult:
        """전체 자동화 워크플로우 실행"""
        start_time = datetime.now()
        
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                # 파일시스템 서버 설정
                if "filesystem" in context.config.mcp.servers:
                    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                    logger.info("Filesystem server configured")

                # 외부 MCP 서버 동적 등록 (OpenAPI/Oracle/Alpaca 등)
                added = configure_external_servers(
                    context,
                    candidates=[
                        "openapi",       # 범용 OpenAPI MCP
                        "oracle",        # 오라클 DB MCP
                        "alpaca",        # 트레이딩 MCP
                        "finnhub",       # 시세/뉴스 MCP
                        "polygon",       # 시세/지표 MCP
                        "edgar",         # 공시/규제 MCP
                        "coinstats",     # 크립토 MCP
                    ],
                )
                if added:
                    logger.info(f"External MCP servers configured: {added}")
                
                logger.info("Starting full automation workflow")
                
                # 1. 병렬로 모든 Agent 실행
                agent_tasks = [
                    self.code_review_agent.review_code(target_path),
                    self.documentation_agent.update_documentation(target_path),
                    self.performance_agent.analyze_performance(target_path),
                    self.security_agent.security_scan(target_path),
                    self.kubernetes_agent.monitor_cluster()  # 🆕 K8s 모니터링 추가
                ]
                
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                # 2. 결과 수집 및 Gemini CLI 명령어 추출
                all_gemini_commands = []
                agent_results_dict = {}
                
                agent_names = ["code_review", "documentation", "performance", "security", "kubernetes"]
                for i, (name, result) in enumerate(zip(agent_names, agent_results)):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {name} failed: {result}")
                        agent_results_dict[name] = {"error": str(result)}
                    else:
                        agent_results_dict[name] = result
                        # Gemini CLI 명령어 수집
                        if hasattr(result, 'gemini_commands'):
                            all_gemini_commands.extend(result.gemini_commands)
                        elif isinstance(result, list):
                            for item in result:
                                if hasattr(item, 'gemini_commands'):
                                    all_gemini_commands.extend(item.gemini_commands)
                
                # 3. Gemini CLI 명령어 실행
                execution_results = []
                if all_gemini_commands:
                    logger.info(f"Executing {len(all_gemini_commands)} Gemini CLI commands")
                    execution_results = await self.gemini_executor.execute_batch_commands(
                        all_gemini_commands
                    )
                
                # 4. 전체 결과 평가
                success = all(not isinstance(result, Exception) for result in agent_results)
                total_duration = (datetime.now() - start_time).total_seconds()
                
                orchestration_result = OrchestrationResult(
                    workflow_type="full_automation",
                    agent_results=agent_results_dict,
                    gemini_commands=all_gemini_commands,
                    execution_results=execution_results,
                    total_duration=total_duration,
                    success=success,
                    timestamp=datetime.now()
                )
                
                self.orchestration_history.append(orchestration_result)
                
                logger.info(f"Full automation completed in {total_duration:.2f}s")
                logger.info(f"Success: {success}")
                
                return orchestration_result
                
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_result = OrchestrationResult(
                workflow_type="full_automation",
                agent_results={"error": str(e)},
                gemini_commands=[],
                execution_results=[],
                total_duration=total_duration,
                success=False,
                timestamp=datetime.now()
            )
            self.orchestration_history.append(error_result)
            return error_result
    
    async def run_kubernetes_workflow(self, app_name: str = "myapp", 
                                    config_path: str = "k8s/") -> OrchestrationResult:
        """Kubernetes 워크플로우 실행 🆕"""
        start_time = datetime.now()
        
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                if "filesystem" in context.config.mcp.servers:
                    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                
                logger.info(f"Starting Kubernetes workflow for {app_name}")
                
                # 1. K8s 배포 전 보안 검증
                security_result = await self.security_agent.security_scan(config_path)
                
                # 2. K8s 애플리케이션 배포
                k8s_deploy_result = await self.kubernetes_agent.deploy_application(app_name, config_path)
                
                # 3. 배포 후 모니터링
                k8s_monitor_result = await self.kubernetes_agent.monitor_cluster()
                
                # 4. 성능 분석
                performance_result = await self.performance_agent.analyze_performance(config_path)
                
                # 모든 Gemini CLI 명령어 수집
                all_gemini_commands = []
                all_gemini_commands.extend(security_result.gemini_commands)
                all_gemini_commands.extend(k8s_deploy_result.gemini_commands)
                all_gemini_commands.extend(k8s_monitor_result.gemini_commands)
                all_gemini_commands.extend(performance_result.gemini_commands)
                
                # Gemini CLI 명령어 실행
                execution_results = []
                if all_gemini_commands:
                    execution_results = await self.gemini_executor.execute_batch_commands(
                        all_gemini_commands
                    )
                
                # 롤백 필요 여부 확인
                should_rollback = (security_result.should_rollback or 
                                 k8s_deploy_result.status == "FAILED")
                
                if should_rollback:
                    rollback_result = await self.kubernetes_agent.rollback_deployment(app_name)
                    all_gemini_commands.extend(rollback_result.gemini_commands)
                    
                    if rollback_result.gemini_commands:
                        rollback_executions = await self.gemini_executor.execute_batch_commands(
                            rollback_result.gemini_commands
                        )
                        execution_results.extend(rollback_executions)
                
                total_duration = (datetime.now() - start_time).total_seconds()
                
                orchestration_result = OrchestrationResult(
                    workflow_type="kubernetes",
                    agent_results={
                        "security": security_result,
                        "kubernetes_deploy": k8s_deploy_result,
                        "kubernetes_monitor": k8s_monitor_result,
                        "performance": performance_result,
                        "rollback": rollback_result if should_rollback else None
                    },
                    gemini_commands=all_gemini_commands,
                    execution_results=execution_results,
                    total_duration=total_duration,
                    success=not should_rollback,
                    timestamp=datetime.now()
                )
                
                self.orchestration_history.append(orchestration_result)
                return orchestration_result
                
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_result = OrchestrationResult(
                workflow_type="kubernetes",
                agent_results={"error": str(e)},
                gemini_commands=[],
                execution_results=[],
                total_duration=total_duration,
                success=False,
                timestamp=datetime.now()
            )
            self.orchestration_history.append(error_result)
            return error_result
    
    async def run_code_review_workflow(self, target_path: str = "srcs") -> OrchestrationResult:
        """코드 리뷰 워크플로우 실행"""
        start_time = datetime.now()
        
        try:
            # 코드 리뷰 실행
            review_result = await self.code_review_agent.review_code(target_path)
            
            # Gemini CLI 명령어 실행
            execution_results = []
            if review_result.gemini_commands:
                execution_results = await self.gemini_executor.execute_batch_commands(
                    review_result.gemini_commands
                )
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            orchestration_result = OrchestrationResult(
                workflow_type="code_review",
                agent_results={"code_review": review_result},
                gemini_commands=review_result.gemini_commands,
                execution_results=execution_results,
                total_duration=total_duration,
                success=True,
                timestamp=datetime.now()
            )
            
            self.orchestration_history.append(orchestration_result)
            return orchestration_result
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_result = OrchestrationResult(
                workflow_type="code_review",
                agent_results={"error": str(e)},
                gemini_commands=[],
                execution_results=[],
                total_duration=total_duration,
                success=False,
                timestamp=datetime.now()
            )
            self.orchestration_history.append(error_result)
            return error_result
    
    async def run_deployment_workflow(self, target_path: str = "srcs") -> OrchestrationResult:
        """배포 워크플로우 실행"""
        start_time = datetime.now()
        
        try:
            # 보안 검증 및 배포 검증
            security_result = await self.security_agent.security_scan(target_path)
            deployment_result = await self.security_agent.verify_deployment(target_path)
            
            # 성능 분석
            performance_result = await self.performance_agent.analyze_performance(target_path)
            
            # 모든 Gemini CLI 명령어 수집
            all_gemini_commands = []
            all_gemini_commands.extend(security_result.gemini_commands)
            all_gemini_commands.extend(deployment_result.gemini_commands)
            all_gemini_commands.extend(performance_result.gemini_commands)
            
            # Gemini CLI 명령어 실행
            execution_results = []
            if all_gemini_commands:
                execution_results = await self.gemini_executor.execute_batch_commands(
                    all_gemini_commands
                )
            
            # 롤백 필요 여부 확인
            should_rollback = (security_result.should_rollback or 
                             deployment_result.should_rollback)
            
            if should_rollback:
                rollback_result = await self.security_agent.auto_rollback()
                all_gemini_commands.extend(rollback_result.gemini_commands)
                
                if rollback_result.gemini_commands:
                    rollback_executions = await self.gemini_executor.execute_batch_commands(
                        rollback_result.gemini_commands
                    )
                    execution_results.extend(rollback_executions)
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            orchestration_result = OrchestrationResult(
                workflow_type="deployment",
                agent_results={
                    "security": security_result,
                    "deployment": deployment_result,
                    "performance": performance_result,
                    "rollback": rollback_result if should_rollback else None
                },
                gemini_commands=all_gemini_commands,
                execution_results=execution_results,
                total_duration=total_duration,
                success=not should_rollback,
                timestamp=datetime.now()
            )
            
            self.orchestration_history.append(orchestration_result)
            return orchestration_result
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_result = OrchestrationResult(
                workflow_type="deployment",
                agent_results={"error": str(e)},
                gemini_commands=[],
                execution_results=[],
                total_duration=total_duration,
                success=False,
                timestamp=datetime.now()
            )
            self.orchestration_history.append(error_result)
            return error_result
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """조율 요약 정보"""
        if not self.orchestration_history:
            return {"message": "No orchestrations performed yet"}
        
        workflow_types = {}
        total_duration = 0
        successful_workflows = 0
        
        for result in self.orchestration_history:
            workflow_type = result.workflow_type
            if workflow_type not in workflow_types:
                workflow_types[workflow_type] = {"total": 0, "success": 0, "failed": 0}
            workflow_types[workflow_type]["total"] += 1
            
            if result.success:
                workflow_types[workflow_type]["success"] += 1
                successful_workflows += 1
            else:
                workflow_types[workflow_type]["failed"] += 1
            
            total_duration += result.total_duration
        
        return {
            "total_orchestrations": len(self.orchestration_history),
            "successful_workflows": successful_workflows,
            "failed_workflows": len(self.orchestration_history) - successful_workflows,
            "success_rate": successful_workflows / len(self.orchestration_history) if self.orchestration_history else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / len(self.orchestration_history) if self.orchestration_history else 0,
            "workflow_types": workflow_types,
            "recent_orchestrations": [
                {
                    "workflow_type": result.workflow_type,
                    "success": result.success,
                    "duration": result.total_duration,
                    "gemini_commands_count": len(result.gemini_commands),
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.orchestration_history[-5:]  # 최근 5개
            ]
        }
    
    def get_agent_summaries(self) -> Dict[str, Any]:
        """각 Agent의 요약 정보"""
        return {
            "code_review": self.code_review_agent.get_review_summary(),
            "documentation": self.documentation_agent.get_documentation_summary(),
            "performance": self.performance_agent.get_performance_summary(),
            "security": self.security_agent.get_security_summary(),
            "kubernetes": self.kubernetes_agent.get_kubernetes_summary(),  # 🆕 K8s 요약 추가
            "gemini_executor": self.gemini_executor.get_execution_summary()
        }


async def main():
    """테스트 실행"""
    orchestrator = MultiAgentOrchestrator()
    
    # 전체 자동화 실행
    result = await orchestrator.run_full_automation()
    print(f"Full automation completed: {result.success}")
    print(f"Duration: {result.total_duration:.2f}s")
    print(f"Gemini commands executed: {len(result.gemini_commands)}")
    
    # K8s 워크플로우 실행
    k8s_result = await orchestrator.run_kubernetes_workflow("testapp")
    print(f"Kubernetes workflow completed: {k8s_result.success}")
    print(f"Duration: {k8s_result.total_duration:.2f}s")
    print(f"K8s commands executed: {len(k8s_result.gemini_commands)}")
    
    # 요약 정보
    summary = orchestrator.get_orchestration_summary()
    print(f"Orchestration summary: {summary}")
    
    # Agent 요약 정보
    agent_summaries = orchestrator.get_agent_summaries()
    print(f"Agent summaries: {agent_summaries}")


if __name__ == "__main__":
    asyncio.run(main()) 