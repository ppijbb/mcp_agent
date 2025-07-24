"""
Multi-Agent Orchestrator
========================

4개 Agent들의 협업을 조율하고 Gemini CLI 명령어를 실행하는 Orchestrator
"""

import asyncio
import json
import schedule
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator

from .agents import (
    CodeReviewAgent,
    DocumentationAgent,
    PerformanceTestAgent,
    SecurityDeploymentAgent
)
from .gemini_cli_executor import GeminiCLIExecutor

@dataclass
class OrchestrationResult:
    """Orchestration 결과"""
    orchestration_id: str
    timestamp: str
    agent_results: Dict[str, Any]
    gemini_commands_executed: List[str]
    overall_status: str
    execution_summary: str

class MultiAgentOrchestrator:
    """Multi-Agent Orchestrator"""
    
    def __init__(self):
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="multi_agent_orchestrator",
            human_input_callback=None
        )
        
        # 4개 Agent 초기화
        self.code_review_agent = CodeReviewAgent()
        self.documentation_agent = DocumentationAgent()
        self.performance_test_agent = PerformanceTestAgent()
        self.security_deployment_agent = SecurityDeploymentAgent()
        
        # Gemini CLI Executor 초기화
        self.gemini_executor = GeminiCLIExecutor()
        
        # Orchestrator 설정
        self.orchestrator = Orchestrator(
            name="automation_orchestrator",
            instruction="""
            당신은 4개 전문 Agent들의 협업을 조율하는 Orchestrator입니다.
            
            Agent 역할:
            1. CodeReviewAgent: 코드 리뷰 및 품질 검토
            2. DocumentationAgent: 자동 문서화
            3. PerformanceTestAgent: 성능 분석 및 테스트
            4. SecurityDeploymentAgent: 보안 스캔 및 배포 검증
            
            워크플로우:
            1. 각 Agent가 병렬로 작업 수행
            2. 결과 수집 및 분석
            3. Gemini CLI 명령어 통합
            4. 최종 실행 및 보고서 생성
            """,
            server_names=["orchestration-mcp", "workflow-mcp"],
        )
        
        self.orchestration_history: List[OrchestrationResult] = []
    
    async def run_full_automation(self, target_paths: List[str] = None) -> OrchestrationResult:
        """전체 자동화 워크플로우 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("Multi-Agent 자동화 워크플로우 시작")
                
                # 1. 병렬로 4개 Agent 실행
                tasks = [
                    self.code_review_agent.review_code(target_paths),
                    self.documentation_agent.update_documentation(target_paths),
                    self.performance_test_agent.analyze_performance(target_paths),
                    self.security_deployment_agent.security_scan(target_paths)
                ]
                
                # 병렬 실행
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 결과 수집
                agent_results = {
                    "code_review": results[0] if not isinstance(results[0], Exception) else None,
                    "documentation": results[1] if not isinstance(results[1], Exception) else None,
                    "performance_test": results[2] if not isinstance(results[2], Exception) else None,
                    "security_deployment": results[3] if not isinstance(results[3], Exception) else None
                }
                
                # 2. Gemini CLI 명령어 통합
                all_gemini_commands = []
                for agent_name, result in agent_results.items():
                    if result and hasattr(result, 'gemini_cli_commands'):
                        all_gemini_commands.extend(result.gemini_cli_commands)
                
                # 3. Gemini CLI 명령어 실행
                executed_commands = []
                for command in all_gemini_commands:
                    try:
                        execution_result = await self.gemini_executor.execute_command(command)
                        executed_commands.append({
                            "command": command,
                            "result": execution_result,
                            "status": "success"
                        })
                    except Exception as e:
                        executed_commands.append({
                            "command": command,
                            "result": str(e),
                            "status": "failed"
                        })
                
                # 4. 전체 상태 분석
                overall_status = self._analyze_overall_status(agent_results, executed_commands)
                
                # 5. 실행 요약 생성
                execution_summary = self._generate_execution_summary(agent_results, executed_commands)
                
                # OrchestrationResult 생성
                orchestration_result = OrchestrationResult(
                    orchestration_id=f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    agent_results=agent_results,
                    gemini_commands_executed=executed_commands,
                    overall_status=overall_status,
                    execution_summary=execution_summary
                )
                
                # 히스토리 저장
                self.orchestration_history.append(orchestration_result)
                
                logger.info(f"Multi-Agent 자동화 완료: {overall_status}")
                
                return orchestration_result
                
        except Exception as e:
            logger.error(f"Multi-Agent 자동화 실패: {e}")
            raise
    
    async def run_code_review_workflow(self, target_paths: List[str] = None) -> OrchestrationResult:
        """코드 리뷰 워크플로우"""
        try:
            # 1. 코드 리뷰
            code_review_result = await self.code_review_agent.review_code(target_paths)
            
            # 2. 심각한 이슈가 있으면 보안 스캔 추가 실행
            critical_issues = self.code_review_agent.get_critical_issues(code_review_result)
            security_result = None
            
            if critical_issues:
                security_result = await self.security_deployment_agent.security_scan(target_paths)
            
            # 3. Gemini CLI 명령어 실행
            gemini_commands = code_review_result.gemini_cli_commands
            if security_result:
                gemini_commands.extend(security_result.gemini_cli_commands)
            
            executed_commands = []
            for command in gemini_commands:
                try:
                    execution_result = await self.gemini_executor.execute_command(command)
                    executed_commands.append({
                        "command": command,
                        "result": execution_result,
                        "status": "success"
                    })
                except Exception as e:
                    executed_commands.append({
                        "command": command,
                        "result": str(e),
                        "status": "failed"
                    })
            
            return OrchestrationResult(
                orchestration_id=f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                agent_results={
                    "code_review": code_review_result,
                    "security_deployment": security_result
                },
                gemini_commands_executed=executed_commands,
                overall_status="completed",
                execution_summary=f"코드 리뷰 완료: {len(critical_issues)}개 심각한 이슈 발견"
            )
            
        except Exception as e:
            print(f"코드 리뷰 워크플로우 실패: {e}")
            raise
    
    async def run_deployment_workflow(self, deployment_id: str = None) -> OrchestrationResult:
        """배포 워크플로우"""
        try:
            # 1. 배포 검증
            deployment_result = await self.security_deployment_agent.verify_deployment(deployment_id)
            
            # 2. 롤백 필요성 확인
            if self.security_deployment_agent.should_rollback(deployment_result):
                rollback_result = await self.security_deployment_agent.auto_rollback(deployment_id)
                
                return OrchestrationResult(
                    orchestration_id=f"deployment_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    agent_results={
                        "deployment_verification": deployment_result,
                        "rollback": rollback_result
                    },
                    gemini_commands_executed=rollback_result.gemini_cli_commands,
                    overall_status="rolled_back",
                    execution_summary="배포 실패로 인한 자동 롤백 실행"
                )
            else:
                # 3. 성능 테스트 실행
                performance_result = await self.performance_test_agent.analyze_performance()
                
                # 4. 문서 업데이트
                documentation_result = await self.documentation_agent.update_documentation()
                
                # 5. Gemini CLI 명령어 실행
                all_commands = []
                all_commands.extend(deployment_result.gemini_cli_commands)
                all_commands.extend(performance_result.gemini_cli_commands)
                all_commands.extend(documentation_result.gemini_cli_commands)
                
                executed_commands = []
                for command in all_commands:
                    try:
                        execution_result = await self.gemini_executor.execute_command(command)
                        executed_commands.append({
                            "command": command,
                            "result": execution_result,
                            "status": "success"
                        })
                    except Exception as e:
                        executed_commands.append({
                            "command": command,
                            "result": str(e),
                            "status": "failed"
                        })
                
                return OrchestrationResult(
                    orchestration_id=f"deployment_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    agent_results={
                        "deployment_verification": deployment_result,
                        "performance_test": performance_result,
                        "documentation": documentation_result
                    },
                    gemini_commands_executed=executed_commands,
                    overall_status="deployed_successfully",
                    execution_summary="배포 성공 및 후속 작업 완료"
                )
                
        except Exception as e:
            print(f"배포 워크플로우 실패: {e}")
            raise
    
    def _analyze_overall_status(self, agent_results: Dict[str, Any], executed_commands: List[Dict]) -> str:
        """전체 상태 분석"""
        # 실패한 Agent 수 확인
        failed_agents = sum(1 for result in agent_results.values() if result is None)
        
        # 실패한 명령어 수 확인
        failed_commands = sum(1 for cmd in executed_commands if cmd["status"] == "failed")
        
        if failed_agents > 2:  # 절반 이상 실패
            return "failed"
        elif failed_agents > 0 or failed_commands > 0:  # 일부 실패
            return "partial_success"
        else:
            return "success"
    
    def _generate_execution_summary(self, agent_results: Dict[str, Any], executed_commands: List[Dict]) -> str:
        """실행 요약 생성"""
        summary = f"""
Multi-Agent 자동화 실행 요약
===========================

📊 Agent 실행 결과:
"""
        
        for agent_name, result in agent_results.items():
            if result:
                summary += f"- {agent_name}: ✅ 성공\n"
            else:
                summary += f"- {agent_name}: ❌ 실패\n"
        
        summary += f"\n🔧 Gemini CLI 명령어 실행: {len(executed_commands)}개\n"
        
        success_commands = sum(1 for cmd in executed_commands if cmd["status"] == "success")
        failed_commands = sum(1 for cmd in executed_commands if cmd["status"] == "failed")
        
        summary += f"- 성공: {success_commands}개\n"
        summary += f"- 실패: {failed_commands}개\n"
        
        return summary
    
    def setup_scheduled_automation(self):
        """스케줄된 자동화 설정"""
        # 매일 새벽 2시 - 전체 자동화
        schedule.every().day.at("02:00").do(
            lambda: asyncio.run(self.run_full_automation())
        )
        
        # 매주 월요일 오전 9시 - 코드 리뷰 워크플로우
        schedule.every().monday.at("09:00").do(
            lambda: asyncio.run(self.run_code_review_workflow())
        )
        
        # 매시간 - 배포 상태 확인
        schedule.every().hour.do(
            lambda: asyncio.run(self.run_deployment_workflow())
        )
    
    def run_scheduler(self):
        """스케줄러 실행"""
        print("Multi-Agent 자동화 스케줄러 시작...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크

# 사용 예시
async def main():
    """사용 예시"""
    orchestrator = MultiAgentOrchestrator()
    
    # 전체 자동화 실행
    result = await orchestrator.run_full_automation()
    print(result.execution_summary)
    
    # 코드 리뷰 워크플로우
    review_result = await orchestrator.run_code_review_workflow()
    print(f"코드 리뷰 완료: {review_result.overall_status}")
    
    # 배포 워크플로우
    deploy_result = await orchestrator.run_deployment_workflow("deployment-123")
    print(f"배포 상태: {deploy_result.overall_status}")

if __name__ == "__main__":
    asyncio.run(main()) 