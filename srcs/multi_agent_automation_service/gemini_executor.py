"""
Gemini CLI Executor

실제 mcp_agent 라이브러리를 사용한 Gemini CLI 명령어 실행기입니다.
"""

import asyncio
import subprocess
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.common.utils import setup_agent_app


@dataclass
class GeminiExecutionResult:
    """Gemini CLI 실행 결과"""
    command: str
    output: str
    error: Optional[str]
    exit_code: int
    execution_time: float
    timestamp: datetime


class GeminiCLIExecutor:
    """Gemini CLI 명령어 실행기 - 실제 mcp_agent 표준 사용"""
    
    def __init__(self):
        self.app = setup_agent_app("gemini_executor_system")
        self.agent = Agent(
            name="gemini_executor",
            instruction="""
            당신은 Gemini CLI 명령어 실행 전문가입니다. 다음을 수행하세요:
            
            1. Agent들이 생성한 Gemini CLI 명령어 검증
            2. 명령어 실행 및 결과 수집
            3. 실행 오류 처리 및 재시도
            4. 실행 결과 분석 및 요약
            5. 배치 명령어 실행 최적화
            
            안전하고 효율적으로 Gemini CLI 명령어를 실행하세요.
            """,
            server_names=["filesystem"],  # 실제 MCP 서버명
        )
        self.execution_history: List[GeminiExecutionResult] = []
        self._check_gemini_installation()
    
    def _check_gemini_installation(self) -> bool:
        """Gemini CLI 설치 확인"""
        try:
            result = subprocess.run(
                ["gemini", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ Gemini CLI installed: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ Gemini CLI not found or not working: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ Gemini CLI not available: {e}")
            return False
    
    async def execute_command(self, command: str, timeout: int = 60) -> GeminiExecutionResult:
        """단일 Gemini CLI 명령어 실행"""
        start_time = datetime.now()
        
        try:
            # 명령어 검증
            if not command.strip().startswith("gemini"):
                raise ValueError("Command must start with 'gemini'")
            
            # 명령어 실행
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            execution_result = GeminiExecutionResult(
                command=command,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                exit_code=result.returncode,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            self.execution_history.append(execution_result)
            
            if result.returncode != 0:
                print(f"⚠️ Command failed: {command}")
                print(f"Error: {result.stderr}")
            else:
                print(f"✅ Command executed successfully: {command}")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = GeminiExecutionResult(
                command=command,
                output="",
                error=f"Command timed out after {timeout} seconds",
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            self.execution_history.append(error_result)
            return error_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = GeminiExecutionResult(
                command=command,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            self.execution_history.append(error_result)
            return error_result
    
    async def execute_batch_commands(self, commands: List[str], 
                                   max_concurrent: int = 3) -> List[GeminiExecutionResult]:
        """배치 명령어 실행"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(command: str) -> GeminiExecutionResult:
            async with semaphore:
                return await self.execute_command(command)
        
        # 병렬 실행
        tasks = [execute_with_semaphore(cmd) for cmd in commands]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        execution_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = GeminiExecutionResult(
                    command=commands[i],
                    output="",
                    error=str(result),
                    exit_code=-1,
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
                execution_results.append(error_result)
            else:
                execution_results.append(result)
        
        return execution_results
    
    async def execute_with_context(self, commands: List[str], 
                                 context: Dict[str, Any]) -> List[GeminiExecutionResult]:
        """컨텍스트를 고려한 명령어 실행"""
        async with self.app.run() as app_context:
            app_logger = app_context.logger
            
            # 컨텍스트 기반 명령어 최적화
            optimized_commands = await self._optimize_commands_with_context(commands, context)
            
            app_logger.info(f"Executing {len(optimized_commands)} commands with context")
            
            results = []
            for command in optimized_commands:
                result = await self.execute_command(command)
                results.append(result)
                
                # 컨텍스트 업데이트
                if result.exit_code == 0:
                    context["last_successful_command"] = command
                    context["last_successful_output"] = result.output
                else:
                    context["last_failed_command"] = command
                    context["last_error"] = result.error
            
            return results
    
    async def _optimize_commands_with_context(self, commands: List[str], 
                                            context: Dict[str, Any]) -> List[str]:
        """컨텍스트를 고려한 명령어 최적화"""
        async with self.agent:
            llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
            
            prompt = f"""
            다음 명령어들을 컨텍스트를 고려하여 최적화하세요:
            
            명령어들:
            {json.dumps(commands, indent=2)}
            
            컨텍스트:
            {json.dumps(context, indent=2)}
            
            다음을 고려하세요:
            1. 중복 명령어 제거
            2. 순서 최적화
            3. 의존성 고려
            4. 효율성 개선
            
            최적화된 명령어 목록을 JSON 배열로 반환하세요.
            """
            
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(model="gpt-4o")
            )
            
            try:
                # JSON 파싱 시도
                optimized_commands = json.loads(result)
                if isinstance(optimized_commands, list):
                    return optimized_commands
            except json.JSONDecodeError:
                pass
            
            # 파싱 실패시 원본 반환
            return commands
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """실행 요약 정보"""
        if not self.execution_history:
            return {"message": "No commands executed yet"}
        
        total_commands = len(self.execution_history)
        successful_commands = sum(1 for result in self.execution_history 
                                if result.exit_code == 0)
        failed_commands = total_commands - successful_commands
        total_execution_time = sum(result.execution_time 
                                  for result in self.execution_history)
        
        # 명령어 타입별 통계
        command_types = {}
        for result in self.execution_history:
            cmd_type = result.command.split()[1] if len(result.command.split()) > 1 else "unknown"
            if cmd_type not in command_types:
                command_types[cmd_type] = {"total": 0, "success": 0, "failed": 0}
            command_types[cmd_type]["total"] += 1
            if result.exit_code == 0:
                command_types[cmd_type]["success"] += 1
            else:
                command_types[cmd_type]["failed"] += 1
        
        return {
            "total_commands": total_commands,
            "successful_commands": successful_commands,
            "failed_commands": failed_commands,
            "success_rate": successful_commands / total_commands if total_commands > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": total_execution_time / total_commands if total_commands > 0 else 0,
            "command_types": command_types,
            "recent_executions": [
                {
                    "command": result.command[:50] + "..." if len(result.command) > 50 else result.command,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.execution_history[-10:]  # 최근 10개
            ]
        }
    
    def get_failed_commands(self) -> List[GeminiExecutionResult]:
        """실패한 명령어 목록"""
        return [result for result in self.execution_history if result.exit_code != 0]
    
    def get_slow_commands(self, threshold: float = 10.0) -> List[GeminiExecutionResult]:
        """느린 명령어 목록"""
        return [result for result in self.execution_history 
                if result.execution_time > threshold]
    
    async def retry_failed_commands(self, max_retries: int = 3) -> List[GeminiExecutionResult]:
        """실패한 명령어 재시도"""
        failed_commands = self.get_failed_commands()
        retry_results = []
        
        for failed_result in failed_commands:
            for attempt in range(max_retries):
                print(f"Retrying command (attempt {attempt + 1}/{max_retries}): {failed_result.command}")
                retry_result = await self.execute_command(failed_result.command)
                
                if retry_result.exit_code == 0:
                    retry_results.append(retry_result)
                    break
                elif attempt == max_retries - 1:
                    retry_results.append(retry_result)
        
        return retry_results


async def main():
    """테스트 실행"""
    executor = GeminiCLIExecutor()
    
    # 단일 명령어 실행
    result = await executor.execute_command("gemini --version")
    print(f"Command executed: {result.command}")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    
    # 배치 명령어 실행
    commands = [
        "gemini --version",
        "gemini help",
        "gemini --help"
    ]
    
    batch_results = await executor.execute_batch_commands(commands)
    print(f"Batch execution completed: {len(batch_results)} commands")
    
    # 요약 정보
    summary = executor.get_execution_summary()
    print(f"Execution summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main()) 