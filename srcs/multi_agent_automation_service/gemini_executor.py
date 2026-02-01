"""
Gemini CLI Executor with MCP Integration

실제 mcp_agent 라이브러리를 사용한 Gemini CLI 명령어 실행기입니다.
MCP(Model Context Protocol) 기반으로 Gemini CLI와 연결하여 도구들을 활용합니다.
"""

import asyncio
import subprocess
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
    mcp_tools_used: List[str]  # 사용된 MCP 도구들


class GeminiCLIExecutor:
    """Gemini CLI 명령어 실행기 - MCP 기반 통합"""

    def __init__(self):
        self.app = setup_agent_app("gemini_executor_system")
        self.agent = Agent(
            name="gemini_executor",
            instruction=(
                "당신은 지시를 엄격히 준수하는 실행 에이전트다.\n"
                "- 목표: 제공된 명령어를 안전하게 검증하고 필요한 MCP 도구로 실행한다.\n"
                "- 원칙: 명확성, 간결성, 결정성. 불필요한 서술 금지.\n"
                "- 금지: 요구된 출력 이외의 사족/사과/추측.\n"
                "- 형식: 요청된 결과만 반환한다."
            ),
            server_names=[
                "filesystem",
                "kubernetes",
                "github",
                "gemini-cli"  # Gemini CLI MCP 서버
            ],
        )
        self.execution_history: List[GeminiExecutionResult] = []
        self.gemini_cli_config = self._setup_gemini_cli_config()
        self._check_gemini_installation()

    def _setup_gemini_cli_config(self) -> Dict[str, Any]:
        """Gemini CLI MCP 설정 구성"""
        return {
            "mcpServers": {
                "kubernetes": {
                    "command": "kubectl",
                    "args": ["proxy", "--port=8001"],
                    "timeout": 30000,
                    "trust": True
                },
                "filesystem": {
                    "command": "mcp-server-filesystem",
                    "args": ["--root", os.getcwd()],
                    "timeout": 15000,
                    "trust": True
                },
                "github": {
                    "command": "docker",
                    "args": [
                        "run", "-i", "--rm",
                        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                        "ghcr.io/github/github-mcp-server"
                    ],
                    "env": {
                        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
                    },
                    "timeout": 30000
                }
            }
        }

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

    async def execute_command_with_mcp(self, command: str, context: Dict[str, Any] = None) -> GeminiExecutionResult:
        """MCP 기반 Gemini CLI 명령어 실행"""
        start_time = datetime.now()

        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger

            try:
                # MCP 서버 설정
                if "filesystem" in context.config.mcp.servers:
                    context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                    logger.info("Filesystem MCP server configured")
                # 외부 MCP 서버 추가 등록: orchestrator와 동일 환경 변수를 사용
                try:
                    from .external_mcp import configure_external_servers
                    added = configure_external_servers(
                        context,
                        candidates=[
                            "openapi", "oracle", "alpaca", "finnhub", "polygon", "edgar", "coinstats"
                        ],
                    )
                    if added:
                        logger.info(f"External MCP servers configured: {added}")
                except Exception as e:
                    logger.warning(f"External MCP configuration skipped: {e}")

                async with self.agent:
                    llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                    # Gemini CLI 명령어를 MCP 도구 호출로 변환
                    mcp_tools_used = []

                    if "kubectl" in command:
                        # Kubernetes 관련 명령어
                        k8s_result = await self._execute_k8s_command(command, context)
                        mcp_tools_used.append("kubernetes")
                        output = k8s_result.get("output", "")
                        error = k8s_result.get("error")
                        exit_code = k8s_result.get("exit_code", 0)

                    elif "gemini" in command:
                        # Gemini CLI 직접 명령어
                        gemini_result = await self._execute_gemini_cli_command(command, context)
                        mcp_tools_used.append("gemini-cli")
                        output = gemini_result.get("output", "")
                        error = gemini_result.get("error")
                        exit_code = gemini_result.get("exit_code", 0)

                    else:
                        # 일반적인 파일시스템 작업
                        fs_result = await self._execute_filesystem_command(command, context)
                        mcp_tools_used.append("filesystem")
                        output = fs_result.get("output", "")
                        error = fs_result.get("error")
                        exit_code = fs_result.get("exit_code", 0)

                    execution_time = (datetime.now() - start_time).total_seconds()

                    execution_result = GeminiExecutionResult(
                        command=command,
                        output=output,
                        error=error,
                        exit_code=exit_code,
                        execution_time=execution_time,
                        timestamp=datetime.now(),
                        mcp_tools_used=mcp_tools_used
                    )

                    self.execution_history.append(execution_result)

                    if exit_code != 0:
                        logger.warning(f"Command failed: {command}")
                        logger.warning(f"Error: {error}")
                    else:
                        logger.info(f"Command executed successfully: {command}")

                    return execution_result

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_result = GeminiExecutionResult(
                    command=command,
                    output="",
                    error=str(e),
                    exit_code=-1,
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    mcp_tools_used=[]
                )
                self.execution_history.append(error_result)
                return error_result

    async def _execute_k8s_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Kubernetes 명령어 MCP 실행"""
        try:
            # kubectl 명령어 파싱
            if "get" in command:
                return await self._execute_k8s_get_command(command, context)
            elif "apply" in command:
                return await self._execute_k8s_apply_command(command, context)
            elif "delete" in command:
                return await self._execute_k8s_delete_command(command, context)
            else:
                return await self._execute_k8s_generic_command(command, context)
        except Exception as e:
            return {"output": "", "error": str(e), "exit_code": 1}

    async def _execute_gemini_cli_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini CLI 명령어 MCP 실행"""
        try:
            # Gemini CLI 명령어를 MCP 도구 호출로 변환
            # 예: gemini -y -p "analyze this code" -> MCP 도구 호출
            prompt = self._extract_prompt_from_gemini_command(command)

            # LLM을 통한 분석 실행
            llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.augmented_generate(
                RequestParams(
                    messages=[{"role": "user", "content": prompt}],
                    tools_choice="auto"
                )
            )

            return {
                "output": result.content,
                "error": None,
                "exit_code": 0
            }
        except Exception as e:
            return {"output": "", "error": str(e), "exit_code": 1}

    async def _execute_filesystem_command(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """파일시스템 명령어 MCP 실행"""
        try:
            # 파일시스템 관련 작업을 MCP 도구로 실행
            if "read" in command or "cat" in command:
                return await self._execute_file_read_command(command, context)
            elif "write" in command or "echo" in command:
                return await self._execute_file_write_command(command, context)
            else:
                return await self._execute_generic_filesystem_command(command, context)
        except Exception as e:
            return {"output": "", "error": str(e), "exit_code": 1}

    def _extract_prompt_from_gemini_command(self, command: str) -> str:
        """Gemini CLI 명령어에서 프롬프트 추출"""
        # gemini -y -p "prompt here" 형태에서 프롬프트 추출
        if "-p" in command:
            parts = command.split("-p")
            if len(parts) > 1:
                prompt = parts[1].strip().strip('"')
                return prompt
        return command

    # 기존 메서드들은 유지하되 MCP 기반으로 수정
    async def execute_command(self, command: str, timeout: int = 60) -> GeminiExecutionResult:
        """단일 Gemini CLI 명령어 실행 (MCP 기반)"""
        return await self.execute_command_with_mcp(command)

    async def execute_batch_commands(self, commands: List[str],
                                   max_concurrent: int = 3) -> List[GeminiExecutionResult]:
        """배치 명령어 실행 (MCP 기반)"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(command: str) -> GeminiExecutionResult:
            async with semaphore:
                return await self.execute_command_with_mcp(command)

        tasks = [execute_with_semaphore(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)

    async def execute_with_context(self, commands: List[str],
                                 context: Dict[str, Any]) -> List[GeminiExecutionResult]:
        """컨텍스트와 함께 명령어 실행 (MCP 기반)"""
        # 컨텍스트를 활용하여 명령어 최적화
        optimized_commands = await self._optimize_commands_with_context(commands, context)

        results = []
        for command in optimized_commands:
            result = await self.execute_command_with_mcp(command, context)
            results.append(result)

        return results

    async def _optimize_commands_with_context(self, commands: List[str],
                                            context: Dict[str, Any]) -> List[str]:
        """컨텍스트를 활용한 명령어 최적화"""
        # LLM을 사용하여 명령어 최적화
        optimization_prompt = f"""
        다음 명령어들을 컨텍스트에 맞게 최적화하세요:

        컨텍스트: {context}
        명령어들: {commands}

        최적화 요구사항:
        1. 중복 제거
        2. 순서 최적화
        3. MCP 도구 활용
        4. 성능 향상

        최적화된 명령어 목록을 JSON 배열로 반환하세요.
        """

        try:
            llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.augmented_generate(
                RequestParams(
                    messages=[{"role": "user", "content": optimization_prompt}],
                    tools_choice="auto"
                )
            )

            # JSON 파싱 시도
            try:
                optimized_commands = json.loads(result.content)
                return optimized_commands
            except json.JSONDecodeError:
                # JSON 파싱 실패시 원본 반환
                return commands

        except Exception as e:
            print(f"명령어 최적화 실패: {e}")
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
