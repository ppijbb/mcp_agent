"""
Kubernetes Agent

실제 mcp_agent 라이브러리를 사용한 Kubernetes 클러스터 제어 전문 Agent입니다.
"""

import asyncio
import os
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from srcs.common.utils import setup_agent_app


@dataclass
class KubernetesResult:
    """Kubernetes 작업 결과"""
    operation_type: str  # DEPLOY, SCALE, ROLLBACK, CONFIG, MONITOR
    target: str
    status: str  # SUCCESS, FAILED, PENDING
    output: str
    error: Optional[str]
    gemini_commands: List[str]
    timestamp: datetime


class KubernetesAgent:
    """Kubernetes 클러스터 제어 전담 Agent - 실제 mcp_agent 표준 사용"""

    def __init__(self):
        self.app = setup_agent_app("kubernetes_system")
        self.agent = Agent(
            name="kubernetes_operator",
            instruction="""
            당신은 전문적인 Kubernetes 클러스터 운영자입니다. 다음을 수행하세요:

            1. Kubernetes 리소스 배포 및 관리
            2. 애플리케이션 스케일링 및 업데이트
            3. 설정 관리 (ConfigMap, Secret)
            4. 모니터링 및 로그 분석
            5. 롤백 및 복구 작업
            6. 클러스터 상태 진단
            7. K8s 작업을 위한 Gemini CLI 명령어 생성

            kubectl, helm, kustomize 등의 도구를 활용하여
            실제 Kubernetes 클러스터를 제어하세요.
            """,
            server_names=["filesystem", "kubernetes", "kubectl"],  # K8s 관련 서버
        )
        self.k8s_history: List[KubernetesResult] = []
        self._check_kubectl_installation()

    def _check_kubectl_installation(self) -> bool:
        """kubectl 설치 확인"""
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False
            )
            if result.returncode == 0:
                print(f"✅ kubectl installed: {result.stdout.strip()}")
                return True
            else:
                print(f"❌ kubectl not found or not working: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"❌ kubectl not available: {e}")
            return False

    async def deploy_application(self, app_name: str, config_path: str = "k8s/") -> KubernetesResult:
        """애플리케이션 배포"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger

            # 파일시스템 서버 설정
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                # 배포 작업 수행
                deploy_prompt = f"""
                다음 애플리케이션을 Kubernetes에 배포하세요: {app_name}
                설정 경로: {config_path}

                다음을 수행하세요:
                1. 배포 매니페스트 검증
                2. 네임스페이스 확인/생성
                3. ConfigMap 및 Secret 적용
                4. Deployment 배포
                5. Service 생성
                6. Ingress 설정 (필요시)
                7. 배포 상태 확인

                각 단계에 대한 구체적인 kubectl 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=deploy_prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # kubectl 명령어 실행
                k8s_commands = self._extract_kubectl_commands(result)
                execution_results = []

                for command in k8s_commands:
                    try:
                        cmd_result = subprocess.run(
                            command.split(),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        execution_results.append({
                            "command": command,
                            "output": cmd_result.stdout,
                            "error": cmd_result.stderr,
                            "exit_code": cmd_result.returncode
                        })
                    except Exception as e:
                        execution_results.append({
                            "command": command,
                            "output": "",
                            "error": str(e),
                            "exit_code": -1
                        })

                # 결과 파싱 및 구조화
                k8s_result = self._parse_kubernetes_result(
                    "DEPLOY", app_name, execution_results, k8s_commands
                )
                self.k8s_history.append(k8s_result)

                return k8s_result

    async def scale_deployment(self, deployment_name: str, namespace: str = "default",
                             replicas: int = 3) -> KubernetesResult:
        """배포 스케일링"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 배포를 스케일링하세요:
                - 배포명: {deployment_name}
                - 네임스페이스: {namespace}
                - 레플리카 수: {replicas}

                다음을 수행하세요:
                1. 현재 배포 상태 확인
                2. 스케일링 명령어 실행
                3. 스케일링 완료 확인
                4. 새로운 상태 검증

                kubectl 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # kubectl 명령어 실행
                k8s_commands = self._extract_kubectl_commands(result)
                execution_results = []

                for command in k8s_commands:
                    try:
                        cmd_result = subprocess.run(
                            command.split(),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        execution_results.append({
                            "command": command,
                            "output": cmd_result.stdout,
                            "error": cmd_result.stderr,
                            "exit_code": cmd_result.returncode
                        })
                    except Exception as e:
                        execution_results.append({
                            "command": command,
                            "output": "",
                            "error": str(e),
                            "exit_code": -1
                        })

                k8s_result = self._parse_kubernetes_result(
                    "SCALE", f"{namespace}/{deployment_name}", execution_results, k8s_commands
                )
                self.k8s_history.append(k8s_result)

                return k8s_result

    async def update_config(self, config_type: str, name: str,
                          config_data: Dict[str, Any], namespace: str = "default") -> KubernetesResult:
        """설정 업데이트 (ConfigMap/Secret)"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 설정을 업데이트하세요:
                - 설정 타입: {config_type}
                - 이름: {name}
                - 네임스페이스: {namespace}
                - 설정 데이터: {json.dumps(config_data, indent=2)}

                다음을 수행하세요:
                1. 기존 설정 확인
                2. 새로운 설정 적용
                3. 설정 업데이트 확인
                4. 관련 Pod 재시작 (필요시)

                kubectl 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # kubectl 명령어 실행
                k8s_commands = self._extract_kubectl_commands(result)
                execution_results = []

                for command in k8s_commands:
                    try:
                        cmd_result = subprocess.run(
                            command.split(),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        execution_results.append({
                            "command": command,
                            "output": cmd_result.stdout,
                            "error": cmd_result.stderr,
                            "exit_code": cmd_result.returncode
                        })
                    except Exception as e:
                        execution_results.append({
                            "command": command,
                            "output": "",
                            "error": str(e),
                            "exit_code": -1
                        })

                k8s_result = self._parse_kubernetes_result(
                    "CONFIG", f"{namespace}/{name}", execution_results, k8s_commands
                )
                self.k8s_history.append(k8s_result)

                return k8s_result

    async def rollback_deployment(self, deployment_name: str, namespace: str = "default",
                                revision: Optional[int] = None) -> KubernetesResult:
        """배포 롤백"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 배포를 롤백하세요:
                - 배포명: {deployment_name}
                - 네임스페이스: {namespace}
                - 리비전: {revision if revision else '이전 버전'}

                다음을 수행하세요:
                1. 배포 히스토리 확인
                2. 롤백 대상 리비전 확인
                3. 롤백 실행
                4. 롤백 완료 확인
                5. 새로운 상태 검증

                kubectl 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # kubectl 명령어 실행
                k8s_commands = self._extract_kubectl_commands(result)
                execution_results = []

                for command in k8s_commands:
                    try:
                        cmd_result = subprocess.run(
                            command.split(),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        execution_results.append({
                            "command": command,
                            "output": cmd_result.stdout,
                            "error": cmd_result.stderr,
                            "exit_code": cmd_result.returncode
                        })
                    except Exception as e:
                        execution_results.append({
                            "command": command,
                            "output": "",
                            "error": str(e),
                            "exit_code": -1
                        })

                k8s_result = self._parse_kubernetes_result(
                    "ROLLBACK", f"{namespace}/{deployment_name}", execution_results, k8s_commands
                )
                self.k8s_history.append(k8s_result)

                return k8s_result

    async def monitor_cluster(self, namespace: str = "all") -> KubernetesResult:
        """클러스터 모니터링"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 클러스터 상태를 모니터링하세요:
                - 네임스페이스: {namespace}

                다음을 확인하세요:
                1. Pod 상태 및 상태
                2. 서비스 상태
                3. 리소스 사용량
                4. 이벤트 및 로그
                5. 네트워크 연결 상태
                6. 스토리지 상태

                kubectl 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # kubectl 명령어 실행
                k8s_commands = self._extract_kubectl_commands(result)
                execution_results = []

                for command in k8s_commands:
                    try:
                        cmd_result = subprocess.run(
                            command.split(),
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        execution_results.append({
                            "command": command,
                            "output": cmd_result.stdout,
                            "error": cmd_result.stderr,
                            "exit_code": cmd_result.returncode
                        })
                    except Exception as e:
                        execution_results.append({
                            "command": command,
                            "output": "",
                            "error": str(e),
                            "exit_code": -1
                        })

                k8s_result = self._parse_kubernetes_result(
                    "MONITOR", namespace, execution_results, k8s_commands
                )
                self.k8s_history.append(k8s_result)

                return k8s_result

    def get_kubernetes_summary(self) -> Dict[str, Any]:
        """Kubernetes 작업 요약 정보"""
        if not self.k8s_history:
            return {"message": "No Kubernetes operations performed yet"}

        operation_types = {}
        total_operations = len(self.k8s_history)
        successful_operations = sum(1 for result in self.k8s_history
                                  if result.status == "SUCCESS")

        for result in self.k8s_history:
            op_type = result.operation_type
            if op_type not in operation_types:
                operation_types[op_type] = {"total": 0, "success": 0, "failed": 0}
            operation_types[op_type]["total"] += 1

            if result.status == "SUCCESS":
                operation_types[op_type]["success"] += 1
            else:
                operation_types[op_type]["failed"] += 1

        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": total_operations - successful_operations,
            "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
            "operation_types": operation_types,
            "recent_operations": [
                {
                    "operation_type": result.operation_type,
                    "target": result.target,
                    "status": result.status,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.k8s_history[-5:]  # 최근 5개
            ]
        }

    def _extract_kubectl_commands(self, result: str) -> List[str]:
        """결과에서 kubectl 명령어 추출"""
        commands = []
        lines = result.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('kubectl ') or line.startswith('helm '):
                commands.append(line)

        return commands

    def _parse_kubernetes_result(self, operation_type: str, target: str,
                               execution_results: List[Dict], k8s_commands: List[str]) -> KubernetesResult:
        """Kubernetes 결과 파싱"""
        # 성공/실패 판단
        all_successful = all(result["exit_code"] == 0 for result in execution_results)
        status = "SUCCESS" if all_successful else "FAILED"

        # 출력 및 오류 수집
        output_lines = []
        error_lines = []

        for result in execution_results:
            if result["output"]:
                output_lines.append(result["output"])
            if result["error"]:
                error_lines.append(result["error"])

        output = "\n".join(output_lines)
        error = "\n".join(error_lines) if error_lines else None

        return KubernetesResult(
            operation_type=operation_type,
            target=target,
            status=status,
            output=output,
            error=error,
            gemini_commands=k8s_commands,
            timestamp=datetime.now()
        )


async def main():
    """테스트 실행"""
    agent = KubernetesAgent()

    # 클러스터 모니터링
    result = await agent.monitor_cluster()
    print(f"Cluster monitoring completed: {result.status}")
    print(f"Target: {result.target}")
    print(f"Generated {len(result.gemini_commands)} kubectl commands")

    # 요약 정보
    summary = agent.get_kubernetes_summary()
    print(f"Kubernetes summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
