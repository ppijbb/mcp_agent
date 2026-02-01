"""
Performance Agent

실제 mcp_agent 라이브러리를 사용한 성능 분석 및 테스트 전문 Agent입니다.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from srcs.common.utils import setup_agent_app


@dataclass
class PerformanceResult:
    """성능 분석 결과"""
    target_path: str
    analysis_type: str  # CODE_ANALYSIS, BENCHMARK, OPTIMIZATION
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    gemini_commands: List[str]
    timestamp: datetime


class PerformanceAgent:
    """성능 분석 및 테스트 전담 Agent - 실제 mcp_agent 표준 사용"""

    def __init__(self):
        self.app = setup_agent_app("performance_system")
        self.agent = Agent(
            name="performance_analyzer",
            instruction="""
            당신은 전문적인 성능 분석가입니다. 다음을 수행하세요:

            1. 코드 성능 분석: 시간복잡도, 공간복잡도, 병목 지점 식별
            2. 메모리 사용량 분석 및 최적화 제안
            3. 알고리즘 효율성 평가
            4. 성능 테스트 케이스 생성
            5. 성능 개선을 위한 Gemini CLI 명령어 생성

            MCP 서버의 도구들을 활용하여 실제 코드를 분석하고,
            구체적인 성능 개선 방안을 제시하세요.
            """,
            server_names=["filesystem", "github"],  # 실제 MCP 서버명
        )
        self.performance_history: List[PerformanceResult] = []

    async def analyze_performance(self, target_path: str = "srcs") -> PerformanceResult:
        """성능 분석 수행"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger

            # 파일시스템 서버 설정
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                # 성능 분석 수행
                analysis_prompt = f"""
                다음 경로의 코드를 성능 관점에서 분석하세요: {target_path}

                다음을 분석하세요:
                1. 시간복잡도 및 공간복잡도
                2. 병목 지점 식별
                3. 메모리 사용량 최적화 기회
                4. 알고리즘 효율성
                5. 성능 개선 제안

                각 발견사항에 대한 구체적인 Gemini CLI 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=analysis_prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                # 결과 파싱 및 구조화
                perf_result = self._parse_performance_result(result, target_path, "CODE_ANALYSIS")
                self.performance_history.append(perf_result)

                return perf_result

    async def generate_tests(self, target_path: str = "srcs") -> PerformanceResult:
        """성능 테스트 생성"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 경로의 코드에 대한 성능 테스트를 생성하세요: {target_path}

                다음을 포함하세요:
                1. 벤치마크 테스트
                2. 부하 테스트
                3. 메모리 누수 테스트
                4. 성능 프로파일링 테스트
                5. 테스트 실행을 위한 Gemini CLI 명령어
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                perf_result = self._parse_performance_result(result, target_path, "BENCHMARK")
                self.performance_history.append(perf_result)

                return perf_result

    async def optimize_code(self, target_path: str = "srcs") -> PerformanceResult:
        """코드 최적화 제안"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 경로의 코드를 최적화하세요: {target_path}

                다음 최적화를 고려하세요:
                1. 알고리즘 개선
                2. 데이터 구조 최적화
                3. 캐싱 전략
                4. 병렬 처리 기회
                5. 메모리 사용량 최적화

                각 최적화에 대한 구체적인 Gemini CLI 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                perf_result = self._parse_performance_result(result, target_path, "OPTIMIZATION")
                self.performance_history.append(perf_result)

                return perf_result

    async def run_benchmarks(self, target_path: str = "srcs") -> PerformanceResult:
        """벤치마크 실행"""
        async with self.app.run() as app_context:
            context = app_context.context

            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)

                prompt = f"""
                다음 경로의 코드에 대한 벤치마크를 실행하세요: {target_path}

                다음을 측정하세요:
                1. 실행 시간
                2. 메모리 사용량
                3. CPU 사용률
                4. I/O 성능
                5. 네트워크 성능 (해당하는 경우)

                벤치마크 실행을 위한 Gemini CLI 명령어를 생성하세요.
                """

                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-5-mini")
                )

                perf_result = self._parse_performance_result(result, target_path, "BENCHMARK")
                self.performance_history.append(perf_result)

                return perf_result

    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 분석 요약 정보"""
        if not self.performance_history:
            return {"message": "No performance analysis performed yet"}

        analysis_types = {}
        total_findings = 0
        total_recommendations = 0

        for result in self.performance_history:
            analysis_type = result.analysis_type
            if analysis_type not in analysis_types:
                analysis_types[analysis_type] = 0
            analysis_types[analysis_type] += 1

            total_findings += len(result.findings)
            total_recommendations += len(result.recommendations)

        return {
            "total_analyses": len(self.performance_history),
            "analysis_types": analysis_types,
            "total_findings": total_findings,
            "total_recommendations": total_recommendations,
            "recent_analyses": [
                {
                    "target_path": result.target_path,
                    "analysis_type": result.analysis_type,
                    "findings_count": len(result.findings),
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.performance_history[-5:]  # 최근 5개
            ]
        }

    def get_critical_bottlenecks(self) -> List[Dict[str, Any]]:
        """중요한 병목 지점 식별"""
        critical_findings = []

        for result in self.performance_history:
            for finding in result.findings:
                if finding.get("severity") in ["HIGH", "CRITICAL"]:
                    critical_findings.append({
                        "target_path": result.target_path,
                        "finding": finding,
                        "timestamp": result.timestamp.isoformat()
                    })

        return critical_findings

    def _parse_performance_result(self, result: str, target_path: str, analysis_type: str) -> PerformanceResult:
        """성능 분석 결과 파싱"""
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        findings = []
        recommendations = []
        gemini_commands = []

        # 간단한 파싱 예시
        lines = result.split('\n')
        current_section = None

        for line in lines:
            if "## 발견사항" in line or "## Findings" in line:
                current_section = "findings"
            elif "## 개선제안" in line or "## Recommendations" in line:
                current_section = "recommendations"
            elif "## Gemini CLI 명령어" in line:
                current_section = "commands"
            elif line.strip().startswith('-') and current_section:
                content = line.strip()[1:].strip()
                if current_section == "findings":
                    findings.append({"description": content, "severity": "MEDIUM"})
                elif current_section == "recommendations":
                    recommendations.append(content)
                elif current_section == "commands":
                    gemini_commands.append(content)

        return PerformanceResult(
            target_path=target_path,
            analysis_type=analysis_type,
            findings=findings,
            recommendations=recommendations,
            gemini_commands=gemini_commands,
            timestamp=datetime.now()
        )


async def main():
    """테스트 실행"""
    agent = PerformanceAgent()

    # 성능 분석
    result = await agent.analyze_performance()
    print(f"Performance analysis completed for: {result.target_path}")
    print(f"Found {len(result.findings)} issues")
    print(f"Generated {len(result.gemini_commands)} Gemini CLI commands")

    # 요약 정보
    summary = agent.get_performance_summary()
    print(f"Performance summary: {summary}")

    # 중요 병목 지점
    bottlenecks = agent.get_critical_bottlenecks()
    print(f"Critical bottlenecks: {len(bottlenecks)}")


if __name__ == "__main__":
    asyncio.run(main())
