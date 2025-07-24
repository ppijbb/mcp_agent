"""
성능 및 테스트 Agent
===================

성능 분석, 테스트 생성, 병목 지점 발견
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

@dataclass
class PerformanceTestResult:
    """성능 테스트 결과"""
    test_id: str
    timestamp: str
    performance_metrics: Dict[str, Any]
    bottlenecks_found: List[Dict[str, Any]]
    optimization_suggestions: List[str]
    test_coverage: float
    new_tests_created: List[str]
    gemini_cli_commands: List[str]

class PerformanceTestAgent:
    """성능 및 테스트 전담 Agent"""
    
    def __init__(self):
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="performance_test_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="performance_tester",
            instruction="""
            당신은 전문적인 성능 분석가이자 테스트 엔지니어입니다. 다음을 수행하세요:
            
            1. 코드 성능 분석 (CPU, 메모리, 네트워크 사용량)
            2. 병목 지점 발견 및 최적화 제안
            3. 자동 테스트 케이스 생성
            4. 테스트 커버리지 분석
            5. 성능 벤치마크 실행
            6. Gemini CLI 명령어 생성 (실제 최적화 및 테스트 실행용)
            
            모든 성능 분석은 정확하고 실행 가능한 개선 방안을 제시해야 합니다.
            """,
            server_names=["performance-mcp", "testing-mcp", "monitoring-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        self.test_history: List[PerformanceTestResult] = []
    
    async def analyze_performance(self, target_paths: List[str] = None) -> PerformanceTestResult:
        """성능 분석 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("성능 분석 시작")
                
                # 1. 성능 분석 요청
                analysis_prompt = f"""
                다음 경로의 코드 성능을 분석해주세요: {target_paths or ['현재 디렉토리']}
                
                다음을 수행하세요:
                1. CPU 사용량 분석
                2. 메모리 사용량 분석
                3. 네트워크 I/O 분석
                4. 데이터베이스 쿼리 성능 분석
                5. 병목 지점 발견
                6. 최적화 방안 제안
                7. Gemini CLI 명령어 생성 (실제 최적화 실행용)
                
                결과를 JSON 형태로 반환하세요:
                {{
                    "performance_metrics": {{
                        "cpu_usage": "평균 CPU 사용률",
                        "memory_usage": "평균 메모리 사용률",
                        "response_time": "평균 응답 시간",
                        "throughput": "처리량"
                    }},
                    "bottlenecks_found": [
                        {{
                            "type": "병목 타입 (CPU/Memory/Network/DB)",
                            "location": "발생 위치",
                            "severity": "high/medium/low",
                            "description": "병목 설명",
                            "impact": "성능 영향도"
                        }}
                    ],
                    "optimization_suggestions": ["최적화 제안 목록"],
                    "test_coverage": 0.85,
                    "new_tests_created": ["새로 생성된 테스트 목록"],
                    "gemini_cli_commands": [
                        "gemini '특정 함수의 성능을 최적화해줘'",
                        "gemini '메모리 누수 문제를 해결해줘'",
                        "gemini '데이터베이스 쿼리를 최적화해줘'"
                    ]
                }}
                """
                
                # Agent 실행
                result = await context.call_tool(
                    "performance_analysis",
                    {
                        "prompt": analysis_prompt,
                        "target_paths": target_paths
                    }
                )
                
                # 결과 파싱
                perf_data = json.loads(result.get("content", "{}"))
                
                # PerformanceTestResult 생성
                perf_result = PerformanceTestResult(
                    test_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    performance_metrics=perf_data.get("performance_metrics", {}),
                    bottlenecks_found=perf_data.get("bottlenecks_found", []),
                    optimization_suggestions=perf_data.get("optimization_suggestions", []),
                    test_coverage=perf_data.get("test_coverage", 0.0),
                    new_tests_created=perf_data.get("new_tests_created", []),
                    gemini_cli_commands=perf_data.get("gemini_cli_commands", [])
                )
                
                # 히스토리 저장
                self.test_history.append(perf_result)
                
                logger.info(f"성능 분석 완료: {len(perf_result.bottlenecks_found)}개 병목 지점 발견")
                
                return perf_result
                
        except Exception as e:
            logger.error(f"성능 분석 실패: {e}")
            raise
    
    async def generate_tests(self, target_paths: List[str] = None) -> PerformanceTestResult:
        """자동 테스트 생성"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 코드 분석
                code_analysis = await context.call_tool(
                    "analyze_code_for_testing",
                    {"target_paths": target_paths}
                )
                
                # 테스트 생성 요청
                test_prompt = f"""
                다음 코드를 분석하여 테스트 케이스를 생성해주세요:
                {code_analysis}
                
                다음을 포함하세요:
                1. 단위 테스트 (Unit Tests)
                2. 통합 테스트 (Integration Tests)
                3. 성능 테스트 (Performance Tests)
                4. 보안 테스트 (Security Tests)
                5. Gemini CLI 명령어 (실제 테스트 실행용)
                """
                
                result = await context.call_tool(
                    "generate_tests",
                    {"prompt": test_prompt}
                )
                
                # 결과 처리
                test_data = json.loads(result.get("content", "{}"))
                
                return PerformanceTestResult(
                    test_id=f"test_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    performance_metrics={},
                    bottlenecks_found=[],
                    optimization_suggestions=[],
                    test_coverage=test_data.get("test_coverage", 0.0),
                    new_tests_created=test_data.get("new_tests_created", []),
                    gemini_cli_commands=test_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"테스트 생성 실패: {e}")
            raise
    
    async def run_benchmarks(self) -> PerformanceTestResult:
        """성능 벤치마크 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 벤치마크 실행
                benchmark_result = await context.call_tool(
                    "run_performance_benchmarks",
                    {}
                )
                
                # 벤치마크 분석
                analysis_prompt = f"""
                다음 벤치마크 결과를 분석해주세요:
                {benchmark_result}
                
                다음을 분석하세요:
                1. 성능 지표 (CPU, Memory, Response Time)
                2. 이전 벤치마크와의 비교
                3. 성능 개선/악화 지점
                4. 최적화 권장사항
                5. Gemini CLI 명령어 (실제 최적화 실행용)
                """
                
                result = await context.call_tool(
                    "analyze_benchmarks",
                    {"prompt": analysis_prompt}
                )
                
                # 결과 처리
                bench_data = json.loads(result.get("content", "{}"))
                
                return PerformanceTestResult(
                    test_id=f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    performance_metrics=bench_data.get("performance_metrics", {}),
                    bottlenecks_found=bench_data.get("bottlenecks_found", []),
                    optimization_suggestions=bench_data.get("optimization_suggestions", []),
                    test_coverage=0.0,
                    new_tests_created=[],
                    gemini_cli_commands=bench_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"벤치마크 실행 실패: {e}")
            raise
    
    async def optimize_code(self, target_paths: List[str] = None) -> PerformanceTestResult:
        """코드 최적화 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 최적화 전 성능 측정
                before_perf = await self.analyze_performance(target_paths)
                
                # 최적화 실행
                optimization_result = await context.call_tool(
                    "optimize_code",
                    {
                        "target_paths": target_paths,
                        "bottlenecks": before_perf.bottlenecks_found
                    }
                )
                
                # 최적화 후 성능 측정
                after_perf = await self.analyze_performance(target_paths)
                
                # 개선 효과 분석
                improvement = {
                    "cpu_improvement": f"{((before_perf.performance_metrics.get('cpu_usage', 0) - after_perf.performance_metrics.get('cpu_usage', 0)) / before_perf.performance_metrics.get('cpu_usage', 1)) * 100:.2f}%",
                    "memory_improvement": f"{((before_perf.performance_metrics.get('memory_usage', 0) - after_perf.performance_metrics.get('memory_usage', 0)) / before_perf.performance_metrics.get('memory_usage', 1)) * 100:.2f}%",
                    "response_time_improvement": f"{((before_perf.performance_metrics.get('response_time', 0) - after_perf.performance_metrics.get('response_time', 0)) / before_perf.performance_metrics.get('response_time', 1)) * 100:.2f}%"
                }
                
                return PerformanceTestResult(
                    test_id=f"optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    performance_metrics=after_perf.performance_metrics,
                    bottlenecks_found=after_perf.bottlenecks_found,
                    optimization_suggestions=[f"성능 개선 효과: {improvement}"],
                    test_coverage=after_perf.test_coverage,
                    new_tests_created=after_perf.new_tests_created,
                    gemini_cli_commands=after_perf.gemini_cli_commands
                )
                
        except Exception as e:
            print(f"코드 최적화 실패: {e}")
            raise
    
    def get_performance_summary(self, perf_result: PerformanceTestResult) -> str:
        """성능 분석 결과 요약"""
        summary = f"""
성능 분석 결과 요약
==================

📊 성능 지표:
- CPU 사용률: {perf_result.performance_metrics.get('cpu_usage', 'N/A')}
- 메모리 사용률: {perf_result.performance_metrics.get('memory_usage', 'N/A')}
- 응답 시간: {perf_result.performance_metrics.get('response_time', 'N/A')}
- 처리량: {perf_result.performance_metrics.get('throughput', 'N/A')}

🚨 병목 지점: {len(perf_result.bottlenecks_found)}개
💡 최적화 제안: {len(perf_result.optimization_suggestions)}개
🧪 테스트 커버리지: {perf_result.test_coverage:.2f}%
📝 새로 생성된 테스트: {len(perf_result.new_tests_created)}개

주요 병목 지점:
"""
        
        for bottleneck in perf_result.bottlenecks_found[:5]:  # 상위 5개만
            summary += f"- {bottleneck['type']}: {bottleneck['description']}\n"
        
        summary += f"\nGemini CLI 명령어 ({len(perf_result.gemini_cli_commands)}개):\n"
        for cmd in perf_result.gemini_cli_commands[:3]:  # 상위 3개만
            summary += f"- {cmd}\n"
        
        return summary
    
    def get_critical_bottlenecks(self, perf_result: PerformanceTestResult) -> List[Dict[str, Any]]:
        """심각한 병목 지점만 필터링"""
        return [
            bottleneck for bottleneck in perf_result.bottlenecks_found
            if bottleneck.get("severity") == "high"
        ]

# 사용 예시
async def main():
    """사용 예시"""
    agent = PerformanceTestAgent()
    
    # 성능 분석
    perf_result = await agent.analyze_performance()
    print(agent.get_performance_summary(perf_result))
    
    # 테스트 생성
    test_result = await agent.generate_tests()
    print(f"생성된 테스트: {len(test_result.new_tests_created)}개")
    
    # 벤치마크 실행
    bench_result = await agent.run_benchmarks()
    print(f"벤치마크 완료: {bench_result.performance_metrics}")
    
    # 코드 최적화
    optimize_result = await agent.optimize_code()
    print(f"최적화 완료: {optimize_result.optimization_suggestions}")

if __name__ == "__main__":
    asyncio.run(main()) 