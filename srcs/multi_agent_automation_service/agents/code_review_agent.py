"""
Code Review Agent

실제 mcp_agent 라이브러리를 사용한 코드 리뷰 전문 Agent입니다.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from srcs.common.utils import setup_agent_app, save_report


@dataclass
class CodeReviewResult:
    """코드 리뷰 결과"""
    file_path: str
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    gemini_commands: List[str]
    timestamp: datetime


class CodeReviewAgent:
    """코드 리뷰 전담 Agent - 실제 mcp_agent 표준 사용"""
    
    def __init__(self):
        self.app = setup_agent_app("code_review_system")
        self.agent = Agent(
            name="code_reviewer",
            instruction=(
                "역할: 코드 리뷰 에이전트. 다음을 수행하라.\n"
                "1) 코드 품질(가독성/성능/보안/유지보수성) 평가\n"
                "2) 잠재 버그/취약점 식별\n"
                "3) 표준 준수 여부 확인\n"
                "4) 실행 가능한 개선 제안\n"
                "5) 문제별 구체적 Gemini CLI 명령어 생성\n"
                "형식: 섹션 헤더와 리스트를 사용하되 군더더기 없이 간결하게. 불필요한 텍스트 금지."
            ),
            server_names=["filesystem", "github"],  # 실제 MCP 서버명
        )
        self.review_history: List[CodeReviewResult] = []
    
    async def review_code(self, target_path: str = ".", file_pattern: str = "*.py") -> CodeReviewResult:
        """코드 리뷰 수행"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger
            
            # 파일시스템 서버 설정
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                # 코드 리뷰 수행
                review_prompt = f"""
                다음 경로의 코드를 리뷰하세요: {target_path}
                파일 패턴: {file_pattern}
                
                다음 형식으로 결과를 제공하세요:
                
                ## 발견된 문제점
                - [심각도] 문제 설명
                
                ## 개선 제안
                - 구체적인 개선 방안
                
                ## Gemini CLI 명령어
                - 발견된 각 문제점에 대한 Gemini CLI 명령어
                """
                
                result = await llm.generate_str(
                    message=review_prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                # 결과 파싱 및 구조화
                review_result = self._parse_review_result(result, target_path)
                self.review_history.append(review_result)
                
                return review_result
    
    async def review_specific_file(self, file_path: str) -> CodeReviewResult:
        """특정 파일 리뷰"""
        return await self.review_code(target_path=file_path)
    
    async def review_recent_changes(self, days: int = 7) -> List[CodeReviewResult]:
        """최근 변경사항 리뷰"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                최근 {days}일간 변경된 파일들을 찾아서 리뷰하세요.
                각 파일별로 문제점과 개선사항을 분석하고,
                Gemini CLI 명령어를 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                # 결과 파싱
                review_results = self._parse_multiple_reviews(result)
                self.review_history.extend(review_results)
                
                return review_results
    
    def get_review_summary(self) -> Dict[str, Any]:
        """리뷰 요약 정보"""
        if not self.review_history:
            return {"message": "No reviews performed yet"}
        
        total_files = len(self.review_history)
        total_issues = sum(len(result.issues) for result in self.review_history)
        critical_issues = sum(1 for result in self.review_history 
                            for issue in result.issues 
                            if issue.get("severity") == "CRITICAL")
        
        return {
            "total_files_reviewed": total_files,
            "total_issues_found": total_issues,
            "critical_issues": critical_issues,
            "review_history": [
                {
                    "file_path": result.file_path,
                    "severity": result.severity,
                    "issues_count": len(result.issues),
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.review_history
            ]
        }
    
    def _parse_review_result(self, result: str, file_path: str) -> CodeReviewResult:
        """리뷰 결과 파싱"""
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        issues = []
        suggestions = []
        gemini_commands = []
        
        # 간단한 파싱 예시
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            if "## 발견된 문제점" in line:
                current_section = "issues"
            elif "## 개선 제안" in line:
                current_section = "suggestions"
            elif "## Gemini CLI 명령어" in line:
                current_section = "commands"
            elif line.strip().startswith('-') and current_section:
                content = line.strip()[1:].strip()
                if current_section == "issues":
                    issues.append({"description": content, "severity": "MEDIUM"})
                elif current_section == "suggestions":
                    suggestions.append(content)
                elif current_section == "commands":
                    gemini_commands.append(content)
        
        # 심각도 결정
        severity = "LOW"
        if any("CRITICAL" in issue.get("description", "") for issue in issues):
            severity = "CRITICAL"
        elif any("HIGH" in issue.get("description", "") for issue in issues):
            severity = "HIGH"
        elif any("MEDIUM" in issue.get("description", "") for issue in issues):
            severity = "MEDIUM"
        
        return CodeReviewResult(
            file_path=file_path,
            issues=issues,
            suggestions=suggestions,
            severity=severity,
            gemini_commands=gemini_commands,
            timestamp=datetime.now()
        )
    
    def _parse_multiple_reviews(self, result: str) -> List[CodeReviewResult]:
        """다중 리뷰 결과 파싱"""
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        return [self._parse_review_result(result, "multiple_files")]


async def main():
    """테스트 실행"""
    agent = CodeReviewAgent()
    
    # 전체 코드 리뷰
    result = await agent.review_code()
    print(f"Code review completed for: {result.file_path}")
    print(f"Found {len(result.issues)} issues")
    print(f"Generated {len(result.gemini_commands)} Gemini CLI commands")
    
    # 요약 정보
    summary = agent.get_review_summary()
    print(f"Review summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main()) 