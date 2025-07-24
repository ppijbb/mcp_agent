"""
코드 리뷰 Agent
==============

코드 품질 검토, 보안 취약점 발견, 개선사항 제안
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
class CodeReviewResult:
    """코드 리뷰 결과"""
    review_id: str
    timestamp: str
    files_reviewed: List[str]
    issues_found: List[Dict[str, Any]]
    security_vulnerabilities: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    code_quality_score: float
    gemini_cli_commands: List[str]

class CodeReviewAgent:
    """코드 리뷰 전담 Agent"""
    
    def __init__(self):
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="code_review_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="code_reviewer",
            instruction="""
            당신은 전문적인 코드 리뷰어입니다. 다음을 수행하세요:
            
            1. 코드 품질 검토 (가독성, 성능, 유지보수성)
            2. 보안 취약점 발견 (SQL 인젝션, XSS, 인증 취약점 등)
            3. 코딩 표준 준수 여부 확인
            4. 개선사항 제안
            5. Gemini CLI 명령어 생성 (실제 수정 작업용)
            
            모든 결과는 구조화된 형태로 반환하고, Gemini CLI에서 실행할 수 있는 명령어를 포함하세요.
            """,
            server_names=["git-mcp", "code-analysis-mcp", "security-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        self.review_history: List[CodeReviewResult] = []
    
    async def review_code(self, target_paths: List[str] = None) -> CodeReviewResult:
        """코드 리뷰 실행"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("코드 리뷰 시작")
                
                # 1. 코드 분석 요청
                analysis_prompt = f"""
                다음 경로의 코드를 리뷰해주세요: {target_paths or ['현재 디렉토리']}
                
                다음을 수행하세요:
                1. 코드 품질 분석 (가독성, 성능, 유지보수성)
                2. 보안 취약점 스캔
                3. 코딩 표준 준수 여부 확인
                4. 개선사항 제안
                5. Gemini CLI에서 실행할 수 있는 수정 명령어 생성
                
                결과를 JSON 형태로 반환하세요:
                {{
                    "files_reviewed": ["파일 목록"],
                    "issues_found": [
                        {{
                            "file": "파일명",
                            "line": "라인번호", 
                            "severity": "high/medium/low",
                            "description": "문제 설명",
                            "suggestion": "해결 방안"
                        }}
                    ],
                    "security_vulnerabilities": [
                        {{
                            "type": "취약점 타입",
                            "file": "파일명",
                            "line": "라인번호",
                            "description": "취약점 설명",
                            "fix_command": "Gemini CLI 수정 명령어"
                        }}
                    ],
                    "improvement_suggestions": ["개선 제안 목록"],
                    "code_quality_score": 0.85,
                    "gemini_cli_commands": [
                        "gemini '특정 파일의 특정 라인을 수정해줘'",
                        "gemini '보안 취약점을 수정해줘'"
                    ]
                }}
                """
                
                # Agent 실행
                result = await context.call_tool(
                    "code_review_analysis",
                    {
                        "prompt": analysis_prompt,
                        "target_paths": target_paths
                    }
                )
                
                # 결과 파싱
                review_data = json.loads(result.get("content", "{}"))
                
                # CodeReviewResult 생성
                review_result = CodeReviewResult(
                    review_id=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    files_reviewed=review_data.get("files_reviewed", []),
                    issues_found=review_data.get("issues_found", []),
                    security_vulnerabilities=review_data.get("security_vulnerabilities", []),
                    improvement_suggestions=review_data.get("improvement_suggestions", []),
                    code_quality_score=review_data.get("code_quality_score", 0.0),
                    gemini_cli_commands=review_data.get("gemini_cli_commands", [])
                )
                
                # 히스토리 저장
                self.review_history.append(review_result)
                
                logger.info(f"코드 리뷰 완료: {len(review_result.files_reviewed)}개 파일 검토")
                
                return review_result
                
        except Exception as e:
            logger.error(f"코드 리뷰 실패: {e}")
            raise
    
    async def review_specific_file(self, file_path: str) -> CodeReviewResult:
        """특정 파일 리뷰"""
        return await self.review_code([file_path])
    
    async def review_recent_changes(self, days: int = 1) -> CodeReviewResult:
        """최근 변경사항 리뷰"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # Git 최근 변경사항 조회
                git_result = await context.call_tool(
                    "git_recent_changes",
                    {"days": days}
                )
                
                changed_files = git_result.get("changed_files", [])
                
                return await self.review_code(changed_files)
                
        except Exception as e:
            print(f"최근 변경사항 리뷰 실패: {e}")
            raise
    
    def get_review_summary(self, review_result: CodeReviewResult) -> str:
        """리뷰 결과 요약"""
        summary = f"""
코드 리뷰 결과 요약
==================

📁 검토된 파일: {len(review_result.files_reviewed)}개
🔍 발견된 이슈: {len(review_result.issues_found)}개
🚨 보안 취약점: {len(review_result.security_vulnerabilities)}개
💡 개선 제안: {len(review_result.improvement_suggestions)}개
⭐ 코드 품질 점수: {review_result.code_quality_score:.2f}/1.0

주요 이슈:
"""
        
        for issue in review_result.issues_found[:5]:  # 상위 5개만
            summary += f"- {issue['file']}:{issue['line']} - {issue['description']}\n"
        
        summary += f"\nGemini CLI 명령어 ({len(review_result.gemini_cli_commands)}개):\n"
        for cmd in review_result.gemini_cli_commands[:3]:  # 상위 3개만
            summary += f"- {cmd}\n"
        
        return summary
    
    def get_critical_issues(self, review_result: CodeReviewResult) -> List[Dict[str, Any]]:
        """심각한 이슈만 필터링"""
        critical_issues = []
        
        # High severity 이슈
        critical_issues.extend([
            issue for issue in review_result.issues_found 
            if issue.get("severity") == "high"
        ])
        
        # 보안 취약점
        critical_issues.extend(review_result.security_vulnerabilities)
        
        return critical_issues

# 사용 예시
async def main():
    """사용 예시"""
    agent = CodeReviewAgent()
    
    # 전체 코드 리뷰
    result = await agent.review_code()
    
    # 결과 출력
    print(agent.get_review_summary(result))
    
    # 심각한 이슈 확인
    critical_issues = agent.get_critical_issues(result)
    if critical_issues:
        print(f"\n🚨 심각한 이슈 {len(critical_issues)}개 발견!")
        for issue in critical_issues:
            print(f"- {issue.get('description', 'Unknown issue')}")

if __name__ == "__main__":
    asyncio.run(main()) 