"""
자동 문서화 Agent
================

코드 변경사항 분석 및 자동 문서 업데이트
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
class DocumentationResult:
    """문서화 결과"""
    doc_id: str
    timestamp: str
    files_updated: List[str]
    new_docs_created: List[str]
    api_docs_updated: List[str]
    readme_updated: bool
    changelog_updated: bool
    gemini_cli_commands: List[str]

class DocumentationAgent:
    """자동 문서화 전담 Agent"""
    
    def __init__(self):
        # mcp_agent App 초기화
        self.app = MCPApp(
            name="documentation_agent",
            human_input_callback=None
        )
        
        # Agent 설정
        self.agent = Agent(
            name="documentation_writer",
            instruction="""
            당신은 전문적인 기술 문서 작성자입니다. 다음을 수행하세요:
            
            1. 코드 변경사항 분석
            2. API 문서 자동 업데이트
            3. README.md 업데이트
            4. CHANGELOG.md 업데이트
            5. 새로운 기능에 대한 문서 생성
            6. Gemini CLI 명령어 생성 (실제 문서 수정용)
            
            모든 문서는 명확하고 이해하기 쉽게 작성하고, 개발자가 바로 사용할 수 있도록 하세요.
            """,
            server_names=["git-mcp", "file-system-mcp", "code-analysis-mcp"],
            llm_factory=lambda: OpenAIAugmentedLLM(
                model="gpt-4",
            ),
        )
        
        self.doc_history: List[DocumentationResult] = []
    
    async def update_documentation(self, target_paths: List[str] = None) -> DocumentationResult:
        """문서 자동 업데이트"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                logger = app_context.logger
                
                logger.info("문서 자동 업데이트 시작")
                
                # 1. 코드 변경사항 분석
                analysis_prompt = f"""
                다음 경로의 코드 변경사항을 분석하여 문서를 업데이트해주세요: {target_paths or ['현재 디렉토리']}
                
                다음을 수행하세요:
                1. 새로운 API 엔드포인트나 함수 발견
                2. 변경된 설정이나 환경 변수 확인
                3. 새로운 의존성이나 라이브러리 추가 확인
                4. 기존 문서와의 차이점 분석
                5. Gemini CLI 명령어 생성 (실제 문서 수정용)
                
                결과를 JSON 형태로 반환하세요:
                {{
                    "files_updated": ["업데이트된 파일 목록"],
                    "new_docs_created": ["새로 생성된 문서 목록"],
                    "api_docs_updated": ["업데이트된 API 문서 목록"],
                    "readme_updated": true,
                    "changelog_updated": true,
                    "gemini_cli_commands": [
                        "gemini 'README.md에 새로운 API 엔드포인트 정보를 추가해줘'",
                        "gemini 'CHANGELOG.md에 최근 변경사항을 추가해줘'",
                        "gemini '새로운 설정 파일에 대한 문서를 생성해줘'"
                    ]
                }}
                """
                
                # Agent 실행
                result = await context.call_tool(
                    "documentation_analysis",
                    {
                        "prompt": analysis_prompt,
                        "target_paths": target_paths
                    }
                )
                
                # 결과 파싱
                doc_data = json.loads(result.get("content", "{}"))
                
                # DocumentationResult 생성
                doc_result = DocumentationResult(
                    doc_id=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    files_updated=doc_data.get("files_updated", []),
                    new_docs_created=doc_data.get("new_docs_created", []),
                    api_docs_updated=doc_data.get("api_docs_updated", []),
                    readme_updated=doc_data.get("readme_updated", False),
                    changelog_updated=doc_data.get("changelog_updated", False),
                    gemini_cli_commands=doc_data.get("gemini_cli_commands", [])
                )
                
                # 히스토리 저장
                self.doc_history.append(doc_result)
                
                logger.info(f"문서 업데이트 완료: {len(doc_result.files_updated)}개 파일 업데이트")
                
                return doc_result
                
        except Exception as e:
            logger.error(f"문서 업데이트 실패: {e}")
            raise
    
    async def update_api_documentation(self) -> DocumentationResult:
        """API 문서 자동 업데이트"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # API 엔드포인트 스캔
                api_result = await context.call_tool(
                    "scan_api_endpoints",
                    {}
                )
                
                api_endpoints = api_result.get("endpoints", [])
                
                # API 문서 업데이트 요청
                api_doc_prompt = f"""
                다음 API 엔드포인트들을 분석하여 문서를 업데이트해주세요:
                {api_endpoints}
                
                다음을 포함하세요:
                1. 각 엔드포인트의 설명
                2. 요청/응답 예시
                3. 파라미터 설명
                4. 에러 코드 설명
                5. Gemini CLI 명령어 (실제 문서 수정용)
                """
                
                result = await context.call_tool(
                    "update_api_docs",
                    {"prompt": api_doc_prompt}
                )
                
                # 결과 처리
                doc_data = json.loads(result.get("content", "{}"))
                
                return DocumentationResult(
                    doc_id=f"api_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    files_updated=doc_data.get("files_updated", []),
                    new_docs_created=doc_data.get("new_docs_created", []),
                    api_docs_updated=doc_data.get("api_docs_updated", []),
                    readme_updated=False,
                    changelog_updated=False,
                    gemini_cli_commands=doc_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"API 문서 업데이트 실패: {e}")
            raise
    
    async def update_readme(self) -> DocumentationResult:
        """README.md 자동 업데이트"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # 프로젝트 구조 분석
                project_analysis = await context.call_tool(
                    "analyze_project_structure",
                    {}
                )
                
                # README 업데이트 요청
                readme_prompt = f"""
                프로젝트 구조를 분석하여 README.md를 업데이트해주세요:
                {project_analysis}
                
                다음을 포함하세요:
                1. 프로젝트 개요
                2. 설치 방법
                3. 사용법
                4. API 문서 링크
                5. 기여 방법
                6. 라이선스 정보
                7. Gemini CLI 명령어 (실제 README 수정용)
                """
                
                result = await context.call_tool(
                    "update_readme",
                    {"prompt": readme_prompt}
                )
                
                # 결과 처리
                doc_data = json.loads(result.get("content", "{}"))
                
                return DocumentationResult(
                    doc_id=f"readme_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    files_updated=doc_data.get("files_updated", []),
                    new_docs_created=doc_data.get("new_docs_created", []),
                    api_docs_updated=doc_data.get("api_docs_updated", []),
                    readme_updated=True,
                    changelog_updated=False,
                    gemini_cli_commands=doc_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"README 업데이트 실패: {e}")
            raise
    
    async def update_changelog(self) -> DocumentationResult:
        """CHANGELOG.md 자동 업데이트"""
        try:
            async with self.app.run() as app_context:
                context = app_context.context
                
                # Git 커밋 히스토리 분석
                git_history = await context.call_tool(
                    "analyze_git_history",
                    {"days": 7}  # 최근 7일
                )
                
                # CHANGELOG 업데이트 요청
                changelog_prompt = f"""
                Git 커밋 히스토리를 분석하여 CHANGELOG.md를 업데이트해주세요:
                {git_history}
                
                다음을 포함하세요:
                1. 새로운 기능 (Features)
                2. 버그 수정 (Bug Fixes)
                3. 개선사항 (Improvements)
                4. 변경사항 (Changes)
                5. Gemini CLI 명령어 (실제 CHANGELOG 수정용)
                """
                
                result = await context.call_tool(
                    "update_changelog",
                    {"prompt": changelog_prompt}
                )
                
                # 결과 처리
                doc_data = json.loads(result.get("content", "{}"))
                
                return DocumentationResult(
                    doc_id=f"changelog_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    files_updated=doc_data.get("files_updated", []),
                    new_docs_created=doc_data.get("new_docs_created", []),
                    api_docs_updated=doc_data.get("api_docs_updated", []),
                    readme_updated=False,
                    changelog_updated=True,
                    gemini_cli_commands=doc_data.get("gemini_cli_commands", [])
                )
                
        except Exception as e:
            print(f"CHANGELOG 업데이트 실패: {e}")
            raise
    
    def get_documentation_summary(self, doc_result: DocumentationResult) -> str:
        """문서화 결과 요약"""
        summary = f"""
문서 자동 업데이트 결과 요약
==========================

📝 업데이트된 파일: {len(doc_result.files_updated)}개
📄 새로 생성된 문서: {len(doc_result.new_docs_created)}개
🔗 API 문서 업데이트: {len(doc_result.api_docs_updated)}개
📖 README 업데이트: {'✅' if doc_result.readme_updated else '❌'}
📋 CHANGELOG 업데이트: {'✅' if doc_result.changelog_updated else '❌'}

업데이트된 파일:
"""
        
        for file in doc_result.files_updated[:5]:  # 상위 5개만
            summary += f"- {file}\n"
        
        summary += f"\nGemini CLI 명령어 ({len(doc_result.gemini_cli_commands)}개):\n"
        for cmd in doc_result.gemini_cli_commands[:3]:  # 상위 3개만
            summary += f"- {cmd}\n"
        
        return summary

# 사용 예시
async def main():
    """사용 예시"""
    agent = DocumentationAgent()
    
    # 전체 문서 업데이트
    result = await agent.update_documentation()
    
    # 결과 출력
    print(agent.get_documentation_summary(result))
    
    # 특정 문서 업데이트
    readme_result = await agent.update_readme()
    changelog_result = await agent.update_changelog()
    api_result = await agent.update_api_documentation()

if __name__ == "__main__":
    asyncio.run(main()) 