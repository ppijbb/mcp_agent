"""
Documentation Agent

실제 mcp_agent 라이브러리를 사용한 자동 문서화 전문 Agent입니다.
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
from srcs.common.utils import setup_agent_app, save_report


@dataclass
class DocumentationResult:
    """문서화 결과"""
    file_path: str
    doc_type: str  # README, API, CHANGELOG, etc.
    content: str
    gemini_commands: List[str]
    timestamp: datetime


class DocumentationAgent:
    """자동 문서화 전담 Agent - 실제 mcp_agent 표준 사용"""
    
    def __init__(self):
        self.app = setup_agent_app("documentation_system")
        self.agent = Agent(
            name="documentation_writer",
            instruction="""
            당신은 전문적인 기술 문서 작성자입니다. 다음을 수행하세요:
            
            1. 코드 분석을 통한 자동 문서 생성
            2. README.md 파일 업데이트 및 개선
            3. API 문서 자동 생성
            4. CHANGELOG.md 업데이트
            5. 문서화 작업에 대한 Gemini CLI 명령어 생성
            
            MCP 서버의 도구들을 활용하여 실제 코드를 분석하고,
            고품질의 문서를 생성하세요.
            """,
            server_names=["filesystem", "github"],  # 실제 MCP 서버명
        )
        self.documentation_history: List[DocumentationResult] = []
    
    async def update_documentation(self, target_path: str = ".") -> List[DocumentationResult]:
        """전체 문서화 업데이트"""
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger
            
            # 파일시스템 서버 설정
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                # 문서화 작업 수행
                doc_prompt = f"""
                다음 경로의 프로젝트를 분석하여 문서를 업데이트하세요: {target_path}
                
                다음 문서들을 생성/업데이트하세요:
                1. README.md - 프로젝트 개요, 설치, 사용법
                2. API 문서 - 주요 함수와 클래스 설명
                3. CHANGELOG.md - 최근 변경사항
                4. CONTRIBUTING.md - 기여 가이드
                
                각 문서에 대한 Gemini CLI 명령어도 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=doc_prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                # 결과 파싱 및 구조화
                doc_results = self._parse_documentation_result(result, target_path)
                self.documentation_history.extend(doc_results)
                
                return doc_results
    
    async def update_readme(self, project_path: str = ".") -> DocumentationResult:
        """README.md 업데이트"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                다음 프로젝트의 README.md를 분석하고 개선하세요: {project_path}
                
                다음을 포함하세요:
                - 프로젝트 개요 및 목적
                - 설치 방법
                - 사용 예제
                - API 개요
                - 기여 방법
                - 라이선스 정보
                
                Gemini CLI 명령어도 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                doc_result = self._parse_single_documentation(result, "README.md", project_path)
                self.documentation_history.append(doc_result)
                
                return doc_result
    
    async def update_api_documentation(self, source_path: str = "srcs") -> DocumentationResult:
        """API 문서 생성"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                다음 소스 코드를 분석하여 API 문서를 생성하세요: {source_path}
                
                다음을 포함하세요:
                - 클래스 및 함수 설명
                - 매개변수 및 반환값
                - 사용 예제
                - 의존성 정보
                
                Gemini CLI 명령어도 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                doc_result = self._parse_single_documentation(result, "API.md", source_path)
                self.documentation_history.append(doc_result)
                
                return doc_result
    
    async def update_changelog(self, project_path: str = ".") -> DocumentationResult:
        """CHANGELOG.md 업데이트"""
        async with self.app.run() as app_context:
            context = app_context.context
            
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
            async with self.agent:
                llm = await self.agent.attach_llm(OpenAIAugmentedLLM)
                
                prompt = f"""
                다음 프로젝트의 최근 변경사항을 분석하여 CHANGELOG.md를 업데이트하세요: {project_path}
                
                다음 형식을 사용하세요:
                ## [버전] - YYYY-MM-DD
                ### Added
                - 새로운 기능
                ### Changed
                - 변경된 기능
                ### Fixed
                - 수정된 버그
                
                Gemini CLI 명령어도 생성하세요.
                """
                
                result = await llm.generate_str(
                    message=prompt,
                    request_params=RequestParams(model="gpt-4o")
                )
                
                doc_result = self._parse_single_documentation(result, "CHANGELOG.md", project_path)
                self.documentation_history.append(doc_result)
                
                return doc_result
    
    def get_documentation_summary(self) -> Dict[str, Any]:
        """문서화 요약 정보"""
        if not self.documentation_history:
            return {"message": "No documentation generated yet"}
        
        doc_types = {}
        for result in self.documentation_history:
            doc_type = result.doc_type
            if doc_type not in doc_types:
                doc_types[doc_type] = 0
            doc_types[doc_type] += 1
        
        return {
            "total_documents": len(self.documentation_history),
            "document_types": doc_types,
            "recent_documents": [
                {
                    "file_path": result.file_path,
                    "doc_type": result.doc_type,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.documentation_history[-5:]  # 최근 5개
            ]
        }
    
    def _parse_documentation_result(self, result: str, target_path: str) -> List[DocumentationResult]:
        """문서화 결과 파싱"""
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        doc_results = []
        
        # 간단한 파싱 예시
        sections = result.split('##')
        for section in sections:
            if 'README' in section or 'API' in section or 'CHANGELOG' in section:
                doc_type = "README.md" if "README" in section else "API.md" if "API" in section else "CHANGELOG.md"
                doc_results.append(self._parse_single_documentation(section, doc_type, target_path))
        
        return doc_results if doc_results else [self._parse_single_documentation(result, "GENERAL.md", target_path)]
    
    def _parse_single_documentation(self, content: str, doc_type: str, file_path: str) -> DocumentationResult:
        """단일 문서 결과 파싱"""
        # 간단한 파싱 예시
        lines = content.split('\n')
        gemini_commands = []
        
        for line in lines:
            if line.strip().startswith('gemini') or 'gemini' in line.lower():
                gemini_commands.append(line.strip())
        
        return DocumentationResult(
            file_path=file_path,
            doc_type=doc_type,
            content=content,
            gemini_commands=gemini_commands,
            timestamp=datetime.now()
        )


async def main():
    """테스트 실행"""
    agent = DocumentationAgent()
    
    # README 업데이트
    result = await agent.update_readme()
    print(f"README updated for: {result.file_path}")
    print(f"Generated {len(result.gemini_commands)} Gemini CLI commands")
    
    # 요약 정보
    summary = agent.get_documentation_summary()
    print(f"Documentation summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main()) 