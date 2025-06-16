"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator


class PRDWriterAgent:
    """PRD 작성 전문 Agent"""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.agent_instance = self._create_agent_instance()

    def _create_agent_instance(self) -> Agent:
        """PRD 작성 Agent의 기본 인스턴스 생성"""
        return self.create_agent(self.output_path)

    @staticmethod
    def create_agent(output_path: str) -> Agent:
        """
        PRD 작성 Agent 생성 (기존 호환성 유지)
        
        Args:
            output_path: 결과물 저장 경로
            
        Returns:
            Agent: 설정된 PRD 작성 Agent
        """
        
        instruction = f"""
        You are a product requirements document (PRD) writer. 
        Your task is to create a comprehensive PRD based on the provided Figma design analysis and user requirements.
        
        **CRITICAL: Use the following markdown structure for the PRD.**
        
        # Product Requirements Document: [Product Name]

        ## 1. Overview
        - **Product Vision**: 
        - **Goals & Objectives**: 
        - **Target Audience**: 

        ## 2. User Requirements
        - **User Personas**: 
        - **User Stories / Use Cases**: 
        - **Functional Requirements**: 

        ## 3. Design & UX Requirements
        - **Key UI/UX Principles**: 
        - **Wireframes & Mockups**: (Reference the Figma analysis)
        - **Accessibility**: 

        ## 4. Technical Specifications
        - **System Architecture**: 
        - **Data Model**: 
        - **Integrations**: 

        ## 5. Success Metrics
        - **Key Performance Indicators (KPIs)**: 
        - **Analytics & Tracking**: 

        ## 6. Future Considerations
        - **Roadmap**: 
        - **Out of Scope**: 

        **Final Output**: The final PRD should be saved to {output_path}.
        """
        
        return Agent(
            name="prd_writer",
            instruction=instruction,
            server_names=["filesystem"]
        )

    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "📋 디자인 분석을 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent (ReAct 패턴 적용)"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Figma 분석 결과 및 요구사항을 종합하여 PRD 초안 생성",
            "표준화된 PRD 템플릿에 따라 구조화된 문서 작성",
            "제품 비전, 목표, 사용자 요구사항 등 핵심 섹션 정의",
            "기술 사양 및 성공 지표 포함",
            "결과물을 마크다운 파일로 저장"
        ]

    @staticmethod
    def get_required_sections() -> list[str]:
        """PRD 필수 섹션 목록 반환"""
        return [
            "Overview", "User Requirements", "Design & UX Requirements",
            "Technical Specifications", "Success Metrics", "Future Considerations"
        ] 