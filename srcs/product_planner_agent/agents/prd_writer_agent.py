"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime

logger = get_logger("prd_writer_agent")


class PRDWriterAgent:
    """PRD 작성 전문 Agent"""

    def __init__(self, output_path: str, orchestrator: Optional[Orchestrator] = None):
        self.output_path = output_path
        self.orchestrator = orchestrator
        self.agent_instance = self._create_agent_instance()
        if orchestrator:
            self.llm = orchestrator.llm_factory()
        else:
            self.llm = None

    def _create_agent_instance(self) -> Agent:
        """PRD 작성 Agent의 기본 인스턴스 생성"""
        return self.create_agent(self.output_path)

    async def write_prd(self, figma_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Figma 분석 결과를 바탕으로 PRD 작성
        
        Args:
            figma_analysis_result: FigmaAnalyzerAgent의 분석 결과
            
        Returns:
            Dict[str, Any]: PRD 작성 결과 및 파일 정보
        """
        logger.info("📋 Starting PRD writing based on Figma analysis")
        
        try:
            # 1. 분석 결과에서 핵심 정보 추출
            extracted_info = await self._extract_key_information(figma_analysis_result)
            
            # 2. PRD 섹션별 내용 생성
            prd_sections = await self._generate_prd_sections(extracted_info)
            
            # 3. 완전한 PRD 문서 조합
            complete_prd = await self._assemble_complete_prd(prd_sections)
            
            # 4. PRD 파일 저장
            file_info = await self._save_prd_file(complete_prd)
            
            result = {
                "prd_content": complete_prd,
                "file_info": file_info,
                "sections": prd_sections,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ PRD writing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"💥 Error in PRD writing: {e}", exc_info=True)
            # No fallback - raise the actual error for proper handling
            raise RuntimeError(f"PRD writing failed: {e}") from e

    async def _extract_key_information(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과에서 PRD 작성에 필요한 핵심 정보 추출"""
        if not self.llm:
            return {"error": "No LLM available for information extraction"}
            
        prompt = f"""
        Extract key information from the Figma analysis for PRD writing.
        
        Analysis Result: {json.dumps(analysis_result, indent=2)}
        
        Extract and structure:
        1. **Product Overview**: What is this product about?
        2. **Target Users**: Who will use this product?
        3. **Core Features**: What are the main functionalities?
        4. **User Goals**: What do users want to accomplish?
        5. **Technical Context**: What technical considerations are important?
        6. **Design Principles**: What design approach is being used?
        
        Format as structured JSON for easy processing.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return {"extracted_info": result, "status": "extracted"}
        except Exception as e:
            logger.warning(f"Information extraction failed: {e}")
            return {
                "status": "extraction_limited",
                "basic_info": {
                    "product_type": "Digital Product",
                    "target_users": "End users",
                    "core_features": ["User interface", "Core functionality"],
                    "user_goals": ["Task completion", "Information access"],
                    "technical_approach": "Modern web application"
                }
            }

    async def _generate_prd_sections(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """PRD의 각 섹션별 내용 생성"""
        if not self.llm:
            return {"error": "No LLM available for section generation"}
            
        sections = {}
        section_prompts = {
            "overview": f"""
            Based on the extracted information, write the Overview section of the PRD.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - Product Vision (2-3 sentences)
            - Goals & Objectives (3-5 key goals)
            - Target Audience (primary user segments)
            
            Write in professional, clear language suitable for stakeholders.
            """,
            
            "user_requirements": f"""
            Write the User Requirements section based on the analysis.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - User Personas (2-3 key personas)
            - User Stories (5-8 key user stories in "As a... I want... So that..." format)
            - Functional Requirements (detailed feature requirements)
            
            Focus on user needs and expected behaviors.
            """,
            
            "design_ux_requirements": f"""
            Write the Design & UX Requirements section.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - Key UI/UX Principles
            - Wireframes & Mockups reference
            - Accessibility requirements (WCAG compliance)
            - Responsive design considerations
            
            Reference the Figma analysis findings.
            """,
            
            "technical_specifications": f"""
            Write the Technical Specifications section.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - System Architecture recommendations
            - Data Model requirements
            - Integration needs (APIs, third-party services)
            - Performance requirements
            - Security considerations
            
            Be specific and actionable for development teams.
            """,
            
            "success_metrics": f"""
            Write the Success Metrics section.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - Key Performance Indicators (KPIs)
            - Analytics & Tracking requirements
            - Success criteria and targets
            - Measurement methodologies
            
            Focus on measurable outcomes.
            """,
            
            "future_considerations": f"""
            Write the Future Considerations section.
            
            Information: {json.dumps(extracted_info, indent=2)}
            
            Include:
            - Product Roadmap (next 6-12 months)
            - Out of Scope items (what's not included in v1)
            - Potential enhancements
            - Scalability considerations
            
            Think strategically about product evolution.
            """
        }
        
        for section_name, prompt in section_prompts.items():
            try:
                section_content = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
                sections[section_name] = section_content
                logger.info(f"✅ Generated {section_name} section")
            except Exception as e:
                logger.warning(f"Failed to generate {section_name} section: {e}")
                sections[section_name] = f"[{section_name.replace('_', ' ').title()} section - content generation failed]"
        
        return sections

    async def _assemble_complete_prd(self, sections: Dict[str, Any]) -> str:
        """PRD 섹션들을 완전한 문서로 조합"""
        if not self.llm:
            return self._create_basic_prd_template(sections)
            
        prompt = f"""
        Assemble the PRD sections into a complete, professional Product Requirements Document.
        
        Sections: {json.dumps(sections, indent=2)}
        
        Create a well-structured markdown document with:
        1. Clear headings and subheadings
        2. Consistent formatting
        3. Professional tone
        4. Logical flow between sections
        5. Executive summary at the beginning
        
        The document should be ready for stakeholder review and development team use.
        """
        
        try:
            complete_prd = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return complete_prd
        except Exception as e:
            logger.warning(f"PRD assembly failed: {e}")
            return self._create_basic_prd_template(sections)

    def _create_basic_prd_template(self, sections: Dict[str, Any]) -> str:
        """기본 PRD 템플릿 생성 (LLM 실패 시 사용)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prd_content = f"""# Product Requirements Document

**Generated:** {timestamp}
**Status:** Draft

## Executive Summary

This PRD outlines the requirements for a digital product based on design analysis and user research.

## 1. Overview

{sections.get('overview', '[Overview section not available]')}

## 2. User Requirements

{sections.get('user_requirements', '[User Requirements section not available]')}

## 3. Design & UX Requirements

{sections.get('design_ux_requirements', '[Design & UX Requirements section not available]')}

## 4. Technical Specifications

{sections.get('technical_specifications', '[Technical Specifications section not available]')}

## 5. Success Metrics

{sections.get('success_metrics', '[Success Metrics section not available]')}

## 6. Future Considerations

{sections.get('future_considerations', '[Future Considerations section not available]')}

---

*This PRD was generated automatically based on Figma design analysis.*
*Please review and refine as needed for your specific requirements.*
"""
        return prd_content

    async def _save_prd_file(self, prd_content: str) -> Dict[str, Any]:
        """PRD 파일을 지정된 경로에 저장"""
        try:
            # 출력 디렉토리 생성
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 파일 저장
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(prd_content)
            
            file_size = len(prd_content.encode('utf-8'))
            
            return {
                "file_path": self.output_path,
                "file_size": file_size,
                "status": "saved",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to save PRD file: {e}")
            return {
                "file_path": self.output_path,
                "status": "save_failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }



    async def refine_prd_section(self, section_name: str, current_content: str, additional_requirements: str) -> Dict[str, Any]:
        """특정 PRD 섹션 개선"""
        logger.info(f"🔄 Refining PRD section: {section_name}")
        
        if not self.llm:
            return {"error": "No LLM available for section refinement"}
            
        prompt = f"""
        Refine the following PRD section based on additional requirements.
        
        Section: {section_name}
        Current Content: {current_content}
        Additional Requirements: {additional_requirements}
        
        Improve the section by:
        1. Incorporating the additional requirements
        2. Enhancing clarity and specificity
        3. Ensuring consistency with PRD standards
        4. Adding missing details or considerations
        
        Return the refined section content.
        """
        
        try:
            refined_content = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {
                "section_name": section_name,
                "refined_content": refined_content,
                "status": "refined",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Section refinement failed: {e}")
            return {
                "section_name": section_name,
                "status": "refinement_failed",
                "error": str(e),
                "original_content": current_content
            }

    async def validate_prd_completeness(self, prd_content: str) -> Dict[str, Any]:
        """PRD 완성도 검증"""
        logger.info("🔍 Validating PRD completeness")
        
        if not self.llm:
            return {"error": "No LLM available for validation"}
            
        prompt = f"""
        Validate the completeness and quality of this PRD.
        
        PRD Content: {prd_content}
        
        Check for:
        1. **Required Sections**: All essential PRD sections present
        2. **Content Quality**: Sufficient detail and clarity
        3. **Consistency**: Coherent throughout the document
        4. **Actionability**: Clear enough for development teams
        5. **Missing Elements**: What's missing or needs improvement
        
        Provide a validation report with scores (1-10) and specific recommendations.
        """
        
        try:
            validation_result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return {
                "validation_report": validation_result,
                "status": "validated",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"PRD validation failed: {e}")
            return {
                "status": "validation_failed",
                "error": str(e),
                "basic_check": {
                    "word_count": len(prd_content.split()),
                    "section_count": prd_content.count('#'),
                    "note": "Basic metrics only - detailed validation unavailable"
                }
            }

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
            "결과물을 마크다운 파일로 저장",
            "PRD 섹션별 개선 및 검증 기능",
            "완성도 검증 및 품질 평가"
        ]

    @staticmethod
    def get_required_sections() -> list[str]:
        """PRD 필수 섹션 목록 반환"""
        return [
            "Overview", "User Requirements", "Design & UX Requirements",
            "Technical Specifications", "Success Metrics", "Future Considerations"
        ] 