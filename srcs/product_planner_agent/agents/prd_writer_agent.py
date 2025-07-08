"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime
import aiohttp

logger = get_logger("prd_writer_agent")

# Helper function to create the HTTP client session
async def get_http_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession()


class PRDWriterAgent:
    """
    Agent responsible for drafting the PRD document based on various inputs.
    Now enhanced with Figma and Notion context.
    """
    def __init__(self, 
                 google_drive_mcp_url: str = "http://localhost:3001",
                 figma_mcp_url: str = "http://localhost:3003",
                 notion_mcp_url: str = "http://localhost:3004"):
        self.google_drive_mcp_url = google_drive_mcp_url
        self.figma_mcp_url = figma_mcp_url
        self.notion_mcp_url = notion_mcp_url

    async def _get_figma_summary(self, figma_file_id: str) -> Optional[Dict[str, Any]]:
        if not figma_file_id:
            return None
        try:
            url = f"{self.figma_mcp_url}/file-summary?fileId={figma_file_id}"
            async with get_http_session() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            print(f"Error getting Figma summary: {e}")
            return {"error": str(e)}

    async def _get_notion_content(self, notion_page_id: str) -> Optional[Dict[str, Any]]:
        if not notion_page_id:
            return None
        try:
            url = f"{self.notion_mcp_url}/page-content?pageId={notion_page_id}"
            async with get_http_session() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            print(f"Error getting Notion content: {e}")
            return {"error": str(e)}

    async def draft_prd(self, 
                      product_brief: Dict[str, Any], 
                      feedback: Optional[str] = None,
                      figma_file_id: Optional[str] = None,
                      notion_page_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Drafts the PRD using the product brief and feedback.
        Now includes context from Figma and Notion.
        """
        
        figma_context = await self._get_figma_summary(figma_file_id)
        notion_context = await self._get_notion_content(notion_page_id)
        
        prompt = f"""
        As a Senior Product Manager, your task is to write a detailed Product Requirements Document (PRD).

        Use the following inputs to create the PRD:

        1.  **Core Product Brief**:
            ```json
            {json.dumps(product_brief, indent=2)}
            ```
        
        2.  **Figma Design Prototype Summary** (if available):
            Analyze the following summary of the Figma design. Pay attention to user flows, component names, and screen layouts to define the user experience and functional requirements.
            ```json
            {json.dumps(figma_context, indent=2) if figma_context else "No Figma design context provided."}
            ```

        3.  **Notion Planning Document** (if available):
            Incorporate the background, goals, and technical notes from the following Notion document into the PRD.
            ```text
            {json.dumps(notion_context, indent=2) if notion_context else "No Notion planning context provided."}
            ```

        4.  **Additional Feedback** (if available):
            ```
            {feedback if feedback else "No additional feedback provided."}
            ```

        **PRD Structure**:
        Your PRD must follow this structure precisely. Fill in every section with relevant details derived from the provided context.

        - **1. Introduction**:
          - **1.1. Problem Statement**: What user problem are we solving?
          - **1.2. Goal**: What is the primary business objective? What are the key success metrics (KPIs)?
          - **1.3. Target Audience**: Who are the primary users?
        - **2. Product Requirements**:
          - **2.1. User Stories**: Write detailed user stories (As a [user type], I want to [action] so that [benefit]). Infer these from the product brief, Figma flows, and Notion doc.
          - **2.2. Functional Requirements**: List specific features (e.g., 'User authentication', 'Dashboard view'). Use details from the Figma components and Notion specs.
          - **2.3. Non-Functional Requirements**: (e.g., Performance, Security, Usability).
        - **3. Design & UX**:
          - **3.1. Design Mockups**: Reference the Figma file ID ({figma_file_id or 'N/A'}).
          - **3.2. User Flow**: Describe the high-level user journey, referencing specific frames from the Figma summary.
        - **4. Assumptions and Constraints**: List any assumptions made or technical constraints identified.
        
        Generate the PRD in a structured JSON format.
        """
        llm = OpenAIAugmentedLLM()
        prd_json_str = await llm.generate_str(
            message=prompt,
            request_params=RequestParams(
                model="gemini-2.5-flash-lite-preview-06-07",
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        )
        
        try:
            prd_data = json.loads(prd_json_str)
            return prd_data
        except json.JSONDecodeError:
            # Fallback if the LLM output is not valid JSON
            print("Error: PRD output is not valid JSON. Returning raw text.")
            return {"error": "Invalid JSON output from LLM", "raw_text": prd_json_str}

    async def save_prd(self, prd_data: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """PRD 파일을 Google Drive에 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"PRD_{timestamp}.md"
        
        upload_url = f"{self.google_drive_mcp_url}/upload"
        payload = {
            "fileName": file_name,
            "content": prd_content
        }
        
        try:
            async with get_http_session() as session:
                async with session.post(upload_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if result.get("success"):
                        file_id = result.get("fileId")
                        logger.info(f"Successfully uploaded PRD to Google Drive. File ID: {file_id}")
                        return {
                            "drive_file_id": file_id,
                            "file_url": f"https://docs.google.com/document/d/{file_id}",
                            "status": "uploaded",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        raise Exception(f"MCP upload failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"Failed to save PRD file to Google Drive: {e}")
            return {
                "file_name": file_name,
                "status": "upload_failed",
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
    def create_agent() -> Agent:
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

        **Final Output**: The final PRD should be saved to {self.google_drive_mcp_url}.
        """
        
        return Agent(
            name="prd_writer",
            instruction=instruction,
            server_names=["fetch", "filesystem"]  # Filesystem might still be used by underlying tools
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