"""
Coordinator Agent
Multi-Agent 간 소통, 워크플로우 조율 및 작업 협조를 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from typing import Dict, List, Any
import json
import asyncio

from .figma_analyzer_agent import FigmaAnalyzerAgent
from .prd_writer_agent import PRDWriterAgent


class CoordinatorAgent:
    """Agent 간 조율 및 워크플로우 관리를 위한 ReAct 기반 실행 Agent"""

    def __init__(self, orchestrator: Orchestrator, agents: Dict[str, Agent]):
        self.orchestrator = orchestrator
        self.agents = agents
        self.agent_instance = self._create_agent_instance()
        
        # 실제 존재하는 에이전트 이름들
        self.available_agents = list(agents.keys())
        print(f"🔍 Available agents: {self.available_agents}")
        
        # LLM 인스턴스 생성
        self.llm = None

    def _create_agent_instance(self) -> Agent:
        """
        조율 Agent의 기본 인스턴스 생성
        """
        instruction = self._get_base_instruction()
        return Agent(
            name="coordinator_agent",
            instruction=instruction,
            server_names=["filesystem"]
        )

    async def run(self, initial_task: str) -> str:
        """
        간소화된 ReAct 패턴을 사용하여 제품 기획 워크플로우를 실행합니다.
        """
        print("🚀 CoordinatorAgent: ReAct 워크플로우 시작...")
        
        # LLM 초기화
        if not self.llm:
            self.llm = await self.agent_instance.attach_llm(OpenAIAugmentedLLM)
        
        final_report = ""
        
        # 단계 1: Figma 분석
        print("\n📋 Step 1: Figma Design Analysis")
        try:
            figma_analysis_prompt = f"""
            Analyze the Figma design at this URL: {initial_task}
            
            Provide a comprehensive analysis including:
            1. UI/UX Design Overview
            2. Component Structure
            3. Design System Elements
            4. User Flow Analysis
            5. Technical Requirements
            6. Accessibility Considerations
            
            Format your response as a detailed markdown report.
            """
            
            figma_result = await self.llm.generate_str(
                message=figma_analysis_prompt,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            print("✅ Figma analysis completed")
            final_report += f"\n\n## Figma Design Analysis\n{figma_result}"
            
        except Exception as e:
            error_msg = f"Error in Figma analysis: {str(e)}"
            print(f"❌ {error_msg}")
            final_report += f"\n\n## Figma Analysis Error\n{error_msg}"

        # 단계 2: PRD 작성
        print("\n📝 Step 2: Product Requirements Document")
        try:
            prd_prompt = f"""
            Based on the following Figma analysis, create a comprehensive Product Requirements Document (PRD).
            
            Figma Analysis:
            {figma_result if 'figma_result' in locals() else 'Analysis not available'}
            
            Create a PRD with these sections:
            1. Executive Summary
            2. Product Overview
            3. User Stories & Use Cases
            4. Functional Requirements
            5. Non-Functional Requirements
            6. Technical Specifications
            7. Success Metrics
            8. Timeline & Milestones
            9. Risk Assessment
            10. Appendix
            
            Format as a professional markdown document.
            """
            
            prd_result = await self.llm.generate_str(
                message=prd_prompt,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            print("✅ PRD creation completed")
            final_report += f"\n\n## Product Requirements Document\n{prd_result}"
            
        except Exception as e:
            error_msg = f"Error in PRD creation: {str(e)}"
            print(f"❌ {error_msg}")
            final_report += f"\n\n## PRD Creation Error\n{error_msg}"

        # 단계 3: 비즈니스 계획
        print("\n💼 Step 3: Business Planning")
        try:
            business_prompt = f"""
            Based on the PRD and Figma analysis, create a comprehensive business plan.
            
            Include:
            1. Market Analysis
            2. Competitive Landscape
            3. Business Model
            4. Revenue Strategy
            5. Go-to-Market Plan
            6. Marketing Strategy
            7. Operations Plan
            8. Financial Projections
            9. Risk Management
            10. Implementation Roadmap
            
            Format as a strategic business document in markdown.
            """
            
            business_result = await self.llm.generate_str(
                message=business_prompt,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            print("✅ Business planning completed")
            final_report += f"\n\n## Business Plan\n{business_result}"
            
        except Exception as e:
            error_msg = f"Error in business planning: {str(e)}"
            print(f"❌ {error_msg}")
            final_report += f"\n\n## Business Planning Error\n{error_msg}"

        # 단계 4: 최종 종합
        print("\n🎯 Step 4: Final Integration")
        try:
            integration_prompt = f"""
            Create a final executive summary that integrates all the analysis and planning work.
            
            Provide:
            1. Project Overview
            2. Key Findings Summary
            3. Strategic Recommendations
            4. Next Steps
            5. Success Criteria
            6. Resource Requirements
            
            Keep it concise but comprehensive.
            """
            
            integration_result = await self.llm.generate_str(
                message=integration_prompt,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            print("✅ Final integration completed")
            final_report = f"# Product Planning Report\n\n## Executive Summary\n{integration_result}\n{final_report}"
            
        except Exception as e:
            error_msg = f"Error in final integration: {str(e)}"
            print(f"❌ {error_msg}")
            final_report += f"\n\n## Integration Error\n{error_msg}"

        # 파일 저장 시도
        try:
            # 출력 경로 추출 (initial_task에서)
            if "output should be saved to:" in initial_task:
                output_path = initial_task.split("output should be saved to:")[1].strip().split()[0]
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_report)
                print(f"✅ Report saved to: {output_path}")
                
        except Exception as e:
            print(f"⚠️ Could not save report to file: {str(e)}")

        print("\n🎉 CoordinatorAgent: ReAct 워크플로우 완료!")
        return final_report

    @staticmethod
    def _get_base_instruction() -> str:
        """
        Agent의 기본 지시사항을 반환합니다.
        """
        return """
        You are the coordination maestro for a multi-agent product planning system. 
        Your role is to orchestrate a comprehensive product planning workflow that includes:
        1. Figma design analysis
        2. Product requirements documentation
        3. Business strategy planning
        4. Final integration and recommendations
        
        You work systematically through each phase, ensuring quality and completeness at each step.
        """

    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🎯 Multi-Agent 간 소통, 워크플로우 조율 및 작업 협조를 관리하는 중앙 조율 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Multi-Agent 워크플로우 및 작업 순서 조율",
            "Agent 간 소통 및 정보 공유 촉진",
            "진행 상황 모니터링 및 품질 표준 보장",
            "Agent 작업 간 충돌 및 종속성 해결",
            "프로젝트 일정 및 마일스톤 조율"
        ]
    
    @staticmethod
    def get_workflow_phases() -> dict[str, dict[str, Any]]:
        """워크플로우 단계별 정보 반환"""
        return {
            "phase_1_analysis": {
                "name": "Figma Design Analysis",
                "description": "Comprehensive analysis of Figma design",
                "duration": "30 minutes"
            },
            "phase_2_prd": {
                "name": "Product Requirements Document", 
                "description": "Detailed PRD creation based on design analysis",
                "duration": "45 minutes"
            },
            "phase_3_business": {
                "name": "Business Planning",
                "description": "Strategic business plan development",
                "duration": "30 minutes"
            },
            "phase_4_integration": {
                "name": "Final Integration",
                "description": "Executive summary and recommendations",
                "duration": "15 minutes"
            }
        }
    
    @staticmethod
    def get_coordination_principles() -> list[str]:
        """조율 원칙 목록 반환"""
        return [
            "순차적 실행: 각 단계의 결과가 다음 단계의 입력이 됨",
            "품질 우선: 각 단계에서 완전한 결과물 생성",
            "오류 처리: 단계별 오류 발생 시 적절한 대응",
            "결과 통합: 모든 단계의 결과를 최종 보고서로 통합",
            "파일 저장: 최종 결과물을 지정된 경로에 저장"
        ]
    
    @staticmethod
    def get_success_metrics() -> list[str]:
        """성공 지표 목록 반환"""
        return [
            "모든 워크플로우 단계 완료",
            "각 단계별 품질 있는 결과물 생성",
            "최종 보고서 파일 저장 성공",
            "오류 발생 시 적절한 처리 및 계속 진행",
            "사용자 요구사항 충족"
        ] 