"""
PRD Writer Agent
디자인 분석 결과를 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent
"""

from mcp_agent.agents.agent import Agent


class PRDWriterAgent:
    """제품 요구사항 문서 작성 전문 Agent"""
    
    @staticmethod
    def create_agent(output_path: str) -> Agent:
        """
        PRD 작성 Agent 생성
        
        Args:
            output_path: PRD 파일 저장 경로
            
        Returns:
            Agent: 설정된 PRD 작성 Agent
        """
        
        instruction = f"""
        You are a professional product requirements document writer.
        
        Based on the Figma design analysis provided, create a comprehensive PRD with these sections:
        
        1. **Executive Summary**
           - Product vision and goals
           - Target audience
           - Key success metrics
        
        2. **Product Overview**
           - Problem statement
           - Solution approach
           - Competitive analysis (if applicable)
        
        3. **User Stories and Use Cases**
           - Primary user personas
           - User journey mapping
           - Core user scenarios
        
        4. **Functional Requirements**
           - Feature specifications
           - User interface requirements
           - Integration requirements
        
        5. **Technical Specifications**
           - Technology stack recommendations
           - Architecture requirements
           - Performance requirements
           - Security considerations
        
        6. **Success Metrics and KPIs**
           - User engagement metrics
           - Business metrics
           - Technical performance metrics
        
        7. **Implementation Timeline**
           - Development phases
           - Milestone definitions
           - Risk assessment
        
        **Critical Requirements**:
        - Save the final PRD to: {output_path}
        - Use professional markdown formatting
        - Include actionable acceptance criteria
        - Validate file creation success
        
        **Quality Checks**:
        - Ensure all sections are complete
        - Verify markdown formatting
        - Confirm file is saved successfully
        
        Make it actionable and ready for development teams."""
        
        return Agent(
            name="prd_writer",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "📋 디자인 분석을 바탕으로 전문적인 제품 요구사항 문서를 작성하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "종합적인 PRD 문서 작성",
            "사용자 스토리 및 사용 사례 정의",
            "기능적 요구사항 명세",
            "기술적 사양 및 아키텍처 권장사항",
            "성공 지표 및 KPI 정의",
            "구현 일정 및 마일스톤 계획"
        ]
    
    @staticmethod
    def get_required_sections() -> list[str]:
        """PRD 필수 섹션 목록 반환"""
        return [
            "Executive Summary",
            "Product Overview", 
            "User Stories",
            "Functional Requirements",
            "Technical Specifications",
            "Success Metrics"
        ] 