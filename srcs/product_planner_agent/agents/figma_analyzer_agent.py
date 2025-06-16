"""
Figma Analyzer Agent
Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent
"""

from mcp_agent.agents.agent import Agent


class FigmaAnalyzerAgent:
    """Figma 디자인 분석 전문 Agent"""
    
    @staticmethod
    def create_agent(figma_url: str) -> Agent:
        """
        Figma 분석 Agent 생성
        
        Args:
            figma_url: 분석할 Figma URL
            
        Returns:
            Agent: 설정된 Figma 분석 Agent
        """
        
        instruction = f"""
        You are a Figma design analyst. Analyze the Figma design at: {figma_url}

        **CRITICAL: First validate the Figma URL accessibility before proceeding.**
        
        If the URL is accessible, provide comprehensive analysis on:
        
        1. **Design Structure Analysis**:
           - Page hierarchy and organization
           - Component library usage
           - Layout patterns (grid, flexbox, etc.)
           - Responsive design considerations
        
        2. **UI/UX Analysis**:
           - User interface patterns and interactions
           - Navigation flow and user journey
           - Accessibility considerations
           - Design system consistency
        
        3. **Technical Requirements**:
           - Required frontend technologies
           - Component architecture recommendations  
           - Integration points and APIs needed
           - Performance considerations
        
        4. **Content Analysis**:
           - Text content and copy requirements
           - Image and media requirements
           - Data structure needs
        
        **Output Format**: Provide structured analysis in markdown format.
        **Validation**: If URL is not accessible, clearly state the limitation and provide analysis based on URL structure and common Figma patterns.
        
        Be specific, actionable, and focus on extracting product requirements from the design."""
        
        return Agent(
            name="figma_analyzer",
            instruction=instruction,
            server_names=["fetch", "filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🎨 Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent"
    
    @staticmethod 
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "디자인 구조 및 계층 분석",
            "UI/UX 패턴 및 상호작용 분석", 
            "기술적 요구사항 도출",
            "콘텐츠 및 데이터 구조 분석",
            "접근성 및 반응형 디자인 고려사항"
        ] 