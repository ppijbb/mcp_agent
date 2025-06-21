"""
Figma Analyzer Agent
Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator


class FigmaAnalyzerAgent:
    """Figma 디자인 분석 전문 Agent"""

    def __init__(self, figma_url: str):
        self.figma_url = figma_url
        self.agent_instance = self._create_agent_instance()

    def _create_agent_instance(self) -> Agent:
        """Figma 분석 Agent의 기본 인스턴스 생성"""
        return self.create_agent(self.figma_url)

    @staticmethod
    def create_agent(figma_url: str) -> Agent:
        """
        Figma 분석 Agent 생성 (기존 호환성 유지)
        
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
        
        Example for inaccessible URL:
        "The Figma URL [URL] is not accessible. Based on the URL structure, this appears to be a [file/prototype]. A typical Figma file includes pages, frames, and components. Analysis is limited without access."
        """
        
        return Agent(
            name="figma_analyzer",
            instruction=instruction,
            server_names=["figma-dev-mode", "fetch", "filesystem"]
        )

    async def analyze_figma_for_prd(self, figma_api_key: str | None = None, figma_file_id: str | None = None, figma_node_id: str | None = None) -> dict:
        """Figma 디자인을 분석해 PRD 작성을 위한 구조화 데이터를 반환합니다.

        실제 Figma API 호출 로직은 TODO 이며, 더 이상 목업 데이터를 반환하지 않습니다.
        환경변수에서 FIGMA_API_KEY / FIGMA_FILE_ID / FIGMA_NODE_ID 값을 자동으로 로드합니다.
        """
        import os

        figma_api_key = figma_api_key or os.getenv("FIGMA_API_KEY")
        figma_file_id = figma_file_id or os.getenv("FIGMA_FILE_ID")
        figma_node_id = figma_node_id or os.getenv("FIGMA_NODE_ID")

        if not all([figma_api_key, figma_file_id, figma_node_id]):
            raise RuntimeError("FIGMA_API_KEY, FIGMA_FILE_ID, FIGMA_NODE_ID must be provided via args or environment variables.")

        print(f"🎨 Figma 분석 시작: file_id={figma_file_id}, node_id={figma_node_id}")

        # TODO: 실제 Figma API 호출 및 분석 로직 구현 후 결과 반환
        raise NotImplementedError("Figma API integration not yet implemented. Provide actual implementation to remove this exception.")

    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🎨 Figma 디자인을 분석하여 구조화된 요구사항을 추출하는 전문 Agent (ReAct 패턴 적용)"
    
    @staticmethod 
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Figma URL 유효성 검증 및 접근성 확인",
            "디자인 구조 및 컴포넌트 계층 분석",
            "UI/UX 패턴 및 사용자 인터랙션 흐름 추출",
            "기술적 요구사항 및 구현 스펙 도출",
            "콘텐츠 구조 및 데이터 요구사항 문서화"
        ] 