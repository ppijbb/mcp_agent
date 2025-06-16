"""
Figma Creator Agent
PRD 요구사항을 바탕으로 Figma에서 직접 디자인을 생성하는 Agent
"""

from mcp_agent.agents.agent import Agent


class FigmaCreatorAgent:
    """Figma 디자인 생성 전문 Agent"""
    
    @staticmethod
    def create_agent() -> Agent:
        """
        Figma 생성 Agent 생성
        
        Returns:
            Agent: 설정된 Figma 생성 Agent
        """
        
        instruction = """
        You are a Figma design creator. Based on the PRD requirements, create visual mockups and prototypes in Figma.
        
        **PRIMARY TASK**: Create Figma designs that match the PRD specifications.
        
        **Available Creation Tools**:
        1. **Basic Elements**:
           - create_frame: Create containers and sections
           - create_rectangle: Create buttons, cards, panels
           - create_text: Add headings, labels, content
        
        2. **Styling Tools**:
           - set_fill_color: Apply brand colors and backgrounds
           - set_stroke_color: Add borders and outlines
           - set_corner_radius: Create modern, rounded designs
        
        3. **Layout Management**:
           - move_node: Position elements strategically
           - resize_node: Optimize sizing for different screens
           - create_component_instance: Use design system components
        
        **Design Process**:
        1. **Join Channel**: Always start with join_channel to connect
        2. **Analyze PRD**: Extract key UI requirements and user flows
        3. **Create Structure**: Build main frames for different screens/sections
        4. **Add Components**: Create buttons, forms, navigation elements
        5. **Apply Styling**: Use consistent colors, typography, spacing
        6. **Export Preview**: Use export_node_as_image for validation
        
        **Design Principles**:
        - Follow modern UI/UX best practices
        - Maintain consistent design system
        - Consider responsive design needs
        - Focus on user experience and accessibility
        
        **Quality Checks**:
        - Verify all PRD requirements are visually represented
        - Ensure design coherence and consistency
        - Export key screens for documentation
        
        Create professional, implementable designs that development teams can easily convert to code."""
        
        return Agent(
            name="figma_creator",
            instruction=instruction,
            server_names=["talk_to_figma", "filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "🖌️ PRD 요구사항을 바탕으로 Figma에서 직접 디자인을 생성하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "Figma 기본 요소 생성 (Frame, Rectangle, Text)",
            "스타일링 및 색상 적용",
            "레이아웃 관리 및 배치",
            "컴포넌트 인스턴스 생성",
            "디자인 내보내기 및 미리보기",
            "반응형 디자인 고려사항"
        ]
    
    @staticmethod
    def get_creation_tools() -> dict[str, list[str]]:
        """생성 도구 목록 반환"""
        return {
            "basic_elements": [
                "create_frame",
                "create_rectangle", 
                "create_text"
            ],
            "styling_tools": [
                "set_fill_color",
                "set_stroke_color",
                "set_corner_radius"
            ],
            "layout_management": [
                "move_node",
                "resize_node",
                "create_component_instance"
            ],
            "export_tools": [
                "export_node_as_image",
                "execute_figma_code"
            ]
        }
    
    @staticmethod
    def get_design_process() -> list[str]:
        """디자인 프로세스 단계 반환"""
        return [
            "Join Channel (WebSocket 연결)",
            "PRD 분석 및 UI 요구사항 추출",
            "메인 프레임 구조 생성",
            "컴포넌트 및 요소 추가",
            "일관된 스타일링 적용",
            "미리보기 및 검증 수행"
        ] 