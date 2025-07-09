from typing import List, Dict, Tuple
from srcs.core.agent.base import BaseAgent
from mcp_agent.agents.agent import Agent as MCP_Agent

class ProductPlannerAgent(BaseAgent):
    """프로덕트 기획자 Agent - 기업급 기능을 위해 BaseAgent를 상속합니다."""

    def __init__(self):
        """
        ProductPlannerAgent를 초기화합니다.
        개발 계획에 명시된 서버 이름을 포함합니다.
        """
        super().__init__(
            name="product_planner",
            instruction="Figma 디자인과 Notion 문서를 연동하여 프로덕트 기획 업무를 자동화합니다.",
            server_names=["figma-dev-mode", "notion-api", "filesystem"]
        )
        self.figma_client = None  # To be implemented via figma_integration.py
        self.notion_client = None # To be implemented via notion_integration.py
        
        # 에이전트 및 평가자 설정
        self.sub_agents = self._create_agents()
        self.evaluator = self._create_evaluator()
        self.orchestrator = self.get_orchestrator(self.sub_agents)


    def _create_agents(self) -> List[MCP_Agent]:
        """개발 계획에 따라 전문화된 서브 Agent들을 생성합니다."""
        agents_config = [
            {
                "name": "design_analyzer",
                "instruction": "Figma 디자인 분석 및 프로덕트 요구사항 추출",
                "server_names": ["figma-dev-mode", "filesystem"]
            },
            {
                "name": "requirement_synthesizer", 
                "instruction": "디자인 분석으로부터 PRD 및 기술 스펙 생성",
                "server_names": ["notion-api", "filesystem"]
            },
            {
                "name": "roadmap_planner",
                "instruction": "프로덕트 로드맵 및 마일스톤 추적 생성",
                "server_names": ["notion-api", "filesystem"]
            },
            {
                "name": "design_notion_connector",
                "instruction": "디자인 변경사항과 Notion 문서 동기화",
                "server_names": ["figma-dev-mode", "notion-api"]
            }
        ]
        
        return [MCP_Agent(**config) for config in agents_config]

    def _create_evaluator(self) -> MCP_Agent:
        """기업급 품질 평가자를 생성합니다."""
        evaluation_criteria: List[Tuple[str, int, str]] = [
            ("Product Feasibility", 30, "제품 실현 가능성 및 기술적 타당성"),
            ("Market Alignment", 25, "시장 요구사항 및 사용자 니즈 부합도"),
            ("Design Consistency", 20, "디자인 시스템 일관성 및 UX 품질"),
            ("Documentation Quality", 15, "문서화 완성도 및 명확성"),
            ("Timeline Realism", 10, "개발 일정의 현실성 및 리스크 고려")
        ]
        
        # BaseAgent에 평가자 생성 헬퍼가 있다면 사용, 없다면 직접 생성
        # 현재 BaseAgent에는 create_standard_evaluator가 없으므로 직접 구성
        instruction = "다음 기준에 따라 프로덕트 기획의 품질을 평가하고 점수를 매기세요:\n"
        for name, weight, desc in evaluation_criteria:
            instruction += f"- {name} (가중치: {weight}%): {desc}\n"
        instruction += "각 항목에 대해 0-100점 척도로 평가하고, 가중치를 적용하여 총점을 계산하세요. 최종 결과는 JSON 형식으로 제공해야 합니다."

        return MCP_Agent(
            name="quality_evaluator",
            instruction=instruction
        )

    async def run_workflow(self, figma_url: str, notion_page_id: str):
        """
        Product Planner Agent의 핵심 워크플로우.
        Figma 분석 -> Notion 문서 생성을 오케스트레이션합니다.
        (현재는 플레이스홀더)
        """
        self.logger.info(f"Product Planner 워크플로우 시작: Figma URL='{figma_url}', Notion Page ID='{notion_page_id}'")
        
        # TODO: 오케스트레이터를 사용하여 서브 에이전트 워크플로우 실행
        # 예시: plan = await self.orchestrator.plan(...)
        #       result = await self.orchestrator.execute(plan)
        
        await self.logger.log("워크플로우가 아직 구현되지 않았습니다.")
        
        # 임시 반환 값
        return {"status": "pending", "message": "Workflow not implemented yet."}
