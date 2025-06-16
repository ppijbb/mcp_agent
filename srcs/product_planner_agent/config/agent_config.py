"""
Agent 설정 및 팩토리 클래스
모든 Agent의 생성과 설정을 중앙에서 관리
"""

from typing import List, Dict, Any
from datetime import datetime
import os

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from ..agents import (
    FigmaAnalyzerAgent, PRDWriterAgent, FigmaCreatorAgent,
    ConversationAgent, ProjectManagerAgent, KPIAnalystAgent,
    MarketingStrategistAgent, OperationsAgent, NotionDocumentAgent,
    CoordinatorAgent
)


class AgentConfig:
    """Agent 설정 클래스"""
    
    def __init__(self, figma_url: str, output_dir: str = "product_reports"):
        self.figma_url = figma_url
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"product_analysis_{self.timestamp}.md"
        self.output_path = os.path.join(output_dir, self.output_file)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def get_output_info(self) -> Dict[str, str]:
        """출력 파일 정보 반환"""
        return {
            "output_dir": self.output_dir,
            "output_file": self.output_file,
            "output_path": self.output_path,
            "timestamp": self.timestamp
        }


class AgentFactory:
    """Agent 생성 팩토리 클래스"""
    
    def __init__(self, config: AgentConfig, orchestrator: Orchestrator = None):
        self.config = config
        self.orchestrator = orchestrator
        self._agents: Dict[str, Agent] = {}
        self._react_agents: Dict[str, Any] = {}
    
    def create_all_agents_dict(self) -> Dict[str, Agent]:
        """모든 전문 Agent를 생성하여 딕셔너리로 반환합니다."""
        print("🤖 Multi-Agent System 초기화 시작...")
        
        # 1. Figma Analyzer Agent
        figma_analyzer = FigmaAnalyzerAgent(self.config.figma_url)
        self._agents["figma_analyzer_agent"] = figma_analyzer
        print(f"✅ {FigmaAnalyzerAgent.get_description()}")
        
        # 2. PRD Writer Agent  
        prd_writer = PRDWriterAgent(self.config.output_path)
        self._agents["prd_writer_agent"] = prd_writer
        print(f"✅ {PRDWriterAgent.get_description()}")
        
        # 3. Figma Creator Agent
        figma_creator = FigmaCreatorAgent.create_agent()
        self._agents["figma_creator_agent"] = figma_creator
        print(f"✅ {FigmaCreatorAgent.get_description()}")
        
        # 4. Conversation Agent
        conversation_agent = ConversationAgent.create_agent()
        self._agents["conversation_agent"] = conversation_agent
        print(f"✅ {ConversationAgent.get_description()}")
        
        # 5. Project Manager Agent
        project_manager = ProjectManagerAgent.create_agent()
        self._agents["project_manager_agent"] = project_manager
        print(f"✅ {ProjectManagerAgent.get_description()}")
        
        # 6. KPI Analyst Agent
        kpi_analyst = KPIAnalystAgent.create_agent()
        self._agents["kpi_analyst_agent"] = kpi_analyst
        print(f"✅ {KPIAnalystAgent.get_description()}")
        
        # 7. Marketing Strategist Agent
        marketing_strategist = MarketingStrategistAgent.create_agent()
        self._agents["marketing_strategist_agent"] = marketing_strategist
        print(f"✅ {MarketingStrategistAgent.get_description()}")
        
        # 8. Operations Agent
        operations_agent = OperationsAgent.create_agent()
        self._agents["operations_agent"] = operations_agent
        print(f"✅ {OperationsAgent.get_description()}")
        
        # 9. Notion Document Agent
        notion_document = NotionDocumentAgent.create_agent()
        self._agents["notion_document_agent"] = notion_document
        print(f"✅ {NotionDocumentAgent.get_description()}")
        
        print("🎯 Multi-Agent System 초기화 완료!")
        print(f"📊 총 {len(self._agents)}개 전문 Agent가 활성화되었습니다.")
        return self._agents

    def create_react_agents_dict(self) -> Dict[str, Any]:
        """ReAct 패턴을 지원하는 Agent들을 생성하여 딕셔너리로 반환합니다."""
        if not self.orchestrator:
            raise ValueError("ReAct Agent 생성을 위해서는 Orchestrator가 필요합니다.")
        
        print("🔄 ReAct 패턴 Multi-Agent System 초기화 시작...")
        
        # 모든 Agent를 담을 통합 딕셔너리
        all_agents: Dict[str, Any] = self.create_all_agents_dict()

        # Coordinator Agent 생성 및 추가
        # 모든 Agent를 Coordinator에게 전달
        coordinator = CoordinatorAgent(self.orchestrator, all_agents)
        all_agents["coordinator_agent"] = coordinator
        print(f"✅ {CoordinatorAgent.get_description()}")

        self._react_agents = all_agents
        
        print("🎯 Coordinator-led ReAct System 초기화 완료!")
        print(f"📊 총 {len(self._react_agents)}개 Agent가 활성화되었습니다.")
        return self._react_agents
    
    def get_agent(self, name: str) -> Agent:
        """특정 Agent 반환"""
        return self._agents.get(name)
    
    def get_agents_info(self) -> Dict[str, Dict[str, Any]]:
        """모든 Agent 정보 반환"""
        return {
            "figma_analyzer": {
                "description": FigmaAnalyzerAgent.get_description(),
                "capabilities": FigmaAnalyzerAgent.get_capabilities()
            },
            "prd_writer": {
                "description": PRDWriterAgent.get_description(),
                "capabilities": PRDWriterAgent.get_capabilities(),
                "required_sections": PRDWriterAgent.get_required_sections()
            },
            "figma_creator": {
                "description": FigmaCreatorAgent.get_description(),
                "capabilities": FigmaCreatorAgent.get_capabilities(),
                "creation_tools": FigmaCreatorAgent.get_creation_tools(),
                "design_process": FigmaCreatorAgent.get_design_process()
            }
        }


class WorkflowOrchestrator:
    """워크플로우 오케스트레이션 클래스"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=agents,
            plan_type="full"
        )
    
    def create_initial_prompt(self, config: AgentConfig) -> str:
        """ReAct 루프를 시작하기 위한 초기 프롬프트 생성"""
        # 이 프롬프트는 이제 CoordinatorAgent의 첫 번째 THOUGHT를 위해 사용됩니다.
        # 전체 워크플로우를 지시하는 대신, 초기 목표만 설정합니다.
        return f"""Start a new product planning project.
        - **User's Goal**: Analyze the provided Figma URL and generate a complete business plan.
        - **Figma URL**: {config.figma_url}
        - **Output Path**: {config.output_path}
        Your first step is to run the 'Discovery & Requirements' phase.
        """
    
    def print_workflow_info(self, config: AgentConfig):
        """Multi-Agent 워크플로우 정보 출력"""
        print("\n" + "="*80)
        print("🚀 MULTI-AGENT PRODUCT PLANNING SYSTEM")
        print("="*80)
        print(f"📋 분석 대상: {config.figma_url}")
        print(f"📁 출력 디렉토리: {config.output_dir}")
        print(f"📄 결과 파일: {config.output_path}")
        print(f"⏰ 타임스탬프: {config.timestamp}")
        
        print("\n🎯 MULTI-AGENT ECOSYSTEM:")
        print("   💬 ConversationAgent - 사용자 대화 및 요구사항 수집")
        print("   🔍 FigmaAnalyzer - 디자인 분석 및 UI/UX 평가") 
        print("   �� PRDWriter - 제품 요구사항 문서 작성")
        print("   🎨 FigmaCreator - 디자인 생성 및 프로토타이핑")
        print("   📅 ProjectManager - 개발 계획 및 리소스 관리")
        print("   📊 KPIAnalyst - 지표 정의 및 성과 추적")
        print("   📈 MarketingStrategist - 마케팅 전략 및 GTM")
        print("   ⚙️ OperationsAgent - 서비스 운영 및 인프라")
        print("   📚 NotionDocument - 문서화 및 지식 관리")
        print("   🎯 Coordinator - 워크플로우 조율 및 품질 관리")
        
        print("\n🔄 4-PHASE EXECUTION PLAN:")
        print("   ▶️ PHASE 1: Discovery & Requirements (2-3 days)")
        print("      └── 사용자 인터뷰 + 디자인 분석")
        print("   ▶️ PHASE 2: Strategic Planning (3-5 days)")
        print("      └── PRD + KPI 프레임워크 + 마케팅 전략")
        print("   ▶️ PHASE 3: Operational Planning (2-3 days)")
        print("      └── 개발 계획 + 운영 전략")
        print("   ▶️ PHASE 4: Design & Documentation (3-4 days)")
        print("      └── 비주얼 디자인 + 종합 문서화")
        
        print("\n📦 COMPREHENSIVE DELIVERABLES:")
        print("   ✅ 전략 문서 (PRD, KPI, 마케팅)")
        print("   ✅ 프로젝트 관리 (일정, 리소스, 리스크)")
        print("   ✅ 디자인 자산 (목업, 프로토타입)")
        print("   ✅ 종합 문서 (노션 워크스페이스)")
        
        print("="*80) 