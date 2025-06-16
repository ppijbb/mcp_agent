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
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._agents: Dict[str, Agent] = {}
    
    def create_all_agents(self) -> List[Agent]:
        """모든 Agent 생성 - Enhanced Multi-Agent System"""
        print("🤖 Multi-Agent System 초기화 시작...")
        
        # === 기존 Agent들 ===
        # 1. Figma Analyzer Agent
        figma_analyzer = FigmaAnalyzerAgent.create_agent(self.config.figma_url)
        self._agents["figma_analyzer"] = figma_analyzer
        print(f"✅ {FigmaAnalyzerAgent.get_description()}")
        
        # 2. PRD Writer Agent  
        prd_writer = PRDWriterAgent.create_agent(self.config.output_path)
        self._agents["prd_writer"] = prd_writer
        print(f"✅ {PRDWriterAgent.get_description()}")
        
        # 3. Figma Creator Agent
        figma_creator = FigmaCreatorAgent.create_agent()
        self._agents["figma_creator"] = figma_creator
        print(f"✅ {FigmaCreatorAgent.get_description()}")
        
        # === 새로운 Multi-Agent들 ===
        # 4. Conversation Agent
        conversation_agent = ConversationAgent.create_agent()
        self._agents["conversation_agent"] = conversation_agent
        print(f"✅ {ConversationAgent.get_description()}")
        
        # 5. Project Manager Agent
        project_manager = ProjectManagerAgent.create_agent()
        self._agents["project_manager"] = project_manager
        print(f"✅ {ProjectManagerAgent.get_description()}")
        
        # 6. KPI Analyst Agent
        kpi_analyst = KPIAnalystAgent.create_agent()
        self._agents["kpi_analyst"] = kpi_analyst
        print(f"✅ {KPIAnalystAgent.get_description()}")
        
        # 7. Marketing Strategist Agent
        marketing_strategist = MarketingStrategistAgent.create_agent()
        self._agents["marketing_strategist"] = marketing_strategist
        print(f"✅ {MarketingStrategistAgent.get_description()}")
        
        # 8. Operations Agent
        operations_agent = OperationsAgent.create_agent()
        self._agents["operations_agent"] = operations_agent
        print(f"✅ {OperationsAgent.get_description()}")
        
        # 9. Notion Document Agent
        notion_document = NotionDocumentAgent.create_agent()
        self._agents["notion_document"] = notion_document
        print(f"✅ {NotionDocumentAgent.get_description()}")
        
        # 10. Coordinator Agent (중앙 조율)
        coordinator = CoordinatorAgent.create_agent()
        self._agents["coordinator"] = coordinator
        print(f"✅ {CoordinatorAgent.get_description()}")
        
        print("🎯 Multi-Agent System 초기화 완료!")
        print(f"📊 총 {len(self._agents)}개 Agent가 활성화되었습니다.")
        return list(self._agents.values())
    
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
    
    def create_workflow_task(self, config: AgentConfig) -> str:
        """Multi-Agent 워크플로우 작업 정의 생성"""
        return f"""Execute a comprehensive multi-agent product planning workflow that creates a complete business plan with technical specifications, marketing strategy, and operational framework.

        **MULTI-AGENT WORKFLOW EXECUTION**:
        
        **PHASE 1: DISCOVERY & REQUIREMENTS (2-3 days)**
        🎯 **coordinator** - Orchestrate the entire workflow and manage agent communication
        💬 **conversation_agent** - Conduct structured user interviews to gather detailed requirements
        🔍 **figma_analyzer** - Analyze existing Figma design (if provided): {config.figma_url}
        
        **PHASE 2: STRATEGIC PLANNING (3-5 days)**
        📋 **prd_writer** - Create comprehensive PRD based on requirements and design analysis
        📊 **kpi_analyst** - Define success metrics, KPIs, and measurement frameworks
        📈 **marketing_strategist** - Develop go-to-market strategy and marketing plans
        
        **PHASE 3: OPERATIONAL PLANNING (2-3 days)**  
        📅 **project_manager** - Create development timeline, sprints, and resource allocation
        ⚙️ **operations_agent** - Design service operations, infrastructure, and support systems
        
        **PHASE 4: DESIGN & DOCUMENTATION (3-4 days)**
        🎨 **figma_creator** - Create visual designs, mockups, and interactive prototypes
        📚 **notion_document** - Compile all deliverables into comprehensive documentation
        
        **INTER-AGENT COMMUNICATION**:
        - Each agent receives outputs from previous phase agents
        - coordinator ensures consistency and quality across all deliverables
        - Feedback loops enable refinement and optimization
        - Parallel execution where possible to optimize timeline
        
        **COMPREHENSIVE DELIVERABLES**:
        
        **📋 Strategic Documents**:
        - ✅ Detailed PRD with technical specifications
        - ✅ KPI framework and measurement strategy
        - ✅ Marketing strategy and go-to-market plan
        - ✅ Business operations and service strategy
        
        **📅 Project Management**:
        - ✅ Development timeline with sprints and milestones
        - ✅ Resource allocation and team requirements
        - ✅ Risk assessment and mitigation strategies
        - ✅ Budget planning and cost estimates
        
        **🎨 Design Assets**:
        - ✅ Visual mockups and interactive prototypes
        - ✅ Design system and component library
        - ✅ User journey maps and workflow diagrams
        - ✅ Responsive design variants
        
        **📚 Comprehensive Documentation**:
        - ✅ Notion workspace with all project documentation
        - ✅ Knowledge base and team collaboration setup
        - ✅ Process documentation and SOPs
        - ✅ Training materials and onboarding guides
        
        **QUALITY CRITERIA**:
        - All deliverables must be professionally formatted and actionable
        - Cross-agent consistency and integration verification
        - Business viability and technical feasibility validation
        - Ready-to-implement package for development teams
        - Scalable frameworks for future growth
        
        **SUCCESS METRICS**:
        - Complete coverage of all business and technical aspects
        - Seamless integration across all deliverables
        - Executable roadmap with clear next steps
        - Comprehensive documentation for team handoff
        
        Output Path: {config.output_path}
        
        Focus on creating a world-class product planning package that covers every aspect of product development, from user requirements to operational deployment."""
    
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
        print("   📋 PRDWriter - 제품 요구사항 문서 작성")
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