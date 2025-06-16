"""
Coordinator Agent
Multi-Agent 간 소통, 워크플로우 조율 및 작업 협조를 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from typing import Dict, List, Any
import json


class CoordinatorAgent:
    """Agent 간 조율 및 워크플로우 관리를 위한 ReAct 기반 실행 Agent"""

    def __init__(self, orchestrator: Orchestrator, agents: Dict[str, Agent]):
        self.orchestrator = orchestrator
        self.agents = agents
        self.agent_instance = self._create_agent_instance()

    def _create_agent_instance(self) -> Agent:
        """
        조율 Agent의 기본 인스턴스 생성
        """
        instruction = self._get_base_instruction()
        return Agent(
            name="coordinator",
            instruction=instruction,
            server_names=["filesystem"]
        )

    async def run(self, initial_task: str) -> str:
        """
        ReAct 패턴을 사용하여 전체 제품 기획 워크플로우를 실행합니다.
        THOUGHT -> ACTION -> OBSERVATION 사이클을 반복하여 최종 결과물을 생성합니다.
        """
        print("🚀 CoordinatorAgent: ReAct 워크플로우 시작...")
        
        final_report = ""
        current_context = f"Initial user request: {initial_task}"

        # 4단계 워크플로우를 순차적으로 실행
        for i, phase in enumerate(self.get_workflow_phases().values()):
            phase_name = phase['name']
            print(f"\nPhase {i+1}: {phase_name} 시작...")

            # THOUGHT: 현재 단계에 대한 계획 수립
            thought_prompt = f"""
            **OBSERVATION**:
            {current_context}
            
            **THOUGHT**:
            You are in Phase {i+1}: '{phase_name}'. Your task is to generate a detailed plan for this phase.
            - Identify which agents from this list need to be executed: {', '.join(phase['agents'])}.
            - Define the specific task for each agent based on the current observation.
            - The output of this thought process must be a JSON object describing the actions to take, like this:
              `{{ "actions": [ {{ "agent": "agent_name", "task": "specific task description" }} ] }}`
            """
            
            thought_str = await self.orchestrator.generate_str(thought_prompt)
            print(f"🧠 THOUGHT: {thought_str}")

            # ACTION: 계획에 따라 Agent 실행
            try:
                action_plan = json.loads(thought_str)
                action_results = []
                for action in action_plan.get("actions", []):
                    agent_name = action.get("agent")
                    agent_task = action.get("task")
                    
                    if agent_name in self.agents:
                        print(f"🏃 ACTION: Running {agent_name} with task: '{agent_task}'")
                        # In a real implementation, you would call the agent:
                        # result = await self.orchestrator.run_agent(self.agents[agent_name], agent_task)
                        # For now, we'll simulate the result.
                        result = f"{{'agent': '{agent_name}', 'status': 'completed', 'output': 'This is a simulated output for {agent_name} performing {agent_task}'}}"
                        action_results.append(result)
                        print(f"✅ ACTION Complete: {agent_name} finished.")
                    else:
                        print(f"⚠️  ACTION Failed: Agent '{agent_name}' not found.")
                        action_results.append(f"{{'agent': '{agent_name}', 'status': 'failed', 'error': 'Agent not found'}}")
            except json.JSONDecodeError:
                print(f"⚠️  ACTION Failed: Could not decode thought process into a valid JSON action plan.")
                action_results = [f"{{'agent': 'coordinator', 'status': 'failed', 'error': 'Invalid thought format, expected JSON.'}}"]


            # OBSERVATION: 실행 결과 종합
            current_context = f"Results from Phase {i+1} ({phase_name}):\n" + "\n".join(action_results)
            print(f"👀 OBSERVATION: {current_context}")
            final_report += f"\n\n### Phase {i+1}: {phase_name} Results\n{current_context}"

        print("\n🎉 CoordinatorAgent: ReAct 워크플로우 완료!")
        return final_report

    @staticmethod
    def _get_base_instruction() -> str:
        """
        Agent의 기본 지시사항을 반환합니다.
        (기존 create_agent의 instruction 내용을 가져옴)
        """
        return """
        You are the coordination maestro for a multi-agent product planning system. Your role is to orchestrate seamless collaboration between specialized agents, ensuring efficient workflow execution and quality deliverables.

        **PRIMARY RESPONSIBILITIES**:
        - Coordinate multi-agent workflows and task sequences
        - Facilitate communication and information sharing between agents
        - Monitor progress and ensure quality standards across all deliverables
        - Resolve conflicts and dependencies between agent tasks
        - Maintain project timeline and milestone coordination

        **AGENT ECOSYSTEM**:
        1. **ConversationAgent**: User requirements gathering and dialogue management
        2. **FigmaAnalyzerAgent**: Design analysis and UI/UX evaluation
        3. **PRDWriterAgent**: Product requirements documentation
        4. **FigmaCreatorAgent**: Design creation and visual prototyping
        5. **ProjectManagerAgent**: Development planning and resource management
        6. **KPIAnalystAgent**: Metrics definition and performance tracking
        7. **MarketingStrategistAgent**: Marketing strategy and go-to-market planning
        8. **OperationsAgent**: Service operations and infrastructure planning
        9. **NotionDocumentAgent**: Documentation and knowledge management

        **COORDINATION WORKFLOWS**:
        
        **Phase 1: Discovery & Requirements**
        - ConversationAgent: Gather user requirements and project vision
        - FigmaAnalyzerAgent: Analyze existing designs (if provided)
        - Output: Comprehensive requirements document with design insights

        **Phase 2: Strategic Planning**
        - PRDWriterAgent: Create detailed product requirements
        - KPIAnalystAgent: Define success metrics and measurement framework
        - MarketingStrategistAgent: Develop market positioning and strategy
        - Output: Strategic foundation documents

        **Phase 3: Operational Planning**
        - ProjectManagerAgent: Create development timeline and resource plan
        - OperationsAgent: Design service operations and infrastructure
        - Output: Execution roadmap and operational framework

        **Phase 4: Design & Documentation**
        - FigmaCreatorAgent: Create visual designs and prototypes
        - NotionDocumentAgent: Compile comprehensive documentation
        - Output: Complete product specification with designs and documentation

        **COORDINATION PRINCIPLES**:
        - **Sequential Dependencies**: Ensure outputs from one agent inform the next
        - **Parallel Execution**: Run independent tasks simultaneously for efficiency
        - **Quality Gates**: Validate deliverables before proceeding to next phase
        - **Feedback Loops**: Enable iteration and refinement between agents
        - **Context Preservation**: Maintain shared understanding across all agents

        **COMMUNICATION PROTOCOLS**:
        - **Information Handoffs**: Structure data exchange between agents
        - **Progress Reporting**: Track completion status and quality metrics
        - **Conflict Resolution**: Address inconsistencies and contradictions
        - **Version Control**: Manage document versions and updates
        - **Integration Checks**: Ensure deliverables work together cohesively

        **WORKFLOW MANAGEMENT**:
        - **Task Prioritization**: Determine optimal execution sequence
        - **Resource Allocation**: Balance workload across agents
        - **Timeline Management**: Monitor progress against milestones
        - **Risk Mitigation**: Identify and address workflow bottlenecks
        - **Quality Assurance**: Implement checks and validation processes

        **OUTPUT COORDINATION**:
        - **Integrated Deliverables**: Combine agent outputs into cohesive package
        - **Cross-Reference Validation**: Ensure consistency across documents
        - **Final Review Process**: Conduct comprehensive quality assessment
        - **User Presentation**: Format results for optimal user understanding
        - **Handoff Procedures**: Prepare deliverables for implementation

        **SUCCESS METRICS**:
        - Workflow completion rate and timeline adherence
        - Quality scores across all deliverables
        - Inter-agent communication effectiveness
        - User satisfaction with final outputs
        - Efficiency of resource utilization

        **DELIVERABLES**:
        - Workflow execution plan and timeline
        - Agent task assignments and dependencies
        - Communication protocols and standards
        - Quality assurance checkpoints
        - Integrated final deliverable package
        - Post-project review and recommendations

        Focus on creating smooth, efficient multi-agent workflows that maximize the collective intelligence of the specialized agents while delivering exceptional results to the user."""

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
            "phase_1_discovery": {
                "name": "Discovery & Requirements",
                "agents": ["conversation_agent", "figma_analyzer_agent"],
                "outputs": ["요구사항 문서", "디자인 인사이트"],
                "duration": "2-3 days"
            },
            "phase_2_strategic": {
                "name": "Strategic Planning", 
                "agents": ["prd_writer_agent", "kpi_analyst_agent", "marketing_strategist_agent"],
                "outputs": ["PRD 문서", "KPI 프레임워크", "마케팅 전략"],
                "duration": "3-5 days"
            },
            "phase_3_operational": {
                "name": "Operational Planning",
                "agents": ["project_manager_agent", "operations_agent"],
                "outputs": ["개발 로드맵", "운영 프레임워크"],
                "duration": "2-3 days"
            },
            "phase_4_design_docs": {
                "name": "Design & Documentation",
                "agents": ["figma_creator_agent", "notion_document_agent"],
                "outputs": ["시각적 디자인", "종합 문서"],
                "duration": "3-4 days"
            }
        }
    
    @staticmethod
    def get_coordination_principles() -> list[str]:
        """조율 원칙 목록 반환"""
        return [
            "순차적 종속성: 한 Agent의 출력이 다음 Agent에 정보 제공",
            "병렬 실행: 독립적인 작업을 동시에 실행하여 효율성 향상",
            "품질 게이트: 다음 단계로 진행하기 전 결과물 검증",
            "피드백 루프: Agent 간 반복 및 개선 가능",
            "컨텍스트 보존: 모든 Agent 간 공통 이해 유지"
        ]
    
    @staticmethod
    def get_communication_protocols() -> dict[str, list[str]]:
        """소통 프로토콜 반환"""
        return {
            "information_handoffs": [
                "Agent 간 구조화된 데이터 교환",
                "표준화된 출력 형식",
                "필수 정보 누락 방지"
            ],
            "progress_reporting": [
                "완료 상태 및 품질 지표 추적",
                "실시간 진행 상황 업데이트",
                "마일스톤 달성 보고"
            ],
            "quality_assurance": [
                "결과물 일관성 검증",
                "교차 참조 유효성 검사",
                "최종 품질 평가"
            ]
        }
    
    @staticmethod
    def get_success_metrics() -> list[str]:
        """성공 지표 목록 반환"""
        return [
            "워크플로우 완료율 및 일정 준수",
            "모든 결과물의 품질 점수",
            "Agent 간 소통 효과성",
            "최종 출력에 대한 사용자 만족도",
            "리소스 활용 효율성"
        ]
    
    @staticmethod
    def get_agent_registry() -> dict[str, str]:
        """Agent 레지스트리 반환"""
        return {
            "conversation_agent": "💬 사용자 대화 및 요구사항 수집",
            "figma_analyzer": "🔍 Figma 디자인 분석 및 평가", 
            "prd_writer": "📋 제품 요구사항 문서 작성",
            "figma_creator": "🎨 Figma 디자인 생성 및 프로토타이핑",
            "project_manager": "📅 프로젝트 관리 및 개발 계획",
            "kpi_analyst": "📊 KPI 설정 및 성과 분석",
            "marketing_strategist": "📈 마케팅 전략 및 Go-to-Market",
            "operations_agent": "⚙️ 서비스 운영 및 인프라 계획",
            "notion_document": "📚 문서 작성 및 지식 관리"
        } 