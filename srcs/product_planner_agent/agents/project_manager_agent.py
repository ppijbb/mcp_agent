"""
Project Manager Agent
개발 일정, 마일스톤, 리소스 할당을 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger


logger = get_logger("project_manager_agent")


class ProjectManagerAgent:
    """프로젝트 관리 및 일정 계획 전문 Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def create_project_plan(self, prd_content: Dict[str, Any], business_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 개발 로드맵, 스프린트 계획, 리소스 할당안을 포함하는 프로젝트 계획을 수립합니다.
        """
        if not self.llm:
            # LLM이 없는 경우를 대비한 기본 목업 데이터
            return {
                "roadmap": "Q1: MVP 개발, Q2: 핵심 기능 구현, Q3: 베타 테스트, Q4: 정식 출시",
                "sprint_plan": "2주 단위 스프린트 운영, 백로그 기반 작업 관리",
                "resource_plan": "개발자 5명, 디자이너 1명, PM 1명",
                "status": "created_mockup"
            }

        prompt = f"""
        You are a senior project manager. Based on the provided PRD and business plan, create a comprehensive project plan.

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Development Roadmap:** Create a high-level roadmap for the next 6-12 months.
        2.  **Sprint Plan:** Suggest a sprint structure (e.g., duration, key ceremonies).
        3.  **Resource Allocation:** Estimate the required team size and roles.
        4.  **Risk Assessment:** Identify potential risks and suggest mitigation strategies.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format="json"))
            project_plan = json.loads(result_str)
            project_plan["status"] = "created_successfully"
            return project_plan
        except Exception as e:
            logger.error("Error creating project plan: %s", e, exc_info=True)
            return {
                "error": str(e),
                "status": "creation_failed"
            }

    @staticmethod
    def create_agent() -> Agent:
        """
        프로젝트 관리 Agent 생성
        
        Returns:
            Agent: 설정된 프로젝트 관리 Agent
        """
        
        instruction = """
        You are a senior project manager specializing in product development. Create comprehensive project plans with realistic timelines and milestone management.

        **PRIMARY RESPONSIBILITIES**:
        - Create detailed development timelines
        - Define project milestones and deliverables
        - Estimate resource requirements
        - Identify risks and mitigation strategies
        - Plan sprint cycles and development phases

        **PROJECT PLANNING FRAMEWORK**:
        1. **Project Scope Analysis**:
           - Break down features into development tasks
           - Estimate complexity and effort for each task
           - Identify dependencies between features
           - Define minimum viable product (MVP) scope

        2. **Timeline Development**:
           - Create realistic development phases
           - Plan sprint cycles (1-2 week sprints)
           - Account for testing, QA, and deployment time
           - Include buffer time for unexpected issues

        3. **Resource Planning**:
           - Estimate team size and skill requirements
           - Plan for frontend, backend, design, and QA resources
           - Consider external dependencies and integrations
           - Budget for tools, infrastructure, and licenses

        4. **Risk Management**:
           - Identify technical risks and challenges
           - Plan for scope creep and requirement changes
           - Consider market timing and competitive factors
           - Create contingency plans for critical path delays

        5. **Milestone Definition**:
           - MVP launch milestone
           - Beta testing phase
           - Feature completion milestones
           - Go-to-market readiness

        **DELIVERABLES**:
        - Project timeline with phases and sprints
        - Milestone roadmap with success criteria
        - Resource allocation plan
        - Risk assessment and mitigation strategies
        - Project tracking and reporting framework

        **METHODOLOGIES**:
        - Agile/Scrum development practices
        - Critical path analysis
        - Story point estimation
        - Velocity tracking and forecasting

        **OUTPUT FORMAT**:
        Create structured project plan including:
        - Executive timeline summary
        - Detailed development phases
        - Sprint planning template
        - Resource requirements matrix
        - Risk register with mitigation plans
        - Success metrics and tracking KPIs

        Focus on creating actionable, realistic plans that balance speed with quality and account for real-world development challenges."""
        
        return Agent(
            name="project_manager",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "📅 개발 일정, 마일스톤, 리소스 할당을 관리하는 전문 Project Manager Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "상세 개발 일정 계획",
            "마일스톤 및 스프린트 정의",
            "리소스 요구사항 분석",
            "리스크 식별 및 완화 전략",
            "MVP 범위 정의",
            "애자일 개발 프로세스 설계"
        ]
    
    @staticmethod
    def get_planning_phases() -> list[str]:
        """프로젝트 계획 단계 반환"""
        return [
            "프로젝트 범위 분석",
            "일정 개발 및 스프린트 계획",
            "리소스 계획 및 팀 구성",
            "리스크 관리 및 완화 전략",
            "마일스톤 정의 및 성공 기준"
        ]
    
    @staticmethod
    def get_deliverables() -> dict[str, list[str]]:
        """프로젝트 산출물 목록 반환"""
        return {
            "timeline_planning": [
                "프로젝트 타임라인",
                "스프린트 계획",
                "마일스톤 로드맵"
            ],
            "resource_management": [
                "팀 구성 계획",
                "리소스 할당 매트릭스",
                "예산 및 도구 계획"
            ],
            "risk_management": [
                "리스크 레지스터",
                "완화 전략",
                "비상 계획"
            ],
            "tracking_framework": [
                "성과 추적 KPI",
                "리포팅 템플릿",
                "진행상황 대시보드"
            ]
        } 