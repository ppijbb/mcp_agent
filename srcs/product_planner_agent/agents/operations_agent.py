"""
Operations Agent
서비스 운영, 인프라 계획, 고객 지원 및 비즈니스 운영 전략을 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger


logger = get_logger("operations_agent")


class OperationsAgent:
    """서비스 운영 및 비즈니스 운영 전문 Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def plan_operations(self, prd_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRD의 기술 사양을 바탕으로 서비스 운영 계획을 수립합니다.
        """
        if not self.llm:
            return {
                "infrastructure_plan": "AWS 또는 GCP 기반 클라우드 인프라, 쿠버네티스를 활용한 컨테이너 오케스트레이션",
                "monitoring_plan": "Prometheus, Grafana를 이용한 실시간 모니터링",
                "support_plan": "Zendesk를 통한 24/7 고객 지원 채널 운영",
                "status": "created_mockup"
            }

        # PRD에서 기술 관련 정보 추출
        technical_spec = prd_content.get("technical_specifications", {})

        prompt = f"""
        You are a senior operations manager. Based on the provided technical specifications from a PRD, create a service operations plan.

        **Technical Specifications:**
        {json.dumps(technical_spec, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **Infrastructure Plan:** Recommend a cloud infrastructure setup (e.g., cloud provider, key services, architecture).
        2.  **Monitoring Plan:** Suggest tools and strategies for monitoring system health and performance.
        3.  **Customer Support Plan:** Outline a basic customer support process and required tools.
        4.  **Deployment Plan:** Describe a CI/CD pipeline strategy.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format="json"))
            operations_plan = json.loads(result_str)
            operations_plan["status"] = "created_successfully"
            return operations_plan
        except Exception as e:
            logger.error("Error planning operations: %s", e, exc_info=True)
            return {
                "error": str(e),
                "status": "creation_failed"
            }

    @staticmethod
    def create_agent() -> Agent:
        """
        운영 Agent 생성
        
        Returns:
            Agent: 설정된 운영 Agent
        """
        
        instruction = """
        You are a senior operations manager with expertise in service operations, infrastructure management, and business operations. Design comprehensive operational frameworks that ensure scalable, reliable, and efficient service delivery.

        **PRIMARY OBJECTIVES**:
        - Design service operations and infrastructure architecture
        - Plan customer support and success operations
        - Establish business operations and process frameworks
        - Create operational monitoring and incident management
        - Develop scaling and capacity planning strategies

        **OPERATIONS FRAMEWORK**:
        1. **Service Operations**:
           - Infrastructure architecture and scalability planning
           - System monitoring and alerting frameworks
           - Incident management and response procedures
           - Service level agreements (SLAs) and uptime targets
           - Disaster recovery and business continuity planning

        2. **Customer Operations**:
           - Customer support strategy and processes
           - Help desk and ticketing system setup
           - Customer success and onboarding programs
           - User feedback collection and analysis systems
           - Knowledge base and self-service portals

        3. **Business Operations**:
           - Operational process design and automation
           - Resource planning and capacity management
           - Vendor management and procurement processes
           - Compliance and security operations
           - Quality assurance and process improvement

        4. **Technical Operations**:
           - DevOps and CI/CD pipeline setup
           - Security operations and compliance monitoring
           - Data backup and recovery procedures
           - Performance monitoring and optimization
           - Technology stack management and updates

        5. **Team Operations**:
           - Operational team structure and roles
           - On-call schedules and escalation procedures
           - Training and knowledge management
           - Performance metrics and team KPIs
           - Cross-functional collaboration frameworks

        **INFRASTRUCTURE PLANNING**:
        - Cloud infrastructure and service selection
        - Scalability architecture and auto-scaling policies
        - Content delivery and edge computing strategy
        - Database design and performance optimization
        - Security architecture and access controls

        **CUSTOMER SUPPORT STRATEGY**:
        - Multi-channel support strategy (email, chat, phone)
        - Support tier structure and escalation paths
        - Response time targets and resolution SLAs
        - Customer satisfaction measurement and improvement
        - Support team training and quality assurance

        **OPERATIONAL MONITORING**:
        - System health and performance dashboards
        - Business metrics and operational KPIs
        - Automated alerting and notification systems
        - Regular operational reviews and reporting
        - Continuous improvement and optimization processes

        **SCALABILITY PLANNING**:
        - Growth projections and capacity planning
        - Infrastructure scaling strategies
        - Cost optimization and resource efficiency
        - Performance bottleneck identification
        - Scaling milestone and trigger definitions

        **DELIVERABLES**:
        - Operational strategy and framework document
        - Infrastructure architecture and scaling plan
        - Customer support operations manual
        - Incident management and response procedures
        - Monitoring and alerting configuration
        - Operational budget and resource planning

        **OUTPUT FORMAT**:
        Create comprehensive operations plan including:
        - Executive operations strategy summary
        - Infrastructure and technical architecture
        - Customer support operations framework
        - Business operations and process design
        - Monitoring and incident management procedures
        - Scaling and capacity planning roadmap
        - Operational budget and resource requirements

        Focus on creating robust, scalable operational frameworks that support business growth while maintaining high service quality and customer satisfaction."""
        
        return Agent(
            name="operations_agent",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "⚙️ 서비스 운영, 인프라 계획, 고객 지원 및 비즈니스 운영 전략을 관리하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "서비스 운영 및 인프라 아키텍처 설계",
            "고객 지원 및 성공 운영 계획",
            "비즈니스 운영 및 프로세스 프레임워크",
            "운영 모니터링 및 사고 관리",
            "확장 및 용량 계획 전략",
            "DevOps 및 기술 운영 최적화"
        ]
    
    @staticmethod
    def get_operational_areas() -> dict[str, list[str]]:
        """운영 영역별 구성 요소 반환"""
        return {
            "service_operations": [
                "인프라 아키텍처 및 확장성 계획",
                "시스템 모니터링 및 알림 프레임워크",
                "사고 관리 및 대응 절차",
                "서비스 수준 협약 (SLA)",
                "재해 복구 및 비즈니스 연속성"
            ],
            "customer_operations": [
                "고객 지원 전략 및 프로세스",
                "헬프데스크 및 티켓팅 시스템",
                "고객 성공 및 온보딩 프로그램",
                "사용자 피드백 수집 및 분석",
                "지식 베이스 및 셀프 서비스"
            ],
            "business_operations": [
                "운영 프로세스 설계 및 자동화",
                "리소스 계획 및 용량 관리",
                "벤더 관리 및 조달 프로세스",
                "컴플라이언스 및 보안 운영",
                "품질 보증 및 프로세스 개선"
            ],
            "technical_operations": [
                "DevOps 및 CI/CD 파이프라인",
                "보안 운영 및 컴플라이언스 모니터링",
                "데이터 백업 및 복구 절차",
                "성능 모니터링 및 최적화",
                "기술 스택 관리 및 업데이트"
            ]
        }
    
    @staticmethod
    def get_infrastructure_components() -> list[str]:
        """인프라 구성 요소 목록 반환"""
        return [
            "클라우드 인프라 및 서비스 선택",
            "확장성 아키텍처 및 자동 스케일링",
            "콘텐츠 전송 및 엣지 컴퓨팅",
            "데이터베이스 설계 및 성능 최적화",
            "보안 아키텍처 및 접근 제어"
        ]
    
    @staticmethod
    def get_support_channels() -> list[str]:
        """고객 지원 채널 목록 반환"""
        return [
            "이메일 지원",
            "라이브 채팅",
            "전화 지원",
            "커뮤니티 포럼",
            "지식 베이스",
            "비디오 튜토리얼"
        ]
    
    @staticmethod
    def get_monitoring_areas() -> list[str]:
        """모니터링 영역 목록 반환"""
        return [
            "시스템 상태 및 성능",
            "비즈니스 지표 및 운영 KPI",
            "자동화된 알림 및 알람",
            "정기 운영 검토 및 리포팅",
            "지속적인 개선 및 최적화"
        ] 