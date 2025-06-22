"""
KPI Analyst Agent
핵심 성과 지표 설정, 분석 프레임워크 및 성능 추적 시스템을 관리하는 Agent
"""

from mcp_agent.agents.agent import Agent
from typing import Dict, Any
import json
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger


logger = get_logger("kpi_analyst_agent")


class KPIAnalystAgent:
    """KPI 설정 및 성과 분석 전문 Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.agent_instance = self.create_agent()

    async def define_kpis(self, prd_content: Dict[str, Any], business_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRD와 비즈니스 계획을 바탕으로 제품의 성공을 측정할 핵심 성과 지표(KPI)를 정의합니다.
        """
        if not self.llm:
            return {
                "user_engagement_kpis": ["Daily Active Users (DAU)", "Monthly Active Users (MAU)"],
                "business_kpis": ["Conversion Rate", "Customer Lifetime Value (CLV)"],
                "status": "created_mockup"
            }

        prompt = f"""
        You are a senior data analyst. Based on the provided PRD and business plan, define key performance indicators (KPIs).

        **PRD Content:**
        {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        **Business Plan:**
        {json.dumps(business_plan, indent=2, ensure_ascii=False)}

        **Instructions:**
        1.  **North Star Metric:** Define one primary "North Star" metric that captures the core value of the product.
        2.  **User-centric KPIs:** Define 3-5 KPIs related to user engagement, retention, and satisfaction.
        3.  **Business-centric KPIs:** Define 3-5 KPIs related to revenue, growth, and market position.
        4.  **Measurement Plan:** Briefly describe how each KPI could be measured.

        Provide the output in a structured JSON format.
        """
        
        try:
            result_str = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.5, response_format="json"))
            kpi_definition = json.loads(result_str)
            kpi_definition["status"] = "created_successfully"
            return kpi_definition
        except Exception as e:
            logger.error("Error defining KPIs: %s", e, exc_info=True)
            return {
                "error": str(e),
                "status": "creation_failed"
            }

    @staticmethod
    def create_agent() -> Agent:
        """
        KPI 분석 Agent 생성
        
        Returns:
            Agent: 설정된 KPI 분석 Agent
        """
        
        instruction = """
        You are a senior data analyst and KPI specialist. Design comprehensive measurement frameworks that align with business objectives and drive data-driven decision making.

        **PRIMARY OBJECTIVES**:
        - Define meaningful KPIs aligned with business goals
        - Create measurement frameworks and analytics strategy
        - Design performance tracking and reporting systems
        - Establish baseline metrics and target setting
        - Plan data collection and analysis processes

        **KPI FRAMEWORK DESIGN**:
        1. **Business Alignment**:
           - Connect KPIs to strategic business objectives
           - Define leading vs lagging indicators
           - Establish KPI hierarchy (North Star → Key → Supporting metrics)
           - Ensure measurability and actionability

        2. **User Behavior Metrics**:
           - User acquisition and activation rates
           - User engagement and retention metrics
           - Feature adoption and usage patterns
           - Customer satisfaction and NPS scores
           - User journey and conversion funnels

        3. **Product Performance Metrics**:
           - Feature performance and adoption rates
           - Product quality metrics (crash rates, load times)
           - User experience metrics (task completion rates)
           - A/B testing and experimentation framework
           - Product-market fit indicators

        4. **Business Metrics**:
           - Revenue and monetization KPIs
           - Customer lifetime value (CLV)
           - Customer acquisition cost (CAC)
           - Monthly/Annual recurring revenue (MRR/ARR)
           - Churn rates and retention cohorts

        5. **Operational Metrics**:
           - Development velocity and cycle time
           - Bug resolution and quality metrics
           - Support ticket volume and resolution time
           - System uptime and performance
           - Cost per feature/user metrics

        **ANALYTICS IMPLEMENTATION**:
        - Define tracking events and user properties
        - Design dashboard and reporting structure
        - Plan data collection implementation
        - Establish data quality and validation processes
        - Create automated alerting and monitoring

        **MEASUREMENT STRATEGY**:
        - Baseline establishment and benchmarking
        - Target setting and goal definition
        - Regular review cycles and optimization
        - Cohort analysis and trend identification
        - Predictive analytics and forecasting

        **DELIVERABLES**:
        - KPI definition document with rationale
        - Measurement framework and hierarchy
        - Analytics implementation plan
        - Dashboard and reporting specifications
        - Data collection and validation strategy
        - Performance review and optimization process

        **OUTPUT FORMAT**:
        Create comprehensive KPI framework including:
        - Executive KPI summary dashboard
        - Detailed metric definitions and calculations
        - Implementation roadmap and requirements
        - Target setting and baseline establishment
        - Reporting schedule and review processes
        - Success criteria and optimization triggers

        Focus on creating actionable, business-relevant metrics that drive meaningful insights and decision-making."""
        
        return Agent(
            name="kpi_analyst",
            instruction=instruction,
            server_names=["filesystem"]
        )
    
    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "📊 핵심 성과 지표 설정, 분석 프레임워크 및 성능 추적 시스템을 관리하는 Agent"
    
    @staticmethod
    def get_capabilities() -> list[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "비즈니스 목표 기반 KPI 정의",
            "측정 프레임워크 및 분석 전략 수립",
            "성능 추적 및 리포팅 시스템 설계",
            "기준 지표 설정 및 목표 수립",
            "데이터 수집 및 분석 프로세스 계획",
            "대시보드 및 모니터링 시스템 구축"
        ]
    
    @staticmethod
    def get_kpi_categories() -> dict[str, list[str]]:
        """KPI 카테고리별 지표 목록 반환"""
        return {
            "user_behavior": [
                "사용자 획득률 (User Acquisition)",
                "사용자 활성화율 (User Activation)", 
                "사용자 참여도 (User Engagement)",
                "사용자 유지율 (User Retention)",
                "기능 채택률 (Feature Adoption)"
            ],
            "product_performance": [
                "기능 성능 지표",
                "제품 품질 지표 (충돌률, 로딩 시간)",
                "사용자 경험 지표 (작업 완료율)",
                "A/B 테스트 결과",
                "제품-시장 적합성 지표"
            ],
            "business_metrics": [
                "수익 및 수익화 KPI",
                "고객 생애 가치 (CLV)",
                "고객 획득 비용 (CAC)",
                "월간/연간 반복 수익 (MRR/ARR)",
                "이탈률 및 유지 코호트"
            ],
            "operational_metrics": [
                "개발 속도 및 사이클 타임",
                "버그 해결 및 품질 지표",
                "지원 티켓 처리 지표",
                "시스템 가동시간 및 성능",
                "기능/사용자당 비용"
            ]
        }
    
    @staticmethod
    def get_measurement_framework() -> list[str]:
        """측정 프레임워크 구성 요소 반환"""
        return [
            "비즈니스 목표와 KPI 연결",
            "선행 지표 vs 후행 지표 정의",
            "KPI 계층 구조 설정",
            "측정 가능성 및 실행 가능성 확보",
            "기준선 설정 및 목표 정의",
            "정기 검토 및 최적화 프로세스"
        ] 