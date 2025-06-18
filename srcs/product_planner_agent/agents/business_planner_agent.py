"""
Business Planner Agent
PRD를 바탕으로 비즈니스 전략과 실행 계획을 수립하는 Agent
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime, timedelta

logger = get_logger("business_planner_agent")


class BusinessPlannerAgent:
    """비즈니스 기획 전문 Agent"""

    def __init__(self, llm=None, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator
        self.llm = llm or (orchestrator.llm_factory() if orchestrator else None)
        self.agent_instance = self._create_agent_instance()

    def _create_agent_instance(self) -> Agent:
        """비즈니스 기획 Agent의 기본 인스턴스 생성"""
        return self.create_agent()

    async def create_business_plan(self, prd_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        PRD를 바탕으로 종합적인 비즈니스 계획 수립
        
        Args:
            prd_content: PRDWriterAgent의 결과물
            
        Returns:
            Dict[str, Any]: 비즈니스 계획 결과
        """
        logger.info("💼 Starting business plan creation based on PRD")
        
        try:
            # 1. PRD에서 비즈니스 관련 정보 추출
            business_context = await self._extract_business_context(prd_content)
            
            # 2. 시장 분석 수행
            market_analysis = await self._conduct_market_analysis(business_context)
            
            # 3. 비즈니스 모델 설계
            business_model = await self._design_business_model(business_context, market_analysis)
            
            # 4. Go-to-Market 전략 수립
            gtm_strategy = await self._develop_gtm_strategy(business_model)
            
            # 5. 재무 계획 수립
            financial_plan = await self._create_financial_plan(business_model)
            
            # 6. 리스크 분석 및 대응 방안
            risk_analysis = await self._analyze_risks_and_mitigation(business_model)
            
            # 7. 실행 로드맵 생성
            execution_roadmap = await self._create_execution_roadmap(gtm_strategy, financial_plan)
            
            business_plan = {
                "business_context": business_context,
                "market_analysis": market_analysis,
                "business_model": business_model,
                "gtm_strategy": gtm_strategy,
                "financial_plan": financial_plan,
                "risk_analysis": risk_analysis,
                "execution_roadmap": execution_roadmap,
                "plan_timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info("✅ Business plan creation completed successfully")
            return business_plan
            
        except Exception as e:
            logger.error(f"💥 Error in business plan creation: {e}", exc_info=True)
            return await self._generate_fallback_business_plan(prd_content, str(e))

    async def _extract_business_context(self, prd_content: Dict[str, Any]) -> Dict[str, Any]:
        """PRD에서 비즈니스 관련 컨텍스트 추출"""
        if not self.llm:
            return {"error": "No LLM available for context extraction"}
            
        prompt = f"""
        Extract business-relevant context from the PRD for business planning.
        
        PRD Content: {json.dumps(prd_content, indent=2)}
        
        Extract and analyze:
        1. **Product Value Proposition**: What unique value does this product provide?
        2. **Target Market**: Who are the customers and what's the market size?
        3. **Revenue Opportunities**: How can this product generate revenue?
        4. **Competitive Landscape**: What are the competitive considerations?
        5. **Business Objectives**: What are the key business goals?
        6. **Resource Requirements**: What resources are needed for success?
        
        Format as structured business context for strategic planning.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return {"context": result, "status": "extracted"}
        except Exception as e:
            logger.warning(f"Business context extraction failed: {e}")
            return {
                "status": "extraction_limited",
                "basic_context": {
                    "product_type": "Digital product",
                    "target_market": "General users",
                    "value_proposition": "Improved user experience",
                    "revenue_model": "To be determined",
                    "competition": "Market analysis needed"
                }
            }

    async def _conduct_market_analysis(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """시장 분석 수행"""
        if not self.llm:
            return {"error": "No LLM available for market analysis"}
            
        prompt = f"""
        Conduct a comprehensive market analysis based on the business context.
        
        Business Context: {json.dumps(business_context, indent=2)}
        
        Analyze:
        1. **Market Size & Growth**: Total Addressable Market (TAM), Serviceable Available Market (SAM)
        2. **Target Customer Segments**: Detailed customer personas and segments
        3. **Competitive Analysis**: Direct and indirect competitors, competitive advantages
        4. **Market Trends**: Industry trends affecting the product
        5. **Market Entry Strategy**: How to enter and capture market share
        6. **Barriers to Entry**: Challenges and obstacles to consider
        
        Provide actionable market insights for business strategy.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {"analysis": result, "status": "completed"}
        except Exception as e:
            logger.warning(f"Market analysis failed: {e}")
            return {
                "status": "analysis_limited",
                "basic_analysis": {
                    "market_size": "Medium to large market opportunity",
                    "customer_segments": ["Primary users", "Secondary users"],
                    "competition": "Moderate competition expected",
                    "trends": ["Digital transformation", "User experience focus"],
                    "entry_strategy": "Product-led growth approach"
                }
            }

    async def _design_business_model(self, business_context: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """비즈니스 모델 설계"""
        if not self.llm:
            return {"error": "No LLM available for business model design"}
            
        prompt = f"""
        Design a comprehensive business model based on context and market analysis.
        
        Business Context: {json.dumps(business_context, indent=2)}
        Market Analysis: {json.dumps(market_analysis, indent=2)}
        
        Design:
        1. **Revenue Streams**: How the business will generate revenue
        2. **Value Propositions**: Unique value for each customer segment
        3. **Customer Segments**: Detailed target customer groups
        4. **Channels**: How to reach and deliver value to customers
        5. **Customer Relationships**: Type of relationships with each segment
        6. **Key Resources**: Critical assets required for the business
        7. **Key Activities**: Most important activities for success
        8. **Key Partnerships**: Strategic partnerships needed
        9. **Cost Structure**: Major cost drivers and structure
        
        Use Business Model Canvas framework for structured output.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {"model": result, "status": "designed"}
        except Exception as e:
            logger.warning(f"Business model design failed: {e}")
            return {
                "status": "design_limited",
                "basic_model": {
                    "revenue_streams": ["Subscription", "Transaction fees", "Premium features"],
                    "value_proposition": "Enhanced user experience and productivity",
                    "customer_segments": ["Individual users", "Business users"],
                    "channels": ["Direct online", "Partner channels"],
                    "cost_structure": ["Development", "Marketing", "Operations"]
                }
            }

    async def _develop_gtm_strategy(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """Go-to-Market 전략 수립"""
        if not self.llm:
            return {"error": "No LLM available for GTM strategy"}
            
        prompt = f"""
        Develop a comprehensive Go-to-Market strategy based on the business model.
        
        Business Model: {json.dumps(business_model, indent=2)}
        
        Develop:
        1. **Target Customer Definition**: Ideal customer profile and personas
        2. **Product Positioning**: How to position the product in the market
        3. **Pricing Strategy**: Pricing model and competitive pricing
        4. **Marketing Channels**: Customer acquisition channels and tactics
        5. **Sales Strategy**: Sales process and team structure
        6. **Launch Plan**: Product launch timeline and milestones
        7. **Customer Success**: Onboarding and retention strategies
        8. **Metrics & KPIs**: Success metrics for GTM execution
        
        Create an actionable GTM playbook.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {"strategy": result, "status": "developed"}
        except Exception as e:
            logger.warning(f"GTM strategy development failed: {e}")
            return {
                "status": "strategy_limited",
                "basic_strategy": {
                    "target_customers": "Early adopters and tech-savvy users",
                    "positioning": "User-friendly and efficient solution",
                    "pricing": "Competitive pricing with value-based tiers",
                    "marketing": ["Content marketing", "Social media", "SEO"],
                    "sales": "Product-led growth with sales support",
                    "launch": "Phased rollout with beta testing"
                }
            }

    async def _create_financial_plan(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """재무 계획 수립"""
        if not self.llm:
            return {"error": "No LLM available for financial planning"}
            
        prompt = f"""
        Create a comprehensive financial plan based on the business model.
        
        Business Model: {json.dumps(business_model, indent=2)}
        
        Plan:
        1. **Revenue Projections**: 3-year revenue forecast by stream
        2. **Cost Structure**: Detailed cost breakdown and projections
        3. **Funding Requirements**: Capital needs and funding strategy
        4. **Unit Economics**: Customer acquisition cost, lifetime value
        5. **Break-even Analysis**: When the business becomes profitable
        6. **Cash Flow Projections**: Monthly cash flow for first 18 months
        7. **Key Financial Metrics**: Important ratios and benchmarks
        8. **Scenario Planning**: Best case, worst case, realistic scenarios
        
        Provide realistic financial projections with clear assumptions.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
            return {"plan": result, "status": "created"}
        except Exception as e:
            logger.warning(f"Financial planning failed: {e}")
            return {
                "status": "planning_limited",
                "basic_plan": {
                    "revenue_model": "Subscription-based with growth projections",
                    "initial_costs": ["Development", "Marketing", "Operations"],
                    "funding_needs": "Seed funding for initial 12-18 months",
                    "break_even": "Expected within 18-24 months",
                    "key_metrics": ["MRR", "CAC", "LTV", "Churn rate"]
                }
            }

    async def _analyze_risks_and_mitigation(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """리스크 분석 및 대응 방안"""
        if not self.llm:
            return {"error": "No LLM available for risk analysis"}
            
        prompt = f"""
        Analyze business risks and develop mitigation strategies.
        
        Business Model: {json.dumps(business_model, indent=2)}
        
        Analyze:
        1. **Market Risks**: Competition, market changes, customer behavior
        2. **Technical Risks**: Technology failures, scalability issues
        3. **Financial Risks**: Funding, cash flow, revenue shortfalls
        4. **Operational Risks**: Team, processes, execution challenges
        5. **Regulatory Risks**: Compliance, legal, privacy concerns
        6. **Strategic Risks**: Partnership, positioning, strategic decisions
        
        For each risk category:
        - Identify specific risks
        - Assess probability and impact
        - Develop mitigation strategies
        - Create contingency plans
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {"analysis": result, "status": "analyzed"}
        except Exception as e:
            logger.warning(f"Risk analysis failed: {e}")
            return {
                "status": "analysis_limited",
                "basic_risks": {
                    "market_risks": ["Competitive pressure", "Market adoption"],
                    "technical_risks": ["Scalability", "Security"],
                    "financial_risks": ["Funding gaps", "Revenue targets"],
                    "operational_risks": ["Team scaling", "Process efficiency"],
                    "mitigation": "Regular monitoring and adaptive strategies"
                }
            }

    async def _create_execution_roadmap(self, gtm_strategy: Dict[str, Any], financial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """실행 로드맵 생성"""
        if not self.llm:
            return {"error": "No LLM available for roadmap creation"}
            
        prompt = f"""
        Create a detailed execution roadmap for business plan implementation.
        
        GTM Strategy: {json.dumps(gtm_strategy, indent=2)}
        Financial Plan: {json.dumps(financial_plan, indent=2)}
        
        Create:
        1. **Phase 1 (0-6 months)**: Launch preparation and initial execution
        2. **Phase 2 (6-12 months)**: Market entry and early growth
        3. **Phase 3 (12-18 months)**: Scale and optimization
        4. **Phase 4 (18-24 months)**: Expansion and maturity
        
        For each phase include:
        - Key objectives and deliverables
        - Critical milestones and success metrics
        - Resource requirements and team needs
        - Budget allocation and financial targets
        - Risk mitigation activities
        - Dependencies and critical path items
        
        Provide actionable timeline with clear accountability.
        """
        
        try:
            result = await self.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
            return {"roadmap": result, "status": "created"}
        except Exception as e:
            logger.warning(f"Roadmap creation failed: {e}")
            return {
                "status": "roadmap_limited",
                "basic_roadmap": {
                    "phase_1": "Product development and team building",
                    "phase_2": "Market launch and customer acquisition",
                    "phase_3": "Growth optimization and feature expansion",
                    "phase_4": "Market expansion and scaling",
                    "timeline": "24-month execution plan"
                }
            }

    async def _generate_fallback_business_plan(self, prd_content: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """비즈니스 계획 생성 실패 시 기본 계획 생성"""
        return {
            "status": "fallback_plan",
            "error": error_msg,
            "basic_business_plan": {
                "executive_summary": {
                    "product_overview": "Digital product with modern user interface",
                    "market_opportunity": "Growing market for user-friendly solutions",
                    "business_model": "Subscription-based with freemium options",
                    "funding_needs": "Seed funding for initial development and marketing"
                },
                "market_strategy": {
                    "target_market": "Tech-savvy early adopters",
                    "competitive_advantage": "Superior user experience and design",
                    "go_to_market": "Product-led growth strategy",
                    "pricing": "Competitive pricing with value tiers"
                },
                "financial_overview": {
                    "revenue_model": "Recurring subscription revenue",
                    "cost_structure": "Development, marketing, and operational costs",
                    "break_even": "18-24 months post-launch",
                    "funding_requirement": "12-18 months runway"
                },
                "execution_plan": {
                    "phase_1": "Product development and testing (0-6 months)",
                    "phase_2": "Market launch and initial traction (6-12 months)",
                    "phase_3": "Growth and optimization (12-18 months)",
                    "phase_4": "Scale and expansion (18+ months)"
                },
                "key_risks": [
                    "Market competition and adoption challenges",
                    "Technical development and scalability risks",
                    "Funding and cash flow management",
                    "Team building and execution capabilities"
                ],
                "success_metrics": [
                    "Monthly Recurring Revenue (MRR)",
                    "Customer Acquisition Cost (CAC)",
                    "Customer Lifetime Value (LTV)",
                    "Product-market fit indicators"
                ]
            },
            "recommendations": [
                "Conduct detailed market research and customer validation",
                "Develop minimum viable product (MVP) for testing",
                "Build strategic partnerships for market entry",
                "Establish clear metrics and tracking systems",
                "Plan for iterative development and improvement"
            ],
            "plan_timestamp": datetime.now().isoformat(),
            "note": "This is a fallback business plan. Please refine with specific market research and business requirements."
        }

    @staticmethod
    def create_agent() -> Agent:
        """
        비즈니스 기획 Agent 생성
        
        Returns:
            Agent: 설정된 비즈니스 기획 Agent
        """
        
        instruction = """
        You are a business strategy and planning expert. Your role is to develop comprehensive business plans based on product requirements and market analysis.
        
        **Core Responsibilities:**
        1. **Business Model Design**: Create sustainable and scalable business models
        2. **Market Analysis**: Conduct thorough market research and competitive analysis
        3. **Financial Planning**: Develop realistic financial projections and funding strategies
        4. **Go-to-Market Strategy**: Design effective customer acquisition and market entry strategies
        5. **Risk Management**: Identify and mitigate business risks
        6. **Execution Planning**: Create actionable roadmaps and implementation plans
        
        **Analysis Framework:**
        - Use structured business frameworks (Business Model Canvas, Porter's Five Forces, etc.)
        - Focus on data-driven insights and realistic assumptions
        - Consider multiple scenarios and contingency planning
        - Emphasize measurable outcomes and success metrics
        
        **Output Quality:**
        - Provide actionable and implementable recommendations
        - Include clear timelines and resource requirements
        - Consider both short-term execution and long-term strategy
        - Address stakeholder concerns and investor requirements
        
        Always think strategically while remaining practical and execution-focused.
        """
        
        return Agent(
            name="business_planner",
            instruction=instruction,
            server_names=["fetch", "filesystem"]
        )

    @staticmethod
    def get_description() -> str:
        """Agent 설명 반환"""
        return "💼 PRD를 바탕으로 종합적인 비즈니스 전략과 실행 계획을 수립하는 전문 Agent"

    @staticmethod
    def get_capabilities() -> List[str]:
        """Agent 주요 기능 목록 반환"""
        return [
            "비즈니스 모델 설계 및 검증",
            "시장 분석 및 경쟁 환경 조사",
            "Go-to-Market 전략 수립",
            "재무 계획 및 투자 전략 개발",
            "리스크 분석 및 대응 방안 수립",
            "실행 로드맵 및 마일스톤 계획",
            "비즈니스 가정 검증 및 테스트 방법론",
            "성과 지표 및 KPI 정의"
        ]

    @staticmethod
    def get_planning_frameworks() -> List[str]:
        """사용하는 기획 프레임워크 목록"""
        return [
            "Business Model Canvas",
            "Porter's Five Forces",
            "SWOT Analysis",
            "Market Sizing (TAM/SAM/SOM)",
            "Customer Development",
            "Lean Startup Methodology",
            "OKRs (Objectives and Key Results)",
            "Financial Modeling and Projections"
        ] 