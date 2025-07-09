"""
Business Planner Agent
PRD를 바탕으로 비즈니스 전략과 실행 계획을 수립하는 Agent
"""

from srcs.core.agent.base import BaseAgent
from srcs.core.errors import APIError, WorkflowError
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.logging.logger import get_logger
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from mcp_agent.context import AgentContext
from srcs.product_planner_agent.prompt import PROMPT
from srcs.product_planner_agent.utils.llm_utils import get_llm_factory

logger = get_logger("business_planner_agent")

class BusinessPlannerAgent(BaseAgent):
    """비즈니스 기획 전문 Agent"""

    def __init__(self):
        super().__init__("business_planner_agent")

    async def run_workflow(self, context: AgentContext) -> Dict[str, Any]:
        """
        PRD를 바탕으로 종합적인 비즈니스 계획 수립
        """
        prd_content = context.get("prd_content")
        if not prd_content:
            raise WorkflowError("PRD content not provided in the context.")

        self.logger.info("💼 Starting business plan creation based on PRD")
        
        try:
            business_context = await self._extract_business_context(prd_content)
            market_analysis = await self._conduct_market_analysis(business_context)
            business_model = await self._design_business_model(business_context, market_analysis)
            gtm_strategy = await self._develop_gtm_strategy(business_model)
            financial_plan = await self._create_financial_plan(business_model)
            risk_analysis = await self._analyze_risks_and_mitigation(business_model)
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
            
            self.logger.info("✅ Business plan creation completed successfully")
            context.set("business_plan", business_plan)
            return business_plan
        except Exception as e:
            raise WorkflowError(f"Business plan creation failed: {e}") from e

    async def _extract_business_context(self, prd_content: Dict[str, Any]) -> Dict[str, Any]:
        """PRD에서 비즈니스 관련 컨텍스트 추출"""
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
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
        return {"context": result, "status": "extracted"}

    async def _conduct_market_analysis(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """시장 분석 수행"""
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
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
        return {"analysis": result, "status": "completed"}

    async def _design_business_model(self, business_context: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """비즈니스 모델 설계"""
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
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
        return {"model": result, "status": "designed"}

    async def _develop_gtm_strategy(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """Go-to-Market 전략 수립"""
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
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
        return {"strategy": result, "status": "developed"}

    async def _create_financial_plan(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """재무 계획 수립"""
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
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.3))
        return {"plan": result, "status": "created"}

    async def _analyze_risks_and_mitigation(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """리스크 분석 및 대응 방안"""
        prompt = f"""
        Analyze business risks and develop mitigation strategies.
        
        Business Model: {json.dumps(business_model, indent=2)}
        
        Analyze:
        1. **Market Risks**: Competition, market changes, customer behavior
        2. **Technical Risks**: Technology failures, scalability issues
        3. **Financial Risks**: Funding gaps, revenue targets
        4. **Operational Risks**: Team scaling, process efficiency
        
        Develop mitigation strategies for each identified risk.
        """
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
        return {"analysis": result, "status": "analyzed"}

    async def _create_execution_roadmap(self, gtm_strategy: Dict[str, Any], financial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """실행 로드맵 생성"""
        prompt = f"""
        Create a detailed execution roadmap for business plan implementation.
        
        Go-to-Market Strategy: {json.dumps(gtm_strategy, indent=2)}
        Financial Plan: {json.dumps(financial_plan, indent=2)}
        
        Create a roadmap with:
        1. **Phased Rollout**: Key phases of product launch and market expansion
        2. **Key Milestones**: Measurable milestones for each phase
        3. **Timeline**: Realistic timeline for the next 12-18 months
        4. **Team & Resources**: Required team and resources for each phase
        5. **Dependencies**: Critical dependencies between tasks and teams
        
        Provide a structured roadmap with clear action items.
        """
        
        result = await self.app.llm.generate_str(prompt, request_params=RequestParams(temperature=0.4))
        return {"roadmap": result, "status": "created"} 