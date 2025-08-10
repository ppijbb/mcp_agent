"""
Business Planner Agent
PRD를 바탕으로 비즈니스 전략과 실행 계획을 수립하는 Agent
"""

from typing import Dict, Any
import json
from datetime import datetime

from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.product_planner_agent.agents.base_agent_simple import BaseAgentSimple as BaseAgent
from srcs.product_planner_agent.utils.logger import get_product_planner_logger
from srcs.product_planner_agent.utils.llm_utils import get_llm_factory
from srcs.product_planner_agent.utils.cached_llm import CachedLLM
from srcs.product_planner_agent.utils.errors import WorkflowError

logger = get_product_planner_logger("agent.business_planner")


class BusinessPlannerAgent(BaseAgent):
    """비즈니스 기획 전문 Agent (spec-only, no fallback)."""

    def __init__(self):
        super().__init__("business_planner_agent")
        # Cache-wrapped LLM instance (centralized model selection via factory)
        self.llm = CachedLLM(get_llm_factory()())
        logger.info("BusinessPlannerAgent initialized.")

    async def run_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """PRD를 바탕으로 종합적인 비즈니스 계획 수립."""
        prd_content = context.get("prd_content")
        if not prd_content:
            raise WorkflowError("PRD content not provided in the context.")

        logger.info("💼 Starting business plan creation based on PRD")

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
            "status": "completed",
        }

        logger.info("✅ Business plan creation completed successfully")
        context["business_plan"] = business_plan
        return business_plan

    async def _extract_business_context(self, prd_content: Dict[str, Any]) -> Dict[str, Any]:
        """PRD에서 비즈니스 관련 컨텍스트 추출 (JSON enforced)."""
        prompt = f"""
        Extract business-relevant context from the PRD for business planning.

        PRD Content: {json.dumps(prd_content, indent=2, ensure_ascii=False)}

        Extract and analyze:
        1. Product Value Proposition
        2. Target Market (customers and market size)
        3. Revenue Opportunities
        4. Competitive Landscape
        5. Business Objectives
        6. Resource Requirements

        Respond as a single JSON object with keys:
        {{
          "value_proposition": str,
          "target_market": {{"segments": [str], "size": str}},
          "revenue_opportunities": [str],
          "competitive_landscape": [str],
          "business_objectives": [str],
          "resource_requirements": [str]
        }}
        """
        params = RequestParams(temperature=0.3, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse business context JSON: {e}") from e
        return {"context": parsed, "status": "extracted"}

    async def _conduct_market_analysis(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """시장 분석 수행 (JSON enforced)."""
        prompt = f"""
        Conduct a comprehensive market analysis based on the business context.

        Business Context: {json.dumps(business_context, indent=2, ensure_ascii=False)}

        Analyze and respond as a single JSON object with keys:
        {{
          "tam": str,
          "sam": str,
          "som": str,
          "growth_rate": str,
          "target_segments": [str],
          "competitors": [str],
          "trends": [str],
          "entry_strategy": [str],
          "barriers": [str]
        }}
        """
        params = RequestParams(temperature=0.4, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse market analysis JSON: {e}") from e
        return {"analysis": parsed, "status": "completed"}

    async def _design_business_model(self, business_context: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """비즈니스 모델 설계 (JSON enforced)."""
        prompt = f"""
        Design a comprehensive business model based on context and market analysis.

        Business Context: {json.dumps(business_context, indent=2, ensure_ascii=False)}
        Market Analysis: {json.dumps(market_analysis, indent=2, ensure_ascii=False)}

        Respond as a single JSON object with keys (Business Model Canvas):
        {{
          "revenue_streams": [str],
          "value_propositions": [str],
          "customer_segments": [str],
          "channels": [str],
          "customer_relationships": [str],
          "key_resources": [str],
          "key_activities": [str],
          "key_partnerships": [str],
          "cost_structure": [str]
        }}
        """
        params = RequestParams(temperature=0.4, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse business model JSON: {e}") from e
        return {"model": parsed, "status": "designed"}

    async def _develop_gtm_strategy(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """Go-to-Market 전략 수립 (JSON enforced)."""
        prompt = f"""
        Develop a comprehensive Go-to-Market strategy based on the business model.

        Business Model: {json.dumps(business_model, indent=2, ensure_ascii=False)}

        Respond as a single JSON object with keys:
        {{
          "target_customers": [str],
          "positioning": str,
          "pricing_strategy": str,
          "marketing_channels": [str],
          "sales_strategy": [str],
          "launch_plan": [str],
          "customer_success": [str],
          "kpis": [str]
        }}
        """
        params = RequestParams(temperature=0.4, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse GTM strategy JSON: {e}") from e
        return {"strategy": parsed, "status": "developed"}

    async def _create_financial_plan(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """재무 계획 수립 (JSON enforced)."""
        prompt = f"""
        Create a comprehensive financial plan based on the business model.

        Business Model: {json.dumps(business_model, indent=2, ensure_ascii=False)}

        Respond as a single JSON object with keys:
        {{
          "revenue_projections": {{"year1": str, "year2": str, "year3": str}},
          "cost_structure": [str],
          "funding_requirements": [str],
          "unit_economics": [str],
          "break_even": str,
          "cash_flow": [str],
          "financial_metrics": [str],
          "scenarios": {{"best": [str], "realistic": [str], "worst": [str]}}
        }}
        """
        params = RequestParams(temperature=0.3, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse financial plan JSON: {e}") from e
        return {"plan": parsed, "status": "created"}

    async def _analyze_risks_and_mitigation(self, business_model: Dict[str, Any]) -> Dict[str, Any]:
        """리스크 분석 및 대응 방안 (JSON enforced)."""
        prompt = f"""
        Analyze business risks and develop mitigation strategies.

        Business Model: {json.dumps(business_model, indent=2, ensure_ascii=False)}

        Respond as a single JSON object with keys:
        {{
          "market_risks": [str],
          "technical_risks": [str],
          "financial_risks": [str],
          "operational_risks": [str],
          "mitigations": [str]
        }}
        """
        params = RequestParams(temperature=0.4, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse risk analysis JSON: {e}") from e
        return {"analysis": parsed, "status": "analyzed"}

    async def _create_execution_roadmap(self, gtm_strategy: Dict[str, Any], financial_plan: Dict[str, Any]) -> Dict[str, Any]:
        """실행 로드맵 생성 (JSON enforced)."""
        prompt = f"""
        Create a detailed execution roadmap for business plan implementation.

        Go-to-Market Strategy: {json.dumps(gtm_strategy, indent=2, ensure_ascii=False)}
        Financial Plan: {json.dumps(financial_plan, indent=2, ensure_ascii=False)}

        Respond as a single JSON object with keys:
        {{
          "phases": [str],
          "milestones": [str],
          "timeline": str,
          "team_and_resources": [str],
          "dependencies": [str]
        }}
        """
        params = RequestParams(temperature=0.4, response_format={"type": "json_object"})
        result_str = await self.llm.generate_str(prompt, request_params=params)
        try:
            parsed = json.loads(result_str)
        except json.JSONDecodeError as e:
            raise WorkflowError(f"Failed to parse execution roadmap JSON: {e}") from e
        return {"roadmap": parsed, "status": "created"}