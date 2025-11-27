"""
Revenue Operations Intelligence Agent (RevOps)

매출 파이프라인 최적화, 예측, 수익성 분석을 수행하는 enterprise급 multi-agent 시스템.

핵심 기능:
1. 매출 파이프라인 분석 및 병목 지점 식별
2. AI 기반 매출 예측 및 목표 달성 가능성 평가
3. 거래 분석, 위험 평가, 우선순위화
4. 지역/세그먼트 최적화 및 할당 최적화
5. 이탈 위험 고객 식별 및 예방 전략
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.llm.fallback_llm import create_fallback_orchestrator_llm_factory
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from srcs.common.utils import setup_agent_app, save_report
from srcs.core.config.loader import settings
from srcs.core.errors import WorkflowError, APIError


class RevenueOperationsIntelligenceAgent:
    """
    Revenue Operations Intelligence Agent (RevOps)
    
    매출 파이프라인을 분석하고 최적화하여 매출 증대를 달성합니다.
    """
    
    def __init__(self, output_dir: str = "revops_intelligence_reports"):
        """
        Revenue Operations Intelligence Agent 초기화
        
        Args:
            output_dir: 리포트 저장 디렉토리
        """
        self.app = setup_agent_app("revenue_operations_intelligence_system")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_revops_workflow(
        self,
        business_context: Dict[str, Any],
        save_to_file: bool = False
    ) -> Dict[str, Any]:
        """
        RevOps 워크플로우 실행 (동기)
        
        Args:
            business_context: 비즈니스 컨텍스트 정보
                - company_name: 회사명
                - revenue_target: 매출 목표 (선택)
                - sales_team_size: 영업팀 규모 (선택)
                - current_pipeline_value: 현재 파이프라인 가치 (선택)
                - time_period: 분석 기간 (선택)
                - territories: 지역/세그먼트 정보 (선택)
            save_to_file: 파일 저장 여부
        
        Returns:
            dict: 실행 결과 및 생성된 콘텐츠
        """
        try:
            validated_context = self._validate_business_context(business_context)
            
            result = asyncio.run(
                self._async_workflow(validated_context, save_to_file)
            )
            return {
                'success': True,
                'message': 'Revenue operations intelligence workflow completed successfully',
                'output_dir': self.output_dir if save_to_file else None,
                'content': result,
                'save_to_file': save_to_file
            }
        except WorkflowError as e:
            return {
                'success': False,
                'message': f'Workflow error: {str(e)}',
                'error': str(e),
                'error_type': 'WorkflowError',
                'save_to_file': save_to_file
            }
        except APIError as e:
            return {
                'success': False,
                'message': f'API error during execution: {str(e)}',
                'error': str(e),
                'error_type': 'APIError',
                'save_to_file': save_to_file
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Unexpected error during RevOps workflow execution: {str(e)}',
                'error': str(e),
                'error_type': type(e).__name__,
                'save_to_file': save_to_file
            }
    
    async def _async_workflow(
        self,
        business_context: Dict[str, Any],
        save_to_file: bool
    ) -> Dict[str, Any]:
        """
        비동기 워크플로우 실행
        
        Args:
            business_context: 검증된 비즈니스 컨텍스트
            save_to_file: 파일 저장 여부
        
        Returns:
            dict: 생성된 분석 결과
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
        
        async with self.app.run() as app_context:
            context = app_context.context
            logger = app_context.logger
            
            # Configure filesystem server
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
            
            # Create all RevOps agents
            agents = self._create_revops_agents(business_context)
            
            # Create orchestrator
            orchestrator = orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(
    primary_model="gemini-2.5-flash-lite",
    logger_instance=logger
)
Orchestrator(
                llm_factory=orchestrator_llm_factory,
                available_agents=list(agents.values()),
                plan_type="full",
            )
            
            # Create task
            task = self._create_task(business_context, timestamp, save_to_file)
            
            # Execute workflow
            logger.info("Starting revenue operations intelligence workflow")
            
            # Model 설정: settings에서 직접 가져오거나 agent가 판단
            # Fallback 없음: 설정이 없으면 WorkflowError 발생
            model_name = None
            if hasattr(settings, 'llm') and hasattr(settings.llm, 'default_model'):
                model_name = settings.llm.default_model
            
            if not model_name:
                raise WorkflowError("LLM model not configured. Please set llm.default_model in settings.")
            
            try:
                result = await orchestrator.generate_str(
                    message=task,
                    request_params=RequestParams(model=model_name)
                )
                
                logger.info("Revenue operations intelligence workflow completed successfully")
            except WorkflowError:
                raise
            except APIError:
                raise
            except Exception as e:
                logger.error(f"Error during workflow execution: {str(e)}", exc_info=True)
                raise WorkflowError(f"Unexpected error in workflow execution: {str(e)}") from e
            
            if save_to_file:
                logger.info(f"All deliverables saved in {self.output_dir}/")
            
            return {
                'business_context': business_context,
                'analysis': result,
                'timestamp': timestamp
            }
    
    def _validate_business_context(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        비즈니스 컨텍스트 검증
        
        Args:
            business_context: 입력된 비즈니스 컨텍스트
        
        Returns:
            dict: 검증된 비즈니스 컨텍스트
        
        Raises:
            WorkflowError: 비즈니스 컨텍스트가 유효하지 않은 경우
        """
        if not isinstance(business_context, dict):
            raise WorkflowError("business_context must be a dictionary")
        
        # Fallback 없음: agent가 동적으로 판단하여 필요한 정보를 요청하거나 추론
        return business_context
    
    def _create_revops_agents(
        self,
        business_context: Dict[str, Any]
    ) -> Dict[str, Agent]:
        """
        Revenue Operations Intelligence를 위한 모든 agent 생성
        
        Args:
            business_context: 검증된 비즈니스 컨텍스트
        
        Returns:
            dict: 생성된 agent 딕셔너리
        """
        agents = {}
        
        # 1. Pipeline Analyzer Agent
        agents['pipeline_analyzer'] = Agent(
            name="pipeline_analyzer",
            instruction=self._create_pipeline_analyzer_instruction(business_context),
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # 2. Revenue Forecaster Agent
        agents['revenue_forecaster'] = Agent(
            name="revenue_forecaster",
            instruction=self._create_revenue_forecaster_instruction(business_context),
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # 3. Deal Intelligence Agent
        agents['deal_intelligence'] = Agent(
            name="deal_intelligence",
            instruction=self._create_deal_intelligence_instruction(business_context),
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # 4. Territory Optimizer Agent
        agents['territory_optimizer'] = Agent(
            name="territory_optimizer",
            instruction=self._create_territory_optimizer_instruction(business_context),
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # 5. Churn Predictor Agent
        agents['churn_predictor'] = Agent(
            name="churn_predictor",
            instruction=self._create_churn_predictor_instruction(business_context),
            server_names=["filesystem", "g-search", "fetch"],
        )
        
        # 6. Quality Evaluator Agent
        agents['quality_evaluator'] = Agent(
            name="revops_quality_evaluator",
            instruction=self._create_quality_evaluator_instruction(),
        )
        
        # 7. Quality Controller (EvaluatorOptimizerLLM)
        agents['quality_controller'] =         evaluator_llm_factory = create_fallback_orchestrator_llm_factory(
            primary_model="gemini-2.5-flash-lite",
            logger_instance=logger
        )
        EvaluatorOptimizerLLM(
            optimizer=agents['pipeline_analyzer'],
            evaluator=agents['quality_evaluator'],
            llm_factory=evaluator_llm_factory,
            min_rating=QualityRating.GOOD,
        )
        
        return agents
    
    def _create_pipeline_analyzer_instruction(
        self,
        business_context: Dict[str, Any]
    ) -> str:
        """
        Pipeline Analyzer Agent instruction 동적 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
        
        Returns:
            str: agent instruction
        """
        instruction = f"""You are an expert revenue operations analyst specializing in sales pipeline analysis and optimization.

Business Context:
{self._format_business_context_for_instruction(business_context)}

Your task is to analyze the sales pipeline comprehensively:

1. Pipeline Structure Analysis:
   - Stage-by-stage pipeline breakdown and value distribution
   - Conversion rates at each pipeline stage
   - Average sales cycle length by stage
   - Win/loss rates and reasons
   - Pipeline velocity and momentum indicators
   - Bottleneck identification and impact assessment

2. Deal Flow Analysis:
   - Deal volume and value trends over time
   - Deal source analysis and attribution
   - Deal progression patterns and timing
   - Stalled deal identification and root causes
   - Deal aging analysis and risk assessment
   - Pipeline coverage ratios and health metrics

3. Performance Metrics:
   - Sales team performance by individual and segment
   - Activity metrics (calls, meetings, demos, proposals)
   - Conversion funnel metrics and drop-off points
   - Time-to-close analysis by deal type
   - Revenue per sales rep and productivity metrics
   - Pipeline-to-quota ratios and attainment tracking

4. Risk Assessment:
   - At-risk deals identification and scoring
   - Deal slippage probability and impact
   - Competitive threat analysis
   - Budget and approval process risks
   - Timing and urgency factors
   - Resource and capacity constraints

5. Optimization Opportunities:
   - Pipeline stage conversion improvement areas
   - Sales process efficiency enhancements
   - Resource allocation optimization
   - Deal prioritization and focus recommendations
   - Sales methodology and training needs
   - Technology and tool utilization gaps

6. Forecasting Accuracy:
   - Historical forecast accuracy analysis
   - Forecast bias identification and correction
   - Confidence intervals and probability distributions
   - Scenario planning and sensitivity analysis
   - Risk-adjusted revenue projections

Use MCP tools (g-search, fetch) to research industry benchmarks, best practices, and comparable company metrics.
Base all analysis on real data and research, not assumptions.

Ensure analysis is:
- Data-driven and quantitative
- Actionable with specific recommendations
- Based on real pipeline data and research
- Free of generic or templated content
- Production-ready for executive decision making

Output format: Comprehensive markdown report with data visualizations, metrics, and actionable insights.
"""
        return instruction
    
    def _create_revenue_forecaster_instruction(
        self,
        business_context: Dict[str, Any]
    ) -> str:
        """
        Revenue Forecaster Agent instruction 동적 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
        
        Returns:
            str: agent instruction
        """
        instruction = f"""You are an expert revenue forecasting analyst specializing in AI-powered revenue prediction and goal achievement analysis.

Business Context:
{self._format_business_context_for_instruction(business_context)}

Your task is to create accurate revenue forecasts and assess goal achievement probability:

1. Revenue Forecasting:
   - Historical revenue trend analysis and pattern recognition
   - Seasonality and cyclical pattern identification
   - Market condition and economic factor impact
   - Pipeline-based revenue projections
   - Statistical forecasting models (time series, regression, ML-based)
   - Confidence intervals and probability distributions

2. Goal Achievement Analysis:
   - Revenue target vs. forecast gap analysis
   - Goal achievement probability assessment
   - Scenario planning (best case, base case, worst case)
   - Required pipeline growth to achieve targets
   - Sales velocity and conversion rate requirements
   - Resource and capacity needs assessment

3. Predictive Analytics:
   - Deal close probability modeling
   - Revenue timing and recognition forecasting
   - Customer acquisition and expansion predictions
   - Churn impact on revenue projections
   - Market trend and competitive impact analysis
   - Economic and industry factor considerations

4. Risk Analysis:
   - Forecast risk factors and uncertainties
   - Downside scenario probability and impact
   - External risk factors (market, competition, economy)
   - Internal risk factors (team, product, operations)
   - Mitigation strategies and contingency planning
   - Early warning indicators and triggers

5. Action Planning:
   - Gap-closing strategies and tactics
   - Pipeline acceleration recommendations
   - Sales process optimization opportunities
   - Resource allocation adjustments
   - Target adjustment recommendations if needed
   - Success probability improvement actions

6. Measurement and Tracking:
   - Forecast accuracy metrics and KPIs
   - Leading indicators and early signals
   - Real-time forecast updates and adjustments
   - Variance analysis and root cause identification
   - Continuous improvement and model refinement

Use MCP tools (g-search, fetch) to research forecasting methodologies, industry benchmarks, and economic indicators.
Base all forecasts on real data, statistical methods, and market research.

Ensure forecasts are:
- Statistically sound and methodologically rigorous
- Transparent with assumptions and confidence levels
- Actionable with clear recommendations
- Based on real data and research
- Production-ready for executive decision making

Output format: Detailed forecast report with scenarios, probabilities, and actionable recommendations.
"""
        return instruction
    
    def _create_deal_intelligence_instruction(
        self,
        business_context: Dict[str, Any]
    ) -> str:
        """
        Deal Intelligence Agent instruction 동적 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
        
        Returns:
            str: agent instruction
        """
        instruction = f"""You are an expert deal intelligence analyst specializing in deal analysis, risk assessment, and prioritization.

Business Context:
{self._format_business_context_for_instruction(business_context)}

Your task is to analyze deals comprehensively and provide actionable intelligence:

1. Deal Analysis:
   - Deal value, size, and strategic importance assessment
   - Deal stage and progression analysis
   - Deal health and momentum indicators
   - Stakeholder and decision-maker mapping
   - Competitive landscape and positioning
   - Deal timeline and urgency factors

2. Win Probability Assessment:
   - Historical win rate analysis by deal characteristics
   - Deal scoring and win probability modeling
   - Risk factors and probability adjustments
   - Competitive threat assessment
   - Internal capability and resource alignment
   - Customer fit and need alignment

3. Deal Prioritization:
   - Strategic value and impact scoring
   - Revenue potential and profitability analysis
   - Resource requirement and ROI assessment
   - Timing and urgency prioritization
   - Risk-adjusted value calculation
   - Portfolio optimization recommendations

4. Risk Assessment:
   - Deal risk factors and severity analysis
   - Competitive displacement risk
   - Budget and approval process risks
   - Technical and implementation risks
   - Relationship and stakeholder risks
   - Timing and market condition risks

5. Action Recommendations:
   - Deal acceleration strategies
   - Risk mitigation tactics
   - Resource allocation recommendations
   - Stakeholder engagement plans
   - Competitive response strategies
   - Deal-specific next steps

6. Performance Tracking:
   - Deal progression monitoring
   - Win/loss analysis and learning
   - Forecast accuracy by deal type
   - Sales rep performance by deal
   - Process efficiency and optimization
   - Best practice identification

Use MCP tools (g-search, fetch) to research industry benchmarks, competitive intelligence, and best practices.
Base all analysis on real deal data and research.

Ensure analysis is:
- Deal-specific and actionable
- Risk-aware and probability-based
- Strategic and revenue-focused
- Based on real data and research
- Production-ready for sales team execution

Output format: Deal intelligence report with scoring, prioritization, and action plans.
"""
        return instruction
    
    def _create_territory_optimizer_instruction(
        self,
        business_context: Dict[str, Any]
    ) -> str:
        """
        Territory Optimizer Agent instruction 동적 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
        
        Returns:
            str: agent instruction
        """
        instruction = f"""You are an expert territory and segment optimization analyst specializing in sales territory design and resource allocation.

Business Context:
{self._format_business_context_for_instruction(business_context)}

Your task is to optimize territory and segment allocation:

1. Territory Analysis:
   - Current territory performance and productivity
   - Geographic and market opportunity mapping
   - Customer concentration and distribution
   - Competitive landscape by territory
   - Market maturity and growth potential
   - Resource allocation and coverage analysis

2. Segment Optimization:
   - Customer segment performance and profitability
   - Segment growth potential and opportunity sizing
   - Segment-specific sales approach effectiveness
   - Cross-segment synergies and opportunities
   - Segment resource requirements and ROI
   - Segment prioritization and focus areas

3. Allocation Optimization:
   - Sales rep assignment and territory balancing
   - Workload distribution and capacity planning
   - Skill and experience matching
   - Geographic and travel optimization
   - Account coverage and relationship management
   - Resource efficiency and productivity maximization

4. Performance Optimization:
   - Territory-specific performance benchmarks
   - Productivity improvement opportunities
   - Best practice identification and sharing
   - Training and development needs
   - Technology and tool utilization
   - Process standardization and optimization

5. Growth Planning:
   - Territory expansion opportunities
   - New market entry strategies
   - Segment expansion and penetration
   - Resource scaling and hiring plans
   - Market development priorities
   - Investment allocation recommendations

6. Measurement and Tracking:
   - Territory performance metrics and KPIs
   - Productivity and efficiency tracking
   - ROI and profitability measurement
   - Goal attainment and quota performance
   - Continuous optimization and adjustment
   - Best practice documentation

Use MCP tools (g-search, fetch) to research market data, geographic intelligence, and industry benchmarks.
Base all optimization on real performance data and market research.

Ensure optimization is:
- Data-driven and quantitative
- Balanced and fair for sales team
- Strategic and growth-oriented
- Based on real data and research
- Production-ready for implementation

Output format: Territory optimization report with allocation recommendations and performance projections.
"""
        return instruction
    
    def _create_churn_predictor_instruction(
        self,
        business_context: Dict[str, Any]
    ) -> str:
        """
        Churn Predictor Agent instruction 동적 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
        
        Returns:
            str: agent instruction
        """
        instruction = f"""You are an expert churn prediction and prevention analyst specializing in customer retention and revenue protection.

Business Context:
{self._format_business_context_for_instruction(business_context)}

Your task is to identify churn risks and develop prevention strategies:

1. Churn Risk Identification:
   - Customer health score calculation and monitoring
   - Early warning indicators and signals
   - Behavioral pattern analysis and anomaly detection
   - Engagement and usage trend analysis
   - Support and satisfaction metrics
   - Competitive threat and market condition factors

2. Risk Scoring and Prioritization:
   - Churn probability modeling and scoring
   - Revenue at risk calculation
   - Customer value and LTV consideration
   - Urgency and timing factors
   - Intervention feasibility assessment
   - Resource requirement estimation

3. Root Cause Analysis:
   - Churn reason identification and categorization
   - Product and feature gap analysis
   - Service and support quality issues
   - Pricing and value perception problems
   - Competitive displacement factors
   - Relationship and communication breakdowns

4. Prevention Strategies:
   - Proactive intervention playbooks
   - Customer success and engagement programs
   - Product adoption and usage optimization
   - Value realization and ROI demonstration
   - Relationship strengthening initiatives
   - Competitive defense strategies

5. Retention Actions:
   - Personalized retention offers and incentives
   - Product and feature recommendations
   - Training and onboarding improvements
   - Account management and support enhancements
   - Contract and pricing adjustments
   - Executive engagement and escalation

6. Measurement and Optimization:
   - Retention program effectiveness tracking
   - Churn prediction accuracy measurement
   - Intervention success rates
   - Revenue saved and protected
   - Process improvement and optimization
   - Best practice identification

Use MCP tools (g-search, fetch) to research churn prevention best practices, industry benchmarks, and retention strategies.
Base all analysis on real customer data and research.

Ensure analysis is:
- Predictive and proactive
- Actionable with specific interventions
- Customer-centric and value-focused
- Based on real data and research
- Production-ready for customer success teams

Output format: Churn risk report with prioritized customer list, risk scores, and prevention action plans.
"""
        return instruction
    
    def _create_quality_evaluator_instruction(self) -> str:
        """
        Quality Evaluator Agent instruction 생성
        
        Returns:
            str: agent instruction
        """
        instruction = """You are a quality assurance expert evaluating revenue operations intelligence deliverables.

Evaluate the RevOps analysis based on:

1. Analysis Quality (30%):
   - Data accuracy and completeness
   - Methodological rigor and statistical soundness
   - Depth and comprehensiveness of analysis
   - Actionability and specificity of recommendations
   - Research-based insights and benchmarks

2. Business Value (30%):
   - Revenue impact potential
   - Practical applicability and feasibility
   - ROI and cost-benefit analysis
   - Strategic alignment and priorities
   - Implementation readiness

3. Forecasting Accuracy (20%):
   - Forecast methodology soundness
   - Confidence intervals and risk assessment
   - Historical accuracy validation
   - Scenario planning completeness
   - Goal achievement probability assessment

4. Actionability (20%):
   - Specific and detailed recommendations
   - Clear next steps and priorities
   - Resource and timeline requirements
   - Success metrics and KPIs
   - Implementation roadmap clarity

Provide EXCELLENT, GOOD, FAIR, or POOR ratings with specific improvement recommendations.
Focus on actionable feedback that improves revenue operations effectiveness.
"""
        return instruction
    
    def _format_business_context_for_instruction(self, business_context: Dict[str, Any]) -> str:
        """
        비즈니스 컨텍스트를 instruction용 포맷으로 변환
        Fallback 없음: agent가 동적으로 판단하여 필요한 정보를 추론하거나 요청
        
        Args:
            business_context: 비즈니스 컨텍스트 딕셔너리
        
        Returns:
            str: 포맷된 비즈니스 컨텍스트 문자열
        """
        lines = []
        
        # 모든 필드를 agent가 동적으로 판단하도록 제공
        if 'company_name' in business_context:
            lines.append(f"- Company Name: {business_context['company_name']}")
        if 'revenue_target' in business_context:
            lines.append(f"- Revenue Target: {business_context['revenue_target']}")
        if 'sales_team_size' in business_context:
            lines.append(f"- Sales Team Size: {business_context['sales_team_size']}")
        if 'current_pipeline_value' in business_context:
            lines.append(f"- Current Pipeline Value: {business_context['current_pipeline_value']}")
        if 'time_period' in business_context:
            lines.append(f"- Analysis Time Period: {business_context['time_period']}")
        if 'territories' in business_context:
            territories = business_context['territories']
            if isinstance(territories, list):
                lines.append(f"- Territories: {', '.join(str(t) for t in territories)}")
            elif territories:
                lines.append(f"- Territories: {territories}")
        
        # 추가 필드가 있으면 모두 포함
        for key, value in business_context.items():
            if key not in ['company_name', 'revenue_target', 'sales_team_size', 'current_pipeline_value', 'time_period', 'territories']:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Fallback 없음: agent가 동적으로 판단
        # 정보가 없으면 agent가 MCP 도구를 사용하여 조사하도록 instruction에 명시
        formatted_context = "\n".join(lines)
        return formatted_context
    
    def _create_task(
        self,
        business_context: Dict[str, Any],
        timestamp: str,
        save_to_file: bool
    ) -> str:
        """
        워크플로우 task 생성
        
        Args:
            business_context: 비즈니스 컨텍스트
            timestamp: 타임스탬프
            save_to_file: 파일 저장 여부
        
        Returns:
            str: task description
        """
        # Fallback 없음: agent가 동적으로 판단
        task = f"""Execute a comprehensive revenue operations intelligence analysis for the following business:

{self._format_business_context_for_instruction(business_context)}

Workflow Steps:

1. Use the quality_controller (pipeline_analyzer + quality_evaluator) to generate high-quality pipeline analysis:
   - Analyze sales pipeline structure and performance using MCP tools (g-search, fetch)
   - Identify bottlenecks, conversion issues, and optimization opportunities
   - Assess pipeline health and coverage ratios
   - Generate actionable recommendations for pipeline improvement
   - Each analysis must be detailed, data-driven, and based on real research

2. Use the revenue_forecaster to create accurate revenue forecasts:
   - Analyze historical revenue trends and patterns
   - Create statistical forecasts with confidence intervals
   - Assess goal achievement probability and scenarios
   - Identify gap-closing strategies and requirements
   - Use MCP tools to research forecasting methodologies and benchmarks
   - Provide scenario planning (best case, base case, worst case)

3. Use the deal_intelligence to analyze and prioritize deals:
   - Assess deal value, win probability, and strategic importance
   - Score and prioritize deals based on risk-adjusted value
   - Identify at-risk deals and recommend actions
   - Analyze competitive threats and opportunities
   - Use MCP tools to research competitive intelligence and benchmarks
   - Provide deal-specific action plans

4. Use the territory_optimizer to optimize territory and segment allocation:
   - Analyze current territory performance and productivity
   - Identify optimization opportunities for allocation
   - Recommend territory balancing and resource allocation
   - Assess segment performance and growth potential
   - Use MCP tools to research market data and geographic intelligence
   - Provide allocation recommendations with performance projections

5. Use the churn_predictor to identify churn risks and develop prevention strategies:
   - Calculate customer health scores and churn probability
   - Identify at-risk customers and revenue at risk
   - Analyze root causes of churn risk
   - Develop prevention strategies and intervention playbooks
   - Use MCP tools to research churn prevention best practices
   - Provide prioritized customer list with action plans

6. Generate comprehensive deliverables:
   - Executive summary of RevOps analysis
   - Pipeline analysis report with optimization recommendations
   - Revenue forecast with scenarios and goal achievement assessment
   - Deal intelligence report with prioritization and action plans
   - Territory optimization report with allocation recommendations
   - Churn risk report with prevention strategies
   - Implementation roadmap and success metrics

All analysis must be:
- Based on real data and research (use MCP tools)
- Free of dummy or mock data
- Actionable and implementable
- Measurable and trackable
- Production-ready quality

"""
        
        if save_to_file:
            task += f"""
Save all deliverables in the {self.output_dir} directory with appropriate naming:
- pipeline_analysis_{timestamp}.md
- revenue_forecast_{timestamp}.md
- deal_intelligence_{timestamp}.md
- territory_optimization_{timestamp}.md
- churn_risk_report_{timestamp}.md
- executive_summary_{timestamp}.md
- implementation_roadmap_{timestamp}.md
"""
        else:
            task += """
Return the complete analysis for immediate display. Do not save to files.
Provide comprehensive, detailed results including all analyses, forecasts, and recommendations.
"""
        
        return task


async def main():
    """
    Revenue Operations Intelligence Agent 실행 예제
    """
    # 예제 비즈니스 컨텍스트
    business_context = {
        'company_name': 'TechCorp Inc.',
        'revenue_target': '$50M ARR',
        'sales_team_size': 25,
        'current_pipeline_value': '$75M',
        'time_period': 'Q1 2025',
        'territories': ['North America', 'Europe', 'Asia Pacific']
    }
    
    agent = RevenueOperationsIntelligenceAgent()
    result = agent.run_revops_workflow(
        business_context=business_context,
        save_to_file=True
    )
    
    print(f"Workflow completed: {result['success']}")
    if result['success']:
        if 'output_dir' in result:
            output_dir = result['output_dir']
            if output_dir:
                print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

