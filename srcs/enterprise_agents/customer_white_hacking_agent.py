"""
Customer White Hacking Agent

비즈니스 상품에 대해 다양한 고객 페르소나 관점에서 분석하고,
각 페르소나에 최적화된 판매 시나리오를 자동 생성하는 enterprise급 multi-agent 시스템.

핵심 기능:
1. 비즈니스 상품 분석 및 고객 페르소나 생성
2. 각 페르소나별 독립적인 니즈, 페인포인트, 구매 동기 분석
3. 페르소나별 최적 판매 시나리오 작성
4. 시나리오 검증 및 최적화
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.common.llm.fallback_llm import create_fallback_orchestrator_llm_factory
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)
from srcs.common.utils import setup_agent_app
from srcs.core.config.loader import settings
from srcs.core.errors import WorkflowError, APIError


class CustomerWhiteHackingAgent:
    """
    Customer White Hacking Agent

    비즈니스 상품을 입력받아 고객 페르소나를 생성하고,
    각 페르소나에 최적화된 판매 시나리오를 자동 생성합니다.
    """

    def __init__(self, output_dir: str = "customer_white_hacking_reports"):
        """
        Customer White Hacking Agent 초기화

        Args:
            output_dir: 리포트 저장 디렉토리
        """
        self.app = setup_agent_app("customer_white_hacking_system")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_white_hacking_workflow(
        self,
        product_info: Dict[str, Any],
        save_to_file: bool = False
    ) -> Dict[str, Any]:
        """
        Customer White Hacking 워크플로우 실행 (동기)

        Args:
            product_info: 비즈니스 상품 정보
                - name: 상품명
                - description: 상품 설명
                - category: 상품 카테고리
                - target_market: 타겟 시장
                - price_range: 가격대 (선택)
                - features: 주요 기능 (선택)
                - competitors: 경쟁사 정보 (선택)
            save_to_file: 파일 저장 여부

        Returns:
            dict: 실행 결과 및 생성된 콘텐츠
        """
        try:
            # Validate product info before execution
            validated_product = self._validate_product_info(product_info)

            result = asyncio.run(
                self._async_workflow(validated_product, save_to_file)
            )
            return {
                'success': True,
                'message': 'Customer white hacking workflow completed successfully',
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
                'message': f'Unexpected error during white hacking workflow execution: {str(e)}',
                'error': str(e),
                'error_type': type(e).__name__,
                'save_to_file': save_to_file
            }

    async def _async_workflow(
        self,
        product_info: Dict[str, Any],
        save_to_file: bool
    ) -> Dict[str, Any]:
        """
        비동기 워크플로우 실행

        Args:
            product_info: 비즈니스 상품 정보
            save_to_file: 파일 저장 여부

        Returns:
            dict: 생성된 페르소나 및 판매 시나리오
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

            # Create all white hacking agents
            agents = self._create_white_hacking_agents(validated_product)

            # Create orchestrator
            orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(
                primary_model="gemini-2.5-flash-lite",
                logger_instance=logger
            )
            orchestrator = Orchestrator(
                llm_factory=orchestrator_llm_factory,
                available_agents=list(agents.values()),
                plan_type="full",
            )

            # Create task
            task = self._create_task(validated_product, timestamp, save_to_file)

            # Execute workflow
            logger.info("Starting customer white hacking workflow")

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

                logger.info("Customer white hacking workflow completed successfully")
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
                'product_info': validated_product,
                'analysis': result,
                'timestamp': timestamp
            }

    def _validate_product_info(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        상품 정보 검증

        Args:
            product_info: 입력된 상품 정보

        Returns:
            dict: 검증된 상품 정보

        Raises:
            WorkflowError: 상품 정보가 유효하지 않은 경우
        """
        if not isinstance(product_info, dict):
            raise WorkflowError("product_info must be a dictionary")

        # Fallback 없음: agent가 동적으로 판단하여 필요한 정보를 요청하거나 추론
        # 모든 필드는 agent가 필요에 따라 동적으로 처리
        return product_info

    def _create_white_hacking_agents(
        self,
        product_info: Dict[str, Any]
    ) -> Dict[str, Agent]:
        """
        Customer White Hacking을 위한 모든 agent 생성

        Args:
            product_info: 검증된 상품 정보

        Returns:
            dict: 생성된 agent 딕셔너리
        """
        agents = {}

        # 1. Persona Generator Agent
        agents['persona_generator'] = Agent(
            name="persona_generator",
            instruction=self._create_persona_generator_instruction(product_info),
            server_names=["filesystem", "g-search", "fetch"],
        )

        # 2. Persona Analyzer Agent (각 페르소나를 독립적으로 분석)
        agents['persona_analyzer'] = Agent(
            name="persona_analyzer",
            instruction=self._create_persona_analyzer_instruction(product_info),
            server_names=["filesystem", "g-search", "fetch"],
        )

        # 3. Sales Scenario Generator Agent
        agents['sales_scenario_generator'] = Agent(
            name="sales_scenario_generator",
            instruction=self._create_sales_scenario_instruction(product_info),
            server_names=["filesystem", "g-search", "fetch"],
        )

        # 4. Scenario Optimizer Agent
        agents['scenario_optimizer'] = Agent(
            name="scenario_optimizer",
            instruction=self._create_scenario_optimizer_instruction(product_info),
            server_names=["filesystem"],
        )

        # 5. Quality Evaluator Agent
        agents['quality_evaluator'] = Agent(
            name="white_hacking_quality_evaluator",
            instruction=self._create_quality_evaluator_instruction(),
        )

        # 6. Quality Controller (EvaluatorOptimizerLLM)
        agents['quality_controller'] =         evaluator_llm_factory = create_fallback_orchestrator_llm_factory(
            primary_model="gemini-2.5-flash-lite",
            logger_instance=logger
        )
        EvaluatorOptimizerLLM(
            optimizer=agents['persona_generator'],
            evaluator=agents['quality_evaluator'],
            llm_factory=evaluator_llm_factory,
            min_rating=QualityRating.GOOD,
        )

        return agents

    def _create_persona_generator_instruction(
        self,
        product_info: Dict[str, Any]
    ) -> str:
        """
        Persona Generator Agent instruction 동적 생성

        Args:
            product_info: 상품 정보

        Returns:
            str: agent instruction
        """
        # Fallback 없음: agent가 동적으로 판단하여 필요한 정보를 추론하거나 요청
        instruction = f"""You are an expert customer persona analyst specializing in creating comprehensive customer personas for business products.

Analyze the following product and generate diverse customer personas:

Product Information:
{self._format_product_info_for_instruction(product_info)}

Your task is to:

1. Research and analyze the product thoroughly:
   - Use web search to understand market trends and customer reviews
   - Analyze competitor products and positioning
   - Identify market segments and customer types
   - Research pricing strategies and value propositions

2. Generate 3-5 distinct customer personas, each representing a different customer archetype:
   For each persona, provide:
   - Persona Name and Tagline
   - Demographics (age, gender, location, income, education, occupation)
   - Psychographics (values, interests, lifestyle, personality traits)
   - Goals and Motivations (what they want to achieve)
   - Pain Points and Challenges (what problems they face)
   - Buying Behavior (how they research, evaluate, and purchase)
   - Preferred Communication Channels
   - Decision-Making Process (who influences, what criteria, timeline)
   - Budget and Price Sensitivity
   - Technology Adoption Level
   - Current Solutions (what they use now)
   - Objections and Concerns (why they might not buy)
   - Success Criteria (what would make them happy)

3. Ensure personas are:
   - Realistic and based on actual market research
   - Diverse (different demographics, needs, and behaviors)
   - Specific and detailed (not generic)
   - Actionable for sales and marketing teams
   - Based on real customer data and insights when available

4. For each persona, identify:
   - Primary use cases for the product
   - Key value propositions that resonate
   - Potential objections and how to address them
   - Best sales approach and messaging

Use MCP tools (g-search, fetch) to gather real market data, customer reviews, and competitor information.
Do not use dummy or mock data. Base all personas on actual research and analysis.

Output format: Structured JSON with persona details, or detailed markdown format.
"""
        return instruction

    def _create_persona_analyzer_instruction(
        self,
        product_info: Dict[str, Any]
    ) -> str:
        """
        Persona Analyzer Agent instruction 동적 생성

        각 페르소나를 독립적으로 심층 분석합니다.

        Args:
            product_info: 상품 정보

        Returns:
            str: agent instruction
        """
        # Fallback 없음: agent가 동적으로 판단
        instruction = f"""You are an expert customer behavior analyst specializing in deep persona analysis.

Product Information:
{self._format_product_info_for_instruction(product_info)}

Your task is to perform comprehensive, independent analysis of each customer persona generated by the persona_generator agent.

For each persona, conduct an independent deep-dive analysis:

1. Needs Analysis:
   - Primary needs and requirements
   - Secondary needs and nice-to-haves
   - Unstated needs and latent desires
   - Need prioritization and hierarchy
   - Need evolution over time
   - Need intensity and urgency

2. Pain Point Deep Dive:
   - Current pain points and frustrations
   - Root causes of pain points
   - Impact of pain points on business/personal life
   - Cost of inaction (what happens if pain persists)
   - Emotional vs. rational pain points
   - Pain point severity and frequency

3. Purchase Motivation Analysis:
   - Primary purchase drivers
   - Emotional triggers and motivators
   - Rational decision factors
   - Urgency drivers and timing factors
   - Risk factors and concerns
   - Success criteria and expectations

4. Decision-Making Process:
   - Decision timeline and urgency
   - Key stakeholders and influencers
   - Decision criteria and evaluation framework
   - Information sources and research behavior
   - Comparison and evaluation process
   - Approval and procurement process

5. Objection Analysis:
   - Likely objections and concerns
   - Objection root causes
   - Objection severity and impact
   - Objection timing (when they arise)
   - Objection handling strategies
   - Risk mitigation approaches

6. Value Perception:
   - What they value most
   - Value drivers and priorities
   - ROI expectations and calculations
   - Price sensitivity and budget constraints
   - Value comparison framework
   - Willingness to pay analysis

7. Communication Preferences:
   - Preferred communication channels
   - Message tone and style preferences
   - Content format preferences
   - Information consumption patterns
   - Engagement frequency preferences
   - Response time expectations

8. Success Factors:
   - What success looks like for this persona
   - Key success metrics and KPIs
   - Success timeline and milestones
   - Required support and resources
   - Change management needs
   - Adoption and usage patterns

For each persona, provide:
- Independent analysis (do not copy from persona generator)
- Deep insights beyond surface-level information
- Actionable recommendations
- Specific examples and scenarios
- Risk assessments
- Opportunity identification

Use MCP tools (g-search, fetch) to research similar personas, customer reviews, and market insights.
Base all analysis on real data and research, not assumptions.

Ensure each persona analysis is:
- Comprehensive and detailed
- Independent and unique
- Actionable for sales teams
- Based on real research and data
- Free of generic or templated content

Output format: Detailed markdown with clear sections for each persona's analysis.
"""
        return instruction

    def _create_sales_scenario_instruction(
        self,
        product_info: Dict[str, Any]
    ) -> str:
        """
        Sales Scenario Generator Agent instruction 동적 생성

        Args:
            product_info: 상품 정보

        Returns:
            str: agent instruction
        """
        # Fallback 없음: agent가 동적으로 판단
        instruction = f"""You are an expert sales strategist specializing in creating optimized sales scenarios for different customer personas.

Product Information:
{self._format_product_info_for_instruction(product_info)}

Your task is to create comprehensive sales scenarios for each customer persona identified by the persona_generator agent.

For each persona, develop a complete sales scenario including:

1. Sales Approach Strategy:
   - Initial contact method and timing
   - Opening conversation framework
   - Discovery questions tailored to persona
   - Value proposition messaging
   - Objection handling strategies
   - Closing techniques

2. Communication Plan:
   - Preferred communication channels
   - Message tone and style
   - Content types and formats
   - Follow-up sequence and timing
   - Multi-touch engagement strategy

3. Value Proposition:
   - Primary value drivers for this persona
   - ROI and business case framework
   - Proof points and social proof
   - Success stories and case studies
   - Competitive differentiation

4. Sales Process:
   - Stage-by-stage sales process
   - Key milestones and decision points
   - Required resources and materials
   - Stakeholder engagement strategy
   - Timeline and urgency factors

5. Objection Handling:
   - Common objections for this persona
   - Response frameworks and scripts
   - Proof points and evidence
   - Alternative solutions and compromises

6. Closing Strategy:
   - Best closing techniques for this persona
   - Incentives and urgency drivers
   - Risk mitigation strategies
   - Next steps and commitment actions

7. Post-Sale Success:
   - Onboarding and implementation plan
   - Success metrics and KPIs
   - Ongoing engagement strategy
   - Expansion and upsell opportunities

Ensure each scenario is:
- Tailored specifically to the persona's needs and preferences
- Based on proven sales methodologies
- Actionable and detailed
- Realistic and achievable
- Measurable with clear success criteria

Use the persona analysis results from persona_generator agent to create highly personalized scenarios.
Do not create generic sales scripts. Each scenario must be unique to its persona.

Output format: Detailed markdown or structured format with clear sections for each persona.
"""
        return instruction

    def _create_scenario_optimizer_instruction(
        self,
        product_info: Dict[str, Any]
    ) -> str:
        """
        Scenario Optimizer Agent instruction 동적 생성

        Args:
            product_info: 상품 정보

        Returns:
            str: agent instruction
        """
        # Fallback 없음: agent가 동적으로 판단
        instruction = f"""You are a sales optimization expert specializing in validating and optimizing sales scenarios.

Product Information:
{self._format_product_info_for_instruction(product_info)}

Your task is to review, validate, and optimize the sales scenarios created by the sales_scenario_generator agent.

For each sales scenario, perform:

1. Validation:
   - Check alignment with persona characteristics
   - Verify realism and achievability
   - Assess completeness and detail level
   - Evaluate sales methodology soundness
   - Review objection handling effectiveness

2. Optimization:
   - Identify gaps and missing elements
   - Suggest improvements to messaging
   - Recommend better sales techniques
   - Propose additional value propositions
   - Enhance objection handling strategies
   - Optimize timing and sequencing

3. Risk Assessment:
   - Identify potential failure points
   - Assess objection handling weaknesses
   - Evaluate competitive vulnerabilities
   - Review implementation challenges

4. Enhancement Recommendations:
   - Additional resources needed
   - Training requirements
   - Tool and technology needs
   - Process improvements
   - Measurement and tracking enhancements

5. Success Probability:
   - Estimate success likelihood for each scenario
   - Identify highest-probability approaches
   - Recommend prioritization strategy
   - Suggest A/B testing opportunities

Output a comprehensive optimization report with:
- Validation results for each scenario
- Specific optimization recommendations
- Risk mitigation strategies
- Enhanced scenario versions
- Success probability assessments
- Implementation priorities

Ensure all recommendations are:
- Actionable and specific
- Based on sales best practices
- Tailored to the specific persona
- Measurable and trackable
- Realistic and achievable

Output format: Structured markdown with clear sections for validation, optimization, and recommendations.
"""
        return instruction

    def _create_quality_evaluator_instruction(self) -> str:
        """
        Quality Evaluator Agent instruction 생성

        Returns:
            str: agent instruction
        """
        instruction = """You are a quality assurance expert evaluating customer white hacking deliverables.

Evaluate the customer white hacking analysis based on:

1. Persona Quality (30%):
   - Realism and authenticity of personas
   - Depth and detail of persona profiles
   - Diversity and representativeness
   - Research-based insights
   - Actionability for sales teams

2. Sales Scenario Quality (30%):
   - Alignment with persona characteristics
   - Completeness and detail level
   - Sales methodology soundness
   - Personalization and customization
   - Actionability and implementability

3. Analysis Depth (20%):
   - Market research quality
   - Competitive analysis depth
   - Customer insight richness
   - Pain point identification accuracy
   - Value proposition clarity

4. Business Value (20%):
   - Practical applicability
   - Sales effectiveness potential
   - ROI and impact potential
   - Implementation feasibility
   - Measurability and tracking

Provide EXCELLENT, GOOD, FAIR, or POOR ratings with specific improvement recommendations.
Focus on actionable feedback that improves sales effectiveness.
"""
        return instruction

    def _format_product_info_for_instruction(self, product_info: Dict[str, Any]) -> str:
        """
        상품 정보를 instruction용 포맷으로 변환
        Fallback 없음: agent가 동적으로 판단하여 필요한 정보를 추론하거나 요청

        Args:
            product_info: 상품 정보 딕셔너리

        Returns:
            str: 포맷된 상품 정보 문자열
        """
        lines = []

        # 모든 필드를 agent가 동적으로 판단하도록 제공
        # 필드가 없으면 agent가 추론하거나 요청하도록 함
        if 'name' in product_info:
            lines.append(f"- Name: {product_info['name']}")
        if 'description' in product_info:
            lines.append(f"- Description: {product_info['description']}")
        if 'category' in product_info:
            lines.append(f"- Category: {product_info['category']}")
        if 'target_market' in product_info:
            lines.append(f"- Target Market: {product_info['target_market']}")
        if 'price_range' in product_info:
            lines.append(f"- Price Range: {product_info['price_range']}")
        if 'features' in product_info:
            features = product_info['features']
            if isinstance(features, list):
                lines.append(f"- Key Features: {', '.join(str(f) for f in features)}")
            elif features:
                lines.append(f"- Key Features: {features}")
        if 'competitors' in product_info:
            competitors = product_info['competitors']
            if isinstance(competitors, list):
                lines.append(f"- Competitors: {', '.join(str(c) for c in competitors)}")
            elif competitors:
                lines.append(f"- Competitors: {competitors}")

        # 추가 필드가 있으면 모두 포함
        for key, value in product_info.items():
            if key not in ['name', 'description', 'category', 'target_market', 'price_range', 'features', 'competitors']:
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        # Fallback 없음: agent가 동적으로 판단
        # 정보가 없으면 agent가 MCP 도구를 사용하여 조사하도록 instruction에 명시
        # 빈 리스트를 join하면 빈 문자열이 반환되므로 fallback 불필요
        return "\n".join(lines)

    def _create_task(
        self,
        product_info: Dict[str, Any],
        timestamp: str,
        save_to_file: bool
    ) -> str:
        """
        워크플로우 task 생성

        Args:
            product_info: 상품 정보
            timestamp: 타임스탬프
            save_to_file: 파일 저장 여부

        Returns:
            str: task description
        """
        # Fallback 없음: agent가 동적으로 판단
        task = f"""Execute a comprehensive customer white hacking analysis for the following product:

{self._format_product_info_for_instruction(product_info)}

Workflow Steps:

1. Use the quality_controller (persona_generator + quality_evaluator) to generate high-quality customer personas:
   - Research the product and market using MCP tools (g-search, fetch)
   - Analyze competitors and customer reviews
   - Generate 3-5 diverse, realistic customer personas
   - Each persona must be detailed, specific, and based on real research
   - Ensure personas represent different customer archetypes and segments

2. Use the persona_analyzer to perform independent deep-dive analysis for each generated persona:
   - Conduct comprehensive needs analysis for each persona
   - Deep-dive into pain points and root causes
   - Analyze purchase motivations and decision drivers
   - Map decision-making processes and timelines
   - Identify likely objections and concerns
   - Assess value perception and price sensitivity
   - Determine communication preferences and engagement patterns
   - Define success factors and metrics
   - Each persona analysis must be independent and comprehensive
   - Use MCP tools to research and validate insights

3. Use the sales_scenario_generator to create optimized sales scenarios based on persona analysis:
   - Develop a complete sales strategy for each persona
   - Create personalized communication plans
   - Design value propositions tailored to each persona
   - Develop objection handling frameworks
   - Create closing strategies and post-sale success plans
   - Ensure each scenario is unique and persona-specific

4. Use the scenario_optimizer to validate and optimize:
   - Review all sales scenarios for quality and effectiveness
   - Validate alignment with persona characteristics
   - Identify optimization opportunities
   - Assess risks and mitigation strategies
   - Provide enhancement recommendations
   - Estimate success probabilities

5. Generate comprehensive deliverables:
   - Executive summary of personas and scenarios
   - Detailed persona profiles (one per persona)
   - Sales scenario playbooks (one per persona)
   - Optimization recommendations
   - Implementation roadmap
   - Success metrics and KPIs

All analysis must be:
- Based on real market research (use MCP tools)
- Free of dummy or mock data
- Actionable and implementable
- Measurable and trackable
- Production-ready quality

"""

        if save_to_file:
            task += f"""
Save all deliverables in the {self.output_dir} directory with appropriate naming:
- customer_personas_{timestamp}.md
- sales_scenarios_{timestamp}.md
- optimization_report_{timestamp}.md
- executive_summary_{timestamp}.md
- implementation_roadmap_{timestamp}.md
"""
        else:
            task += """
Return the complete analysis for immediate display. Do not save to files.
Provide comprehensive, detailed results including all personas, scenarios, and recommendations.
"""

        return task


async def main():
    """
    Customer White Hacking Agent 실행 예제
    """
    # 예제 상품 정보
    product_info = {
        'name': 'AI-Powered Project Management Tool',
        'description': 'An intelligent project management platform with AI-driven task prioritization, automated resource allocation, and predictive analytics for team productivity.',
        'category': 'SaaS / Project Management',
        'target_market': 'Mid to large enterprises, remote teams, agile organizations',
        'price_range': '$50-$200 per user/month',
        'features': [
            'AI task prioritization',
            'Automated resource allocation',
            'Predictive analytics',
            'Real-time collaboration',
            'Integration with popular tools'
        ],
        'competitors': ['Asana', 'Monday.com', 'Jira', 'ClickUp']
    }

    agent = CustomerWhiteHackingAgent()
    result = agent.run_white_hacking_workflow(
        product_info=product_info,
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

