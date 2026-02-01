"""
Agent Templates Module

Base templates and patterns for creating new agents with standardized structure.
"""

from abc import ABC, abstractmethod

# Defer imports to avoid circular dependencies


def _get_orchestrator_imports():
    try:
        from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator, QualityRating
        from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import EvaluatorOptimizerLLM
        from mcp_agent.workflows.llm.augmented_llm import RequestParams
        from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
        return Orchestrator, QualityRating, EvaluatorOptimizerLLM, RequestParams, OpenAIAugmentedLLM
    except ImportError:
        return None, None, None, None, None


from .config import *
from .utils import *


def _get_llm_imports():
    try:
        from .llm import create_fallback_llm_factory, try_fallback_orchestrator_execution
        return create_fallback_llm_factory, try_fallback_orchestrator_execution
    except ImportError:
        return None, None


class AgentTemplate(ABC):
    """Base template for all agents"""

    def __init__(self, agent_name, company_name=None, custom_scope=None):
        self.agent_name = agent_name
        self.company_name = company_name or "Company"
        self.custom_scope = custom_scope
        self.output_dir = get_output_dir("agent", agent_name)
        self.timestamp = get_timestamp()

    @abstractmethod
    def create_agents(self):
        """Create and return list of specialized agents"""

    @abstractmethod
    def create_evaluator(self):
        """Create and return quality evaluator agent"""

    @abstractmethod
    def define_task(self):
        """Define the main task for orchestrator execution"""

    def setup_app(self):
        """Setup MCP application"""
        return setup_agent_app(f"{self.agent_name}_system")

    def create_orchestrator(self, agents, evaluator):
        """Create orchestrator with agents and evaluator (with fallback support)"""
        # Fallback이 가능한 LLM factory 사용 (common 모듈)
        fallback_llm_factory = create_fallback_llm_factory(
            primary_model="gemini-2.5-flash-lite"
        )

        quality_controller = EvaluatorOptimizerLLM(
            optimizer=agents[0],  # Use first agent as optimizer
            evaluator=evaluator,
            llm_factory=fallback_llm_factory,
            min_rating=QualityRating.GOOD,
        )

        return Orchestrator(
            llm_factory=fallback_llm_factory,
            available_agents=[quality_controller] + agents[1:],
            plan_type="full",
        )

    async def run(self):
        """Main execution method"""
        ensure_output_directory(self.output_dir)

        async with self.setup_app().run() as app:
            context = app.context
            logger = app.logger

            configure_filesystem_server(context, logger)

            # Create agents and orchestrator
            agents = self.create_agents()
            evaluator = self.create_evaluator()
            orchestrator = self.create_orchestrator(agents, evaluator)

            # Execute task
            logger.info(f"Starting {self.agent_name} workflow")
            task = self.define_task()

            try:
                # Fallback 지원 orchestrator 실행 (common 모듈)
                result = await try_fallback_orchestrator_execution(
                    orchestrator=orchestrator,
                    agents=orchestrator.available_agents,
                    task=task,
                    primary_model="gemini-2.5-flash-lite",
                    logger_instance=logger,
                    max_loops=30
                )

                logger.info(f"{self.agent_name} workflow completed successfully")
                logger.info(f"All deliverables saved in {self.output_dir}/")

                # Create summary and KPIs
                self.create_summary()
                self.create_kpis()

                return True

            except Exception as e:
                logger.error(f"Error during {self.agent_name} workflow execution: {str(e)}")
                return False

    def create_summary(self):
        """Create executive summary - to be overridden by subclasses"""

    def create_kpis(self):
        """Create KPI template - to be overridden by subclasses"""


class EnterpriseAgentTemplate(AgentTemplate):
    """Template specifically for enterprise-level agents"""

    def __init__(self, agent_name, company_name=None, business_scope=None):
        super().__init__(agent_name, company_name, business_scope)
        self.business_scope = business_scope or "Global Operations"

    def create_quality_controller(self, optimizer_agent, evaluator_agent):
        """Create enterprise-grade quality controller (with fallback support)"""
        # Fallback이 가능한 LLM factory 사용 (common 모듈)
        fallback_llm_factory = create_fallback_llm_factory(
            primary_model="gemini-2.5-flash-lite"
        )

        return EvaluatorOptimizerLLM(
            optimizer=optimizer_agent,
            evaluator=evaluator_agent,
            llm_factory=fallback_llm_factory,
            min_rating=QualityRating.GOOD,
        )

    def create_standard_evaluator(self, evaluation_criteria):
        """Create standardized enterprise evaluator"""
        criteria_text = self._format_evaluation_criteria(evaluation_criteria)

        return Agent(
            name=f"{self.agent_name}_quality_evaluator",
            instruction=f"""You are a {self.agent_name} expert evaluating enterprise initiatives.

            Evaluate programs based on:

            {criteria_text}

            Provide EXCELLENT, GOOD, FAIR, or POOR ratings with specific improvement recommendations.
            Highlight critical success factors and implementation challenges.
            """,
        )

    def _format_evaluation_criteria(self, criteria):
        """Format evaluation criteria for evaluator instruction"""
        formatted = []
        for i, (category, weight, description) in enumerate(criteria, 1):
            formatted.append(f"{i}. {category} ({weight}%)")
            formatted.append(f"   - {description}")
            formatted.append("")
        return "\n".join(formatted)

    def create_enterprise_summary(self, summary_data):
        """Create enterprise-specific executive summary"""
        return create_executive_summary(
            output_dir=self.output_dir,
            agent_name=self.agent_name,
            company_name=self.company_name,
            timestamp=self.timestamp,
            **summary_data
        )

    def create_enterprise_kpis(self, kpi_structure):
        """Create enterprise-specific KPI template"""
        return create_kpi_template(
            output_dir=self.output_dir,
            agent_name=self.agent_name,
            kpi_structure=kpi_structure,
            timestamp=self.timestamp
        )


class BasicAgentTemplate(AgentTemplate):
    """Template for basic/simple agents"""

    def __init__(self, agent_name, task_description):
        super().__init__(agent_name)
        self.task_description = task_description

    def create_agents(self):
        """Create simple agent list"""
        return [
            Agent(
                name=self.agent_name,
                instruction=self.task_description,
                server_names=DEFAULT_SERVERS,
            )
        ]

    def create_evaluator(self):
        """Create basic evaluator"""
        return Agent(
            name=f"{self.agent_name}_evaluator",
            instruction=f"""Evaluate the quality and effectiveness of {self.agent_name} results.

            Rate as EXCELLENT, GOOD, FAIR, or POOR based on:
            - Accuracy and completeness
            - Clarity and usefulness
            - Actionability of recommendations

            Provide specific feedback for improvement.
            """,
        )

    def define_task(self):
        """Define simple task"""
        return f"Execute {self.agent_name} task: {self.task_description}"


__all__ = [
    "AgentTemplate", "EnterpriseAgentTemplate", "BasicAgentTemplate"
]
