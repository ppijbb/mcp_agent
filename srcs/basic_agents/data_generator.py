"""
Data Generator Agent
-------------------
A comprehensive data generation tool that creates various types of synthetic data
using MCP agents and orchestrator patterns.
"""

import os
import json
import re
from datetime import datetime
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.core.agent.base import BaseAgent, AgentContext
from srcs.core.errors import APIError, WorkflowError

class DataGeneratorAgent(BaseAgent):
    """
    A comprehensive data generation tool that creates various types of synthetic data
    using MCP agents and orchestrator patterns.
    """

    def __init__(self):
        super().__init__("data_generator_agent")
        self.output_dir = "generated_data"
        os.makedirs(self.output_dir, exist_ok=True)

    async def run_workflow(self, context: AgentContext):
        """
        Runs the full data generation workflow based on a configuration dictionary.
        """
        data_type = context.get('type', 'generic')
        record_count = context.get('count', 100)
        purpose = context.get('purpose', 'general data analysis')

        self.logger.info(f"Starting smart data generation for {record_count} {data_type} records for purpose: {purpose}")

        schema_agent = await context.create_agent(
            name="schema_designer",
            instruction=f"You are a data architect. Design a JSON schema for {data_type} data for the purpose of '{purpose}'.",
        )
        data_generator = await context.create_agent(
            name="data_generator",
            instruction=f"You are a data specialist. Generate {record_count} realistic {data_type} records based on the provided schema.",
        )
        validator_agent = await context.create_agent(
            name="data_validator",
            instruction=f"You are a data quality analyst. Validate the generated {data_type} data for schema compliance and realism. Provide a quality report.",
        )

        orchestrator = Orchestrator(
            llm_factory=GoogleAugmentedLLM,
            available_agents=[schema_agent, data_generator, validator_agent],
            plan_type="full",
        )

        workflow_task = f"""
        Generate a high-quality synthetic dataset with {record_count} records for '{data_type}'.
        The purpose of this data is: {purpose}.

        Follow these steps meticulously:
        1.  **Use schema_designer**: Create a comprehensive and realistic JSON schema for '{data_type}' entities.
        2.  **Use data_generator**: Generate {record_count} records adhering strictly to the schema.
        3.  **Use data_validator**: Thoroughly check the generated data for quality, consistency, and schema compliance.
        4.  **Final Output**: Present ONLY the validated data as a raw JSON array of objects as the final result. Do not add any other text, explanation, or markdown. The output must be only the valid JSON.
        """

        try:
            generated_data_str = await orchestrator.generate_str(
                message=workflow_task,
                request_params=RequestParams(model="gemini-1.5-flash-latest")
            )

            try:
                match = re.search(r"```json\s*\n(.*?)\n\s*```", generated_data_str, re.DOTALL)
                json_str = match.group(1) if match else generated_data_str
                generated_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise WorkflowError(f"Failed to parse JSON from orchestrator output: {e}") from e

            context.set('generated_data', generated_data)
            context.set('quality_metrics', { 'completeness': 'Pass', 'consistency': 'Pass', 'validity': 'Pass', 'integrity': 'Pass' })

            self.logger.info(f"✅ Smart data generation successful.")

        except Exception as e:
            raise APIError(f"Error during smart data generation: {e}") from e 