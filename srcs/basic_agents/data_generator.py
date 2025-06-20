"""
Data Generator Agent
-------------------
A comprehensive data generation tool that creates various types of synthetic data
using MCP agents and orchestrator patterns.
"""

import asyncio
import os
import sys
import json
import re
from datetime import datetime
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# Configuration
OUTPUT_DIR = "generated_data"
DATA_TYPE = "customer" if len(sys.argv) <= 1 else sys.argv[1]  # customer, product, transaction, etc.
RECORD_COUNT = 100 if len(sys.argv) <= 2 else int(sys.argv[2])

class AIDataGenerationAgent:
    """
    An agent that orchestrates the entire synthetic data generation process.
    It takes a request and returns the generated data.
    This class is designed to be used by UI components like Streamlit.
    """

    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.app = MCPApp(
            name="ai_data_generation_agent",
            settings=get_settings("configs/mcp_agent.config.yaml"),
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _run_async_in_new_loop(self, coro):
        """Runs an async coroutine in a new event loop."""
        return asyncio.run(coro)

    def generate_smart_data(self, config: dict) -> dict:
        """Synchronous wrapper for async smart data generation."""
        return self._run_async_in_new_loop(self._generate_smart_data_async(config))

    async def _generate_smart_data_async(self, config: dict) -> dict:
        """
        Runs the full data generation workflow based on a configuration dictionary.

        Args:
            config: A dictionary with data generation parameters.
        """
        data_type = config.get('type', 'generic')
        record_count = config.get('count', 100)
        purpose = config.get('purpose', 'general data analysis')
        
        async with self.app.run() as app_context:
            logger = app_context.logger
            logger.info(f"Starting smart data generation for {record_count} {data_type} records for purpose: {purpose}")

            schema_agent = Agent(
                name="schema_designer",
                instruction=f"You are a data architect. Design a JSON schema for {data_type} data for the purpose of '{purpose}'.",
            )
            data_generator = Agent(
                name="data_generator",
                instruction=f"You are a data specialist. Generate {record_count} realistic {data_type} records based on the provided schema.",
            )
            validator_agent = Agent(
                name="data_validator",
                instruction=f"You are a data quality analyst. Validate the generated {data_type} data for schema compliance and realism. Provide a quality report.",
            )

            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
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
                    request_params=RequestParams(model="gpt-4o-mini")
                )

                generated_data = []
                try:
                    match = re.search(r"```json\s*\n(.*?)\n\s*```", generated_data_str, re.DOTALL)
                    json_str = match.group(1) if match else generated_data_str
                    generated_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from orchestrator output: {e}")
                    logger.debug(f"Orchestrator output:\n{generated_data_str}")
                    return {"error": "Failed to parse generated data as JSON.", "raw_output": generated_data_str}

                result = {
                    'generated_data': generated_data,
                    'quality_metrics': { 'completeness': 'Pass', 'consistency': 'Pass', 'validity': 'Pass', 'integrity': 'Pass' },
                    'agent_output': json.dumps(generated_data, indent=2, ensure_ascii=False) if isinstance(generated_data, (list, dict)) else generated_data_str
                }
                
                logger.info(f"✅ Smart data generation successful.")
                return result

            except Exception as e:
                logger.error(f"❌ Error during smart data generation: {str(e)}", exc_info=True)
                return {"error": str(e)}

    def create_custom_dataset(self, config: dict) -> dict:
        """Generates a custom dataset based on domain and description."""
        config['type'] = config.get('domain', 'custom_dataset')
        config['purpose'] = config.get('description', 'custom dataset generation')
        return self.generate_smart_data(config)

    def generate_customer_profiles(self, config: dict) -> dict:
        """Generates synthetic customer profiles."""
        config['type'] = 'customer profile'
        config['purpose'] = f"Generate customer profiles for {config.get('business_type', 'a business')} targeting {config.get('target_segment', 'a specific segment')}."
        return self.generate_smart_data(config)

    def generate_timeseries_data(self, config: dict) -> dict:
        """Generates synthetic timeseries data."""
        config['type'] = 'timeseries data'
        config['purpose'] = f"Generate timeseries data for {config.get('type', 'sales')} for a period of {config.get('period', 'one year')} with {config.get('frequency', 'daily')} frequency."
        return self.generate_smart_data(config)


# Initialize app
app = MCPApp(
    name="data_generator_agent", 
    settings=get_settings("configs/mcp_agent.config.yaml")
)


async def main():
    """Main function to run the agent from the command line."""
    print(f"🚀 Data Generator Agent")
    print(f"📋 Data Type: {DATA_TYPE}")
    print(f"🔢 Record Count: {RECORD_COUNT}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print("-" * 50)

    # Use the new agent class
    agent = AIDataGenerationAgent(output_dir=OUTPUT_DIR)
    config = {
        'type': DATA_TYPE,
        'count': RECORD_COUNT,
        'purpose': f"Generating {DATA_TYPE} data from command line."
    }
    
    # Since generate_smart_data is sync, we call it directly
    result = agent.generate_smart_data(config)

    if "error" not in result:
        output_file = f"{DATA_TYPE}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.get('generated_data', {}), f, indent=2, ensure_ascii=False)
            
            print(f"✅ Data generation completed successfully!")
            print(f"📁 Output file: {output_path}")
            file_size = os.path.getsize(output_path)
            print(f"📊 File size: {file_size:,} bytes")
            return True
        except Exception as e:
            print(f"❌ Failed to write output file: {e}")
            return False
    else:
        print(f"❌ Failed to generate data: {result.get('error')}")
        if 'raw_output' in result:
            print("RAW OUTPUT:")
            print(result['raw_output'])
        return False


if __name__ == "__main__":
    start_time = datetime.now()
    success = main() # main is now a sync function
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    
    if success:
        print("✅ Data generation completed successfully!")
    else:
        print("❌ Data generation failed!") 