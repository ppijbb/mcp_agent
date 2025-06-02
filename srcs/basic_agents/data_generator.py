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

# Initialize app
app = MCPApp(
    name="data_generator_agent", 
    settings=get_settings("configs/mcp_agent.config.yaml")
)


async def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{DATA_TYPE}_data_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_file)

    async with app.run() as generator_app:
        context = generator_app.context
        logger = generator_app.logger

        logger.info(f"Starting data generation for {RECORD_COUNT} {DATA_TYPE} records")

        # Configure filesystem server
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")

        # --- DEFINE AGENTS ---

        # Schema Designer: Defines the data structure
        schema_agent = Agent(
            name="schema_designer",
            instruction=f"""You are an expert data architect. Design a comprehensive JSON schema for {DATA_TYPE} data.
            
            Create a detailed schema that includes:
            1. All relevant fields for {DATA_TYPE} entities
            2. Appropriate data types (string, number, boolean, array, object)
            3. Realistic constraints and formats
            4. Nested objects where appropriate
            
            For example, if designing customer data, include:
            - Personal info (name, email, phone, address)
            - Demographics (age, gender, occupation)
            - Preferences and behavior data
            - Account information
            
            Return the schema as a clean JSON structure with field descriptions.""",
            server_names=["filesystem"],
        )

        # Data Generator: Creates synthetic data based on schema
        data_generator = Agent(
            name="data_generator",
            instruction=f"""You are a synthetic data generation specialist. 
            
            Generate {RECORD_COUNT} realistic {DATA_TYPE} records based on the provided schema.
            
            Requirements:
            1. Create diverse, realistic data that follows real-world patterns
            2. Ensure data consistency (e.g., zip codes match cities)
            3. Include edge cases and variations
            4. Use realistic names, addresses, and other personal data
            5. Make sure all required fields are populated
            6. Generate data in valid JSON format
            
            Focus on creating high-quality, realistic synthetic data that could be used for testing or development.""",
            server_names=["filesystem"],
        )

        # Data Validator: Checks data quality and consistency
        validator_agent = Agent(
            name="data_validator",
            instruction=f"""You are a data quality specialist. Validate the generated {DATA_TYPE} data.
            
            Check for:
            1. JSON format validity
            2. Schema compliance (all required fields present)
            3. Data consistency (e.g., email formats, phone numbers)
            4. Realistic value ranges
            5. No duplicate records (unless intentional)
            6. Proper data types
            
            If issues are found, provide specific feedback on what needs to be fixed.
            Rate the data quality as EXCELLENT, GOOD, FAIR, or POOR with detailed reasoning.""",
            server_names=["filesystem"],
        )

        # File Writer: Saves the final data
        file_writer = Agent(
            name="file_writer",
            instruction=f"""Save the validated {DATA_TYPE} data to the specified file.
            
            Format the data as clean, properly indented JSON.
            Include metadata at the top with:
            - Generation timestamp
            - Data type
            - Record count
            - Schema version
            
            Save to: {output_path}""",
            server_names=["filesystem"],
        )

        # --- SIMPLE WORKFLOW (like basic.py) ---
        logger.info("=== Starting Simple Data Generation Workflow ===")
        
        async with schema_agent:
            logger.info("Step 1: Designing data schema...")
            llm = await schema_agent.attach_llm(OpenAIAugmentedLLM)
            schema_result = await llm.generate_str(
                message=f"Design a comprehensive JSON schema for {DATA_TYPE} data with at least 8-10 relevant fields."
            )
            logger.info(f"Schema designed: {schema_result[:200]}...")

        async with data_generator:
            logger.info("Step 2: Generating synthetic data...")
            llm = await data_generator.attach_llm(GoogleAugmentedLLM)
            data_result = await llm.generate_str(
                message=f"Generate {RECORD_COUNT} {DATA_TYPE} records using this schema:\n{schema_result}"
            )
            logger.info(f"Data generated: {len(data_result)} characters")

        async with validator_agent:
            logger.info("Step 3: Validating data quality...")
            llm = await validator_agent.attach_llm(OpenAIAugmentedLLM)
            validation_result = await llm.generate_str(
                message=f"Validate this {DATA_TYPE} data:\n{data_result[:1000]}..."
            )
            logger.info(f"Validation result: {validation_result}")

        async with file_writer:
            logger.info("Step 4: Saving data to file...")
            llm = await file_writer.attach_llm(OpenAIAugmentedLLM)
            save_result = await llm.generate_str(
                message=f"Save this validated {DATA_TYPE} data to {output_path}:\n{data_result}"
            )
            logger.info(f"File save result: {save_result}")

        # --- ORCHESTRATED WORKFLOW (like agent.py) ---
        logger.info("\n=== Starting Orchestrated Data Generation Workflow ===")
        
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                schema_agent,
                data_generator,
                validator_agent,
                file_writer,
            ],
            plan_type="full",
        )

        orchestrated_task = f"""Create high-quality synthetic {DATA_TYPE} data by following these steps:

        1. Use schema_designer to create a comprehensive JSON schema for {DATA_TYPE} entities
        2. Use data_generator to create {RECORD_COUNT} realistic {DATA_TYPE} records
        3. Use data_validator to check data quality and consistency
        4. Use file_writer to save the final data to: {output_path}

        The final output should be production-ready synthetic data that follows real-world patterns."""

        try:
            await orchestrator.generate_str(
                message=orchestrated_task,
                request_params=RequestParams(model="gpt-4o-mini")
            )

            # Check if file was created
            if os.path.exists(output_path):
                logger.info(f"✅ Data generation completed successfully!")
                logger.info(f"📁 Output file: {output_path}")
                
                # Show file size
                file_size = os.path.getsize(output_path)
                logger.info(f"📊 File size: {file_size:,} bytes")
                
                return True
            else:
                logger.error(f"❌ Failed to create output file: {output_path}")
                return False

        except Exception as e:
            logger.error(f"❌ Error during data generation: {str(e)}")
            return False


if __name__ == "__main__":
    print(f"🚀 Data Generator Agent")
    print(f"📋 Data Type: {DATA_TYPE}")
    print(f"🔢 Record Count: {RECORD_COUNT}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print("-" * 50)
    
    start_time = datetime.now()
    result = asyncio.run(main())
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    
    if result:
        print("✅ Data generation completed successfully!")
    else:
        print("❌ Data generation failed!") 