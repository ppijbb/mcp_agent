"""
Enhanced Data Generator Agent with Meta Synthetic Data Kit Integration
--------------------------------------------------------------------
A comprehensive data generation tool that combines MCP agents with Meta's 
Synthetic Data Kit for high-quality synthetic dataset creation.
"""

import asyncio
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# Configuration
OUTPUT_DIR = "synthetic_datasets"
DATA_TYPE = "qa" if len(sys.argv) <= 1 else sys.argv[1]  # qa, cot, summary, custom
RECORD_COUNT = 100 if len(sys.argv) <= 2 else int(sys.argv[2])
SOURCE_FILE = None if len(sys.argv) <= 3 else sys.argv[3]  # Optional source document

# Initialize app
app = MCPApp(
    name="enhanced_data_generator",
    settings=get_settings("configs/mcp_agent.config.yaml")
)


class SyntheticDataKitIntegration:
    """Integration wrapper for Meta's Synthetic Data Kit"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.setup_directories()
    
    def setup_directories(self):
        """Create the directory structure expected by synthetic-data-kit"""
        dirs = [
            "pdf", "html", "youtube", "docx", "ppt", "txt", 
            "output", "generated", "cleaned", "final"
        ]
        for dir_name in dirs:
            (self.data_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def check_installation(self):
        """Check if synthetic-data-kit is installed"""
        try:
            result = subprocess.run(
                ["synthetic-data-kit", "--help"], 
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_kit(self):
        """Install synthetic-data-kit via pip"""
        try:
            subprocess.run(
                ["pip", "install", "synthetic-data-kit"], 
                check=True, capture_output=True, text=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def system_check(self):
        """Run system check for synthetic-data-kit"""
        try:
            result = subprocess.run(
                ["synthetic-data-kit", "system-check"],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during system check"
    
    def ingest_document(self, file_path: str):
        """Ingest a document using synthetic-data-kit"""
        try:
            result = subprocess.run(
                ["synthetic-data-kit", "ingest", file_path],
                capture_output=True, text=True, timeout=60
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during document ingestion"
    
    def create_dataset(self, input_file: str, data_type: str, num_pairs: int):
        """Create dataset using synthetic-data-kit"""
        try:
            cmd = [
                "synthetic-data-kit", "create", input_file,
                "--type", data_type,
                "-n", str(num_pairs)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during dataset creation"
    
    def curate_dataset(self, input_file: str, threshold: float = 7.0):
        """Curate dataset using synthetic-data-kit"""
        try:
            cmd = [
                "synthetic-data-kit", "curate", input_file,
                "-t", str(threshold)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during dataset curation"
    
    def save_dataset(self, input_file: str, format_type: str = "alpaca", storage: str = "json"):
        """Save dataset in specified format using synthetic-data-kit"""
        try:
            cmd = [
                "synthetic-data-kit", "save-as", input_file,
                "-f", format_type,
                "--storage", storage
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during dataset saving"


class SyntheticDataAgent:
    """
    An enhanced agent that orchestrates the entire synthetic data generation process.
    It takes a simple request and returns the path to the generated data file.
    """

    def __init__(self, output_dir="generated_data"):
        self.output_dir = output_dir
        self.app = MCPApp(
            name="synthetic_data_orchestrator",
            settings=get_settings("configs/mcp_agent.config.yaml"),
        )
        os.makedirs(self.output_dir, exist_ok=True)

    async def run(self, data_type: str, record_count: int, max_refine_iters: int = 2) -> str:
        """
        Runs the full data generation workflow.

        Args:
            data_type: The type of data to generate (e.g., 'customer', 'product').
            record_count: The number of records to generate.

        Returns:
            A message indicating the result, including the path to the output file on success.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{data_type}_data_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_file)

        async with self.app.run() as app_context:
            logger = app_context.logger
            logger.info(f"Starting orchestrated data generation for {record_count} {data_type} records.")

            # Define the agents needed for the workflow
            schema_agent = Agent(
                name="schema_designer",
                instruction=f"You are a data architect. Design a JSON schema for {data_type} data.",
                server_names=[],
            )
            data_generator_agent = Agent(
                name="data_generator",
                instruction=f"You are a data specialist. Generate {record_count} realistic {data_type} records based on a schema.",
                server_names=[],
            )
            validator_agent = Agent(
                name="data_validator",
                instruction=f"You are a data quality analyst. Validate the dataset and provide a concise validation report highlighting issues.",
                server_names=[],
            )
            refiner_agent = Agent(
                name="data_refiner",
                instruction="You are a data specialist. Given a dataset and its validation report, refine the dataset by addressing issues. Output only the refined dataset in valid JSON format.",
                server_names=[],
            )

            # The orchestrator will manage the workflow
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=[schema_agent, data_generator_agent, validator_agent, refiner_agent],
                plan_type="full",
            )

            # The overall objective for the orchestrator
            workflow_task = f"""
            Generate a high-quality synthetic dataset with {record_count} records for the data type '{data_type}'.

            Follow these steps meticulously:
            1.  **Use schema_designer**: Create a comprehensive and realistic JSON schema for '{data_type}' entities. The schema must be detailed.
            2.  **Use data_generator**: Generate {record_count} records adhering strictly to the schema designed in the previous step.
            3.  **Use data_validator**: Thoroughly check the generated data for quality, consistency, and schema compliance. Provide a brief quality report.
            4.  **Use data_refiner**: Given the validation report, refine the dataset by addressing issues.
            5.  **Final Output**: Present the refined JSON data as the final result. Do not include a file writing step in the plan. The data itself is the final output.
            """

            try:
                # Initial dataset generation
                generated_data_str = await orchestrator.generate_str(
                    message=workflow_task,
                    request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07")
                )

                # --- Generator–Validator–Refiner Feedback Loop ---
                # Define validator and refiner agents
                validator_agent = Agent(
                    name="data_validator",
                    instruction=f"You are a data quality analyst. Validate the dataset and provide a concise validation report highlighting issues.",
                    server_names=[],
                )
                refiner_agent = Agent(
                    name="data_refiner",
                    instruction="You are a data specialist. Given a dataset and its validation report, refine the dataset by addressing issues. Output only the refined dataset in valid JSON format.",
                    server_names=[],
                )
                current_data = generated_data_str
                for iteration in range(max_refine_iters):
                    # Validation step
                    async with validator_agent:
                        val_llm = await validator_agent.attach_llm(OpenAIAugmentedLLM)
                        validation_report = await val_llm.generate_str(message=current_data)
                    # Refinement step
                    async with refiner_agent:
                        ref_llm = await refiner_agent.attach_llm(OpenAIAugmentedLLM)
                        current_data = await ref_llm.generate_str(
                            message=f"Dataset: {current_data}\nValidation Report: {validation_report}\nPlease refine the dataset accordingly."
                        )
                final_data = current_data

                # Save the final, refined data to the file
                with open(output_path, 'w', encoding='utf-8') as f:
                    try:
                        parsed_json = json.loads(final_data)
                        json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        f.write(final_data)
                
                file_size = os.path.getsize(output_path)
                success_message = (
                    f"✅ 데이터 생성 및 정제에 성공했습니다!\n\n"
                    f"📁 **파일 경로**: `{output_path}`\n"
                    f"📊 **파일 크기**: {file_size / 1024:.2f} KB"
                )
                logger.info(success_message)
                return success_message

            except Exception as e:
                error_message = f"❌ 데이터 생성 중 오류 발생: {e}"
                logger.error(error_message, exc_info=True)
                return error_message


async def main():
    # Create output directory and SDK integration
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sdk = SyntheticDataKitIntegration(OUTPUT_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{DATA_TYPE}_dataset_{timestamp}"
    
    async with app.run() as generator_app:
        context = generator_app.context
        logger = generator_app.logger

        logger.info(f"🚀 Starting enhanced data generation for {RECORD_COUNT} {DATA_TYPE} records")

        # Configure filesystem server
        if "filesystem" in context.config.mcp.servers:
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured")

        # --- DEFINE AGENTS ---

        # SDK Setup Agent: Manages Synthetic Data Kit installation and setup
        sdk_setup_agent = Agent(
            name="sdk_setup_manager",
            instruction="""You are a system administrator specializing in ML toolkit setup.
            
            Your responsibilities:
            1. Check if synthetic-data-kit is installed
            2. Install it if missing
            3. Verify system requirements
            4. Run system checks
            5. Report any issues or successful setup
            
            Provide clear status updates and troubleshooting guidance.""",
            server_names=["filesystem"],
        )

        # Document Processor: Handles document ingestion using SDK
        doc_processor_agent = Agent(
            name="document_processor",
            instruction=f"""You are a document processing specialist using Meta's Synthetic Data Kit.
            
            Your tasks:
            1. Process input documents (PDF, HTML, DOCX, etc.)
            2. Extract text content using synthetic-data-kit ingest
            3. Prepare documents for synthetic data generation
            4. Handle various file formats efficiently
            
            Focus on clean text extraction and proper formatting for downstream processing.""",
            server_names=["filesystem"],
        )

        # Dataset Creator: Creates synthetic datasets using SDK
        dataset_creator_agent = Agent(
            name="dataset_creator",
            instruction=f"""You are a synthetic dataset creation expert using Meta's toolkit.
            
            Generate {RECORD_COUNT} high-quality {DATA_TYPE} examples:
            
            For QA pairs:
            - Create diverse, challenging questions
            - Provide accurate, detailed answers
            - Ensure educational value
            
            For CoT (Chain of Thought):
            - Include step-by-step reasoning
            - Show logical progression
            - Demonstrate problem-solving process
            
            For Summary:
            - Create concise, informative summaries
            - Capture key points and insights
            - Maintain factual accuracy
            
            Use synthetic-data-kit create command with appropriate parameters.""",
            server_names=["filesystem"],
        )

        # Quality Curator: Curates and filters generated data
        quality_curator_agent = Agent(
            name="quality_curator",
            instruction=f"""You are a data quality specialist using Meta's curation tools.
            
            Your responsibilities:
            1. Evaluate generated {DATA_TYPE} data quality
            2. Apply quality thresholds and filters
            3. Remove low-quality or problematic examples
            4. Ensure consistency and accuracy
            5. Maintain high standards for training data
            
            Use synthetic-data-kit curate with appropriate quality thresholds.
            Aim for excellence in the final dataset.""",
            server_names=["filesystem"],
        )

        # Format Converter: Converts data to various training formats
        format_converter_agent = Agent(
            name="format_converter",
            instruction=f"""You are a data format specialist for ML training pipelines.
            
            Convert the curated {DATA_TYPE} dataset to multiple formats:
            1. Alpaca format for instruction tuning
            2. OpenAI fine-tuning format (JSONL)
            3. ChatML format for conversational AI
            4. HuggingFace dataset format
            
            Ensure compatibility with popular training frameworks.
            Use synthetic-data-kit save-as with different format options.""",
            server_names=["filesystem"],
        )

        # --- ENHANCED WORKFLOW WITH SDK INTEGRATION ---
        logger.info("=== Starting Enhanced Synthetic Data Generation Workflow ===")
        
        # Step 1: Setup and System Check
        async with sdk_setup_agent:
            logger.info("Step 1: Setting up Synthetic Data Kit...")
            llm = await sdk_setup_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Check if SDK is installed
            if not sdk.check_installation():
                logger.info("Installing synthetic-data-kit...")
                if sdk.install_kit():
                    logger.info("✅ Synthetic Data Kit installed successfully")
                else:
                    logger.error("❌ Failed to install Synthetic Data Kit")
                    return False
            
            # Run system check
            success, stdout, stderr = sdk.system_check()
            setup_result = await llm.generate_str(
                message=f"System check result: Success={success}, Output={stdout}, Errors={stderr}. Provide status summary."
            )
            logger.info(f"Setup status: {setup_result}")

        # Step 2: Document Processing (if source file provided)
        if SOURCE_FILE and os.path.exists(SOURCE_FILE):
            async with doc_processor_agent:
                logger.info("Step 2: Processing source document...")
                llm = await doc_processor_agent.attach_llm(GoogleAugmentedLLM)
                
                # Copy source file to appropriate directory
                import shutil
                file_ext = Path(SOURCE_FILE).suffix.lower()
                if file_ext == '.pdf':
                    target_dir = sdk.data_dir / "pdf"
                elif file_ext in ['.html', '.htm']:
                    target_dir = sdk.data_dir / "html"
                elif file_ext == '.docx':
                    target_dir = sdk.data_dir / "docx"
                else:
                    target_dir = sdk.data_dir / "txt"
                
                target_file = target_dir / Path(SOURCE_FILE).name
                shutil.copy2(SOURCE_FILE, target_file)
                
                # Ingest document
                success, stdout, stderr = sdk.ingest_document(str(target_file))
                doc_result = await llm.generate_str(
                    message=f"Document ingestion result: Success={success}, Output={stdout}, Errors={stderr}. Summarize processing status."
                )
                logger.info(f"Document processing: {doc_result}")

        # Step 3: Dataset Creation
        async with dataset_creator_agent:
            logger.info("Step 3: Creating synthetic dataset...")
            llm = await dataset_creator_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Determine input file
            if SOURCE_FILE:
                input_file = str(sdk.data_dir / "output" / f"{Path(SOURCE_FILE).stem}.txt")
            else:
                # Create a sample text file for demonstration
                sample_text = f"Sample content for {DATA_TYPE} generation with {RECORD_COUNT} examples."
                input_file = str(sdk.data_dir / "txt" / f"sample_{DATA_TYPE}.txt")
                with open(input_file, 'w') as f:
                    f.write(sample_text)
                
                # Ingest the sample file
                sdk.ingest_document(input_file)
                input_file = str(sdk.data_dir / "output" / f"sample_{DATA_TYPE}.txt")
            
            # Create dataset
            success, stdout, stderr = sdk.create_dataset(input_file, DATA_TYPE, RECORD_COUNT)
            creation_result = await llm.generate_str(
                message=f"Dataset creation result: Success={success}, Output={stdout}, Errors={stderr}. Analyze generation quality."
            )
            logger.info(f"Dataset creation: {creation_result}")

        # Step 4: Quality Curation
        async with quality_curator_agent:
            logger.info("Step 4: Curating dataset quality...")
            llm = await quality_curator_agent.attach_llm(GoogleAugmentedLLM)
            
            # Find generated file
            generated_file = str(sdk.data_dir / "generated" / f"{Path(input_file).stem}_{DATA_TYPE}_pairs.json")
            
            if os.path.exists(generated_file):
                success, stdout, stderr = sdk.curate_dataset(generated_file, threshold=7.5)
                curation_result = await llm.generate_str(
                    message=f"Dataset curation result: Success={success}, Output={stdout}, Errors={stderr}. Evaluate quality improvements."
                )
                logger.info(f"Quality curation: {curation_result}")
            else:
                logger.warning(f"Generated file not found: {generated_file}")

        # Step 5: Format Conversion
        async with format_converter_agent:
            logger.info("Step 5: Converting to training formats...")
            llm = await format_converter_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Find curated file
            curated_file = str(sdk.data_dir / "cleaned" / f"{Path(input_file).stem}_cleaned.json")
            
            if os.path.exists(curated_file):
                # Convert to multiple formats
                formats = [
                    ("alpaca", "json"),
                    ("ft", "json"),
                    ("chatml", "json"),
                    ("alpaca", "hf")
                ]
                
                conversion_results = []
                for fmt, storage in formats:
                    success, stdout, stderr = sdk.save_dataset(curated_file, fmt, storage)
                    conversion_results.append(f"{fmt}_{storage}: {'✅' if success else '❌'}")
                
                format_result = await llm.generate_str(
                    message=f"Format conversion results: {', '.join(conversion_results)}. Summarize output formats and locations."
                )
                logger.info(f"Format conversion: {format_result}")
            else:
                logger.warning(f"Curated file not found: {curated_file}")

        # --- ORCHESTRATED WORKFLOW ---
        logger.info("\n=== Starting Orchestrated Workflow ===")
        
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                sdk_setup_agent,
                doc_processor_agent,
                dataset_creator_agent,
                quality_curator_agent,
                format_converter_agent,
            ],
            plan_type="full",
        )

        orchestrated_task = f"""Create a high-quality synthetic {DATA_TYPE} dataset using Meta's Synthetic Data Kit:

        1. Use sdk_setup_manager to ensure synthetic-data-kit is properly installed and configured
        2. Use document_processor to ingest and process source documents (if provided)
        3. Use dataset_creator to generate {RECORD_COUNT} high-quality {DATA_TYPE} examples
        4. Use quality_curator to filter and improve dataset quality
        5. Use format_converter to save the dataset in multiple training-ready formats

        The final output should be production-ready synthetic data suitable for LLM fine-tuning."""

        try:
            await orchestrator.generate_str(
                message=orchestrated_task,
                request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07")
            )

            # Check final outputs
            final_dir = sdk.data_dir / "final"
            if final_dir.exists() and any(final_dir.iterdir()):
                logger.info(f"✅ Enhanced data generation completed successfully!")
                logger.info(f"📁 Output directory: {sdk.data_dir}")
                
                # List generated files
                for file_path in final_dir.glob("*"):
                    file_size = file_path.stat().st_size
                    logger.info(f"📊 {file_path.name}: {file_size:,} bytes")
                
                return True
            else:
                logger.error(f"❌ No final output files found in {final_dir}")
                return False

        except Exception as e:
            logger.error(f"❌ Error during enhanced data generation: {str(e)}")
            return False


if __name__ == "__main__":
    print(f"🚀 Enhanced Data Generator with Meta Synthetic Data Kit")
    print(f"📋 Data Type: {DATA_TYPE}")
    print(f"🔢 Record Count: {RECORD_COUNT}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    if SOURCE_FILE:
        print(f"📄 Source File: {SOURCE_FILE}")
    print("-" * 60)
    
    start_time = datetime.now()
    result = asyncio.run(main())
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Total execution time: {duration:.2f} seconds")
    
    if result:
        print("✅ Enhanced data generation completed successfully!")
        print("🎯 Check the synthetic_datasets/data/ directory for outputs")
    else:
        print("❌ Enhanced data generation failed!") 