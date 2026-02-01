"""
Enhanced Data Generator Agent with Meta Synthetic Data Kit Integration
--------------------------------------------------------------------
A comprehensive data generation tool that combines MCP agents with Meta's
Synthetic Data Kit for high-quality synthetic dataset creation.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.llm.fallback_llm import create_fallback_orchestrator_llm_factory
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from srcs.core.config.loader import settings

# Configuration
OUTPUT_DIR = "synthetic_datasets"
DATA_TYPE = "qa" if len(sys.argv) <= 1 else sys.argv[1]  # qa, cot, summary, custom
RECORD_COUNT = 100 if len(sys.argv) <= 2 else int(sys.argv[2])
SOURCE_FILE = None if len(sys.argv) <= 3 else sys.argv[3]  # Optional source document


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
        mcp_servers_config = {
            name: server.model_dump()
            for name, server in settings.mcp_servers.items()
            if server.enabled
        }
        app_config = {
            "name": "enhanced_data_generator_agent",
            "mcp": mcp_servers_config
        }
        self.app = MCPApp(settings=app_config, human_input_callback=None)
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
                instruction="You are a data quality analyst. Validate the dataset and provide a concise validation report highlighting issues.",
                server_names=[],
            )
            refiner_agent = Agent(
                name="data_refiner",
                instruction="You are a data specialist. Given a dataset and its validation report, refine the dataset by addressing issues. Output only the refined dataset in valid JSON format.",
                server_names=[],
            )

            # The orchestrator will manage the workflow
            orchestrator_llm_factory = create_fallback_orchestrator_llm_factory(
                primary_model="gemini-2.5-flash-lite",
                logger_instance=logger
            )
            orchestrator = Orchestrator(
                llm_factory=orchestrator_llm_factory,
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
                    request_params=RequestParams(model="gemini-2.5-flash-lite")
                )

                # --- Generator–Validator–Refiner Feedback Loop ---
                # Define validator and refiner agents
                validator_agent = Agent(
                    name="data_validator",
                    instruction="You are a data quality analyst. Validate the dataset and provide a concise validation report highlighting issues.",
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
                        val_llm = await validator_agent.attach_llm(GoogleAugmentedLLM)
                        validation_report = await val_llm.generate_str(message=current_data)
                    # Refinement step
                    async with refiner_agent:
                        ref_llm = await refiner_agent.attach_llm(GoogleAugmentedLLM)
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

                logger.info(f"Successfully generated and saved refined data to {output_path}")
                return f"Successfully generated and saved refined data to {output_path}"

            except Exception as e:
                logger.error(f"An error occurred during the data generation workflow: {e}")
                return f"Error: {e}"
