"""
Drone Scout - Enterprise Drone Control Agent

A complete MCPAgent implementation for autonomous drone fleet management
with natural language task processing and real-time monitoring.

Features:
- 🚁 Multi-drone fleet coordination
- 🎯 Natural language task definition (Korean/English)
- 📊 Real-time progress monitoring
- 🛡️ Advanced safety systems
- 🔌 Multi-provider hardware support
"""

import asyncio
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.evaluator_optimizer.evaluator_optimizer import (
    EvaluatorOptimizerLLM,
    QualityRating,
)

# Import our models
from .models.drone_data import (
    DronePosition, DroneStatus, DroneTask, TaskResult, RealTimeReport,
    WeatherData, DroneCapability, DroneFleet, DroneAlert
)
from .models.task_types import (
    TaskType, TaskPriority, SensorType, DroneOperationalStatus,
    KOREAN_TASK_KEYWORDS, TASK_SENSOR_REQUIREMENTS, TASK_DURATION_ESTIMATES
)

# Configuration
OUTPUT_DIR = "drone_scout_reports"
DEFAULT_FLEET_SIZE = 5
MISSION_CONTROL_CENTER = "Seoul Drone Operations Center"

app = MCPApp(
    name="drone_scout_system",
    settings=get_settings("configs/mcp_agent.config.yaml"),
    human_input_callback=None
)


async def main(mission: str, result_json_path: str):
    """
    Drone Scout - Enterprise Drone Control Agent System
    
    This function now uses the REAL agent orchestrator to process the user's mission.
    """
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async with app.run() as drone_app:
        context = drone_app.context
        logger = drone_app.logger
        print(f"--- Real Drone Scout Agent System Initialized for mission: {mission} ---")
        
        # Configure servers
        if "filesystem" in context.config.mcp.servers:
            # Note: Ensure the working directory is what you expect.
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            logger.info("Filesystem server configured.")
        
        # All agent definitions from the original implementation are assumed to be here.
        # This includes task_parser_agent, fleet_coordinator_agent, etc.
        # (Code omitted for brevity, but it's the same as the original)
        # --- AGENT DEFINITIONS ( 그대로 유지 ) ---
        task_parser_agent = Agent(
            name="drone_task_parser",
            instruction=f"""You are an expert drone task interpreter for {MISSION_CONTROL_CENTER}. Convert natural language instructions into structured drone tasks. Prioritize safety and regulatory compliance.""",
            server_names=["filesystem", "g-search"],
        )
        fleet_coordinator_agent = Agent(
            name="drone_fleet_coordinator",
            instruction=f"""You are the central fleet coordination system for {MISSION_CONTROL_CENTER}. Manage the status, coordination, and task assignment for a fleet of {DEFAULT_FLEET_SIZE} drones.""",
            server_names=["filesystem", "weather"],
        )
        mission_planner_agent = Agent(
            name="drone_mission_planner",
            instruction=f"""You are an expert mission planning specialist. Create detailed, optimized flight plans, considering waypoints, altitude, weather, and safety. The output must include a clear list of waypoints in the format `WAYPOINT: [longitude, latitude, altitude]` for each step of the flight path.""",
            server_names=["filesystem", "weather", "g-search"],
        )
        realtime_monitor_agent = Agent(
            name="drone_realtime_monitor",
            instruction=f"""You are the real-time monitoring and telemetry specialist. Provide continuous monitoring, progress tracking, and anomaly detection.""",
            server_names=["filesystem"],
        )
        safety_emergency_agent = Agent(
            name="drone_safety_emergency_system",
            instruction=f"""You are the safety officer and emergency response coordinator. Ensure maximum safety and handle all emergency situations.""",
            server_names=["filesystem", "weather"],
        )
        data_analysis_agent = Agent(
            name="drone_data_analysis_specialist",
            instruction=f"""You are the data analysis and reporting expert. Process all collected sensor data and transform it into actionable intelligence.""",
            server_names=["filesystem", "g-search"],
        )
        drone_quality_evaluator = Agent(
            name="drone_operations_quality_evaluator",
            instruction="""You are the quality assurance specialist. Evaluate drone missions based on execution quality, data quality, technical performance, and safety. Provide EXCELLENT, GOOD, FAIR, or POOR ratings.""",
        )
        drone_quality_controller = EvaluatorOptimizerLLM(
            optimizer=task_parser_agent,
            evaluator=drone_quality_evaluator,
            llm_factory=OpenAIAugmentedLLM,
            min_rating=QualityRating.GOOD,
        )

        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                drone_quality_controller,
                fleet_coordinator_agent,
                mission_planner_agent,
                realtime_monitor_agent,
                safety_emergency_agent,
                data_analysis_agent,
            ],
            plan_type="full",
        )
        
        # The user's mission is now the primary driver for the agent task.
        task = f"""Primary Mission: "{mission}"

Execute a comprehensive autonomous drone fleet management operation for {MISSION_CONTROL_CENTER} to accomplish the primary mission.

Your main goal is to interpret, plan, and execute the user's mission. Use your specialized agents to perform the following sub-tasks as needed:
1.  **Interpret & Plan**: Use `drone_task_parser` and `mission_planner_agent` to understand the mission and create an optimized, safe flight plan. The mission plan **must** include a list of geographic waypoints.
2.  **Coordinate & Monitor**: Use `fleet_coordinator_agent` and `realtime_monitor_agent` to manage the drone fleet and track progress.
3.  **Ensure Safety**: Use `safety_emergency_agent` at all times.
4.  **Analyze & Report**: Use `data_analysis_agent` to process any collected data and generate insights.
5.  **Output**: Save all key deliverables to the `{OUTPUT_DIR}` directory, using the timestamp `{timestamp}` in filenames. Crucially, the `mission_plans_{timestamp}.md` file must contain the flight path waypoints.
"""
        
        logger.info(f"Starting real drone agent workflow for mission: {mission}")
        agent_result_string = ""
        final_result = {}

        try:
            agent_result_string = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model="gpt-4o-mini")
            )
            logger.info("Drone Scout workflow completed successfully.")

            # --- PARSE AGENT OUTPUT FOR UI ---
            mission_plan_path_str = os.path.join(OUTPUT_DIR, f"mission_plans_{timestamp}.md")
            
            trajectory = []
            if Path(mission_plan_path_str).exists():
                logger.info(f"Parsing mission plan: {mission_plan_path_str}")
                with open(mission_plan_path_str, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Use regex to find all waypoint lines, robust against formatting variations.
                    # This looks for "[lon, lat, alt]" formats.
                    waypoint_matches = re.findall(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', content)
                    for match in waypoint_matches:
                        try:
                            # Convert matched strings to float: [lon, lat, alt]
                            coords = [float(match[0]), float(match[1]), float(match[2])]
                            trajectory.append(coords)
                        except (ValueError, IndexError):
                            continue # Skip malformed waypoints
                logger.info(f"Extracted {len(trajectory)} waypoints.")
            else:
                logger.warning(f"Mission plan file not found at: {mission_plan_path_str}")

            final_result = {
                "success": True,
                "summary": {
                    "mission": mission,
                    "status": "Completed Successfully",
                    "output_files_location": f"{OUTPUT_DIR}/",
                    "agent_summary": agent_result_string[:500] + "..." # Truncate for display
                },
                "trajectory": trajectory
            }
        except Exception as e:
            logger.error(f"Error during drone workflow or parsing: {str(e)}")
            final_result = {"success": False, "error": str(e), "raw_output": agent_result_string}
        
        # Save the processed, UI-friendly JSON
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        logger.info(f"UI result JSON saved to {result_json_path}")


if __name__ == "__main__":
    # Example of how to run this directly for testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mission = "Fly over Seoul Forest park at 50m altitude and check for unusual activity."
        test_result_path = "test_drone_result.json"
        print(f"--- RUNNING TEST ---")
        print(f"Mission: {test_mission}")
        print(f"Result Path: {test_result_path}")
        asyncio.run(main(mission=test_mission, result_json_path=test_result_path))
        print(f"--- TEST COMPLETE ---")
    else:
        print("This script is meant to be called by the runner. To test, run with `python ... drone_control_agent.py test`")

# The original main function is now replaced by the refactored one.
# No more mock data. No more separate original_main.