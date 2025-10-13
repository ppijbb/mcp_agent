import asyncio
import os
import json
import psutil
from datetime import datetime
from typing import Dict, Any

# Correct imports based on the SEO Doctor Agent
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.config import get_settings
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

class AIOpsOrchestratorAgent:
    """
    AIOps Orchestrator Agent built on the standardized MCP Agent architecture.
    It coordinates multiple specialized virtual agents to diagnose and potentially
    remediate IT infrastructure issues.
    """
    def __init__(self, output_dir: str = "aiops_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize MCPApp, pointing to the central config file
        self.app = MCPApp(
            name="aiops_orchestrator",
            settings=get_settings("configs/mcp_agent.config.yaml"),
            human_input_callback=None
        )
    
    def get_real_system_snapshot(self) -> Dict[str, Any]:
        """Get actual system metrics using psutil"""
        try:
            # Get CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Get memory information
            memory = psutil.virtual_memory()
            
            # Get top processes by CPU usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    proc_info['cpu_percent'] = proc.cpu_percent()
                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Sort by CPU usage and get top 10
            top_processes = sorted(processes, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:10]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "usage_percent": memory.percent
                },
                "top_processes": top_processes,
                "system_info": {
                    "hostname": psutil.os.uname().nodename if hasattr(psutil.os, 'uname') else "unknown",
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                }
            }
        except Exception as e:
            return {"error": f"Failed to get system snapshot: {str(e)}"}

    async def handle_alert(self, alert: Dict[str, Any]):
        """
        Handles an incoming IT alert by orchestrating a team of virtual agents.
        This follows the established ReAct pattern within the MCP framework.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"alert_{alert['id']}_{timestamp}.txt")

        # The app.run() context manages the lifecycle of MCP servers
        async with self.app.run() as app_context:
            logger = app_context.logger
            logger.info(f"AIOps Orchestrator started for alert: {alert['description']}")

            # 1. Define Specialized Virtual Agents for the Task
            monitoring_agent = Agent(
                name="system_monitor",
                instruction=f"""You are a System Monitor. Your task is to analyze system performance
                for the node '{alert['node']}'. Since this is a simulated environment, provide a realistic
                analysis of what could cause high CPU usage (95% for 5 minutes). 
                
                Focus on:
                - Common causes of high CPU usage (runaway processes, memory leaks, infinite loops)
                - Typical processes that consume high CPU (java, python, node, database processes)
                - System monitoring best practices
                - Immediate remediation steps
                
                Provide a detailed technical analysis as if you had access to real system metrics.""",
                server_names=["g-search", "fetch"] # Use stable servers
            )

            rca_agent = Agent(
                name="root_cause_analyst",
                instruction=f"""You are a Root Cause Analyst. Based on the system snapshot,
                determine the likely root cause of the issue. Use 'g-search' to look up error messages
                or unfamiliar process names. Your analysis should be concise and point to a specific cause.""",
                server_names=["g-search", "fetch"]
            )

            # 2. Create an Orchestrator to manage the agents
            orchestrator = Orchestrator(
                llm_factory=GoogleAugmentedLLM,
                available_agents=[monitoring_agent, rca_agent],
                plan_type="full" # Let the LLM create a full plan
            )

            # 3. Define the main task for the Orchestrator
            analysis_task = f"""
            An alert has been triggered: '{alert['description']}' on node '{alert['node']}'.
            
            Follow these steps using the ReAct pattern (Thought -> Action -> Observation):
            1.  **THOUGHT**: Plan to use the `monitoring_scout` to get the system state.
            2.  **ACTION**: Execute the call to the `monitoring_scout`.
            3.  **OBSERVATION**: Analyze the snapshot.
            4.  **THOUGHT**: Based on the snapshot, plan to use the `root_cause_analyst` to find the cause.
            5.  **ACTION**: Execute the analysis.
            6.  **OBSERVATION**: State the final root cause.
            
            Provide a clear, final conclusion about the root cause of the high CPU usage.
            """

            logger.info(f"Executing AIOps analysis task for node {alert['node']}...")
            
            try:
                # 4. Run the orchestration via a dedicated ReAct handler method
                final_result = await self._react_aiops_analysis(
                    orchestrator, analysis_task, logger
                )

                logger.info(f"Analysis complete. Final Result:\n{final_result}")
                
                # 5. Save the report
                with open(report_path, "w") as f:
                    f.write("AIOps Alert Analysis Report\n")
                    f.write("="*30 + "\n")
                    f.write(f"Alert ID: {alert['id']}\n")
                    f.write(f"Node: {alert['node']}\n")
                    f.write(f"Description: {alert['description']}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write("\n--- Analysis Result ---\n")
                    f.write(final_result)
                
                logger.info(f"Report saved to {report_path}")
                return final_result

            except Exception as e:
                error_message = f"AIOps orchestration failed for alert {alert['id']}: {e}"
                logger.error(error_message)
                with open(report_path, "w") as f:
                    f.write(error_message)
                return error_message

    async def _react_aiops_analysis(
        self,
        orchestrator: Orchestrator,
        task: str,
        logger
    ) -> str:
        """
        Real ReAct loop for AIOps analysis based on SEO Doctor pattern.
        Implements THOUGHT → ACTION → OBSERVATION cycle using actual orchestrator calls.
        """
        logger.info("Entering REAL ReAct analysis loop...")

        # 🧠 THOUGHT PHASE: Plan the analysis approach
        thought_task = f"""
        THOUGHT PHASE - AIOps Analysis Planning:
        
        I need to analyze an IT infrastructure alert systematically.
        
        Context: {task}
        
        My approach will be:
        1. First, gather real-time system metrics from the affected node
        2. Analyze the data to identify resource consumption patterns
        3. Determine the root cause of the high CPU usage
        4. Provide actionable remediation steps
        
        I will use the monitoring-scout agent to get system snapshots,
        then use search capabilities to research any unfamiliar processes or error patterns.
        
        What specific system metrics should I gather first?
        """
        
        logger.info("REACT THOUGHT: Planning AIOps analysis approach")
        thought_result = await orchestrator.generate_str(
            message=thought_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.1)
        )
        
        # ⚡ ACTION PHASE: Execute the monitoring and data collection
        # Get real system snapshot
        system_snapshot = self.get_real_system_snapshot()
        
        action_task = f"""
        ACTION PHASE - Execute System Analysis:
        
        Based on my thought process: {thought_result}
        
        REAL SYSTEM SNAPSHOT DATA:
        {json.dumps(system_snapshot, indent=2)}
        
        Now analyze this ACTUAL system data:
        
        1. CURRENT SYSTEM STATE:
        - Current CPU usage: {system_snapshot.get('cpu', {}).get('usage_percent', 'N/A')}%
        - Memory usage: {system_snapshot.get('memory', {}).get('usage_percent', 'N/A')}%
        - Top CPU processes: {[p.get('name', 'unknown') for p in system_snapshot.get('top_processes', [])[:3]]}
        
        2. PROCESS INVESTIGATION:
        - Analyze the top CPU-consuming processes from the real data
        - Research any suspicious or high-consumption processes using search
        - Check for known issues or solutions for these specific processes
        
        3. SYSTEM STATE EVALUATION:
        - Compare current metrics against normal operational baselines
        - Identify if the current state indicates the reported high CPU issue
        - Determine if this is a current problem or if it has been resolved
        
        Execute these monitoring and analysis tasks using the REAL system data provided.
        Provide detailed findings for each step.
        """
        
        logger.info("REACT ACTION: Executing monitoring and data collection")
        action_result = await orchestrator.generate_str(
            message=action_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.2)
        )
        
        # 🔍 OBSERVATION PHASE: Analyze results and generate conclusions
        observation_task = f"""
        OBSERVATION PHASE - Root Cause Analysis and Recommendations:
        
        Based on my analysis execution: {action_result}
        
        Now I need to synthesize the findings and provide conclusions:
        
        1. ROOT CAUSE IDENTIFICATION:
        - What is the primary cause of the high CPU usage?
        - Is this a runaway process, resource leak, or external attack?
        - What evidence supports this conclusion?
        
        2. IMPACT ASSESSMENT:
        - How severe is this issue?
        - What systems or services are affected?
        - What is the urgency level?
        
        3. REMEDIATION PLAN:
        - Immediate actions to resolve the issue
        - Steps to prevent recurrence
        - Monitoring recommendations
        
        4. ESCALATION DECISION:
        - Should this be escalated to human operators?
        - Can this be resolved automatically?
        - What are the risks of automated intervention?
        
        Provide a clear, actionable conclusion with specific next steps.
        Format the response as a structured incident report.
        """
        
        logger.info("REACT OBSERVATION: Analyzing results and generating recommendations")
        observation_result = await orchestrator.generate_str(
            message=observation_task,
            request_params=RequestParams(model="gemini-2.5-flash-lite-preview-06-07", temperature=0.1)
        )
        
        # Combine all ReAct results for comprehensive analysis
        combined_result = f"""
        # 🚨 AIOPS INCIDENT ANALYSIS - REACT REPORT
        
        ## 🧠 THOUGHT PHASE - Analysis Planning
        {thought_result}
        
        ## ⚡ ACTION PHASE - System Investigation
        {action_result}
        
        ## 🔍 OBSERVATION PHASE - Root Cause & Recommendations
        {observation_result}
        
        ---
        Analysis completed using ReAct pattern for infrastructure incident.
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        logger.info("REACT COMPLETE: AIOps analysis using THOUGHT → ACTION → OBSERVATION pattern")
        return combined_result


async def main():
    """Main function to run a demo of the AIOps Orchestrator Agent."""
    agent = AIOpsOrchestratorAgent()
    
    test_alert = {
        "id": "cpu-95",
        "node": "web-server-01",
        "description": "High CPU Usage Detected"
    }
    
    await agent.handle_alert(test_alert)

if __name__ == "__main__":
    asyncio.run(main()) 