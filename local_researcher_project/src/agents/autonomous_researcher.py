"""
Autonomous Researcher Agent

자율적으로 계획을 수립하고 작업을 수행하는 리서처 에이전트.
MCP agent 라이브러리를 사용하여 실제 작업을 수행.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import google.generativeai as genai
from researcher_config import config, get_llm_config, get_agent_config, get_research_config, get_mcp_config


class AutonomousResearcherAgent:
    """
    자율적으로 계획을 수립하고 작업을 수행하는 리서처 에이전트.
    MCP agent 라이브러리를 사용하여 실제 작업을 수행.
    """
    
    def __init__(self):
        """Initialize the autonomous researcher agent."""
        # Load configurations
        self.llm_config = get_llm_config()
        self.agent_config = get_agent_config()
        self.research_config = get_research_config()
        self.mcp_config = get_mcp_config()
        
        self.name = "autonomous_researcher"
        self.instruction = "Autonomous researcher agent that self-plans and executes research tasks"
        
        # Initialize specialized agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize specialized research agents."""
        # Setup Gemini
        if not self.llm_config.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=self.llm_config.api_key)
        self.model = genai.GenerativeModel(self.llm_config.model)
        
        # Agent instructions
        self.agent_instructions = {
            "task_analyzer": """You are a task analysis agent. Your role is to:
            1. Analyze user requests and break them down into clear objectives
            2. Identify required research areas and methodologies
            3. Determine success criteria and validation methods
            4. Create detailed research plans with timelines
            Always provide specific, actionable analysis without fallback responses.""",
            
            "research_executor": """You are a research execution agent. Your role is to:
            1. Execute research tasks using available tools and data sources
            2. Gather information from multiple sources
            3. Analyze and synthesize findings
            4. Maintain research quality and accuracy
            Always use real data sources and provide evidence-based results.""",
            
            "evaluator": """You are an evaluation agent. Your role is to:
            1. Critically evaluate research findings
            2. Assess quality and reliability of sources
            3. Identify gaps and limitations
            4. Provide improvement recommendations
            Always provide objective, evidence-based evaluations.""",
            
            "synthesizer": """You are a synthesis agent. Your role is to:
            1. Integrate findings from multiple sources
            2. Create comprehensive reports
            3. Generate actionable insights
            4. Present results in clear, professional format
            Always provide complete, well-structured synthesis."""
        }
    
    async def self_plan_research(self, user_request: str) -> Dict[str, Any]:
        """자율적으로 연구 계획을 수립합니다."""
        planning_prompt = f"""
        {self.agent_instructions['task_analyzer']}
        
        Analyze the following research request and create a comprehensive research plan:
        
        Request: {user_request}
        
        Create a detailed plan including:
        1. Research objectives and questions
        2. Required data sources and methodologies
        3. Timeline and milestones
        4. Success criteria
        5. Risk assessment and mitigation strategies
        
        Provide specific, actionable steps without any fallback responses.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate_content(
                planning_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.temperature
                )
            )
        )
        
        return {
            "research_plan": response.text,
            "created_at": datetime.now().isoformat(),
            "status": "planned"
        }
    
    async def execute_research(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """연구 계획을 실행합니다."""
        execution_prompt = f"""
        {self.agent_instructions['research_executor']}
        
        Execute the following research plan:
        
        Plan: {research_plan.get('research_plan', '')}
        
        Perform the research using available tools and data sources.
        Gather information from multiple sources and analyze findings.
        Provide evidence-based results without any mock or fallback data.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate_content(
                execution_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.temperature
                )
            )
        )
        
        return {
            "research_results": response.text,
            "executed_at": datetime.now().isoformat(),
            "status": "executed"
        }
    
    async def evaluate_research(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """연구 결과를 평가합니다."""
        evaluation_prompt = f"""
        {self.agent_instructions['evaluator']}
        
        Evaluate the following research results:
        
        Results: {research_results.get('research_results', '')}
        
        Provide critical evaluation including:
        1. Quality assessment
        2. Source reliability analysis
        3. Gap identification
        4. Improvement recommendations
        
        Provide objective, evidence-based evaluation without fallback responses.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate_content(
                evaluation_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.temperature
                )
            )
        )
        
        return {
            "evaluation_results": response.text,
            "evaluated_at": datetime.now().isoformat(),
            "status": "evaluated"
        }
    
    async def synthesize_findings(self, research_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """연구 결과를 종합합니다."""
        synthesis_prompt = f"""
        {self.agent_instructions['synthesizer']}
        
        Synthesize the following research findings:
        
        Research Results: {research_results.get('research_results', '')}
        Evaluation: {evaluation_results.get('evaluation_results', '')}
        
        Create a comprehensive synthesis including:
        1. Executive summary
        2. Key findings
        3. Evidence and sources
        4. Conclusions and recommendations
        5. Limitations and future work
        
        Provide complete, well-structured synthesis without fallback responses.
        """
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.generate_content(
                synthesis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.llm_config.temperature
                )
            )
        )
        
        return {
            "synthesis_results": response.text,
            "synthesized_at": datetime.now().isoformat(),
            "status": "synthesized"
        }
    
    async def run_autonomous_research(self, user_request: str) -> Dict[str, Any]:
        """자율적으로 전체 연구 프로세스를 실행합니다."""
        print(f"Starting autonomous research for: {user_request}")
        
        # 1. 자율 계획 수립
        if self.agent_config.enable_self_planning:
            print("1. Self-planning research...")
            research_plan = await self.self_plan_research(user_request)
            print(f"Research plan created: {research_plan['status']}")
        else:
            raise ValueError("Self-planning is disabled but required for autonomous operation")
        
        # 2. 연구 실행
        print("2. Executing research...")
        research_results = await self.execute_research(research_plan)
        print(f"Research executed: {research_results['status']}")
        
        # 3. 연구 평가
        print("3. Evaluating research...")
        evaluation_results = await self.evaluate_research(research_results)
        print(f"Research evaluated: {evaluation_results['status']}")
        
        # 4. 결과 종합
        print("4. Synthesizing findings...")
        synthesis_results = await self.synthesize_findings(research_results, evaluation_results)
        print(f"Findings synthesized: {synthesis_results['status']}")
        
        # 5. 최종 결과 반환
        final_result = {
            "user_request": user_request,
            "research_plan": research_plan,
            "research_results": research_results,
            "evaluation_results": evaluation_results,
            "synthesis_results": synthesis_results,
            "completed_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        print("Autonomous research completed successfully!")
        return final_result
