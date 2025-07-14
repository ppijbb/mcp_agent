"""
Simulation Engine for the Kimi-K2 Agentic Data Synthesis System

Executes large-scale simulations with multiple agents and scenarios.
"""

from typing import List, Dict, Any, Optional, Callable
from ..models.simulation import SimulationSession, SimulationStep, EnvironmentState, SimulationStatus, StepStatus, StepType
from ..models.domain import Domain, Scenario
from ..models.agent import Agent
from ..models.tool import Tool
import asyncio
import logging
from datetime import datetime
import time
import random

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Engine for executing large-scale agentic simulations.
    
    Responsibilities:
    - Multi-turn scenario execution
    - Agent interaction simulation
    - Environment state management
    - Simulation logging and monitoring
    """
    
    def __init__(self, domain_manager=None, tool_registry=None, agent_factory=None):
        self.domain_manager = domain_manager
        self.tool_registry = tool_registry
        self.agent_factory = agent_factory
        self.active_sessions: Dict[str, SimulationSession] = {}
        self.completed_sessions: Dict[str, SimulationSession] = {}
        self.simulation_callbacks: Dict[str, Callable] = {}
        self.max_concurrent_sessions = 10
        self.session_timeout = 300  # 5 minutes
        
    def create_simulation_session(self, domain_id: str, scenario_id: str,
                                agent_ids: List[str], user_agent_id: Optional[str] = None,
                                metadata: Dict[str, Any] = None) -> Optional[SimulationSession]:
        """Create a new simulation session"""
        # Validate domain and scenario
        if self.domain_manager:
            domain = self.domain_manager.get_domain(domain_id)
            if not domain:
                logger.error(f"Domain not found: {domain_id}")
                return None
            
            scenario = None
            for s in domain.scenarios:
                if s.id == scenario_id:
                    scenario = s
                    break
            
            if not scenario:
                logger.error(f"Scenario not found: {scenario_id}")
                return None
        
        # Validate agents
        if self.agent_factory:
            for agent_id in agent_ids:
                agent = self.agent_factory.get_agent(agent_id)
                if not agent:
                    logger.error(f"Agent not found: {agent_id}")
                    return None
        
        # Create simulation session
        session = SimulationSession(
            domain_id=domain_id,
            scenario_id=scenario_id,
            agent_ids=agent_ids,
            user_agent_id=user_agent_id,
            metadata=metadata or {}
        )
        
        # Initialize environment state
        initial_state = EnvironmentState(
            session_id=session.id,
            current_step=0,
            available_tools=self._get_available_tools(agent_ids),
            active_agents=agent_ids,
            user_context={},
            system_state={"status": "initialized"}
        )
        
        session.add_environment_state(initial_state)
        
        logger.info(f"Created simulation session: {session.id}")
        return session
    
    def _get_available_tools(self, agent_ids: List[str]) -> List[str]:
        """Get available tools for the given agents"""
        if not self.agent_factory:
            return []
        
        available_tools = set()
        for agent_id in agent_ids:
            agent = self.agent_factory.get_agent(agent_id)
            if agent:
                available_tools.update(agent.tool_set)
        
        return list(available_tools)
    
    def start_simulation(self, session_id: str) -> bool:
        """Start a simulation session"""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Simulation session not found: {session_id}")
            return False
        
        if session.status != SimulationStatus.PENDING:
            logger.error(f"Simulation session {session_id} is not in pending status")
            return False
        
        session.start()
        logger.info(f"Started simulation session: {session_id}")
        return True
    
    async def execute_simulation(self, session_id: str) -> Dict[str, Any]:
        """Execute a complete simulation session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        try:
            # Get domain and scenario
            domain = None
            scenario = None
            if self.domain_manager:
                domain = self.domain_manager.get_domain(session.domain_id)
                if domain:
                    for s in domain.scenarios:
                        if s.id == session.scenario_id:
                            scenario = s
                            break
            
            if not scenario:
                session.fail("Scenario not found")
                return {"success": False, "error": "Scenario not found"}
            
            # Execute scenario steps
            for step in scenario.steps:
                step_result = await self._execute_step(session, step)
                if not step_result["success"]:
                    session.fail(f"Step {step.step_number} failed: {step_result['error']}")
                    return step_result
            
            # Complete simulation
            session.complete()
            self.completed_sessions[session_id] = session
            del self.active_sessions[session_id]
            
            logger.info(f"Completed simulation session: {session_id}")
            return {"success": True, "session_id": session_id}
            
        except Exception as e:
            session.fail(f"Simulation execution failed: {str(e)}")
            logger.error(f"Simulation execution failed for session {session_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_step(self, session: SimulationSession, scenario_step) -> Dict[str, Any]:
        """Execute a single scenario step"""
        # Create simulation step
        sim_step = SimulationStep(
            step_number=scenario_step.step_number,
            step_type=StepType.AGENT_ACTION,
            description=scenario_step.description,
            input_data={
                "expected_action": scenario_step.expected_action,
                "required_tools": scenario_step.required_tools,
                "expected_outcome": scenario_step.expected_outcome
            }
        )
        
        session.add_step(sim_step)
        sim_step.start()
        
        try:
            # Find suitable agent for this step
            agent = await self._select_agent_for_step(session, scenario_step)
            if not agent:
                sim_step.fail("No suitable agent found")
                return {"success": False, "error": "No suitable agent found"}
            
            sim_step.agent_id = agent.id
            
            # Execute the step
            step_result = await self._execute_agent_action(session, agent, scenario_step)
            
            if step_result["success"]:
                sim_step.complete(step_result["output"])
                # Update environment state
                self._update_environment_state(session, step_result)
            else:
                sim_step.fail(step_result["error"])
            
            return step_result
            
        except Exception as e:
            sim_step.fail(f"Step execution failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _select_agent_for_step(self, session: SimulationSession, scenario_step) -> Optional[Agent]:
        """Select the most suitable agent for a step"""
        if not self.agent_factory:
            return None
        
        suitable_agents = []
        
        for agent_id in session.agent_ids:
            agent = self.agent_factory.get_agent(agent_id)
            if not agent or not agent.is_active:
                continue
            
            # Check if agent has required tools
            has_required_tools = all(agent.has_tool(tool) for tool in scenario_step.required_tools)
            if not has_required_tools:
                continue
            
            # Calculate suitability score
            suitability_score = self._calculate_agent_suitability(agent, scenario_step)
            suitable_agents.append((agent, suitability_score))
        
        if not suitable_agents:
            return None
        
        # Select agent with highest suitability score
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        return suitable_agents[0][0]
    
    def _calculate_agent_suitability(self, agent: Agent, scenario_step) -> float:
        """Calculate how suitable an agent is for a step"""
        score = 0.0
        
        # Tool availability (40% weight)
        tool_score = len([t for t in scenario_step.required_tools if agent.has_tool(t)]) / len(scenario_step.required_tools)
        score += tool_score * 0.4
        
        # Expertise match (30% weight)
        expertise_score = 0.0
        for area in agent.profile.expertise_areas:
            if area.lower() in scenario_step.description.lower():
                expertise_score += 0.5
        score += min(expertise_score, 1.0) * 0.3
        
        # Performance history (20% weight)
        performance_score = agent.performance_metrics.success_rate
        score += performance_score * 0.2
        
        # Availability (10% weight)
        availability_score = 1.0 if agent.is_active else 0.0
        score += availability_score * 0.1
        
        return score
    
    async def _execute_agent_action(self, session: SimulationSession, agent: Agent, scenario_step) -> Dict[str, Any]:
        """Execute an agent action for a step"""
        try:
            # Simulate agent decision making
            decision = await self._simulate_agent_decision(agent, scenario_step, session)
            
            # Execute tool usage if required
            tool_results = []
            if scenario_step.required_tools:
                for tool_name in scenario_step.required_tools:
                    if agent.has_tool(tool_name):
                        tool_result = await self._simulate_tool_usage(tool_name, scenario_step)
                        tool_results.append(tool_result)
            
            # Generate response
            response = await self._generate_agent_response(agent, scenario_step, decision, tool_results)
            
            return {
                "success": True,
                "output": {
                    "decision": decision,
                    "tool_results": tool_results,
                    "response": response,
                    "agent_id": agent.id
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _simulate_agent_decision(self, agent: Agent, scenario_step, session: SimulationSession) -> Dict[str, Any]:
        """Simulate agent decision making process"""
        # Apply behavior pattern rules
        decisions = []
        for rule in agent.get_rules_by_priority():
            if self._evaluate_rule_condition(rule, scenario_step, session):
                decisions.append({
                    "rule": rule.name,
                    "action": rule.action,
                    "priority": rule.priority
                })
        
        # Generate decision based on personality
        personality_decision = self._generate_personality_decision(agent.behavior_pattern.personality_type, scenario_step)
        
        return {
            "applied_rules": decisions,
            "personality_decision": personality_decision,
            "final_decision": self._combine_decisions(decisions, personality_decision)
        }
    
    def _evaluate_rule_condition(self, rule, scenario_step, session: SimulationSession) -> bool:
        """Evaluate if a rule condition is met"""
        condition = rule.condition.lower()
        
        if "factual information" in condition and "information" in scenario_step.description.lower():
            return True
        elif "complex problems" in condition and len(scenario_step.required_tools) > 2:
            return True
        elif "solving problems" in condition:
            return True
        
        return False
    
    def _generate_personality_decision(self, personality_type, scenario_step) -> str:
        """Generate decision based on personality type"""
        if personality_type.value == "analytical":
            return "analyze_systematically"
        elif personality_type.value == "creative":
            return "explore_creative_solutions"
        elif personality_type.value == "systematic":
            return "follow_established_process"
        elif personality_type.value == "adaptive":
            return "adapt_to_context"
        else:
            return "standard_approach"
    
    def _combine_decisions(self, rule_decisions: List[Dict], personality_decision: str) -> str:
        """Combine rule decisions with personality decision"""
        if rule_decisions:
            # Use highest priority rule action
            highest_priority = max(rule_decisions, key=lambda x: x["priority"])
            return highest_priority["action"]
        else:
            return personality_decision
    
    async def _simulate_tool_usage(self, tool_name: str, scenario_step) -> Dict[str, Any]:
        """Simulate tool usage"""
        # Simulate tool execution time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Generate synthetic tool result
        result = {
            "tool": tool_name,
            "success": random.random() > 0.1,  # 90% success rate
            "result": f"Simulated result for {tool_name}",
            "execution_time": random.uniform(0.1, 2.0)
        }
        
        # Record tool usage if registry is available
        if self.tool_registry:
            self.tool_registry.record_tool_usage(tool_name, result["success"], result["execution_time"])
        
        return result
    
    async def _generate_agent_response(self, agent: Agent, scenario_step, decision: Dict, tool_results: List[Dict]) -> str:
        """Generate agent response based on decision and tool results"""
        # Simulate response generation time
        await asyncio.sleep(random.uniform(0.2, 1.0))
        
        # Generate response based on communication style
        communication_style = agent.behavior_pattern.communication_style.value
        
        if communication_style == "technical":
            response = f"Technical analysis complete. {scenario_step.expected_outcome}"
        elif communication_style == "friendly":
            response = f"Great! I've completed the task. {scenario_step.expected_outcome}"
        elif communication_style == "formal":
            response = f"Task execution completed successfully. {scenario_step.expected_outcome}"
        elif communication_style == "detailed":
            response = f"Detailed analysis performed. Results: {scenario_step.expected_outcome}"
        else:
            response = f"Task completed. {scenario_step.expected_outcome}"
        
        return response
    
    def _update_environment_state(self, session: SimulationSession, step_result: Dict[str, Any]):
        """Update environment state after step execution"""
        current_state = session.get_current_state()
        if not current_state:
            return
        
        # Update current step
        current_state.current_step += 1
        
        # Update system state
        current_state.system_state.update({
            "last_action": step_result["output"]["final_decision"],
            "last_agent": step_result["output"]["agent_id"],
            "step_completed": True
        })
        
        # Add new environment state
        new_state = EnvironmentState(
            session_id=session.id,
            current_step=current_state.current_step,
            environment_variables=current_state.environment_variables.copy(),
            available_tools=current_state.available_tools.copy(),
            active_agents=current_state.active_agents.copy(),
            user_context=current_state.user_context.copy(),
            system_state=current_state.system_state.copy()
        )
        
        session.add_environment_state(new_state)
    
    def get_session(self, session_id: str) -> Optional[SimulationSession]:
        """Get a simulation session by ID"""
        return self.active_sessions.get(session_id) or self.completed_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[SimulationSession]:
        """List all active simulation sessions"""
        return list(self.active_sessions.values())
    
    def list_completed_sessions(self) -> List[SimulationSession]:
        """List all completed simulation sessions"""
        return list(self.completed_sessions.values())
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active simulation session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        session.status = SimulationStatus.CANCELLED
        session.end_time = datetime.utcnow()
        
        # Move to completed sessions
        self.completed_sessions[session_id] = session
        del self.active_sessions[session_id]
        
        logger.info(f"Cancelled simulation session: {session_id}")
        return True
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about simulations"""
        stats = {
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions),
            "total_sessions": len(self.active_sessions) + len(self.completed_sessions),
            "success_rate": 0.0,
            "average_duration": 0.0,
            "sessions_by_status": {}
        }
        
        # Calculate success rate and average duration
        successful_sessions = 0
        total_duration = 0.0
        
        for session in self.completed_sessions.values():
            if session.status == SimulationStatus.COMPLETED:
                successful_sessions += 1
            
            if session.duration:
                total_duration += session.duration
        
        if len(self.completed_sessions) > 0:
            stats["success_rate"] = successful_sessions / len(self.completed_sessions)
            stats["average_duration"] = total_duration / len(self.completed_sessions)
        
        # Sessions by status
        for session in list(self.active_sessions.values()) + list(self.completed_sessions.values()):
            status = session.status.value
            stats["sessions_by_status"][status] = stats["sessions_by_status"].get(status, 0) + 1
        
        return stats
    
    def add_simulation_callback(self, event_type: str, callback: Callable) -> None:
        """Add a callback for simulation events"""
        self.simulation_callbacks[event_type] = callback
    
    def _trigger_callback(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger a simulation callback"""
        callback = self.simulation_callbacks.get(event_type)
        if callback:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback execution failed: {e}") 