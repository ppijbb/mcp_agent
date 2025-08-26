"""
Agent Factory for the Kimi-K2 Agentic Data Synthesis System

Creates and manages diverse agents with different tool sets and behavior patterns.
"""

from typing import List, Dict, Any, Optional
from ..models.agent import Agent, AgentConfig, AgentProfile, BehaviorPattern, Rule, AgentRole, PersonalityType, CommunicationStyle
from ..models.tool import Tool
from ..agents.kimi_k2_agent import KimiK2ConversableAgent # Import the new conversable agent
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating diverse agents with different capabilities and behaviors.
    
    Responsibilities:
    - Agent creation with different profiles
    - Tool set assignment based on domain requirements
    - Behavior pattern generation
    - Agent specialization and customization
    """
    
    def __init__(self, tool_registry: Optional[Any] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.agents: Dict[str, KimiK2ConversableAgent] = {} # Change type hint
        self.agent_configs: Dict[str, AgentConfig] = {} # Store agent configs
        self.agent_templates: Dict[str, Dict[str, Any]] = {}
        self.behavior_templates: Dict[str, BehaviorPattern] = {}
        self.tool_registry = tool_registry
        self.llm_config = llm_config
        
        # Initialize with default behavior patterns
        self._initialize_behavior_patterns()
        self._initialize_agent_templates()
    
    def _initialize_behavior_patterns(self) -> None:
        """Initialize default behavior patterns"""
        self.behavior_templates = {
            "analytical_expert": BehaviorPattern(
                name="Analytical Expert",
                description="Analytical expert who focuses on systematic problem solving",
                personality_type=PersonalityType.ANALYTICAL,
                communication_style=CommunicationStyle.TECHNICAL,
                decision_making_approach="data-driven analysis",
                problem_solving_strategy="systematic decomposition",
                collaboration_preference="structured teamwork",
                rules=[
                    Rule(
                        name="Verify Information",
                        description="Always verify information before providing it",
                        condition="when providing factual information",
                        action="search for verification sources",
                        priority=5
                    ),
                    Rule(
                        name="Systematic Approach",
                        description="Use systematic approach to problem solving",
                        condition="when faced with complex problems",
                        action="break down into smaller components",
                        priority=4
                    )
                ]
            ),
            "creative_problem_solver": BehaviorPattern(
                name="Creative Problem Solver",
                description="Creative problem solver who thinks outside the box",
                personality_type=PersonalityType.CREATIVE,
                communication_style=CommunicationStyle.FRIENDLY,
                decision_making_approach="intuitive exploration",
                problem_solving_strategy="lateral thinking",
                collaboration_preference="brainstorming sessions",
                rules=[
                    Rule(
                        name="Explore Alternatives",
                        description="Always explore multiple solution approaches",
                        condition="when solving problems",
                        action="generate multiple solution options",
                        priority=5
                    ),
                    Rule(
                        name="Encourage Innovation",
                        description="Encourage innovative thinking",
                        condition="when collaborating with others",
                        action="suggest creative approaches",
                        priority=3
                    )
                ]
            ),
            "systematic_coordinator": BehaviorPattern(
                name="Systematic Coordinator",
                description="Systematic coordinator who manages complex workflows",
                personality_type=PersonalityType.SYSTEMATIC,
                communication_style=CommunicationStyle.FORMAL,
                decision_making_approach="process-driven",
                problem_solving_strategy="workflow optimization",
                collaboration_preference="structured coordination",
                rules=[
                    Rule(
                        name="Follow Process",
                        description="Always follow established processes",
                        condition="when executing tasks",
                        action="adhere to defined procedures",
                        priority=5
                    ),
                    Rule(
                        name="Coordinate Resources",
                        description="Coordinate resources effectively",
                        condition="when managing multiple tasks",
                        action="allocate resources optimally",
                        priority=4
                    )
                ]
            ),
            "adaptive_specialist": BehaviorPattern(
                name="Adaptive Specialist",
                description="Adaptive specialist who adjusts to changing situations",
                personality_type=PersonalityType.ADAPTIVE,
                communication_style=CommunicationStyle.CASUAL,
                decision_making_approach="context-aware",
                problem_solving_strategy="adaptive response",
                collaboration_preference="flexible teamwork",
                rules=[
                    Rule(
                        name="Adapt to Context",
                        description="Adapt behavior based on context",
                        condition="when situation changes",
                        action="adjust approach accordingly",
                        priority=5
                    ),
                    Rule(
                        name="Learn from Experience",
                        description="Learn from previous experiences",
                        condition="when encountering similar situations",
                        action="apply learned strategies",
                        priority=4
                    )
                ]
            )
        }
    
    def _initialize_agent_templates(self) -> None:
        """Initialize default agent templates"""
        self.agent_templates = {
            "tech_support_specialist": {
                "name": "Tech Support Specialist",
                "agent_type": "SPECIALIST", # Changed from role
                "expertise_domains": ["networking", "system administration", "troubleshooting"],
                "description": "Agent specializing in technical support", # Added for AgentConfig
                "communication_style": "technical",
                "problem_solving_approach": "systematic",
                "collaboration_style": "mentoring",
                "tool_preferences": ["system_diagnostics", "log_analysis", "remote_access"]
            },
            "data_analyst": {
                "name": "Data Analyst",
                "agent_type": "EXPERT",
                "expertise_domains": ["data analysis", "statistics", "business intelligence"],
                "description": "Agent specializing in data analysis",
                "communication_style": "detailed",
                "problem_solving_approach": "analytical",
                "collaboration_style": "consultative",
                "tool_preferences": ["data_analysis", "spreadsheet_tools", "reporting"]
            },
            "creative_writer": {
                "name": "Creative Writer",
                "agent_type": "SPECIALIST",
                "expertise_domains": ["creative writing", "content creation", "storytelling"],
                "description": "Agent specializing in creative writing",
                "communication_style": "creative",
                "problem_solving_approach": "intuitive",
                "collaboration_style": "brainstorming sessions",
                "tool_preferences": ["text_generation", "style_analysis", "content_editing"]
            },
            "project_coordinator": {
                "name": "Project Coordinator",
                "agent_type": "COORDINATOR",
                "expertise_domains": ["project management", "coordination", "workflow optimization"],
                "description": "Agent specializing in project coordination",
                "communication_style": "formal",
                "problem_solving_approach": "systematic",
                "collaboration_style": "structured coordination",
                "tool_preferences": ["project_management", "scheduling", "communication"]
            }
        }
    
    def create_agent(self, agent_config: AgentConfig) -> KimiK2ConversableAgent:
        """Create a new KimiK2ConversableAgent instance from AgentConfig."""
        if agent_config.agent_id in self.agents:
            logger.warning(f"Agent with ID {agent_config.agent_id} already exists. Returning existing instance.")
            return self.agents[agent_config.agent_id]

        # Use AgentConfig directly to create the conversable agent
        agent_instance = KimiK2ConversableAgent(
            agent_config=agent_config,
            llm_config=self.llm_config,
            tool_registry=self.tool_registry
        )
        
        self.agents[agent_config.agent_id] = agent_instance
        self.agent_configs[agent_config.agent_id] = agent_config
        logger.info(f"Created agent: {agent_config.name} (ID: {agent_config.agent_id})")
        
        return agent_instance
    
    def create_agent_from_template(self, template_name: str, customizations: Optional[Dict[str, Any]] = None) -> Optional[KimiK2ConversableAgent]:
        """Create an agent from a predefined template."""
        if template_name not in self.agent_templates:
            logger.warning(f"Agent template not found: {template_name}")
            return None
        
        template = self.agent_templates[template_name].copy()
        customizations = customizations or {}
        
        # Merge template with customizations
        template.update(customizations)

        # Generate a unique agent_id if not provided
        if 'agent_id' not in template or not template['agent_id']:
            template['agent_id'] = f"{template_name}_{random.randint(1000, 9999)}"
        
        try:
            agent_config = AgentConfig(**template)
            return self.create_agent(agent_config)
        except Exception as e:
            logger.error(f"Failed to create agent from template {template_name}: {e}")
            return None

    def get_agent(self, agent_id: str) -> Optional[KimiK2ConversableAgent]: # Change return type
        """Get an agent by its ID"""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[KimiK2ConversableAgent]: # Change return type
        """Get an agent by its name"""
        for agent_id, agent in self.agents.items():
            if agent.agent_config.name == name:
                return agent
        return None
    
    def list_agents(self, agent_type: Optional[str] = None, active_only: bool = True) -> List[KimiK2ConversableAgent]: # Change param and return type
        """List agents, optionally filtered by role and active status"""
        filtered_agents = []
        for agent in self.agents.values():
            if active_only and not agent.agent_config.is_active: # Assuming AgentConfig has is_active
                continue
            if agent_type and agent.agent_config.agent_type != agent_type.upper(): # Assuming AgentConfig has agent_type
                continue
            filtered_agents.append(agent)
        return filtered_agents

    def update_agent(self, agent_id: str, **kwargs) -> bool:
        """Update an existing agent's configuration."""
        agent_instance = self.agents.get(agent_id)
        if not agent_instance:
            logger.warning(f"Agent with ID {agent_id} not found for update.")
            return False
        
        # Update the underlying AgentConfig object
        for key, value in kwargs.items():
            if hasattr(agent_instance.agent_config, key):
                setattr(agent_instance.agent_config, key, value)
        agent_instance.agent_config.updated_at = datetime.utcnow()
        logger.info(f"Agent {agent_id} configuration updated.")
        return True
    
    def assign_tools_to_agent(self, agent_id: str, tool_ids: List[str]) -> bool:
        """Assign a list of tool IDs to an agent's tool set."""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found. Cannot assign tools.")
            return False
        agent.agent_config.tool_preferences = list(set(tool_ids)) # Update config
        agent._register_tools() # Re-register tools with the conversable agent
        logger.info(f"Tools {tool_ids} assigned to agent {agent_id}.")
        return True
    
    def add_tool_to_agent(self, agent_id: str, tool_id: str) -> bool:
        """Add a single tool to an agent's tool set."""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found. Cannot add tool.")
            return False
        if tool_id not in agent.agent_config.tool_preferences:
            agent.agent_config.tool_preferences.append(tool_id) # Update config
            agent._register_tools() # Re-register tools
            logger.info(f"Tool {tool_id} added to agent {agent_id}.")
            return True
        logger.info(f"Tool {tool_id} already exists for agent {agent_id}.")
        return False
    
    def remove_tool_from_agent(self, agent_id: str, tool_id: str) -> bool:
        """Remove a tool from an agent's tool set."""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found. Cannot remove tool.")
            return False
        if tool_id in agent.agent_config.tool_preferences:
            agent.agent_config.tool_preferences.remove(tool_id) # Update config
            # Note: AutoGen ConversableAgent does not directly support unregistering tools dynamically.
            # For a production system, a more robust re-initialization or tool management might be needed.
            logger.info(f"Tool {tool_id} removed from agent {agent_id}. Re-initializing agent for changes to take effect.")
            # A simple workaround: re-create the agent to reflect tool changes. This might be disruptive.
            # A better approach for dynamic tool removal in AutoGen might involve custom `generate_reply` logic.
            agent_config_current = self.agent_configs.get(agent_id)
            if agent_config_current:
                new_agent = KimiK2ConversableAgent(
                    agent_config=agent_config_current,
                    llm_config=self.llm_config,
                    tool_registry=self.tool_registry
                )
                self.agents[agent_id] = new_agent
                return True
        logger.info(f"Tool {tool_id} not found for agent {agent_id}.")
        return False
    
    def add_rule_to_agent(self, agent_id: str, rule: Rule) -> bool:
        """Add an interaction rule to an agent."""
        agent_config = self.agent_configs.get(agent_id)
        if not agent_config:
            logger.warning(f"Agent {agent_id} config not found. Cannot add rule.")
            return False
        if rule not in agent_config.behavior_pattern.rules: # Assuming comparison works for Rule objects
            agent_config.behavior_pattern.rules.append(rule)
            logger.info(f"Rule {rule.name} added to agent {agent_id}.")
            return True
        logger.info(f"Rule {rule.name} already exists for agent {agent_id}.")
        return False
    
    def get_agents_for_domain(self, domain_id: str) -> List[KimiK2ConversableAgent]: # Change return type
        """Get agents that have expertise in a given domain."""
        return [agent for agent in self.agents.values() 
                if domain_id in agent.agent_config.expertise_domains]

    def get_agents_with_tools(self, tool_ids: List[str]) -> List[KimiK2ConversableAgent]: # Change return type
        """Get agents that possess any of the specified tools."""
        matched_agents = []
        for agent in self.agents.values():
            if any(tool_id in agent.agent_config.tool_preferences for tool_id in tool_ids):
                matched_agents.append(agent)
        return matched_agents

    def create_specialized_agent(self, domain_id: str, required_tools: List[str],
                                behavior_preference: Optional[str] = None) -> Optional[KimiK2ConversableAgent]: # Change return type
        """Create a specialized agent for a given domain and required tools."""
        # This method would typically involve more sophisticated logic for agent specialization
        # For now, it will attempt to create a generic agent based on parameters.
        agent_name = f"SpecializedAgent_{random.randint(1000, 9999)}"
        description = f"Specialized agent for {domain_id} with tools {required_tools}"
        agent_type = "SPECIALIST"
        expertise_domains = [domain_id]
        tool_preferences = required_tools
        communication_style = "technical"
        problem_solving_approach = "goal_oriented"
        collaboration_style = "collaborative"

        behavior_pattern_name = behavior_preference or "analytical_expert"

        # Create AgentConfig
        agent_config = AgentConfig(
            agent_id=agent_name, # Use generated name as ID
            name=agent_name,
            description=description,
            agent_type=agent_type,
            behavior_pattern=behavior_pattern_name,
            expertise_domains=expertise_domains,
            tool_preferences=tool_preferences,
            communication_style=communication_style,
            problem_solving_approach=problem_solving_approach,
            collaboration_style=collaboration_style
        )

        return self.create_agent(agent_config)

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics for all managed agents."""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.agent_config.is_active),
            "agent_types_count": {},
            "domain_expertise_count": {},
            "tool_preference_count": {}
        }
        
        for agent in self.agents.values():
            agent_type = agent.agent_config.agent_type
            stats["agent_types_count"][agent_type] = stats["agent_types_count"].get(agent_type, 0) + 1
            
            for domain in agent.agent_config.expertise_domains:
                stats["domain_expertise_count"][domain] = stats["domain_expertise_count"].get(domain, 0) + 1
            
            for tool in agent.agent_config.tool_preferences:
                stats["tool_preference_count"][tool] = stats["tool_preference_count"].get(tool, 0) + 1
                
        return stats

    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent, making it unavailable for new tasks."""
        agent_config = self.agent_configs.get(agent_id)
        if agent_config:
            agent_config.is_active = False
            logger.info(f"Agent {agent_id} deactivated.")
            return True
        logger.warning(f"Agent {agent_id} not found. Cannot deactivate.")
        return False

    def activate_agent(self, agent_id: str) -> bool:
        """Activate an agent, making it available for tasks again."""
        agent_config = self.agent_configs.get(agent_id)
        if agent_config:
            agent_config.is_active = True
            logger.info(f"Agent {agent_id} activated.")
            return True
        logger.warning(f"Agent {agent_id} not found. Cannot activate.")
        return False

    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Perform a validation check on an agent's configuration and status."""
        agent_config = self.agent_configs.get(agent_id)
        validation_result = {"is_valid": True, "errors": [], "warnings": []}

        if not agent_config:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Agent config for ID {agent_id} not found.")
            return validation_result
        
        if not agent_config.name:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Agent name is missing.")
        
        if not agent_config.expertise_domains:
            validation_result["warnings"].append(f"Agent {agent_id} has no specified expertise domains.")
            
        if not agent_config.tool_preferences:
            validation_result["warnings"].append(f"Agent {agent_id} has no specified tool preferences.")
            
        # Check if behavior pattern exists
        if agent_config.behavior_pattern not in self.behavior_templates:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Behavior pattern '{agent_config.behavior_pattern}' not defined.")
        
        # Check if preferred tools exist in tool_registry (if provided)
        if self.tool_registry:
            for tool_id in agent_config.tool_preferences:
                if not self.tool_registry.get_tool(tool_id):
                    validation_result["warnings"].append(f"Preferred tool '{tool_id}' not found in registry.")
                    
        if not validation_result["errors"] and not validation_result["warnings"]:
            logger.info(f"Agent {agent_id} validated successfully.")
        else:
            logger.warning(f"Agent {agent_id} validation completed with issues: {validation_result}")

        return validation_result 
    
    def _analyze_user_intent(self, user_request: str) -> Dict[str, Any]:
        """Analyze user intent from request"""
        
        request_lower = user_request.lower()
        
        # Intent patterns
        intent_patterns = {
            "file_operation": ["read", "write", "create", "delete", "edit", "save", "open"],
            "code_development": ["code", "program", "debug", "compile", "test", "deploy"],
            "data_analysis": ["analyze", "process", "query", "visualize", "report", "export"],
            "system_administration": ["install", "configure", "monitor", "backup", "restart"],
            "web_interaction": ["browse", "navigate", "click", "input", "submit", "download"],
            "api_integration": ["api", "endpoint", "request", "response", "authentication"]
        }
        
        # Identify primary intent
        primary_intent = "general"
        intent_confidence = 0.0
        
        for intent, patterns in intent_patterns.items():
            pattern_matches = sum(1 for pattern in patterns if pattern in request_lower)
            if pattern_matches > 0:
                confidence = pattern_matches / len(patterns)
                if confidence > intent_confidence:
                    primary_intent = intent
                    intent_confidence = confidence
        
        return {
            "primary_intent": primary_intent,
            "intent_confidence": intent_confidence,
            "keywords_found": [word for word in request_lower.split() if len(word) > 3]
        }
    
    def _map_intent_to_tools(self, intent_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Map user intent to tool requirements"""
        
        intent = intent_analysis["primary_intent"]
        
        # Tool requirement mapping
        tool_mappings = {
            "file_operation": {
                "required_tools": ["file_server"],
                "optional_tools": ["code_editor"],
                "complexity_multiplier": 1.0
            },
            "code_development": {
                "required_tools": ["code_editor", "terminal"],
                "optional_tools": ["file_server", "database"],
                "complexity_multiplier": 1.5
            },
            "data_analysis": {
                "required_tools": ["database", "terminal"],
                "optional_tools": ["file_server", "code_editor"],
                "complexity_multiplier": 1.3
            },
            "system_administration": {
                "required_tools": ["terminal"],
                "optional_tools": ["database", "file_server"],
                "complexity_multiplier": 1.2
            },
            "web_interaction": {
                "required_tools": ["web_browser"],
                "optional_tools": ["api_client"],
                "complexity_multiplier": 1.0
            },
            "api_integration": {
                "required_tools": ["api_client"],
                "optional_tools": ["terminal", "file_server"],
                "complexity_multiplier": 1.4
            }
        }
        
        base_requirements = tool_mappings.get(intent, {
            "required_tools": [],
            "optional_tools": [],
            "complexity_multiplier": 1.0
        })
        
        return {
            "required_tools": base_requirements["required_tools"].copy(),
            "optional_tools": base_requirements["optional_tools"].copy(),
            "complexity_multiplier": base_requirements["complexity_multiplier"]
        }
    
    def _evaluate_tool_fitness(
        self,
        available_tools: List[str],
        tool_requirements: Dict[str, Any],
        agent_config: Any
    ) -> Dict[str, float]:
        """Evaluate fitness of available tools for the requirements"""
        
        tool_fitness = {}
        
        for tool in available_tools:
            fitness_score = 0.0
            
            # Base fitness based on requirements
            if tool in tool_requirements.get("required_tools", []):
                fitness_score += 0.8
            elif tool in tool_requirements.get("optional_tools", []):
                fitness_score += 0.6
            else:
                fitness_score += 0.2
            
            # Agent preference bonus
            if agent_config and hasattr(agent_config, 'tool_preferences') and tool in agent_config.tool_preferences:
                fitness_score += 0.2
            
            # Domain expertise bonus
            if agent_config and hasattr(agent_config, 'expertise_domains'):
                if any(domain in tool for domain in agent_config.expertise_domains):
                    fitness_score += 0.1
            
            # Normalize fitness score
            tool_fitness[tool] = min(1.0, fitness_score)
        
        return tool_fitness
    
    def _select_optimal_tool(self, tool_fitness: Dict[str, float], tool_requirements: Dict[str, Any]) -> str:
        """Select the optimal tool based on fitness scores"""
        
        if not tool_fitness:
            return "unknown"
        
        # Sort tools by fitness score
        sorted_tools = sorted(
            tool_fitness.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top candidate
        return sorted_tools[0][0]
    
    def _generate_tool_selection_reasoning(
        self,
        selected_tool: str,
        intent_analysis: Dict[str, Any],
        tool_requirements: Dict[str, Any],
        tool_fitness: Dict[str, float]
    ) -> str:
        """Generate reasoning for tool selection"""
        
        reasoning_parts = []
        
        # Intent-based reasoning
        primary_intent = intent_analysis["primary_intent"]
        reasoning_parts.append(f"Selected {selected_tool} based on primary intent: {primary_intent}")
        
        # Requirement-based reasoning
        if selected_tool in tool_requirements.get("required_tools", []):
            reasoning_parts.append(f"{selected_tool} is a required tool for this task")
        elif selected_tool in tool_requirements.get("optional_tools", []):
            reasoning_parts.append(f"{selected_tool} is an optional but suitable tool")
        
        # Fitness-based reasoning
        fitness_score = tool_fitness.get(selected_tool, 0.0)
        reasoning_parts.append(f"Tool fitness score: {fitness_score:.2f}")
        
        return ". ".join(reasoning_parts)
 
    def simulate_mcp_tool_selection(
        self,
        agent_id: str,
        user_request: str,
        available_tools: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate MCP tool selection for an agent with fallback handling"""

        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "Agent not found"}

        # Analyze user intent
        intent_analysis = self._analyze_user_intent(user_request)

        # Map intent to tool requirements
        tool_requirements = self._map_intent_to_tools(intent_analysis, context)

        # Evaluate tool fitness
        tool_fitness = self._evaluate_tool_fitness(
            available_tools, tool_requirements, agent.agent_config
        )

        # Select optimal tool
        selected_tool = self._select_optimal_tool(tool_fitness, tool_requirements)

        # Generate selection reasoning
        reasoning = self._generate_tool_selection_reasoning(
            selected_tool, intent_analysis, tool_requirements, tool_fitness
        )

        # Check if fallback is needed
        fallback_info = self._check_fallback_needed(
            selected_tool, tool_fitness, tool_requirements, context
        )

        # Apply fallback strategy if needed
        if fallback_info["fallback_needed"]:
            fallback_result = self._apply_fallback_strategy(
                fallback_info, available_tools, tool_requirements, context
            )
            selected_tool = fallback_result["selected_tool"]
            reasoning = fallback_result["reasoning"]
            tool_fitness = fallback_result["updated_fitness"]

        return {
            "agent_id": agent_id,
            "selected_tool": selected_tool,
            "selection_reasoning": reasoning,
            "confidence_score": tool_fitness.get(selected_tool, 0.0),
            "intent_analysis": intent_analysis,
            "tool_requirements": tool_requirements,
            "tool_fitness_scores": tool_fitness,
            "alternative_tools": [tool for tool, score in tool_fitness.items() if score > 0.5 and tool != selected_tool],
            "fallback_info": fallback_info,
            "fallback_applied": fallback_info["fallback_needed"],
            "selection_context": {
                "user_expertise_level": context.get("user_expertise_level", "intermediate"),
                "task_complexity": context.get("task_complexity", "medium"),
                "available_resources": context.get("available_resources", []),
                "time_constraints": context.get("time_constraints", "moderate"),
                "workspace_type": context.get("workspace_type", "general")
            }
        }

    def _check_fallback_needed(
        self,
        selected_tool: str,
        tool_fitness: Dict[str, float],
        tool_requirements: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if fallback strategy is needed"""
        
        confidence_threshold = context.get("confidence_threshold", 0.6)
        selected_tool_confidence = tool_fitness.get(selected_tool, 0.0)
        
        # Check if selected tool meets confidence threshold
        if selected_tool_confidence < confidence_threshold:
            return {
                "fallback_needed": True,
                "reason": "low_confidence",
                "confidence_gap": confidence_threshold - selected_tool_confidence,
                "strategy": "skip"  # User requested skip strategy for data generation
            }
        
        # Check if required tools are available
        required_tools = tool_requirements.get("required_tools", [])
        available_required = [tool for tool in required_tools if tool in tool_fitness]
        
        if len(available_required) < len(required_tools):
            return {
                "fallback_needed": True,
                "reason": "missing_required_tools",
                "missing_tools": [tool for tool in required_tools if tool not in tool_fitness],
                "strategy": "skip"
            }
        
        # Check context constraints
        if context.get("strict_quality", False) and selected_tool_confidence < 0.8:
            return {
                "fallback_needed": True,
                "reason": "strict_quality_requirement",
                "quality_gap": 0.8 - selected_tool_confidence,
                "strategy": "skip"
            }
        
        return {
            "fallback_needed": False,
            "reason": "no_fallback_needed",
            "strategy": None
        }

    def _apply_fallback_strategy(
        self,
        fallback_info: Dict[str, Any],
        available_tools: List[str],
        tool_requirements: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply fallback strategy based on fallback info"""
        
        strategy = fallback_info.get("strategy", "skip")
        
        if strategy == "skip":
            # User requested skip strategy - return unknown tool to mark data as unsuitable
            return {
                "selected_tool": "unknown",
                "reasoning": f"Fallback applied: {fallback_info.get('reason', 'unknown')}. Data quality insufficient for training.",
                "updated_fitness": {"unknown": 0.0}
            }
        
        elif strategy == "alternative_tool":
            # Find alternative tool with highest fitness
            alternative_tools = [tool for tool in available_tools if tool != "unknown"]
            if alternative_tools:
                # Create a simple fitness mapping for alternative tools
                alt_fitness = {tool: 0.5 for tool in alternative_tools}
                best_alternative = max(alternative_tools, key=lambda t: alt_fitness.get(t, 0.0))
                return {
                    "selected_tool": best_alternative,
                    "reasoning": f"Fallback to alternative tool: {best_alternative}",
                    "updated_fitness": alt_fitness
                }
        
        elif strategy == "degraded_mode":
            # Use selected tool but with degraded confidence
            selected_tool = fallback_info.get("selected_tool", "unknown")
            # Create degraded fitness scores
            degraded_fitness = {tool: 0.3 for tool in available_tools if tool != "unknown"}
            if selected_tool != "unknown":
                degraded_fitness[selected_tool] = 0.4  # Slightly higher for selected tool
            return {
                "selected_tool": selected_tool,
                "reasoning": f"Fallback to degraded mode for tool: {selected_tool}",
                "updated_fitness": degraded_fitness
            }
        
        # Default fallback to skip
        return {
            "selected_tool": "unknown",
            "reasoning": "Fallback strategy failed, skipping data generation",
            "updated_fitness": {"unknown": 0.0}
        }