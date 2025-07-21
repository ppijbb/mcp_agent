"""
Agent Factory for the Kimi-K2 Agentic Data Synthesis System

Creates and manages diverse agents with different tool sets and behavior patterns.
"""

from typing import List, Dict, Any, Optional
from models.agent import Agent, AgentConfig, AgentProfile, BehaviorPattern, Rule, AgentRole, PersonalityType, CommunicationStyle
from models.tool import Tool
from agents.kimi_k2_agent import KimiK2ConversableAgent # Import the new conversable agent
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