"""
Agent Factory for the Kimi-K2 Agentic Data Synthesis System

Creates and manages diverse agents with different tool sets and behavior patterns.
"""

from typing import List, Dict, Any, Optional
from ..models.agent import Agent, AgentProfile, BehaviorPattern, Rule, AgentRole, PersonalityType, CommunicationStyle
from ..models.tool import Tool
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
    
    def __init__(self, tool_registry=None):
        self.agents: Dict[str, Agent] = {}
        self.agent_templates: Dict[str, Dict[str, Any]] = {}
        self.behavior_templates: Dict[str, BehaviorPattern] = {}
        self.tool_registry = tool_registry
        
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
                "role": AgentRole.SPECIALIST,
                "expertise_areas": ["networking", "system administration", "troubleshooting"],
                "background": "10+ years in IT support and system administration",
                "skills": ["network diagnostics", "log analysis", "remote support"],
                "preferences": {
                    "communication_style": "technical",
                    "problem_solving": "systematic"
                },
                "limitations": ["no access to physical hardware", "limited to software issues"],
                "behavior_pattern": "analytical_expert",
                "default_tools": ["system_diagnostics", "log_analysis", "remote_access"]
            },
            "data_analyst": {
                "name": "Data Analyst",
                "role": AgentRole.EXPERT,
                "expertise_areas": ["data analysis", "statistics", "business intelligence"],
                "background": "8+ years in data analysis and business intelligence",
                "skills": ["statistical analysis", "data visualization", "reporting"],
                "preferences": {
                    "communication_style": "detailed",
                    "problem_solving": "analytical"
                },
                "limitations": ["requires structured data", "limited to quantitative analysis"],
                "behavior_pattern": "analytical_expert",
                "default_tools": ["data_analysis", "spreadsheet_tools", "reporting"]
            },
            "creative_writer": {
                "name": "Creative Writer",
                "role": AgentRole.SPECIALIST,
                "expertise_areas": ["creative writing", "content creation", "storytelling"],
                "background": "5+ years in creative writing and content creation",
                "skills": ["storytelling", "content editing", "style adaptation"],
                "preferences": {
                    "communication_style": "creative",
                    "problem_solving": "intuitive"
                },
                "limitations": ["requires creative input", "subjective quality assessment"],
                "behavior_pattern": "creative_problem_solver",
                "default_tools": ["text_generation", "style_analysis", "content_editing"]
            },
            "project_coordinator": {
                "name": "Project Coordinator",
                "role": AgentRole.COORDINATOR,
                "expertise_areas": ["project management", "coordination", "workflow optimization"],
                "background": "7+ years in project management and team coordination",
                "skills": ["project planning", "resource allocation", "progress tracking"],
                "preferences": {
                    "communication_style": "formal",
                    "problem_solving": "systematic"
                },
                "limitations": ["requires clear objectives", "limited to coordination tasks"],
                "behavior_pattern": "systematic_coordinator",
                "default_tools": ["project_management", "scheduling", "communication"]
            }
        }
    
    def create_agent(self, name: str, role: AgentRole, expertise_areas: List[str],
                    background: str, skills: List[str], behavior_pattern_name: str = None,
                    tool_set: List[str] = None, preferences: Dict[str, Any] = None,
                    limitations: List[str] = None) -> Agent:
        """Create a new agent"""
        # Create agent profile
        profile = AgentProfile(
            name=name,
            role=role,
            expertise_areas=expertise_areas,
            background=background,
            skills=skills,
            preferences=preferences or {},
            limitations=limitations or []
        )
        
        # Get or create behavior pattern
        if behavior_pattern_name and behavior_pattern_name in self.behavior_templates:
            behavior_pattern = self.behavior_templates[behavior_pattern_name]
        else:
            # Use default analytical pattern
            behavior_pattern = self.behavior_templates["analytical_expert"]
        
        # Create agent
        agent = Agent(
            name=name,
            profile=profile,
            tool_set=tool_set or [],
            behavior_pattern=behavior_pattern,
            interaction_rules=behavior_pattern.rules.copy()
        )
        
        self.agents[agent.id] = agent
        logger.info(f"Created agent: {name} (ID: {agent.id})")
        
        return agent
    
    def create_agent_from_template(self, template_name: str, customizations: Dict[str, Any] = None) -> Optional[Agent]:
        """Create an agent from a predefined template"""
        if template_name not in self.agent_templates:
            logger.warning(f"Agent template not found: {template_name}")
            return None
        
        template = self.agent_templates[template_name]
        customizations = customizations or {}
        
        # Merge template with customizations
        agent_config = template.copy()
        agent_config.update(customizations)
        
        # Create agent
        agent = self.create_agent(
            name=agent_config["name"],
            role=agent_config["role"],
            expertise_areas=agent_config["expertise_areas"],
            background=agent_config["background"],
            skills=agent_config["skills"],
            behavior_pattern_name=agent_config.get("behavior_pattern"),
            tool_set=agent_config.get("default_tools", []),
            preferences=agent_config.get("preferences", {}),
            limitations=agent_config.get("limitations", [])
        )
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """Get an agent by name"""
        for agent in self.agents.values():
            if agent.name.lower() == name.lower():
                return agent
        return None
    
    def list_agents(self, role: Optional[AgentRole] = None, active_only: bool = True) -> List[Agent]:
        """List agents with optional filtering"""
        agents = list(self.agents.values())
        
        if role:
            agents = [a for a in agents if a.profile.role == role]
        
        if active_only:
            agents = [a for a in agents if a.is_active]
        
        return agents
    
    def update_agent(self, agent_id: str, **kwargs) -> bool:
        """Update an agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        for key, value in kwargs.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
            elif hasattr(agent.profile, key):
                setattr(agent.profile, key, value)
        
        agent.updated_at = datetime.utcnow()
        logger.info(f"Updated agent: {agent.name}")
        return True
    
    def assign_tools_to_agent(self, agent_id: str, tool_ids: List[str]) -> bool:
        """Assign tools to an agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        # Validate tools exist
        if self.tool_registry:
            for tool_id in tool_ids:
                if not self.tool_registry.get_tool(tool_id):
                    logger.warning(f"Tool not found: {tool_id}")
                    return False
        
        agent.tool_set = tool_ids
        agent.updated_at = datetime.utcnow()
        logger.info(f"Assigned {len(tool_ids)} tools to agent: {agent.name}")
        return True
    
    def add_tool_to_agent(self, agent_id: str, tool_id: str) -> bool:
        """Add a tool to an agent's tool set"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        agent.add_tool(tool_id)
        logger.info(f"Added tool {tool_id} to agent: {agent.name}")
        return True
    
    def remove_tool_from_agent(self, agent_id: str, tool_id: str) -> bool:
        """Remove a tool from an agent's tool set"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        agent.remove_tool(tool_id)
        logger.info(f"Removed tool {tool_id} from agent: {agent.name}")
        return True
    
    def add_rule_to_agent(self, agent_id: str, rule: Rule) -> bool:
        """Add an interaction rule to an agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        agent.add_rule(rule)
        logger.info(f"Added rule to agent: {agent.name}")
        return True
    
    def get_agents_for_domain(self, domain: str) -> List[Agent]:
        """Get agents suitable for a specific domain"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.is_active:
                # Check expertise match
                expertise_match = agent.get_expertise_match(domain)
                if expertise_match > 0.3:  # Threshold for domain suitability
                    suitable_agents.append(agent)
        
        # Sort by expertise match
        suitable_agents.sort(key=lambda a: a.get_expertise_match(domain), reverse=True)
        return suitable_agents
    
    def get_agents_with_tools(self, tool_ids: List[str]) -> List[Agent]:
        """Get agents that have access to specific tools"""
        matching_agents = []
        
        for agent in self.agents.values():
            if agent.is_active:
                has_all_tools = all(agent.has_tool(tool_id) for tool_id in tool_ids)
                if has_all_tools:
                    matching_agents.append(agent)
        
        return matching_agents
    
    def create_specialized_agent(self, domain: str, required_tools: List[str],
                                behavior_preference: str = None) -> Optional[Agent]:
        """Create a specialized agent for a specific domain"""
        # Find suitable behavior pattern
        if behavior_preference and behavior_preference in self.behavior_templates:
            behavior_pattern = self.behavior_templates[behavior_preference]
        else:
            # Choose based on domain characteristics
            if domain in ["technology", "engineering", "science"]:
                behavior_pattern = self.behavior_templates["analytical_expert"]
            elif domain in ["creative", "art", "design"]:
                behavior_pattern = self.behavior_templates["creative_problem_solver"]
            elif domain in ["management", "coordination"]:
                behavior_pattern = self.behavior_templates["systematic_coordinator"]
            else:
                behavior_pattern = self.behavior_templates["adaptive_specialist"]
        
        # Create specialized profile
        profile = AgentProfile(
            name=f"{domain.title()} Specialist",
            role=AgentRole.SPECIALIST,
            expertise_areas=[domain],
            background=f"Specialized in {domain} with extensive experience",
            skills=[f"{domain}_analysis", f"{domain}_problem_solving"],
            preferences={
                "communication_style": behavior_pattern.communication_style.value,
                "problem_solving": behavior_pattern.problem_solving_strategy
            },
            limitations=[f"Limited to {domain} domain"]
        )
        
        # Create agent
        agent = Agent(
            name=profile.name,
            profile=profile,
            tool_set=required_tools,
            behavior_pattern=behavior_pattern,
            interaction_rules=behavior_pattern.rules.copy()
        )
        
        self.agents[agent.id] = agent
        logger.info(f"Created specialized agent for {domain}: {agent.name}")
        
        return agent
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about all agents"""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.is_active]),
            "agents_by_role": {},
            "agents_by_personality": {},
            "average_tools_per_agent": 0.0,
            "most_skilled_agents": []
        }
        
        total_tools = 0
        
        for agent in self.agents.values():
            # Role statistics
            role = agent.profile.role.value
            stats["agents_by_role"][role] = stats["agents_by_role"].get(role, 0) + 1
            
            # Personality statistics
            personality = agent.behavior_pattern.personality_type.value
            stats["agents_by_personality"][personality] = stats["agents_by_personality"].get(personality, 0) + 1
            
            # Tool statistics
            total_tools += len(agent.tool_set)
        
        if stats["total_agents"] > 0:
            stats["average_tools_per_agent"] = total_tools / stats["total_agents"]
        
        # Most skilled agents (by number of skills)
        agents_with_skills = [(a, len(a.profile.skills)) for a in self.agents.values()]
        agents_with_skills.sort(key=lambda x: x[1], reverse=True)
        stats["most_skilled_agents"] = [
            {"name": agent.name, "skills_count": skills_count}
            for agent, skills_count in agents_with_skills[:10]
        ]
        
        return stats
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        agent.is_active = False
        agent.updated_at = datetime.utcnow()
        logger.info(f"Deactivated agent: {agent.name}")
        return True
    
    def activate_agent(self, agent_id: str) -> bool:
        """Activate an agent"""
        agent = self.get_agent(agent_id)
        if not agent:
            logger.warning(f"Agent not found: {agent_id}")
            return False
        
        agent.is_active = True
        agent.updated_at = datetime.utcnow()
        logger.info(f"Activated agent: {agent.name}")
        return True
    
    def validate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Validate an agent configuration"""
        agent = self.get_agent(agent_id)
        if not agent:
            return {"valid": False, "errors": ["Agent not found"]}
        
        errors = []
        warnings = []
        
        # Validate agent structure
        if not agent.name:
            errors.append("Agent name is required")
        
        if not agent.profile.expertise_areas:
            warnings.append("Agent has no expertise areas")
        
        if not agent.tool_set:
            warnings.append("Agent has no tools assigned")
        
        # Validate tool availability
        if self.tool_registry:
            for tool_id in agent.tool_set:
                if not self.tool_registry.get_tool(tool_id):
                    warnings.append(f"Tool {tool_id} not found in registry")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "tool_count": len(agent.tool_set),
            "rule_count": len(agent.interaction_rules)
        } 