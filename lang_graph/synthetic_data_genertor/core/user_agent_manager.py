"""
User Agent Manager for the Kimi-K2 Agentic Data Synthesis System

Manages user agents, their behavior patterns, and user request simulation.
"""

from typing import List, Dict, Any, Optional
from ..models.agent import Agent, AgentProfile, BehaviorPattern, AgentRole, PersonalityType, CommunicationStyle
from ..models.simulation import SimulationSession
import random
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UserAgentManager:
    """
    Manages user agents and their interactions.
    
    Responsibilities:
    - User agent creation and management
    - User behavior pattern simulation
    - User request generation
    - User feedback simulation
    """
    
    def __init__(self):
        self.user_agents: Dict[str, Agent] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.user_behavior_patterns: Dict[str, BehaviorPattern] = {}
        self.user_scenarios: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize with default user types
        self._initialize_user_types()
    
    def _initialize_user_types(self) -> None:
        """Initialize default user types and behavior patterns"""
        self.user_behavior_patterns = {
            "novice_user": BehaviorPattern(
                name="Novice User",
                description="Novice user who needs guidance and explanations",
                personality_type=PersonalityType.ADAPTIVE,
                communication_style=CommunicationStyle.FRIENDLY,
                decision_making_approach="seeking guidance",
                problem_solving_strategy="ask for help",
                collaboration_preference="guided assistance"
            ),
            "expert_user": BehaviorPattern(
                name="Expert User",
                description="Expert user who is knowledgeable and efficient",
                personality_type=PersonalityType.ANALYTICAL,
                communication_style=CommunicationStyle.TECHNICAL,
                decision_making_approach="independent analysis",
                problem_solving_strategy="self-directed",
                collaboration_preference="efficient collaboration"
            ),
            "casual_user": BehaviorPattern(
                name="Casual User",
                description="Casual user who prefers simple solutions",
                personality_type=PersonalityType.CREATIVE,
                communication_style=CommunicationStyle.CASUAL,
                decision_making_approach="intuitive choice",
                problem_solving_strategy="simple solutions",
                collaboration_preference="friendly interaction"
            ),
            "business_user": BehaviorPattern(
                name="Business User",
                description="Business user focused on efficiency and results",
                personality_type=PersonalityType.SYSTEMATIC,
                communication_style=CommunicationStyle.FORMAL,
                decision_making_approach="results-oriented",
                problem_solving_strategy="practical approach",
                collaboration_preference="professional interaction"
            )
        }
        
        # Initialize user profiles
        self.user_profiles = {
            "tech_novice": {
                "name": "Tech Novice",
                "role": AgentRole.USER,
                "expertise_areas": ["basic computing"],
                "background": "Limited technical experience, learning new technologies",
                "skills": ["basic computer usage", "web browsing"],
                "preferences": {
                    "communication_style": "friendly",
                    "learning_style": "step_by_step"
                },
                "limitations": ["limited technical knowledge", "needs clear explanations"],
                "behavior_pattern": "novice_user",
                "typical_requests": [
                    "How do I connect to WiFi?",
                    "My computer is running slow, what should I do?",
                    "I can't find my files, can you help?",
                    "What's the best way to organize my photos?"
                ]
            },
            "tech_expert": {
                "name": "Tech Expert",
                "role": AgentRole.USER,
                "expertise_areas": ["advanced computing", "programming", "system administration"],
                "background": "Extensive technical experience, works in IT",
                "skills": ["programming", "system administration", "network configuration"],
                "preferences": {
                    "communication_style": "technical",
                    "efficiency": "high"
                },
                "limitations": ["expects technical accuracy", "values efficiency"],
                "behavior_pattern": "expert_user",
                "typical_requests": [
                    "Analyze this network configuration for security issues",
                    "Optimize this database query for better performance",
                    "Review this code for potential bugs",
                    "Set up automated deployment pipeline"
                ]
            },
            "business_professional": {
                "name": "Business Professional",
                "role": AgentRole.USER,
                "expertise_areas": ["business operations", "project management"],
                "background": "Business professional with focus on efficiency",
                "skills": ["project management", "data analysis", "communication"],
                "preferences": {
                    "communication_style": "formal",
                    "focus": "results"
                },
                "limitations": ["limited technical depth", "time-constrained"],
                "behavior_pattern": "business_user",
                "typical_requests": [
                    "Create a project timeline for this initiative",
                    "Analyze quarterly sales data and provide insights",
                    "Prepare a presentation for stakeholders",
                    "Optimize our customer service workflow"
                ]
            },
            "creative_professional": {
                "name": "Creative Professional",
                "role": AgentRole.USER,
                "expertise_areas": ["creative work", "content creation"],
                "background": "Creative professional working in media and design",
                "skills": ["content creation", "design", "storytelling"],
                "preferences": {
                    "communication_style": "creative",
                    "approach": "innovative"
                },
                "limitations": ["may lack technical precision", "prefers creative solutions"],
                "behavior_pattern": "casual_user",
                "typical_requests": [
                    "Help me brainstorm ideas for a marketing campaign",
                    "Review and improve this content for better engagement",
                    "Suggest creative ways to present this data",
                    "Help me develop a brand voice for my business"
                ]
            }
        }
    
    def create_user_agent(self, profile_name: str, customizations: Dict[str, Any] = None) -> Optional[Agent]:
        """Create a user agent from a predefined profile"""
        if profile_name not in self.user_profiles:
            logger.warning(f"User profile not found: {profile_name}")
            return None
        
        profile_data = self.user_profiles[profile_name].copy()
        customizations = customizations or {}
        
        # Apply customizations
        profile_data.update(customizations)
        
        # Create agent profile
        profile = AgentProfile(
            name=profile_data["name"],
            role=profile_data["role"],
            expertise_areas=profile_data["expertise_areas"],
            background=profile_data["background"],
            skills=profile_data["skills"],
            preferences=profile_data["preferences"],
            limitations=profile_data["limitations"]
        )
        
        # Get behavior pattern
        behavior_pattern_name = profile_data.get("behavior_pattern", "novice_user")
        behavior_pattern = self.user_behavior_patterns.get(behavior_pattern_name)
        
        if not behavior_pattern:
            behavior_pattern = self.user_behavior_patterns["novice_user"]
        
        # Create user agent
        user_agent = Agent(
            name=profile.name,
            profile=profile,
            tool_set=[],  # User agents typically don't have tools
            behavior_pattern=behavior_pattern,
            interaction_rules=[]
        )
        
        self.user_agents[user_agent.id] = user_agent
        
        # Store additional user-specific data
        self.user_scenarios[user_agent.id] = profile_data.get("typical_requests", [])
        
        logger.info(f"Created user agent: {profile.name} (ID: {user_agent.id})")
        return user_agent
    
    def get_user_agent(self, agent_id: str) -> Optional[Agent]:
        """Get a user agent by ID"""
        return self.user_agents.get(agent_id)
    
    def list_user_agents(self) -> List[Agent]:
        """List all user agents"""
        return list(self.user_agents.values())
    
    def generate_user_request(self, user_agent_id: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate a user request based on the agent's profile"""
        user_agent = self.get_user_agent(user_agent_id)
        if not user_agent:
            return None
        
        # Get typical requests for this user
        typical_requests = self.user_scenarios.get(user_agent_id, [])
        
        if not typical_requests:
            # Generate generic request based on expertise areas
            expertise = user_agent.profile.expertise_areas[0] if user_agent.profile.expertise_areas else "general"
            typical_requests = [f"Help me with {expertise}"]
        
        # Select a request (can be enhanced with context-aware selection)
        selected_request = random.choice(typical_requests)
        
        # Customize request based on context
        if context:
            selected_request = self._customize_request(selected_request, context, user_agent)
        
        return selected_request
    
    def _customize_request(self, base_request: str, context: Dict[str, Any], user_agent: Agent) -> str:
        """Customize a request based on context and user agent"""
        # Simple customization - can be enhanced with more sophisticated logic
        if "urgency" in context:
            if context["urgency"] == "high":
                base_request = f"URGENT: {base_request}"
            elif context["urgency"] == "low":
                base_request = f"Whenever you have time: {base_request}"
        
        if "complexity" in context:
            if context["complexity"] == "simple":
                base_request = f"Simple question: {base_request}"
            elif context["complexity"] == "complex":
                base_request = f"Complex issue: {base_request}"
        
        return base_request
    
    def simulate_user_feedback(self, user_agent_id: str, response_quality: float) -> Dict[str, Any]:
        """Simulate user feedback based on response quality and user personality"""
        user_agent = self.get_user_agent(user_agent_id)
        if not user_agent:
            return {"satisfaction": 0.0, "feedback": "User agent not found"}
        
        # Base satisfaction on response quality
        base_satisfaction = response_quality
        
        # Adjust based on personality
        personality = user_agent.behavior_pattern.personality_type.value
        
        if personality == "analytical":
            # Analytical users are more critical
            satisfaction = base_satisfaction * 0.9
            feedback_style = "detailed_analysis"
        elif personality == "creative":
            # Creative users appreciate innovative solutions
            satisfaction = base_satisfaction * 1.1
            feedback_style = "appreciative"
        elif personality == "systematic":
            # Systematic users value consistency
            satisfaction = base_satisfaction * 0.95
            feedback_style = "structured"
        else:  # adaptive
            # Adaptive users are flexible
            satisfaction = base_satisfaction
            feedback_style = "flexible"
        
        # Generate feedback text
        feedback_text = self._generate_feedback_text(satisfaction, feedback_style, user_agent)
        
        return {
            "satisfaction": max(0.0, min(1.0, satisfaction)),
            "feedback": feedback_text,
            "personality": personality,
            "feedback_style": feedback_style
        }
    
    def _generate_feedback_text(self, satisfaction: float, feedback_style: str, user_agent: Agent) -> str:
        """Generate feedback text based on satisfaction and style"""
        if satisfaction >= 0.8:
            if feedback_style == "detailed_analysis":
                return "Excellent analysis. The solution addresses all the technical requirements comprehensively."
            elif feedback_style == "appreciative":
                return "Wow, this is exactly what I was looking for! Very creative approach."
            elif feedback_style == "structured":
                return "Perfect. The systematic approach and clear structure are exactly what I needed."
            else:
                return "Great! This solution works perfectly for my needs."
        
        elif satisfaction >= 0.6:
            if feedback_style == "detailed_analysis":
                return "Good effort, but I need more technical details to fully understand the solution."
            elif feedback_style == "appreciative":
                return "Thanks! It's helpful, though I was hoping for something more innovative."
            elif feedback_style == "structured":
                return "The approach is good, but I need more structure and organization."
            else:
                return "This is helpful, but I need some clarification."
        
        else:
            if feedback_style == "detailed_analysis":
                return "The response lacks the technical depth I require. Please provide more detailed analysis."
            elif feedback_style == "appreciative":
                return "I appreciate the effort, but this doesn't quite meet my creative needs."
            elif feedback_style == "structured":
                return "The response is too disorganized. I need a more systematic approach."
            else:
                return "This doesn't really address my needs. Can you try a different approach?"
    
    def create_user_scenario(self, user_agent_id: str, scenario: Dict[str, Any]) -> bool:
        """Create a custom scenario for a user agent"""
        if user_agent_id not in self.user_agents:
            return False
        
        if user_agent_id not in self.user_scenarios:
            self.user_scenarios[user_agent_id] = []
        
        self.user_scenarios[user_agent_id].append(scenario)
        return True
    
    def get_user_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about user agents"""
        stats = {
            "total_user_agents": len(self.user_agents),
            "agents_by_personality": {},
            "agents_by_expertise": {},
            "average_satisfaction": 0.0
        }
        
        # Count by personality
        for agent in self.user_agents.values():
            personality = agent.behavior_pattern.personality_type.value
            stats["agents_by_personality"][personality] = stats["agents_by_personality"].get(personality, 0) + 1
        
        # Count by expertise
        for agent in self.user_agents.values():
            for expertise in agent.profile.expertise_areas:
                stats["agents_by_expertise"][expertise] = stats["agents_by_expertise"].get(expertise, 0) + 1
        
        return stats
    
    def update_user_agent(self, user_agent_id: str, **kwargs) -> bool:
        """Update a user agent"""
        user_agent = self.get_user_agent(user_agent_id)
        if not user_agent:
            return False
        
        for key, value in kwargs.items():
            if hasattr(user_agent, key):
                setattr(user_agent, key, value)
            elif hasattr(user_agent.profile, key):
                setattr(user_agent.profile, key, value)
        
        user_agent.updated_at = datetime.utcnow()
        return True
    
    def delete_user_agent(self, user_agent_id: str) -> bool:
        """Delete a user agent"""
        if user_agent_id not in self.user_agents:
            return False
        
        agent_name = self.user_agents[user_agent_id].name
        del self.user_agents[user_agent_id]
        
        if user_agent_id in self.user_scenarios:
            del self.user_scenarios[user_agent_id]
        
        logger.info(f"Deleted user agent: {agent_name}")
        return True
    
    def get_user_agent_by_personality(self, personality_type: PersonalityType) -> List[Agent]:
        """Get user agents by personality type"""
        return [
            agent for agent in self.user_agents.values()
            if agent.behavior_pattern.personality_type == personality_type
        ]
    
    def get_user_agent_by_expertise(self, expertise: str) -> List[Agent]:
        """Get user agents by expertise area"""
        return [
            agent for agent in self.user_agents.values()
            if expertise in agent.profile.expertise_areas
        ] 