"""
Agent models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class AgentRole(str, Enum):
    """Agent roles in the system"""
    ASSISTANT = "assistant"
    USER = "user"
    EXPERT = "expert"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class PersonalityType(str, Enum):
    """Agent personality types"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"
    INNOVATIVE = "innovative"


class CommunicationStyle(str, Enum):
    """Agent communication styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    DIRECT = "direct"
    DETAILED = "detailed"


class Rule(BaseModel):
    """Rule for agent behavior"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    condition: str  # Condition when rule applies
    action: str  # Action to take
    priority: int = Field(default=1, ge=1, le=10)
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Always verify information",
                "description": "Always verify information before providing it to users",
                "condition": "when providing factual information",
                "action": "search for verification sources",
                "priority": 5
            }
        }


class BehaviorPattern(BaseModel):
    """Behavior pattern for agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    personality_type: PersonalityType
    communication_style: CommunicationStyle
    decision_making_approach: str
    problem_solving_strategy: str
    collaboration_preference: str
    rules: List[Rule] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Analytical Expert",
                "description": "Analytical expert who focuses on systematic problem solving",
                "personality_type": "analytical",
                "communication_style": "technical",
                "decision_making_approach": "data-driven analysis",
                "problem_solving_strategy": "systematic decomposition",
                "collaboration_preference": "structured teamwork"
            }
        }


class Metrics(BaseModel):
    """Performance metrics for agents"""
    total_sessions: int = 0
    successful_sessions: int = 0
    average_session_duration: float = 0.0
    tool_usage_count: Dict[str, int] = {}
    success_rate: float = 0.0
    user_satisfaction_score: float = 0.0
    response_time_average: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def update_success_rate(self) -> None:
        """Update success rate based on current metrics"""
        if self.total_sessions > 0:
            self.success_rate = self.successful_sessions / self.total_sessions
        self.last_updated = datetime.utcnow()
    
    def add_session(self, successful: bool, duration: float) -> None:
        """Add a new session to metrics"""
        self.total_sessions += 1
        if successful:
            self.successful_sessions += 1
        
        # Update average duration
        self.average_session_duration = (
            (self.average_session_duration * (self.total_sessions - 1)) + duration
        ) / self.total_sessions
        
        self.update_success_rate()
    
    def record_tool_usage(self, tool_name: str) -> None:
        """Record tool usage"""
        self.tool_usage_count[tool_name] = self.tool_usage_count.get(tool_name, 0) + 1
        self.last_updated = datetime.utcnow()


class AgentProfile(BaseModel):
    """Profile information for an agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: AgentRole
    expertise_areas: List[str] = []
    background: str
    skills: List[str] = []
    preferences: Dict[str, Any] = {}
    limitations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Tech Support Specialist",
                "role": "specialist",
                "expertise_areas": ["networking", "system administration", "troubleshooting"],
                "background": "10+ years in IT support and system administration",
                "skills": ["network diagnostics", "log analysis", "remote support"],
                "preferences": {
                    "communication_style": "technical",
                    "problem_solving": "systematic"
                },
                "limitations": ["no access to physical hardware", "limited to software issues"]
            }
        }


class AgentConfig(BaseModel):
    """Configuration for creating agents"""
    agent_id: str
    name: str
    description: str
    agent_type: str = "EXPERT"  # Using string instead of enum for flexibility
    behavior_pattern: str = "COLLABORATIVE"
    expertise_domains: List[str] = []
    tool_preferences: List[str] = []
    communication_style: str = "professional"
    problem_solving_approach: str = "systematic"
    collaboration_style: str = "mentoring"
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "senior_developer",
                "name": "Senior Developer",
                "description": "Experienced software developer with expertise in multiple languages",
                "agent_type": "EXPERT",
                "behavior_pattern": "COLLABORATIVE",
                "expertise_domains": ["web_development", "software_engineering"],
                "tool_preferences": ["code_editor", "terminal", "git"],
                "communication_style": "professional",
                "problem_solving_approach": "systematic",
                "collaboration_style": "mentoring"
            }
        }


class Agent(BaseModel):
    """Agent definition for the synthesis system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    profile: AgentProfile
    tool_set: List[str] = []
    behavior_pattern: BehaviorPattern
    interaction_rules: List[Rule] = []
    performance_metrics: Metrics = Field(default_factory=Metrics)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Network Troubleshooter Agent",
                "profile": {},
                "tool_set": ["network_diagnostics", "system_info", "log_analyzer"],
                "behavior_pattern": {},
                "interaction_rules": [],
                "performance_metrics": {},
                "is_active": True
            }
        }
    
    def add_tool(self, tool_id: str) -> None:
        """Add a tool to the agent's tool set"""
        if tool_id not in self.tool_set:
            self.tool_set.append(tool_id)
            self.updated_at = datetime.utcnow()
    
    def remove_tool(self, tool_id: str) -> None:
        """Remove a tool from the agent's tool set"""
        if tool_id in self.tool_set:
            self.tool_set.remove(tool_id)
            self.updated_at = datetime.utcnow()
    
    def add_rule(self, rule: Rule) -> None:
        """Add an interaction rule to the agent"""
        self.interaction_rules.append(rule)
        self.updated_at = datetime.utcnow()
    
    def get_rules_by_priority(self) -> List[Rule]:
        """Get interaction rules sorted by priority"""
        return sorted(self.interaction_rules, key=lambda x: x.priority, reverse=True)
    
    def has_tool(self, tool_id: str) -> bool:
        """Check if agent has access to a specific tool"""
        return tool_id in self.tool_set
    
    def get_expertise_match(self, domain: str) -> float:
        """Calculate expertise match for a domain (0.0 to 1.0)"""
        if not self.profile.expertise_areas:
            return 0.0
        
        # Simple matching - can be enhanced with semantic similarity
        domain_lower = domain.lower()
        matches = sum(1 for area in self.profile.expertise_areas 
                     if area.lower() in domain_lower or domain_lower in area.lower())
        
        return matches / len(self.profile.expertise_areas)
    
    def update_metrics(self, session_data: Dict[str, Any]) -> None:
        """Update performance metrics with session data"""
        successful = session_data.get('successful', False)
        duration = session_data.get('duration', 0.0)
        tools_used = session_data.get('tools_used', [])
        
        self.performance_metrics.add_session(successful, duration)
        
        for tool in tools_used:
            self.performance_metrics.record_tool_usage(tool)
        
        self.updated_at = datetime.utcnow() 