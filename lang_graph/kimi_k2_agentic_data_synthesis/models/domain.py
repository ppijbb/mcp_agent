"""
Domain models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class ComplexityLevel(str, Enum):
    """Complexity levels for domains and scenarios"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DomainCategory(str, Enum):
    """Domain categories"""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    CREATIVE = "creative"
    SCIENTIFIC = "scientific"
    SOCIAL = "social"
    GAMING = "gaming"
    PRODUCTIVITY = "productivity"


class Criteria(BaseModel):
    """Evaluation criteria for scenarios"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    evaluation_type: str  # "accuracy", "completeness", "creativity", etc.
    scoring_scale: int = Field(default=5, ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Task Completion",
                "description": "Whether the agent successfully completed the assigned task",
                "weight": 0.4,
                "evaluation_type": "accuracy",
                "scoring_scale": 5
            }
        }


class Step(BaseModel):
    """Individual step in a scenario"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    description: str
    expected_action: str
    required_tools: List[str] = []
    expected_outcome: str
    difficulty: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    time_limit: Optional[int] = None  # in seconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "description": "Search for relevant information about the topic",
                "expected_action": "Use search tools to find information",
                "required_tools": ["web_search", "file_search"],
                "expected_outcome": "Found relevant information",
                "difficulty": "intermediate"
            }
        }


class Scenario(BaseModel):
    """Scenario definition for a domain"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain_id: str
    name: str
    description: str
    steps: List[Step]
    expected_outcome: str
    difficulty_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    evaluation_rubric: List[Criteria]
    estimated_duration: int = Field(ge=1)  # in minutes
    required_tools: List[str] = []
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "domain_id": "tech_support",
                "name": "Troubleshoot Network Issues",
                "description": "Help user diagnose and fix network connectivity problems",
                "steps": [],
                "expected_outcome": "Network issues resolved",
                "difficulty_level": "intermediate",
                "evaluation_rubric": [],
                "estimated_duration": 15,
                "required_tools": ["network_diagnostics", "system_info"],
                "tags": ["networking", "troubleshooting"]
            }
        }


class Domain(BaseModel):
    """Domain definition for the synthesis system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    category: DomainCategory
    complexity_level: ComplexityLevel = ComplexityLevel.INTERMEDIATE
    required_tools: List[str] = []
    scenarios: List[Scenario] = []
    evaluation_criteria: List[Criteria] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Technical Support",
                "description": "Technical support and troubleshooting scenarios",
                "category": "technology",
                "complexity_level": "intermediate",
                "required_tools": ["system_diagnostics", "log_analysis", "remote_access"],
                "scenarios": [],
                "evaluation_criteria": [],
                "metadata": {
                    "industry": "IT Support",
                    "target_audience": "Technical Support Engineers"
                }
            }
        }
    
    def add_scenario(self, scenario: Scenario) -> None:
        """Add a scenario to the domain"""
        scenario.domain_id = self.id
        self.scenarios.append(scenario)
        self.updated_at = datetime.utcnow()
    
    def get_scenarios_by_difficulty(self, difficulty: ComplexityLevel) -> List[Scenario]:
        """Get scenarios by difficulty level"""
        return [s for s in self.scenarios if s.difficulty_level == difficulty]
    
    def get_scenarios_by_tools(self, tools: List[str]) -> List[Scenario]:
        """Get scenarios that require specific tools"""
        return [s for s in self.scenarios if any(tool in s.required_tools for tool in tools)] 