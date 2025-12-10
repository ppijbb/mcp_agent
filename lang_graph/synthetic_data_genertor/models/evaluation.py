"""
Evaluation models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class EvaluationType(str, Enum):
    """Types of evaluation"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"
    USER_SATISFACTION = "user_satisfaction"
    TOOL_USAGE = "tool_usage"
    OVERALL = "overall"
    # MCP-specific evaluation types
    TOOL_SELECTION_ACCURACY = "tool_selection_accuracy"
    FUNCTION_CALL_SUCCESS_RATE = "function_call_success_rate"
    PARAMETER_ACCURACY = "parameter_accuracy"
    MCP_COMMUNICATION_RELIABILITY = "mcp_communication_reliability"


class Rubric(BaseModel):
    """Evaluation rubric for scenarios"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    criteria: List[Dict[str, Any]] = []
    total_weight: float = 1.0
    passing_threshold: float = 0.7
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Technical Support Quality Rubric",
                "description": "Evaluation criteria for technical support scenarios",
                "criteria": [
                    {
                        "name": "Problem Identification",
                        "weight": 0.3,
                        "description": "Correctly identifies the root cause of the issue"
                    },
                    {
                        "name": "Solution Effectiveness",
                        "weight": 0.4,
                        "description": "Provides effective solution to the problem"
                    },
                    {
                        "name": "Communication Quality",
                        "weight": 0.3,
                        "description": "Clear and professional communication"
                    }
                ],
                "total_weight": 1.0,
                "passing_threshold": 0.7
            }
        }
    
    def add_criterion(self, criterion: Dict[str, Any]) -> None:
        """Add a criterion to the rubric"""
        self.criteria.append(criterion)
        self.updated_at = datetime.utcnow()
    
    def validate_weights(self) -> bool:
        """Validate that criteria weights sum to total_weight"""
        total = sum(c.get('weight', 0) for c in self.criteria)
        return abs(total - self.total_weight) < 0.001


class QualityScore(BaseModel):
    """Quality score for evaluation results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluation_type: EvaluationType
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "evaluation_type": "accuracy",
                "score": 0.85,
                "confidence": 0.9,
                "reasoning": "Agent correctly identified the network issue and provided appropriate solution",
                "evidence": ["Correctly diagnosed DNS issue", "Provided working solution"]
            }
        }


class EvaluationRubric(BaseModel):
    """Rubric for evaluation configuration"""
    dimensions: List[Dict[str, Any]] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "dimensions": [
                    {
                        "name": "tool_usage_effectiveness",
                        "description": "How effectively tools are used to solve the problem",
                        "weight": 0.3,
                        "criteria": ["Appropriate tool selection", "Correct tool usage", "Efficient workflow"]
                    }
                ]
            }
        }


class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    evaluation_id: str
    name: str
    description: str
    rubric: EvaluationRubric
    llm_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    class Config:
        json_schema_extra = {
            "example": {
                "evaluation_id": "comprehensive_evaluation",
                "name": "Comprehensive Tool Usage Evaluation",
                "description": "Multi-dimensional evaluation of tool usage and collaboration",
                "rubric": {},
                "llm_model": "gemini-2.5-flash-lite",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }


class HumanFeedback(BaseModel):
    """Model for capturing human feedback on simulation results."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str
    rating: float = Field(ge=0.0, le=1.0, description="Overall quality rating from the human reviewer.")
    qualitative_feedback: str = Field(description="Detailed textual feedback from the reviewer.")
    bias_flags: List[str] = Field(default_factory=list, description="Flags for any detected biases (e.g., gender, race).")
    suggested_edits: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "reviewer_id": "human_expert_01",
                "rating": 0.9,
                "qualitative_feedback": "The agent's reasoning was sound, but the final code snippet had a minor, non-critical bug.",
                "bias_flags": [],
                "suggested_edits": "Change line 42 to `x > 0` instead of `x >= 0`."
            }
        }


class EvaluationResult(BaseModel):
    """Complete evaluation result"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    simulation_id: str
    rubric_id: str
    evaluator_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    individual_scores: List[QualityScore] = []
    passed: bool = False
    feedback: str
    recommendations: List[str] = []
    evaluation_time: float = 0.0  # in seconds
    human_feedback: Optional[HumanFeedback] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_id": "sim_123",
                "rubric_id": "rubric_456",
                "evaluator_id": "evaluator_789",
                "overall_score": 0.82,
                "passed": True,
                "feedback": "Good performance overall with room for improvement in communication",
                "recommendations": ["Improve response time", "Add more detailed explanations"]
            }
        }
    
    def add_score(self, score: QualityScore) -> None:
        """Add an individual quality score"""
        self.individual_scores.append(score)
    
    def calculate_overall_score(self) -> float:
        """Calculate overall score from individual scores"""
        if not self.individual_scores:
            return 0.0
        
        # Simple average - can be enhanced with weighted scoring
        total_score = sum(score.score for score in self.individual_scores)
        return total_score / len(self.individual_scores)
    
    def determine_pass_fail(self, threshold: float = 0.7) -> bool:
        """Determine if the evaluation passed based on threshold"""
        self.passed = self.overall_score >= threshold
        return self.passed
    
    def get_score_by_type(self, evaluation_type: EvaluationType) -> Optional[QualityScore]:
        """Get score by evaluation type"""
        for score in self.individual_scores:
            if score.evaluation_type == evaluation_type:
                return score
        return None
    
    def get_evidence_summary(self) -> List[str]:
        """Get summary of all evidence from individual scores"""
        evidence = []
        for score in self.individual_scores:
            evidence.extend(score.evidence)
        return evidence
    
    def get_recommendations_by_score(self) -> List[str]:
        """Generate recommendations based on low scores"""
        recommendations = []
        
        for score in self.individual_scores:
            if score.score < 0.6:
                if score.evaluation_type == EvaluationType.ACCURACY:
                    recommendations.append("Improve accuracy in problem diagnosis")
                elif score.evaluation_type == EvaluationType.COMPLETENESS:
                    recommendations.append("Provide more comprehensive solutions")
                elif score.evaluation_type == EvaluationType.CREATIVITY:
                    recommendations.append("Consider alternative approaches")
                elif score.evaluation_type == EvaluationType.EFFICIENCY:
                    recommendations.append("Optimize solution efficiency")
                elif score.evaluation_type == EvaluationType.USER_SATISFACTION:
                    recommendations.append("Improve user communication")
                elif score.evaluation_type == EvaluationType.TOOL_USAGE:
                    recommendations.append("Better utilize available tools")
        
        return recommendations 