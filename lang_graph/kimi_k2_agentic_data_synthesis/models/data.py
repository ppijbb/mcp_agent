"""
Data models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class DataFormat(str, Enum):
    """Supported data formats for export"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    PICKLE = "pickle"


class DataQuality(str, Enum):
    """Data quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"


class Metadata(BaseModel):
    """Metadata for training data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain_id: str
    scenario_id: str
    agent_ids: List[str] = []
    simulation_id: str
    evaluation_id: Optional[str] = None
    quality_score: Optional[float] = None
    data_format: DataFormat = DataFormat.JSON
    version: str = "1.0.0"
    tags: List[str] = []
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "domain_id": "tech_support",
                "scenario_id": "network_troubleshooting",
                "agent_ids": ["agent_1", "agent_2"],
                "simulation_id": "sim_123",
                "quality_score": 0.85,
                "data_format": "json",
                "version": "1.0.0",
                "tags": ["networking", "troubleshooting", "high_quality"]
            }
        }


class TrainingData(BaseModel):
    """Individual training data entry"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Metadata
    conversation_history: List[Dict[str, Any]] = []
    tool_usage_log: List[Dict[str, Any]] = []
    final_outcome: Dict[str, Any] = {}
    quality_metrics: Dict[str, float] = {}
    is_valid: bool = True
    validation_errors: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {},
                "conversation_history": [
                    {"role": "user", "content": "My internet is not working"},
                    {"role": "assistant", "content": "Let me help you troubleshoot this issue"}
                ],
                "tool_usage_log": [
                    {"tool": "network_diagnostics", "parameters": {}, "result": "DNS issue detected"}
                ],
                "final_outcome": {"status": "resolved", "solution": "Changed DNS settings"},
                "quality_metrics": {"accuracy": 0.9, "completeness": 0.8}
            }
        }
    
    def add_conversation_turn(self, role: str, content: str, **kwargs) -> None:
        """Add a conversation turn to the history"""
        turn = {"role": role, "content": content, **kwargs}
        self.conversation_history.append(turn)
    
    def add_tool_usage(self, tool: str, parameters: Dict[str, Any], result: Any) -> None:
        """Add a tool usage entry to the log"""
        usage = {
            "tool": tool,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.tool_usage_log.append(usage)
    
    def set_final_outcome(self, outcome: Dict[str, Any]) -> None:
        """Set the final outcome of the training data"""
        self.final_outcome = outcome
    
    def add_quality_metric(self, metric: str, value: float) -> None:
        """Add a quality metric"""
        self.quality_metrics[metric] = value
    
    def validate_data(self) -> bool:
        """Validate the training data"""
        self.validation_errors = []
        
        # Check required fields
        if not self.conversation_history:
            self.validation_errors.append("Conversation history is empty")
        
        if not self.metadata.domain_id:
            self.validation_errors.append("Domain ID is required")
        
        if not self.metadata.scenario_id:
            self.validation_errors.append("Scenario ID is required")
        
        # Check conversation structure
        for i, turn in enumerate(self.conversation_history):
            if "role" not in turn or "content" not in turn:
                self.validation_errors.append(f"Invalid conversation turn at index {i}")
        
        # Check tool usage structure
        for i, usage in enumerate(self.tool_usage_log):
            if "tool" not in usage or "parameters" not in usage:
                self.validation_errors.append(f"Invalid tool usage at index {i}")
        
        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid
    
    def get_conversation_text(self) -> str:
        """Get the full conversation as text"""
        text_parts = []
        for turn in self.conversation_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)
    
    def get_tool_usage_summary(self) -> Dict[str, int]:
        """Get summary of tool usage"""
        summary = {}
        for usage in self.tool_usage_log:
            tool = usage.get("tool", "unknown")
            summary[tool] = summary.get(tool, 0) + 1
        return summary


class DataExportConfig(BaseModel):
    """Configuration for data export"""
    formats: List[str] = ["json", "jsonl", "csv"]
    include_metadata: bool = True
    include_evaluations: bool = True
    split_ratios: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
    compression: bool = False
    metadata_fields: List[str] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "formats": ["json", "jsonl", "csv"],
                "include_metadata": True,
                "include_evaluations": True,
                "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "compression": True,
                "metadata_fields": ["simulation_id", "domain", "agents", "tools_used"]
            }
        }


class DataBatch(BaseModel):
    """Batch of training data"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    training_data: List[TrainingData] = []
    batch_size: int = 0
    quality_threshold: float = 0.7
    passed_validation: int = 0
    failed_validation: int = 0
    average_quality_score: float = 0.0
    domain_distribution: Dict[str, int] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Tech Support Batch 001",
                "description": "Technical support scenarios for network troubleshooting",
                "batch_size": 100,
                "quality_threshold": 0.7,
                "passed_validation": 95,
                "failed_validation": 5
            }
        }
    
    def add_training_data(self, data: TrainingData) -> None:
        """Add training data to the batch"""
        self.training_data.append(data)
        self.batch_size = len(self.training_data)
        self.updated_at = datetime.utcnow()
    
    def validate_batch(self) -> Dict[str, Any]:
        """Validate all training data in the batch"""
        results = {
            "total": len(self.training_data),
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for data in self.training_data:
            if data.validate_data():
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].extend(data.validation_errors)
        
        self.passed_validation = results["passed"]
        self.failed_validation = results["failed"]
        
        return results
    
    def calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics for the batch"""
        if not self.training_data:
            return {}
        
        # Calculate average quality score
        valid_data = [d for d in self.training_data if d.is_valid]
        if valid_data:
            total_score = sum(d.quality_metrics.get("overall", 0.0) for d in valid_data)
            self.average_quality_score = total_score / len(valid_data)
        
        # Calculate domain distribution
        domain_counts = {}
        for data in self.training_data:
            domain = data.metadata.domain_id
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        self.domain_distribution = domain_counts
        
        return {
            "average_quality_score": self.average_quality_score,
            "validation_rate": self.passed_validation / len(self.training_data) if self.training_data else 0.0,
            "domain_diversity": len(self.domain_distribution)
        }
    
    def filter_by_quality(self, threshold: Optional[float] = None) -> List[TrainingData]:
        """Filter training data by quality threshold"""
        if threshold is None:
            threshold = self.quality_threshold
        
        return [
            data for data in self.training_data
            if data.is_valid and data.quality_metrics.get("overall", 0.0) >= threshold
        ]
    
    def get_domain_data(self, domain_id: str) -> List[TrainingData]:
        """Get training data for a specific domain"""
        return [
            data for data in self.training_data
            if data.metadata.domain_id == domain_id
        ]
    
    def export_batch(self, format: DataFormat = DataFormat.JSON) -> str:
        """Export batch to specified format"""
        # This would be implemented based on the specific format requirements
        # For now, return a placeholder
        return f"Exported {len(self.training_data)} records in {format} format" 