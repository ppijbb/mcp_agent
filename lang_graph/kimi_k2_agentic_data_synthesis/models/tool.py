"""
Tool models for the Kimi-K2 Agentic Data Synthesis System
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class ToolType(str, Enum):
    """Types of tools supported by the system"""
    MCP = "mcp"  # Model Context Protocol tools
    SYNTHETIC = "synthetic"  # Synthetic tools for simulation
    EXTERNAL = "external"  # External API tools
    CUSTOM = "custom"  # Custom tools


class ParameterType(str, Enum):
    """Parameter types for tool parameters"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"


class ToolParameter(BaseModel):
    """Parameter definition for a tool"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # regex pattern for validation
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "query",
                "type": "string",
                "description": "Search query string",
                "required": True,
                "pattern": "^[a-zA-Z0-9\\s]+$"
            }
        }


class ToolExample(BaseModel):
    """Example usage of a tool"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    input_parameters: Dict[str, Any]
    expected_output: str
    tags: List[str] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Search for Python documentation",
                "description": "Search for Python programming documentation",
                "input_parameters": {
                    "query": "Python list methods",
                    "max_results": 5
                },
                "expected_output": "List of Python list methods with descriptions",
                "tags": ["python", "documentation", "search"]
            }
        }


class Tool(BaseModel):
    """Tool definition for the synthesis system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ToolType
    description: str
    parameters: List[ToolParameter] = []
    return_type: str
    domain_compatibility: List[str] = []
    usage_examples: List[ToolExample] = []
    version: str = "1.0.0"
    author: str = ""
    documentation_url: Optional[str] = None
    is_active: bool = True
    usage_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "web_search",
                "type": "mcp",
                "description": "Search the web for information",
                "parameters": [],
                "return_type": "search_results",
                "domain_compatibility": ["general", "research", "information"],
                "usage_examples": [],
                "version": "1.0.0",
                "author": "Kimi-K2 Team"
            }
        }
    
    def add_example(self, example: ToolExample) -> None:
        """Add a usage example to the tool"""
        self.usage_examples.append(example)
        self.updated_at = datetime.utcnow()
    
    def update_usage_stats(self, success: bool, response_time: float) -> None:
        """Update tool usage statistics"""
        self.usage_count += 1
        if success:
            self.success_rate = ((self.success_rate * (self.usage_count - 1)) + 1.0) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count
        
        self.average_response_time = (
            (self.average_response_time * (self.usage_count - 1)) + response_time
        ) / self.usage_count
        self.updated_at = datetime.utcnow()
    
    def is_compatible_with_domain(self, domain_id: str) -> bool:
        """Check if tool is compatible with a specific domain"""
        return domain_id in self.domain_compatibility
    
    def get_required_parameters(self) -> List[ToolParameter]:
        """Get list of required parameters"""
        return [p for p in self.parameters if p.required]
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters against tool definition"""
        required_params = self.get_required_parameters()
        
        # Check required parameters
        for param in required_params:
            if param.name not in params:
                return False
        
        # Check parameter types and constraints
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                
                # Type validation
                if param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return False
                elif param.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
                    return False
                elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return False
                elif param.type == ParameterType.STRING and not isinstance(value, str):
                    return False
                
                # Value constraints
                if param.allowed_values and value not in param.allowed_values:
                    return False
                
                if param.min_value is not None and value < param.min_value:
                    return False
                
                if param.max_value is not None and value > param.max_value:
                    return False
        
        return True 