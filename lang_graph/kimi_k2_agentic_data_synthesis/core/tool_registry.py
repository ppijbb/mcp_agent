"""
Tool Registry for the Kimi-K2 Agentic Data Synthesis System

Manages MCP tools, synthetic tools, and their metadata for the synthesis system.
"""

from typing import List, Dict, Any, Optional, Set
from ..models.tool import Tool, ToolType, ToolParameter, ToolExample, ParameterType
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages tools for the agentic data synthesis system.
    
    Responsibilities:
    - Tool registration and management
    - Tool metadata and usage tracking
    - Tool compatibility validation
    - Tool categorization and search
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[str, Set[str]] = {}
        self.domain_tool_mapping: Dict[str, Set[str]] = {}
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with default tools
        self._initialize_default_tools()
    
    def _initialize_default_tools(self) -> None:
        """Initialize the system with default tools"""
        default_tools = [
            {
                "name": "web_search",
                "type": ToolType.MCP,
                "description": "Search the web for information",
                "parameters": [
                    ToolParameter(
                        name="query",
                        type=ParameterType.STRING,
                        description="Search query string",
                        required=True
                    ),
                    ToolParameter(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum number of results",
                        required=False,
                        default_value=5,
                        min_value=1,
                        max_value=20
                    )
                ],
                "return_type": "search_results",
                "domain_compatibility": ["general", "research", "information"]
            },
            {
                "name": "file_search",
                "type": ToolType.MCP,
                "description": "Search for files in the system",
                "parameters": [
                    ToolParameter(
                        name="path",
                        type=ParameterType.STRING,
                        description="Search path",
                        required=True
                    ),
                    ToolParameter(
                        name="pattern",
                        type=ParameterType.STRING,
                        description="File pattern to search for",
                        required=False
                    )
                ],
                "return_type": "file_list",
                "domain_compatibility": ["development", "system_admin", "file_management"]
            },
            {
                "name": "code_analysis",
                "type": ToolType.SYNTHETIC,
                "description": "Analyze code for issues and improvements",
                "parameters": [
                    ToolParameter(
                        name="code",
                        type=ParameterType.STRING,
                        description="Code to analyze",
                        required=True
                    ),
                    ToolParameter(
                        name="language",
                        type=ParameterType.STRING,
                        description="Programming language",
                        required=True,
                        allowed_values=["python", "javascript", "java", "cpp"]
                    )
                ],
                "return_type": "analysis_report",
                "domain_compatibility": ["development", "code_review", "quality_assurance"]
            },
            {
                "name": "data_analysis",
                "type": ToolType.SYNTHETIC,
                "description": "Perform data analysis and generate insights",
                "parameters": [
                    ToolParameter(
                        name="data",
                        type=ParameterType.OBJECT,
                        description="Data to analyze",
                        required=True
                    ),
                    ToolParameter(
                        name="analysis_type",
                        type=ParameterType.STRING,
                        description="Type of analysis to perform",
                        required=True,
                        allowed_values=["descriptive", "predictive", "diagnostic"]
                    )
                ],
                "return_type": "analysis_results",
                "domain_compatibility": ["data_science", "business_intelligence", "research"]
            }
        ]
        
        for tool_config in default_tools:
            self.register_tool(**tool_config)
    
    def register_tool(self, name: str, type: ToolType, description: str,
                     parameters: List[ToolParameter] = None, return_type: str = "any",
                     domain_compatibility: List[str] = None, version: str = "1.0.0",
                     author: str = "", documentation_url: str = None) -> Tool:
        """Register a new tool"""
        tool = Tool(
            name=name,
            type=type,
            description=description,
            parameters=parameters or [],
            return_type=return_type,
            domain_compatibility=domain_compatibility or [],
            version=version,
            author=author,
            documentation_url=documentation_url
        )
        
        self.tools[tool.id] = tool
        
        # Update category mapping
        for domain in tool.domain_compatibility:
            if domain not in self.domain_tool_mapping:
                self.domain_tool_mapping[domain] = set()
            self.domain_tool_mapping[domain].add(tool.id)
        
        # Initialize usage stats
        self.tool_usage_stats[tool.id] = {
            "total_usage": 0,
            "successful_usage": 0,
            "failed_usage": 0,
            "average_response_time": 0.0,
            "last_used": None
        }
        
        logger.info(f"Registered tool: {name} (ID: {tool.id})")
        return tool
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID"""
        return self.tools.get(tool_id)
    
    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        for tool in self.tools.values():
            if tool.name.lower() == name.lower():
                return tool
        return None
    
    def list_tools(self, tool_type: Optional[ToolType] = None,
                   domain: Optional[str] = None, active_only: bool = True) -> List[Tool]:
        """List tools with optional filtering"""
        tools = list(self.tools.values())
        
        if tool_type:
            tools = [t for t in tools if t.type == tool_type]
        
        if domain:
            tools = [t for t in tools if domain in t.domain_compatibility]
        
        if active_only:
            tools = [t for t in tools if t.is_active]
        
        return tools
    
    def update_tool(self, tool_id: str, **kwargs) -> bool:
        """Update a tool"""
        tool = self.get_tool(tool_id)
        if not tool:
            logger.warning(f"Tool not found: {tool_id}")
            return False
        
        for key, value in kwargs.items():
            if hasattr(tool, key):
                setattr(tool, key, value)
        
        tool.updated_at = datetime.utcnow()
        logger.info(f"Updated tool: {tool.name}")
        return True
    
    def deactivate_tool(self, tool_id: str) -> bool:
        """Deactivate a tool"""
        tool = self.get_tool(tool_id)
        if not tool:
            logger.warning(f"Tool not found: {tool_id}")
            return False
        
        tool.is_active = False
        tool.updated_at = datetime.utcnow()
        logger.info(f"Deactivated tool: {tool.name}")
        return True
    
    def activate_tool(self, tool_id: str) -> bool:
        """Activate a tool"""
        tool = self.get_tool(tool_id)
        if not tool:
            logger.warning(f"Tool not found: {tool_id}")
            return False
        
        tool.is_active = True
        tool.updated_at = datetime.utcnow()
        logger.info(f"Activated tool: {tool.name}")
        return True
    
    def add_tool_example(self, tool_id: str, example: ToolExample) -> bool:
        """Add an example to a tool"""
        tool = self.get_tool(tool_id)
        if not tool:
            logger.warning(f"Tool not found: {tool_id}")
            return False
        
        tool.add_example(example)
        logger.info(f"Added example to tool: {tool.name}")
        return True
    
    def get_tools_for_domain(self, domain: str) -> List[Tool]:
        """Get tools compatible with a specific domain"""
        tool_ids = self.domain_tool_mapping.get(domain, set())
        return [self.tools[tid] for tid in tool_ids if tid in self.tools and self.tools[tid].is_active]
    
    def validate_tool_usage(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool usage with given parameters"""
        tool = self.get_tool(tool_id)
        if not tool:
            return {"valid": False, "error": "Tool not found"}
        
        if not tool.is_active:
            return {"valid": False, "error": "Tool is not active"}
        
        # Validate parameters
        if not tool.validate_parameters(parameters):
            return {"valid": False, "error": "Invalid parameters"}
        
        return {"valid": True, "tool": tool}
    
    def record_tool_usage(self, tool_id: str, success: bool, response_time: float) -> bool:
        """Record tool usage statistics"""
        tool = self.get_tool(tool_id)
        if not tool:
            return False
        
        # Update tool statistics
        tool.update_usage_stats(success, response_time)
        
        # Update registry statistics
        if tool_id in self.tool_usage_stats:
            stats = self.tool_usage_stats[tool_id]
            stats["total_usage"] += 1
            if success:
                stats["successful_usage"] += 1
            else:
                stats["failed_usage"] += 1
            
            # Update average response time
            current_avg = stats["average_response_time"]
            total_usage = stats["total_usage"]
            stats["average_response_time"] = (
                (current_avg * (total_usage - 1)) + response_time
            ) / total_usage
            
            stats["last_used"] = datetime.utcnow()
        
        return True
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about all tools"""
        stats = {
            "total_tools": len(self.tools),
            "active_tools": len([t for t in self.tools.values() if t.is_active]),
            "tools_by_type": {},
            "tools_by_domain": {},
            "most_used_tools": [],
            "highest_success_rate": []
        }
        
        # Tools by type
        for tool in self.tools.values():
            tool_type = tool.type.value
            stats["tools_by_type"][tool_type] = stats["tools_by_type"].get(tool_type, 0) + 1
        
        # Tools by domain
        for domain, tool_ids in self.domain_tool_mapping.items():
            stats["tools_by_domain"][domain] = len(tool_ids)
        
        # Most used tools
        usage_data = []
        for tool_id, usage_stats in self.tool_usage_stats.items():
            tool = self.tools.get(tool_id)
            if tool:
                usage_data.append({
                    "tool_id": tool_id,
                    "tool_name": tool.name,
                    "total_usage": usage_stats["total_usage"],
                    "success_rate": tool.success_rate,
                    "average_response_time": tool.average_response_time
                })
        
        # Sort by usage and success rate
        usage_data.sort(key=lambda x: x["total_usage"], reverse=True)
        stats["most_used_tools"] = usage_data[:10]
        
        usage_data.sort(key=lambda x: x["success_rate"], reverse=True)
        stats["highest_success_rate"] = usage_data[:10]
        
        return stats
    
    def search_tools(self, query: str, search_fields: List[str] = None) -> List[Tool]:
        """Search tools by query"""
        if search_fields is None:
            search_fields = ["name", "description"]
        
        query_lower = query.lower()
        matching_tools = []
        
        for tool in self.tools.values():
            for field in search_fields:
                if hasattr(tool, field):
                    field_value = getattr(tool, field)
                    if isinstance(field_value, str) and query_lower in field_value.lower():
                        matching_tools.append(tool)
                        break
        
        return matching_tools
    
    def get_tool_compatibility_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get tool compatibility matrix"""
        matrix = {}
        
        for tool in self.tools.values():
            matrix[tool.name] = {}
            for other_tool in self.tools.values():
                # Check if tools share any domains
                shared_domains = set(tool.domain_compatibility) & set(other_tool.domain_compatibility)
                matrix[tool.name][other_tool.name] = len(shared_domains) > 0
        
        return matrix
    
    def export_tool(self, tool_id: str, format: str = "json") -> Optional[str]:
        """Export a tool to specified format"""
        tool = self.get_tool(tool_id)
        if not tool:
            return None
        
        if format.lower() == "json":
            return tool.model_dump_json(indent=2)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return None
    
    def import_tool(self, tool_data: Dict[str, Any]) -> Optional[Tool]:
        """Import a tool from data"""
        try:
            tool = Tool(**tool_data)
            self.tools[tool.id] = tool
            
            # Update mappings
            for domain in tool.domain_compatibility:
                if domain not in self.domain_tool_mapping:
                    self.domain_tool_mapping[domain] = set()
                self.domain_tool_mapping[domain].add(tool.id)
            
            # Initialize usage stats
            self.tool_usage_stats[tool.id] = {
                "total_usage": 0,
                "successful_usage": 0,
                "failed_usage": 0,
                "average_response_time": 0.0,
                "last_used": None
            }
            
            logger.info(f"Imported tool: {tool.name}")
            return tool
        except Exception as e:
            logger.error(f"Failed to import tool: {e}")
            return None
    
    def validate_tool(self, tool_id: str) -> Dict[str, Any]:
        """Validate a tool configuration"""
        tool = self.get_tool(tool_id)
        if not tool:
            return {"valid": False, "errors": ["Tool not found"]}
        
        errors = []
        warnings = []
        
        # Validate tool structure
        if not tool.name:
            errors.append("Tool name is required")
        
        if not tool.description:
            warnings.append("Tool description is empty")
        
        # Validate parameters
        for param in tool.parameters:
            if not param.name:
                errors.append(f"Parameter name is required")
            
            if param.required and param.default_value is not None:
                warnings.append(f"Required parameter '{param.name}' has default value")
        
        # Validate domain compatibility
        if not tool.domain_compatibility:
            warnings.append("Tool has no domain compatibility specified")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "parameter_count": len(tool.parameters),
            "example_count": len(tool.usage_examples)
        } 