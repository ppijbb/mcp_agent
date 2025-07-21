"""
Tool Registry for the Kimi-K2 Agentic Data Synthesis System

Manages MCP tools, synthetic tools, and their metadata for the synthesis system.
"""

from typing import List, Dict, Any, Optional, Set
from models.tool import Tool, ToolType, ToolParameter, ToolExample, ParameterType
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
                "tool_id": "web_search",
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
                "tool_id": "file_search",
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
                "tool_id": "code_editor",
                "name": "code_editor",
                "type": ToolType.MCP,
                "description": "Multi-language code editor with syntax highlighting",
                "parameters": [
                    ToolParameter(
                        name="action",
                        type=ParameterType.STRING,
                        description="Action to perform (open, edit, save, close)",
                        required=True
                    ),
                    ToolParameter(
                        name="file_path",
                        type=ParameterType.STRING,
                        description="File path to work with",
                        required=False
                    ),
                    ToolParameter(
                        name="content",
                        type=ParameterType.STRING,
                        description="Content to write or edit",
                        required=False
                    )
                ],
                "return_type": "operation_result",
                "domain_compatibility": ["development", "programming", "coding"]
            },
            {
                "tool_id": "terminal",
                "name": "terminal",
                "type": ToolType.MCP,
                "description": "Command line interface for system operations",
                "parameters": [
                    ToolParameter(
                        name="command",
                        type=ParameterType.STRING,
                        description="Command to execute",
                        required=True
                    ),
                    ToolParameter(
                        name="working_dir",
                        type=ParameterType.STRING,
                        description="Working directory for command execution",
                        required=False
                    )
                ],
                "return_type": "command_output",
                "domain_compatibility": ["development", "system_admin", "automation"]
            },
            {
                "tool_id": "python",
                "name": "python",
                "type": ToolType.SYNTHETIC,
                "description": "Python programming language interpreter",
                "parameters": [
                    ToolParameter(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code to execute",
                        required=True
                    ),
                    ToolParameter(
                        name="timeout",
                        type=ParameterType.INTEGER,
                        description="Execution timeout in seconds",
                        required=False,
                        default_value=30
                    )
                ],
                "return_type": "execution_result",
                "domain_compatibility": ["data_analysis", "programming", "automation"]
            }
        ]
        
        for tool_config in default_tools:
            self.register_tool(**tool_config)
    
    def register_tool(self, tool_id: str, name: str, type: ToolType, description: str,
                     parameters: List[ToolParameter] = None, return_type: str = "any",
                     domain_compatibility: List[str] = None, version: str = "1.0.0",
                     author: str = "", documentation_url: str = None) -> Tool:
        """Register a new tool with explicit tool_id"""
        tool = Tool(
            id=tool_id,  # Use the provided tool_id
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
        
        self.tools[tool_id] = tool
        
        # Update category mapping
        for domain in tool.domain_compatibility:
            if domain not in self.domain_tool_mapping:
                self.domain_tool_mapping[domain] = set()
            self.domain_tool_mapping[domain].add(tool_id)
        
        # Initialize usage stats
        self.tool_usage_stats[tool_id] = {
            "total_usage": 0,
            "successful_usage": 0,
            "failed_usage": 0,
            "average_response_time": 0.0,
            "last_used": None
        }
        
        logger.info(f"Registered tool: {name} (ID: {tool_id})")
        return tool
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID or name"""
        # First try by ID
        if tool_id in self.tools:
            return self.tools[tool_id]
        
        # If not found by ID, try by name
        for tool in self.tools.values():
            if tool.name.lower() == tool_id.lower():
                return tool
        
        return None
    
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

    def execute_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        tool = self.get_tool(tool_id)
        if not tool:
            return {"error": f"Tool {tool_id} not found"}
        
        if not tool.is_active:
            return {"error": f"Tool {tool_id} is not active"}
        
        # Validate parameters
        validation_result = self.validate_tool_usage(tool_id, parameters)
        if not validation_result["valid"]:
            return {"error": validation_result["error"]}
        
        try:
            # Record usage start
            start_time = datetime.utcnow()
            
            # Execute based on tool type
            if tool.type == ToolType.MCP:
                result = self._execute_mcp_tool(tool, parameters)
            elif tool.type == ToolType.SYNTHETIC:
                result = self._execute_synthetic_tool(tool, parameters)
            else:
                result = {"error": f"Unsupported tool type: {tool.type}"}
            
            # Record usage end
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            # Update usage statistics
            self._update_tool_usage_stats(tool_id, True, response_time)
            
            return result
            
        except Exception as e:
            # Record failed usage
            self._update_tool_usage_stats(tool_id, False, 0.0)
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _execute_mcp_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool (placeholder implementation)"""
        # In a real implementation, this would connect to actual MCP servers
        return {
            "status": "success",
            "tool": tool.name,
            "result": f"Simulated MCP tool execution for {tool.name}",
            "parameters": parameters
        }
    
    def _execute_synthetic_tool(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute synthetic tool (simulated implementation)"""
        import time
        import random
        
        # Simulate execution time
        execution_time = random.uniform(0.1, 2.0)
        time.sleep(execution_time)
        
        # Simulate different tool behaviors
        if tool.name == "python":
            code = parameters.get("code", "")
            return {
                "status": "success",
                "tool": "python",
                "result": f"Executed Python code: {code[:100]}...",
                "execution_time": execution_time
            }
        elif tool.name == "code_editor":
            action = parameters.get("action", "")
            file_path = parameters.get("file_path", "")
            return {
                "status": "success",
                "tool": "code_editor",
                "result": f"Performed {action} on {file_path}",
                "execution_time": execution_time
            }
        elif tool.name == "terminal":
            command = parameters.get("command", "")
            return {
                "status": "success",
                "tool": "terminal",
                "result": f"Executed command: {command}",
                "execution_time": execution_time
            }
        else:
            return {
                "status": "success",
                "tool": tool.name,
                "result": f"Simulated execution of {tool.name}",
                "execution_time": execution_time
            }
    
    def _update_tool_usage_stats(self, tool_id: str, success: bool, response_time: float) -> None:
        """Update tool usage statistics"""
        if tool_id not in self.tool_usage_stats:
            return
        
        stats = self.tool_usage_stats[tool_id]
        stats["total_usage"] += 1
        
        if success:
            stats["successful_usage"] += 1
        else:
            stats["failed_usage"] += 1
        
        # Update average response time
        if stats["total_usage"] > 0:
            current_avg = stats["average_response_time"]
            stats["average_response_time"] = (
                (current_avg * (stats["total_usage"] - 1) + response_time) / stats["total_usage"]
            )
        
        stats["last_used"] = datetime.utcnow()
        
        # Update tool's own stats
        tool = self.get_tool(tool_id)
        if tool:
            tool.update_usage_stats(success, response_time) 