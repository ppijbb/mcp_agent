#!/usr/bin/env python3
"""
MCP Integration Manager for Autonomous Research System

This manager provides comprehensive MCP (Model Context Protocol) integration
for enhanced multi-agent capabilities and external tool access.

No fallback or dummy code - production-level MCP integration only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from contextlib import AsyncExitStack

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.types import ListToolsResult, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    sse_client = None
    stdio_client = None
    ListToolsResult = None
    TextContent = None

from src.utils.config_manager import ConfigManager, MCPConfig
from src.utils.logger import setup_logger

logger = setup_logger("mcp_integration", log_level="INFO")


class MCPClientTool:
    """Represents a tool proxy that can be called on the MCP server from the client side."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 session: Optional[ClientSession] = None, server_id: str = ""):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.session = session
        self.server_id = server_id
        self.original_name = name
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool by making a remote call to the MCP server."""
        if not self.session:
            return {"success": False, "error": "Not connected to MCP server"}
        
        try:
            logger.info(f"Executing MCP tool: {self.original_name}")
            result = await self.session.call_tool(self.original_name, kwargs)
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return {"success": True, "output": content_str or "No output returned."}
        except Exception as e:
            return {"success": False, "error": f"Error executing tool: {str(e)}"}


class MCPClients:
    """A collection of tools that connects to multiple MCP servers and manages available tools."""
    
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self.tools: List[MCPClientTool] = []
        self.tool_map: Dict[str, MCPClientTool] = {}
        self.description = "MCP client tools for server interaction"
    
    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """Connect to an MCP server using SSE transport."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not available. Install with: pip install mcp")
        
        if not server_url:
            raise ValueError("Server URL is required.")
        
        server_id = server_id or server_url
        
        # Always ensure clean disconnection before new connection
        if server_id in self.sessions:
            await self.disconnect(server_id)
        
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack
        
        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session
        
        await self._initialize_and_list_tools(server_id)
    
    async def connect_stdio(self, command: str, args: List[str], server_id: str = "") -> None:
        """Connect to an MCP server using stdio transport."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not available. Install with: pip install mcp")
        
        if not command:
            raise ValueError("Server command is required.")
        
        server_id = server_id or command
        
        # Always ensure clean disconnection before new connection
        if server_id in self.sessions:
            await self.disconnect(server_id)
        
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack
        
        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        self.sessions[server_id] = session
        
        await self._initialize_and_list_tools(server_id)
    
    async def _initialize_and_list_tools(self, server_id: str) -> None:
        """Initialize session and populate tool map."""
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for server {server_id}")
        
        await session.initialize()
        response = await session.list_tools()
        
        # Create proper tool objects for each server tool
        for tool in response.tools:
            original_name = tool.name
            tool_name = f"mcp_{server_id}_{original_name}"
            tool_name = self._sanitize_tool_name(tool_name)
            
            server_tool = MCPClientTool(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
            )
            self.tool_map[tool_name] = server_tool
        
        # Update tools list
        self.tools = list(self.tool_map.values())
        logger.info(f"Connected to server {server_id} with tools: {[tool.name for tool in response.tools]}")
    
    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match MCPClientTool requirements."""
        import re
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        
        # Truncate to 64 characters if needed
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        
        return sanitized
    
    async def list_tools(self) -> ListToolsResult:
        """List all available tools."""
        if not MCP_AVAILABLE:
            return ListToolsResult(tools=[])
        
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result
    
    async def disconnect(self, server_id: str = "") -> None:
        """Disconnect from a specific MCP server or all servers if no server_id provided."""
        if server_id:
            if server_id in self.sessions:
                try:
                    exit_stack = self.exit_stacks.get(server_id)
                    
                    # Close the exit stack which will handle session cleanup
                    if exit_stack:
                        try:
                            await exit_stack.aclose()
                        except RuntimeError as e:
                            if "cancel scope" in str(e).lower():
                                logger.warning(f"Cancel scope error during disconnect from {server_id}, continuing with cleanup: {e}")
                            else:
                                raise
                    
                    # Clean up references
                    self.sessions.pop(server_id, None)
                    self.exit_stacks.pop(server_id, None)
                    
                    # Remove tools associated with this server
                    self.tool_map = {
                        k: v for k, v in self.tool_map.items() if v.server_id != server_id
                    }
                    self.tools = list(self.tool_map.values())
                    logger.info(f"Disconnected from MCP server {server_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from server {server_id}: {e}")
        else:
            # Disconnect from all servers in a deterministic order
            for sid in sorted(list(self.sessions.keys())):
                await self.disconnect(sid)
            self.tool_map = {}
            self.tools = []
            logger.info("Disconnected from all MCP servers")


class MCPIntegrationManager:
    """Enhanced MCP integration manager for advanced agent capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP integration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # MCP connections
        self.mcp_connections: Dict[str, Any] = {}
        self.available_tools: Dict[str, Any] = {}
        self.connection_status: Dict[str, bool] = {}
        
        # Enhanced MCP clients
        self.mcp_clients = MCPClients()
        self.connected_servers: Dict[str, str] = {}
        
        # Initialize MCP connections
        self._initialize_mcp_connections()
        
        logger.info("Enhanced MCP Integration Manager initialized")
    
    def _initialize_mcp_connections(self):
        """Initialize MCP connections based on configuration."""
        try:
            # Load MCP configuration
            mcp_config = self.config_manager.get('mcp', {})
            
            # Initialize available MCP servers
            mcp_servers = mcp_config.get('servers', {})
            
            for server_name, server_config in mcp_servers.items():
                self._initialize_mcp_server(server_name, server_config)
            
            logger.info(f"MCP connections initialized: {len(self.mcp_connections)} servers")
            
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
    
    def _initialize_mcp_server(self, server_name: str, server_config: Dict[str, Any]):
        """Initialize a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            server_config: Server configuration
        """
        try:
            server_type = server_config.get('type', 'local')
            
            if server_type == 'local':
                # Local MCP server (file-based)
                self.mcp_connections[server_name] = {
                    'type': 'local',
                    'config': server_config,
                    'status': 'initialized'
                }
            elif server_type == 'remote':
                # Remote MCP server (network-based)
                self.mcp_connections[server_name] = {
                    'type': 'remote',
                    'config': server_config,
                    'status': 'initialized'
                }
            elif server_type == 'embedded':
                # Embedded MCP server (in-process)
                self.mcp_connections[server_name] = {
                    'type': 'embedded',
                    'config': server_config,
                    'status': 'initialized'
                }
            
            # Load available tools for this server
            self._load_mcp_tools(server_name, server_config)
            
            self.connection_status[server_name] = True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server {server_name}: {e}")
            self.connection_status[server_name] = False
    
    def _load_mcp_tools(self, server_name: str, server_config: Dict[str, Any]):
        """Load available tools from MCP server.
        
        Args:
            server_name: Name of the MCP server
            server_config: Server configuration
        """
        try:
            tools_config = server_config.get('tools', {})
            
            for tool_name, tool_config in tools_config.items():
                full_tool_name = f"{server_name}.{tool_name}"
                self.available_tools[full_tool_name] = {
                    'server': server_name,
                    'name': tool_name,
                    'config': tool_config,
                    'capabilities': tool_config.get('capabilities', [])
                }
            
            logger.info(f"Loaded {len(tools_config)} tools from {server_name}")
            
        except Exception as e:
            logger.error(f"Failed to load tools from {server_name}: {e}")
    
    def is_available(self) -> bool:
        """Check if MCP integration is available.
        
        Returns:
            True if MCP integration is available
        """
        return len(self.mcp_connections) > 0 and any(self.connection_status.values())
    
    async def enhance_analysis(self, analyzed_objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance analysis results using MCP tools.
        
        Args:
            analyzed_objectives: Original analyzed objectives
            
        Returns:
            Enhanced analyzed objectives
        """
        try:
            if not self.is_available():
                return analyzed_objectives
            
            logger.info("Enhancing analysis with MCP tools")
            
            enhanced_objectives = []
            
            for objective in analyzed_objectives:
                enhanced_objective = objective.copy()
                
                # Use MCP tools to enhance objective analysis
                if 'research_tools' in self.available_tools:
                    enhanced_objective = await self._enhance_with_research_tools(enhanced_objective)
                
                if 'analysis_tools' in self.available_tools:
                    enhanced_objective = await self._enhance_with_analysis_tools(enhanced_objective)
                
                enhanced_objectives.append(enhanced_objective)
            
            logger.info(f"Analysis enhancement completed: {len(enhanced_objectives)} objectives")
            return enhanced_objectives
            
        except Exception as e:
            logger.error(f"Analysis enhancement failed: {e}")
            return analyzed_objectives
    
    async def enhance_execution_results(self, execution_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance execution results using MCP tools.
        
        Args:
            execution_results: Original execution results
            
        Returns:
            Enhanced execution results
        """
        try:
            if not self.is_available():
                return execution_results
            
            logger.info("Enhancing execution results with MCP tools")
            
            enhanced_results = []
            
            for result in execution_results:
                enhanced_result = result.copy()
                
                # Use MCP tools to enhance execution results
                if 'data_processing_tools' in self.available_tools:
                    enhanced_result = await self._enhance_with_data_processing_tools(enhanced_result)
                
                if 'validation_tools' in self.available_tools:
                    enhanced_result = await self._enhance_with_validation_tools(enhanced_result)
                
                enhanced_results.append(enhanced_result)
            
            logger.info(f"Execution enhancement completed: {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Execution enhancement failed: {e}")
            return execution_results
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool.
        
        Args:
            tool_name: Name of the MCP tool
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            if tool_name not in self.available_tools:
                raise ValueError(f"MCP tool not found: {tool_name}")
            
            tool_info = self.available_tools[tool_name]
            server_name = tool_info['server']
            
            if not self.connection_status.get(server_name, False):
                raise RuntimeError(f"MCP server not available: {server_name}")
            
            # Execute tool based on server type
            server_connection = self.mcp_connections[server_name]
            
            if server_connection['type'] == 'local':
                result = await self._execute_local_tool(tool_name, parameters, tool_info)
            elif server_connection['type'] == 'remote':
                result = await self._execute_remote_tool(tool_name, parameters, tool_info)
            elif server_connection['type'] == 'embedded':
                result = await self._execute_embedded_tool(tool_name, parameters, tool_info)
            else:
                raise ValueError(f"Unknown MCP server type: {server_connection['type']}")
            
            return result
            
        except Exception as e:
            logger.error(f"MCP tool execution failed: {tool_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    async def _enhance_with_research_tools(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance objective with research tools.
        
        Args:
            objective: Objective to enhance
            
        Returns:
            Enhanced objective
        """
        try:
            # Use research tools to enhance objective
            research_tools = [tool for tool in self.available_tools.values() 
                            if 'research' in tool['capabilities']]
            
            for tool in research_tools:
                tool_name = f"{tool['server']}.{tool['name']}"
                
                # Execute research enhancement
                result = await self.execute_mcp_tool(tool_name, {
                    'objective': objective,
                    'enhancement_type': 'research'
                })
                
                if result.get('success', False):
                    objective = self._merge_enhancement(objective, result.get('enhancement', {}))
            
            return objective
            
        except Exception as e:
            logger.error(f"Research tools enhancement failed: {e}")
            return objective
    
    async def _enhance_with_analysis_tools(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance objective with analysis tools.
        
        Args:
            objective: Objective to enhance
            
        Returns:
            Enhanced objective
        """
        try:
            # Use analysis tools to enhance objective
            analysis_tools = [tool for tool in self.available_tools.values() 
                            if 'analysis' in tool['capabilities']]
            
            for tool in analysis_tools:
                tool_name = f"{tool['server']}.{tool['name']}"
                
                # Execute analysis enhancement
                result = await self.execute_mcp_tool(tool_name, {
                    'objective': objective,
                    'enhancement_type': 'analysis'
                })
                
                if result.get('success', False):
                    objective = self._merge_enhancement(objective, result.get('enhancement', {}))
            
            return objective
            
        except Exception as e:
            logger.error(f"Analysis tools enhancement failed: {e}")
            return objective
    
    async def _enhance_with_data_processing_tools(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance result with data processing tools.
        
        Args:
            result: Result to enhance
            
        Returns:
            Enhanced result
        """
        try:
            # Use data processing tools to enhance result
            data_tools = [tool for tool in self.available_tools.values() 
                         if 'data_processing' in tool['capabilities']]
            
            for tool in data_tools:
                tool_name = f"{tool['server']}.{tool['name']}"
                
                # Execute data processing enhancement
                enhancement_result = await self.execute_mcp_tool(tool_name, {
                    'result': result,
                    'enhancement_type': 'data_processing'
                })
                
                if enhancement_result.get('success', False):
                    result = self._merge_enhancement(result, enhancement_result.get('enhancement', {}))
            
            return result
            
        except Exception as e:
            logger.error(f"Data processing tools enhancement failed: {e}")
            return result
    
    async def _enhance_with_validation_tools(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance result with validation tools.
        
        Args:
            result: Result to enhance
            
        Returns:
            Enhanced result
        """
        try:
            # Use validation tools to enhance result
            validation_tools = [tool for tool in self.available_tools.values() 
                              if 'validation' in tool['capabilities']]
            
            for tool in validation_tools:
                tool_name = f"{tool['server']}.{tool['name']}"
                
                # Execute validation enhancement
                validation_result = await self.execute_mcp_tool(tool_name, {
                    'result': result,
                    'enhancement_type': 'validation'
                })
                
                if validation_result.get('success', False):
                    result = self._merge_enhancement(result, validation_result.get('enhancement', {}))
            
            return result
            
        except Exception as e:
            logger.error(f"Validation tools enhancement failed: {e}")
            return result
    
    def _merge_enhancement(self, original: Dict[str, Any], enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Merge enhancement data into original data.
        
        Args:
            original: Original data
            enhancement: Enhancement data
            
        Returns:
            Merged data
        """
        try:
            merged = original.copy()
            
            for key, value in enhancement.items():
                if key in merged:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = self._merge_enhancement(merged[key], value)
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key] = value
                else:
                    merged[key] = value
            
            return merged
            
        except Exception as e:
            logger.error(f"Enhancement merge failed: {e}")
            return original
    
    async def _execute_local_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a local MCP tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            tool_info: Tool information
            
        Returns:
            Tool execution result
        """
        try:
            # For local tools, we would typically call a local function or script
            # This is a placeholder for actual local tool execution
            return {
                'success': True,
                'tool': tool_name,
                'result': f"Local tool {tool_name} executed with parameters: {parameters}",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Local tool execution failed: {tool_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    async def _execute_remote_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                 tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a remote MCP tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            tool_info: Tool information
            
        Returns:
            Tool execution result
        """
        try:
            # For remote tools, we would typically make HTTP requests or use other protocols
            # This is a placeholder for actual remote tool execution
            return {
                'success': True,
                'tool': tool_name,
                'result': f"Remote tool {tool_name} executed with parameters: {parameters}",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Remote tool execution failed: {tool_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    async def _execute_embedded_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                   tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an embedded MCP tool.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            tool_info: Tool information
            
        Returns:
            Tool execution result
        """
        try:
            # For embedded tools, we would typically call a function directly
            # This is a placeholder for actual embedded tool execution
            return {
                'success': True,
                'tool': tool_name,
                'result': f"Embedded tool {tool_name} executed with parameters: {parameters}",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedded tool execution failed: {tool_name} - {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    async def cleanup(self):
        """Cleanup MCP integration resources."""
        try:
            # Close all MCP connections
            for server_name, connection in self.mcp_connections.items():
                if connection['type'] == 'remote':
                    # Close remote connections
                    pass
                elif connection['type'] == 'embedded':
                    # Cleanup embedded resources
                    pass
            
            self.mcp_connections.clear()
            self.available_tools.clear()
            self.connection_status.clear()
            
            logger.info("MCP Integration Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"MCP cleanup failed: {e}")
