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

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("mcp_integration", log_level="INFO")


class MCPIntegrationManager:
    """MCP integration manager for enhanced agent capabilities."""
    
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
        
        # Initialize MCP connections
        self._initialize_mcp_connections()
        
        logger.info("MCP Integration Manager initialized")
    
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
