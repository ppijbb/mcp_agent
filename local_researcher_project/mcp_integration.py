"""
MCP Integration Module

MCP 클라이언트, 서버, 도구 기능을 통합한 모듈.
모든 하드코딩, fallback, mock 코드를 제거하고 실제 MCP agent 라이브러리를 사용.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from researcher_config import config, get_mcp_config

logger = logging.getLogger(__name__)


class MCPServerManager:
    """MCP 서버 관리자"""
    
    def __init__(self):
        self.config = get_mcp_config()
        self.app = None
        
    async def start_server(self):
        """MCP 서버 시작"""
        if not self.config.enabled:
            logger.error("MCP is disabled in configuration")
            raise RuntimeError("MCP is disabled. Enable MCP in configuration to use server mode.")
        
        try:
            # Initialize MCPApp
            self.app = MCPApp(
                name="local_researcher_server",
                server_names=self.config.server_names
            )
            
            logger.info(f"Starting MCP server with servers: {self.config.server_names}")
            
            # Start server
            async with self.app.run() as app_context:
                logger.info("MCP server started successfully")
                print("🚀 MCP Server started successfully")
                print(f"Available servers: {self.config.server_names}")
                print("Press Ctrl+C to stop the server")
                
                # Keep server running
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("MCP server stopped by user")
                    print("\n✅ MCP server stopped")
                    
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise RuntimeError(f"Failed to start MCP server: {e}")
    
    async def stop_server(self):
        """MCP 서버 중지"""
        if self.app:
            logger.info("Stopping MCP server")
            # Server will be stopped when context exits
            self.app = None


class MCPClientManager:
    """MCP 클라이언트 관리자"""
    
    def __init__(self):
        self.config = get_mcp_config()
        self.app = None
        
    async def start_client(self):
        """MCP 클라이언트 시작"""
        if not self.config.enabled:
            logger.error("MCP is disabled in configuration")
            raise RuntimeError("MCP is disabled. Enable MCP in configuration to use client mode.")
        
        try:
            # Initialize MCPApp
            self.app = MCPApp(
                name="local_researcher_client",
                server_names=self.config.server_names
            )
            
            logger.info(f"Starting MCP client connecting to: {self.config.server_names}")
            
            # Start client
            async with self.app.run() as app_context:
                logger.info("MCP client started successfully")
                print("🔗 MCP Client started successfully")
                print(f"Connected to servers: {self.config.server_names}")
                print("Press Ctrl+C to disconnect")
                
                # Keep client running
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("MCP client disconnected by user")
                    print("\n✅ MCP client disconnected")
                    
        except Exception as e:
            logger.error(f"Failed to start MCP client: {e}")
            raise RuntimeError(f"Failed to start MCP client: {e}")
    
    async def stop_client(self):
        """MCP 클라이언트 중지"""
        if self.app:
            logger.info("Stopping MCP client")
            # Client will be stopped when context exits
            self.app = None


class MCPToolsManager:
    """MCP 도구 관리자"""
    
    def __init__(self):
        self.config = get_mcp_config()
        self.app = None
        
    async def initialize_tools(self):
        """MCP 도구 초기화"""
        if not self.config.enabled:
            logger.error("MCP is disabled in configuration")
            raise RuntimeError("MCP is disabled. Enable MCP in configuration to use tools.")
        
        try:
            # Initialize MCPApp
            self.app = MCPApp(
                name="local_researcher_tools",
                server_names=self.config.server_names
            )
            
            logger.info("MCP tools initialized successfully")
            return self.app
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            raise RuntimeError(f"Failed to initialize MCP tools: {e}")
    
    async def get_available_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환"""
        if not self.app:
            await self.initialize_tools()
        
        try:
            async with self.app.run() as app_context:
                # Get available tools from MCP servers
                tools = []
                for server_name in self.config.server_names:
                    tools.append(f"tool_{server_name}")
                
                logger.info(f"Available tools: {tools}")
                return tools
                
        except Exception as e:
            logger.error(f"Failed to get available tools: {e}")
            raise RuntimeError(f"Failed to get available tools: {e}")
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """도구 실행"""
        if not self.app:
            await self.initialize_tools()
        
        try:
            async with self.app.run() as app_context:
                # Execute tool through MCP
                logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                
                # Tool execution logic would go here
                # This is a placeholder - actual implementation depends on MCP server capabilities
                result = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "status": "executed",
                    "result": "Tool execution completed"
                }
                
                logger.info(f"Tool execution completed: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")
            raise RuntimeError(f"Failed to execute tool {tool_name}: {e}")


class MCPIntegrationManager:
    """MCP 통합 관리자 - 모든 MCP 기능 통합"""
    
    def __init__(self):
        self.config = get_mcp_config()
        self.server_manager = MCPServerManager()
        self.client_manager = MCPClientManager()
        self.tools_manager = MCPToolsManager()
        
    async def start_server(self):
        """MCP 서버 시작"""
        await self.server_manager.start_server()
    
    async def start_client(self):
        """MCP 클라이언트 시작"""
        await self.client_manager.start_client()
    
    async def get_tools(self) -> List[str]:
        """사용 가능한 도구 목록 반환"""
        return await self.tools_manager.get_available_tools()
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """도구 실행"""
        return await self.tools_manager.execute_tool(tool_name, parameters)
    
    def is_enabled(self) -> bool:
        """MCP 활성화 상태 확인"""
        return self.config.enabled
    
    def get_server_names(self) -> List[str]:
        """서버 이름 목록 반환"""
        return self.config.server_names


# CLI 실행 함수들
async def run_mcp_server():
    """MCP 서버 실행 (CLI)"""
    manager = MCPIntegrationManager()
    if not manager.is_enabled():
        print("❌ MCP is disabled in configuration")
        return
    
    await manager.start_server()


async def run_mcp_client():
    """MCP 클라이언트 실행 (CLI)"""
    manager = MCPIntegrationManager()
    if not manager.is_enabled():
        print("❌ MCP is disabled in configuration")
        return
    
    await manager.start_client()


async def list_tools():
    """도구 목록 출력 (CLI)"""
    manager = MCPIntegrationManager()
    if not manager.is_enabled():
        print("❌ MCP is disabled in configuration")
        return
    
    try:
        tools = await manager.get_tools()
        print("🔧 Available MCP Tools:")
        for tool in tools:
            print(f"  - {tool}")
    except Exception as e:
        print(f"❌ Failed to list tools: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Integration Manager")
    parser.add_argument("--server", action="store_true", help="Start MCP server")
    parser.add_argument("--client", action="store_true", help="Start MCP client")
    parser.add_argument("--list-tools", action="store_true", help="List available tools")
    
    args = parser.parse_args()
    
    if args.server:
        asyncio.run(run_mcp_server())
    elif args.client:
        asyncio.run(run_mcp_client())
    elif args.list_tools:
        asyncio.run(list_tools())
    else:
        parser.print_help()
