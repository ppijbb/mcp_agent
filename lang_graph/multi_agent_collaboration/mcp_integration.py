"""
MCP (Model Context Protocol) 통합 모듈
에이전트가 외부 도구와의 연동을 표준화하여 다양한 기능에 접근할 수 있도록 합니다.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolType(Enum):
    """MCP 도구 타입 정의"""
    SEARCH = "search"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    DATABASE = "database"
    CUSTOM = "custom"

@dataclass
class MCPTool:
    """MCP 도구 정의"""
    name: str
    description: str
    tool_type: MCPToolType
    parameters: Dict[str, Any]
    handler: Callable
    required_permissions: List[str]
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """도구를 딕셔너리로 변환"""
        return asdict(self)

class MCPRegistry:
    """MCP 도구 레지스트리"""
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._permissions: Dict[str, List[str]] = {}
    
    def register_tool(self, tool: MCPTool) -> None:
        """도구 등록"""
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting...")
        self._tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """도구 조회"""
        return self._tools.get(name)
    
    def list_tools(self, agent_id: str = None) -> List[MCPTool]:
        """사용 가능한 도구 목록 조회"""
        if agent_id is None:
            return list(self._tools.values())
        
        # 에이전트별 권한에 따른 도구 필터링
        agent_permissions = self._permissions.get(agent_id, [])
        return [
            tool for tool in self._tools.values()
            if all(perm in agent_permissions for perm in tool.required_permissions)
        ]
    
    def grant_permissions(self, agent_id: str, permissions: List[str]) -> None:
        """에이전트에게 권한 부여"""
        if agent_id not in self._permissions:
            self._permissions[agent_id] = []
        self._permissions[agent_id].extend(permissions)
        logger.info(f"Granted permissions {permissions} to agent {agent_id}")
    
    def revoke_permissions(self, agent_id: str, permissions: List[str]) -> None:
        """에이전트의 권한 회수"""
        if agent_id in self._permissions:
            for perm in permissions:
                if perm in self._permissions[agent_id]:
                    self._permissions[agent_id].remove(perm)
            logger.info(f"Revoked permissions {permissions} from agent {agent_id}")

class MCPExecutor:
    """MCP 도구 실행기"""
    
    def __init__(self, registry: MCPRegistry):
        self.registry = registry
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          agent_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """도구 실행"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found",
                "tool_name": tool_name
            }
        
        # 권한 확인
        agent_permissions = self.registry._permissions.get(agent_id, [])
        if not all(perm in agent_permissions for perm in tool.required_permissions):
            return {
                "success": False,
                "error": f"Insufficient permissions for tool {tool_name}",
                "tool_name": tool_name,
                "required": tool.required_permissions,
                "granted": agent_permissions
            }
        
        try:
            # 도구 실행
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**parameters)
            else:
                result = tool.handler(**parameters)
            
            execution_record = {
                "tool_name": tool_name,
                "agent_id": agent_id,
                "parameters": parameters,
                "result": result,
                "timestamp": asyncio.get_event_loop().time(),
                "context": context
            }
            self.execution_history.append(execution_record)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name,
                "execution_id": len(self.execution_history) - 1
            }
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    def get_execution_history(self, agent_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """실행 히스토리 조회"""
        history = self.execution_history[-limit:] if limit else self.execution_history
        if agent_id:
            return [record for record in history if record["agent_id"] == agent_id]
        return history

# 전역 MCP 레지스트리 및 실행기 인스턴스
mcp_registry = MCPRegistry()
mcp_executor = MCPExecutor(mcp_registry)

# 기본 도구들 등록
def register_default_tools():
    """기본 MCP 도구들 등록"""
    
    # 검색 도구
    def search_handler(query: str, max_results: int = 5) -> Dict[str, Any]:
        # 실제 검색 로직은 기존 search_tool과 연동
        return {
            "query": query,
            "max_results": max_results,
            "status": "search_ready"
        }
    
    search_tool = MCPTool(
        name="web_search",
        description="웹 검색을 수행하여 관련 정보를 찾습니다",
        tool_type=MCPToolType.SEARCH,
        parameters={
            "query": {"type": "string", "description": "검색 쿼리"},
            "max_results": {"type": "integer", "description": "최대 결과 수", "default": 5}
        },
        handler=search_handler,
        required_permissions=["search"]
    )
    
    # 코드 실행 도구
    def code_execution_handler(code: str, timeout: int = 15) -> Dict[str, Any]:
        # 기존 mcp_tool_executor와 연동
        return {
            "code": code,
            "timeout": timeout,
            "status": "execution_ready"
        }
    
    code_tool = MCPTool(
        name="code_execution",
        description="Python 코드를 안전하게 실행합니다",
        tool_type=MCPToolType.CODE_EXECUTION,
        parameters={
            "code": {"type": "string", "description": "실행할 Python 코드"},
            "timeout": {"type": "integer", "description": "실행 제한 시간(초)", "default": 15}
        },
        handler=code_execution_handler,
        required_permissions=["code_execution"]
    )
    
    # 도구 등록
    mcp_registry.register_tool(search_tool)
    mcp_registry.register_tool(code_tool)
    
    logger.info("Default MCP tools registered successfully")

# 기본 도구 등록
register_default_tools()
