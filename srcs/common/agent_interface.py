"""
공통 Agent 인터페이스 및 메타데이터 정의

모든 agent 타입에 대한 표준 인터페이스를 제공합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import asyncio


class AgentType(Enum):
    """Agent 타입 정의"""
    MCP_AGENT = "mcp_agent"
    LANGGRAPH_AGENT = "langgraph_agent"
    CRON_AGENT = "cron_agent"
    SPARKLEFORGE_AGENT = "sparkleforge_agent"
    CREWAI_AGENT = "crewai_agent"
    CUSTOM_AGENT = "custom_agent"


class AgentStatus(Enum):
    """Agent 상태"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentMetadata:
    """Agent 메타데이터"""
    agent_id: str
    agent_name: str
    agent_type: AgentType
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None  # 모듈 경로 또는 실행 스크립트
    config_schema: Optional[Dict[str, Any]] = None
    a2a_endpoint: Optional[str] = None  # A2A 통신 엔드포인트
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "capabilities": self.capabilities,
            "requirements": self.requirements,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "a2a_endpoint": self.a2a_endpoint,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class AgentExecutionResult:
    """Agent 실행 결과"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent(ABC):
    """모든 Agent의 기본 추상 클래스"""
    
    def __init__(self, metadata: AgentMetadata):
        self.metadata = metadata
        self.status = AgentStatus.IDLE
        self.current_result: Optional[AgentExecutionResult] = None
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> AgentExecutionResult:
        """
        Agent 실행 (비동기)
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            AgentExecutionResult: 실행 결과
        """
        pass
    
    def execute_sync(self, input_data: Dict[str, Any]) -> AgentExecutionResult:
        """
        Agent 실행 (동기)
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            AgentExecutionResult: 실행 결과
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.execute(input_data))
            return result
        finally:
            loop.close()
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Agent의 능력 목록 반환"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 검증"""
        pass
    
    def get_status(self) -> AgentStatus:
        """현재 상태 반환"""
        return self.status
    
    def get_metadata(self) -> AgentMetadata:
        """메타데이터 반환"""
        return self.metadata


class AgentRunner:
    """Agent 실행을 위한 러너 클래스"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
    
    async def run(self, input_data: Dict[str, Any]) -> AgentExecutionResult:
        """
        Agent 실행
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            AgentExecutionResult: 실행 결과
        """
        if not self.agent.validate_input(input_data):
            return AgentExecutionResult(
                success=False,
                error="Invalid input data",
                metadata={"agent_id": self.agent.metadata.agent_id}
            )
        
        self.agent.status = AgentStatus.RUNNING
        start_time = datetime.now()
        
        try:
            result = await self.agent.execute(input_data)
            self.agent.current_result = result
            self.agent.status = AgentStatus.COMPLETED if result.success else AgentStatus.ERROR
            return result
        except Exception as e:
            self.agent.status = AgentStatus.ERROR
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                metadata={"agent_id": self.agent.metadata.agent_id}
            )
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.agent.current_result:
                self.agent.current_result.execution_time = execution_time
    
    def run_sync(self, input_data: Dict[str, Any]) -> AgentExecutionResult:
        """
        Agent 실행 (동기)
        
        Args:
            input_data: 입력 데이터
            
        Returns:
            AgentExecutionResult: 실행 결과
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run(input_data))
        finally:
            loop.close()

