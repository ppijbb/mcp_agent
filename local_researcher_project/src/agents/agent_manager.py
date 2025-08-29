"""
Agent Manager for Local Researcher

This module manages research agents and coordinates their activities
for research tasks.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

# Try to import asyncio, fallback to synchronous version if not available
try:
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    import time

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class AgentType(Enum):
    """Types of research agents."""
    TOPIC_ANALYZER = "topic_analyzer"
    SOURCE_DISCOVERER = "source_discoverer"
    CONTENT_GATHERER = "content_gatherer"
    CONTENT_ANALYZER = "content_analyzer"
    FACT_CHECKER = "fact_checker"
    QUALITY_ASSESSOR = "quality_assessor"
    REPORT_GENERATOR = "report_generator"


@dataclass
class AgentConfig:
    """Configuration for a research agent."""
    name: str
    agent_type: AgentType
    model: str
    max_tokens: int = 2000
    timeout: int = 300
    retry_count: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result of agent execution."""
    agent_name: str
    success: bool
    output: Any
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all research agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.status = AgentStatus.IDLE
        self.last_execution = None
        self.execution_count = 0
        self.error_count = 0
        
        logger.info(f"Initialized agent: {config.name}")
    
    @abstractmethod
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Execute the agent's main functionality.
        
        Args:
            input_data: Input data for the agent
            **kwargs: Additional arguments
            
        Returns:
            Agent execution result
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.
        
        Returns:
            Status information dictionary
        """
        return {
            "name": self.config.name,
            "type": self.config.agent_type.value,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "last_execution": self.last_execution,
            "execution_count": self.execution_count,
            "error_count": self.error_count
        }
    
    def reset(self):
        """Reset agent state."""
        self.status = AgentStatus.IDLE
        self.error_count = 0
        logger.info(f"Reset agent: {self.config.name}")


class TopicAnalyzerAgent(BaseAgent):
    """Agent for analyzing research topics."""
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Analyze a research topic.
        
        Args:
            input_data: Research topic string
            **kwargs: Additional arguments
            
        Returns:
            Topic analysis result
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        
        try:
            logger.info(f"Topic analyzer executing for: {input_data}")
            
            # Mock implementation - replace with actual logic
            if ASYNC_AVAILABLE:
                await asyncio.sleep(2)
            else:
                time.sleep(2)
            
            result = {
                "topic": input_data,
                "keywords": ["AI", "machine learning", "research"],
                "complexity": "medium",
                "estimated_sources": 15,
                "analysis_complete": True
            }
            
            self.status = AgentStatus.IDLE
            self.last_execution = start_time
            self.execution_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            self.last_execution = start_time
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Topic analyzer failed: {e}")
            
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e)
            )


class SourceDiscovererAgent(BaseAgent):
    """Agent for discovering research sources."""
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Discover research sources.
        
        Args:
            input_data: Topic analysis result
            **kwargs: Additional arguments
            
        Returns:
            Source discovery result
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        
        try:
            logger.info(f"Source discoverer executing")
            
            # Mock implementation - replace with actual logic
            if ASYNC_AVAILABLE:
                await asyncio.sleep(3)
            else:
                time.sleep(3)
            
            result = {
                "sources_found": 12,
                "source_types": ["academic", "news", "web"],
                "source_urls": [
                    "https://example.com/paper1",
                    "https://example.com/paper2",
                    "https://example.com/news1"
                ],
                "discovery_complete": True
            }
            
            self.status = AgentStatus.IDLE
            self.last_execution = start_time
            self.execution_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            self.last_execution = start_time
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Source discoverer failed: {e}")
            
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e)
            )


class ContentGathererAgent(BaseAgent):
    """Agent for gathering content from sources."""
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Gather content from sources.
        
        Args:
            input_data: Source discovery result
            **kwargs: Additional arguments
            
        Returns:
            Content gathering result
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        
        try:
            logger.info(f"Content gatherer executing")
            
            # Mock implementation - replace with actual logic
            if ASYNC_AVAILABLE:
                await asyncio.sleep(5)
            else:
                time.sleep(5)
            
            result = {
                "content_collected": True,
                "total_content_length": 25000,
                "sources_processed": 12,
                "content_summaries": [
                    "Summary of paper 1...",
                    "Summary of paper 2...",
                    "Summary of news 1..."
                ],
                "gathering_complete": True
            }
            
            self.status = AgentStatus.IDLE
            self.last_execution = start_time
            self.execution_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            self.last_execution = start_time
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Content gatherer failed: {e}")
            
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e)
            )


class ContentAnalyzerAgent(BaseAgent):
    """Agent for analyzing gathered content."""
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Analyze gathered content.
        
        Args:
            input_data: Content gathering result
            **kwargs: Additional arguments
            
        Returns:
            Content analysis result
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        
        try:
            logger.info(f"Content analyzer executing")
            
            # Mock implementation - replace with actual logic
            if ASYNC_AVAILABLE:
                await asyncio.sleep(4)
            else:
                time.sleep(4)
            
            result = {
                "analysis_complete": True,
                "key_insights": [
                    "AI is rapidly advancing in research",
                    "Machine learning shows promising results",
                    "New methodologies are emerging"
                ],
                "confidence_score": 0.85,
                "trends_identified": ["automation", "efficiency", "innovation"]
            }
            
            self.status = AgentStatus.IDLE
            self.last_execution = start_time
            self.execution_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            self.last_execution = start_time
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Content analyzer failed: {e}")
            
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e)
            )


class ReportGeneratorAgent(BaseAgent):
    """Agent for generating research reports."""
    
    async def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Generate research report.
        
        Args:
            input_data: Content analysis result
            **kwargs: Additional arguments
            
        Returns:
            Report generation result
        """
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        
        try:
            logger.info(f"Report generator executing")
            
            # Mock implementation - replace with actual logic
            if ASYNC_AVAILABLE:
                await asyncio.sleep(3)
            else:
                time.sleep(3)
            
            result = {
                "report_generated": True,
                "report_path": "outputs/research_report.md",
                "report_length": 1500,
                "report_sections": [
                    "Executive Summary",
                    "Introduction",
                    "Methodology",
                    "Findings",
                    "Conclusion"
                ],
                "generation_complete": True
            }
            
            self.status = AgentStatus.IDLE
            self.last_execution = start_time
            self.execution_count += 1
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.config.name,
                success=True,
                output=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error_count += 1
            self.last_execution = start_time
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Report generator failed: {e}")
            
            return AgentResult(
                agent_name=self.config.name,
                success=False,
                output=None,
                execution_time=execution_time,
                error_message=str(e)
            )


class AgentManager:
    """Manages research agents and their execution."""
    
    def __init__(self, config_manager):
        """Initialize the agent manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Load agent configurations
        self._load_agent_configs()
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("Agent Manager initialized")
    
    def _load_agent_configs(self):
        """Load agent configurations from config."""
        agent_configs = self.config_manager.get("open_deep_research.agents", {})
        
        for agent_name, config_data in agent_configs.items():
            if config_data.get("enabled", True):
                agent_type = AgentType(agent_name)
                
                config = AgentConfig(
                    name=agent_name,
                    agent_type=agent_type,
                    model=config_data.get("model", "gpt-4"),
                    max_tokens=config_data.get("max_tokens", 2000),
                    timeout=config_data.get("timeout", 300),
                    retry_count=config_data.get("retry_count", 3),
                    enabled=config_data.get("enabled", True),
                    metadata=config_data
                )
                
                self.agent_configs[agent_name] = config
        
        logger.info(f"Loaded {len(self.agent_configs)} agent configurations")
    
    def _initialize_agents(self):
        """Initialize agent instances."""
        agent_classes = {
            AgentType.TOPIC_ANALYZER: TopicAnalyzerAgent,
            AgentType.SOURCE_DISCOVERER: SourceDiscovererAgent,
            AgentType.CONTENT_GATHERER: ContentGathererAgent,
            AgentType.CONTENT_ANALYZER: ContentAnalyzerAgent,
            AgentType.REPORT_GENERATOR: ReportGeneratorAgent
        }
        
        for agent_name, config in self.agent_configs.items():
            if config.agent_type in agent_classes:
                agent_class = agent_classes[config.agent_type]
                agent = agent_class(config)
                self.agents[agent_name] = agent
                
                logger.info(f"Initialized agent: {agent_name}")
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def execute_agent(
        self,
        agent_name: str,
        input_data: Any,
        **kwargs
    ) -> Optional[AgentResult]:
        """Execute a specific agent.
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            **kwargs: Additional arguments
            
        Returns:
            Agent execution result or None if agent not found
        """
        if agent_name not in self.agents:
            logger.error(f"Agent not found: {agent_name}")
            return None
        
        agent = self.agents[agent_name]
        
        if not agent.config.enabled:
            logger.warning(f"Agent is disabled: {agent_name}")
            return None
        
        if agent.status == AgentStatus.BUSY:
            logger.warning(f"Agent is busy: {agent_name}")
            return None
        
        try:
            result = await agent.execute(input_data, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Agent execution failed: {agent_name}, Error: {e}")
            return None
    
    async def execute_agent_chain(
        self,
        agent_names: List[str],
        initial_input: Any,
        **kwargs
    ) -> List[AgentResult]:
        """Execute a chain of agents sequentially.
        
        Args:
            agent_names: List of agent names to execute in order
            initial_input: Initial input for the first agent
            **kwargs: Additional arguments
            
        Returns:
            List of agent execution results
        """
        results = []
        current_input = initial_input
        
        for agent_name in agent_names:
            logger.info(f"Executing agent in chain: {agent_name}")
            
            result = await self.execute_agent(agent_name, current_input, **kwargs)
            
            if result is None:
                logger.error(f"Agent chain failed at: {agent_name}")
                break
            
            results.append(result)
            
            if not result.success:
                logger.error(f"Agent chain failed at: {agent_name}")
                break
            
            # Use output as input for next agent
            current_input = result.output
        
        return results
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent status or None if not found
        """
        if agent_name not in self.agents:
            return None
        
        return self.agents[agent_name].get_status()
    
    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents.
        
        Returns:
            Dictionary of agent statuses
        """
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
    
    def enable_agent(self, agent_name: str) -> bool:
        """Enable a specific agent.
        
        Args:
            agent_name: Name of the agent to enable
            
        Returns:
            True if enabled successfully, False otherwise
        """
        if agent_name in self.agents:
            self.agents[agent_name].config.enabled = True
            logger.info(f"Enabled agent: {agent_name}")
            return True
        return False
    
    def disable_agent(self, agent_name: str) -> bool:
        """Disable a specific agent.
        
        Args:
            agent_name: Name of the agent to disable
            
        Returns:
            True if disabled successfully, False otherwise
        """
        if agent_name in self.agents:
            self.agents[agent_name].config.enabled = False
            logger.info(f"Disabled agent: {agent_name}")
            return True
        return False
    
    def reset_agent(self, agent_name: str) -> bool:
        """Reset a specific agent.
        
        Args:
            agent_name: Name of the agent to reset
            
        Returns:
            True if reset successfully, False otherwise
        """
        if agent_name in self.agents:
            self.agents[agent_name].reset()
            return True
        return False
    
    def list_agents(self) -> List[str]:
        """List all available agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent information or None if not found
        """
        if agent_name not in self.agents:
            return None
        
        agent = self.agents[agent_name]
        config = agent.config
        
        return {
            "name": config.name,
            "type": config.agent_type.value,
            "model": config.model,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "retry_count": config.retry_count,
            "enabled": config.enabled,
            "status": agent.status.value,
            "last_execution": agent.last_execution,
            "execution_count": agent.execution_count,
            "error_count": agent.error_count,
            "metadata": config.metadata
        }
