"""
Base Agent Class for GraphRAG Agents

This module provides a common base class for all GraphRAG agents,
containing shared functionality like logging, metrics, and common utilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
except ImportError:
    BaseModel = object
    Field = None
    field_validator = None
    ConfigDict = None

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
except ImportError:
    Console = None
    Progress = None
    TaskID = None

# LangChain imports are optional for basic functionality
try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_core.cache import BaseCache
    from langchain_community.cache import RedisCache, SQLiteCache
    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseLanguageModel = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    Document = None
    VectorStore = None
    Chroma = None
    FAISS = None
    BaseCache = None
    RedisCache = None
    SQLiteCache = None
    LANGCHAIN_AVAILABLE = False


class BaseAgentConfig:
    """Base configuration for all agents"""
    def __init__(self, **kwargs):
        # Default values
        self.model_name = kwargs.get("model_name", "gpt-4o-mini")
        self.openai_api_key = kwargs.get("openai_api_key", "")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1000)
        self.timeout = kwargs.get("timeout", 30)
        self.retry_attempts = kwargs.get("retry_attempts", 3)
        self.enable_caching = kwargs.get("enable_caching", True)
        self.cache_ttl = kwargs.get("cache_ttl", 3600)
        self.enable_metrics = kwargs.get("enable_metrics", True)
        self.metrics_retention_days = kwargs.get("metrics_retention_days", 30)
        self.log_level = kwargs.get("log_level", "INFO")
        self.enable_rich_logging = kwargs.get("enable_rich_logging", True)
        
        # Set any additional kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4000, ge=100, le=8000, description="Maximum tokens")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, ge=60, le=86400, description="Cache TTL in seconds")
    max_concurrency: int = Field(default=4, ge=1, le=20, description="Maximum concurrent operations")
    
    # Advanced features
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    enable_metrics_tracking: bool = Field(default=True, description="Enable metrics tracking")
    
    @field_validator('model_name')
    def validate_model_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class BaseAgent(ABC):
    """
    Base class for all GraphRAG agents
    
    Provides common functionality including:
    - Structured logging setup
    - Metrics tracking
    - Console output
    - Common utilities
    - Error handling
    """
    
    def __init__(self, config: BaseAgentConfig):
        self.config = config
        self.console = Console()
        self._setup_logging()
        self._setup_metrics()
        self._setup_components()
        
        self.logger.info("Agent initialized", 
                        agent_type=self.__class__.__name__,
                        config=config.model_dump())
    
    def _setup_logging(self):
        """Setup structured logging for the agent"""
        if not self.config.enable_structured_logging:
            return
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _setup_metrics(self):
        """Setup performance metrics tracking"""
        if not self.config.enable_metrics_tracking:
            return
        
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'last_operation_time': None
        }
    
    def _setup_components(self):
        """Setup common components"""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model_kwargs={
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Initialize cache if enabled
            self.cache = None
            if self.config.enable_caching:
                try:
                    self.cache = RedisCache(ttl=self.config.cache_ttl)
                except:
                    self.cache = SQLiteCache()
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error("Component initialization failed", error=str(e))
            raise
    
    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update performance metrics"""
        if not self.config.enable_metrics_tracking:
            return
        
        self.metrics['total_operations'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['total_operations']
        )
        self.metrics['last_operation_time'] = datetime.now().isoformat()
        
        if success:
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['failed_operations'] += 1
            self.metrics['errors'] += 1
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        if not self.config.enable_metrics_tracking:
            return
        
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'last_operation_time': None
        }
        self.logger.info("Metrics reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.config.enable_metrics_tracking:
            return {}
        return self.metrics.copy()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        summary = self.config.model_dump()
        summary['agent_type'] = self.__class__.__name__
        summary['status'] = 'ready'
        
        if self.config.enable_metrics_tracking:
            summary['metrics'] = self.get_metrics()
        
        return summary
    
    async def test_connectivity(self) -> bool:
        """Test if the agent can connect to required services"""
        try:
            # Test LLM connectivity
            test_response = await self.llm.ainvoke("Hello")
            if not test_response or not test_response.content:
                return False
            
            # Test embeddings
            test_embedding = await self.embeddings.aembed_query("test")
            if not test_embedding:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Connectivity test failed", error=str(e))
            return False
    
    def _convert_to_networkx(self, knowledge_graph: Any) -> nx.Graph:
        """Convert knowledge graph to NetworkX graph"""
        try:
            G = nx.Graph()
            
            # Add nodes
            for node in knowledge_graph.nodes:
                node_id = getattr(node, 'id', str(node))
                node_data = {
                    'title': getattr(node, 'title', ''),
                    'type': getattr(node, 'type', ''),
                    'description': getattr(node, 'description', '')
                }
                G.add_node(node_id, **node_data)
            
            # Add edges
            for edge in knowledge_graph.edges:
                source_id = getattr(edge.source, 'id', str(edge.source))
                target_id = getattr(edge.target, 'id', str(edge.target))
                edge_data = {
                    'type': getattr(edge, 'type', ''),
                    'description': getattr(edge, 'description', '')
                }
                G.add_edge(source_id, target_id, **edge_data)
            
            return G
            
        except Exception as e:
            self.logger.error("Graph conversion failed", error=str(e))
            return nx.Graph()
    
    def _get_graph_size(self, knowledge_graph: Any) -> Dict[str, int]:
        """Get basic graph size metrics"""
        try:
            num_nodes = len(knowledge_graph.nodes) if hasattr(knowledge_graph, 'nodes') else 0
            num_edges = len(knowledge_graph.edges) if hasattr(knowledge_graph, 'edges') else 0
            return {'nodes': num_nodes, 'edges': num_edges}
        except Exception as e:
            self.logger.error("Graph size calculation failed", error=str(e))
            return {'nodes': 0, 'edges': 0}
    
    def _format_node_content(self, node: Any) -> str:
        """Format a node's content for the vector store"""
        content_parts = []
        
        # Add title/name
        if hasattr(node, 'title') and node.title:
            content_parts.append(f"Entity: {node.title}")
        elif hasattr(node, 'name') and node.name:
            content_parts.append(f"Entity: {node.name}")
        elif hasattr(node, 'id') and node.id:
            content_parts.append(f"Entity ID: {node.id}")
        
        # Add description
        if hasattr(node, 'description') and node.description:
            content_parts.append(f"Description: {node.description}")
        
        # Add type
        if hasattr(node, 'type') and node.type:
            content_parts.append(f"Type: {node.type}")
        
        # Add properties if available
        if hasattr(node, 'properties') and node.properties:
            for key, value in node.properties.items():
                if value and str(value).strip():
                    content_parts.append(f"{key}: {value}")
        
        return "\n".join(content_parts) if content_parts else f"Entity: {str(node)}"
    
    def _format_edge_content(self, edge: Any) -> str:
        """Format an edge's content for the vector store"""
        content_parts = []
        
        # Add relationship description
        if hasattr(edge, 'description') and edge.description:
            content_parts.append(f"Relationship: {edge.description}")
        
        # Add source and target
        if hasattr(edge, 'source') and edge.source:
            source_name = getattr(edge.source, 'title', getattr(edge.source, 'name', str(edge.source)))
            content_parts.append(f"From: {source_name}")
        
        if hasattr(edge, 'target') and edge.target:
            target_name = getattr(edge.target, 'title', getattr(edge.target, 'name', str(edge.target)))
            content_parts.append(f"To: {target_name}")
        
        # Add type
        if hasattr(edge, 'type') and edge.type:
            content_parts.append(f"Type: {edge.type}")
        
        return "\n".join(content_parts) if content_parts else f"Relationship: {str(edge)}"
    
    def _clean_text_data(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove problematic characters
        text = text.replace('\x00', '').replace('\ufffd', '')
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _calculate_text_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate text quality metrics"""
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'quality_score': 0.0
            }
        
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Simple quality score based on length and structure
        quality_score = min(1.0, (word_count / 100) * (sentence_count / 10))
        
        return {
            'length': len(text),
            'word_count': word_count,
            'sentence_count': sentence_count,
            'quality_score': quality_score
        }
    
    def _generate_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check cache for key"""
        if not self.cache:
            return None
        
        try:
            return self.cache.get(key)
        except Exception as e:
            self.logger.error("Cache retrieval failed", key=key, error=str(e))
            return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set cache value for key"""
        if not self.cache:
            return
        
        try:
            self.cache.set(key, value)
        except Exception as e:
            self.logger.error("Cache storage failed", key=key, error=str(e))
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Main processing method - must be implemented by subclasses"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.model_name})"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} - {self.config.model_name}"
