"""
GraphRAG Agents

This module contains the core agent implementations and workflows for the GraphRAG system.
"""

from .graph_generator import GraphGeneratorNode
from .rag_agent import RAGAgentNode
from .workflow import GraphRAGWorkflow
from .graphrag_agent import GraphRAGAgent
from .natural_language_agent import NaturalLanguageAgent
from .llm_processor import LLMProcessor

# Import intelligent agent components with error handling
try:
    from .intelligent_agent import IntelligentGraphRAGAgent
    INTELLIGENT_AGENT_AVAILABLE = True
except ImportError:
    INTELLIGENT_AGENT_AVAILABLE = False

try:
    from .autonomous_behavior import AutonomousBehaviorEngine
    AUTONOMOUS_BEHAVIOR_AVAILABLE = True
except ImportError:
    AUTONOMOUS_BEHAVIOR_AVAILABLE = False

__all__ = [
    "GraphGeneratorNode",
    "RAGAgentNode",
    "GraphRAGWorkflow",
    "GraphRAGAgent",
    "NaturalLanguageAgent",
    "LLMProcessor"
]

# Add intelligent components if available
if INTELLIGENT_AGENT_AVAILABLE:
    __all__.append("IntelligentGraphRAGAgent")

if AUTONOMOUS_BEHAVIOR_AVAILABLE:
    __all__.append("AutonomousBehaviorEngine")