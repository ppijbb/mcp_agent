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

try:
    from .ontology_builder import OntologyBuilder
    ONTOLOGY_BUILDER_AVAILABLE = True
except ImportError:
    ONTOLOGY_BUILDER_AVAILABLE = False

try:
    from .query_translator import QueryTranslator
    QUERY_TRANSLATOR_AVAILABLE = True
except ImportError:
    QUERY_TRANSLATOR_AVAILABLE = False

try:
    from .goal_ontology_builder import GoalOntologyBuilder
    GOAL_ONTOLOGY_BUILDER_AVAILABLE = True
except ImportError:
    GOAL_ONTOLOGY_BUILDER_AVAILABLE = False

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

if ONTOLOGY_BUILDER_AVAILABLE:
    __all__.append("OntologyBuilder")

if QUERY_TRANSLATOR_AVAILABLE:
    __all__.append("QueryTranslator")

if GOAL_ONTOLOGY_BUILDER_AVAILABLE:
    __all__.append("GoalOntologyBuilder")

try:
    from .task_executor import TaskExecutor, ExecutionResult
    TASK_EXECUTOR_AVAILABLE = True
except ImportError:
    TASK_EXECUTOR_AVAILABLE = False

if TASK_EXECUTOR_AVAILABLE:
    __all__.extend(["TaskExecutor", "ExecutionResult"])

try:
    from .goal_path_finder import GoalPathFinder, AchievementPath
    GOAL_PATH_FINDER_AVAILABLE = True
except ImportError:
    GOAL_PATH_FINDER_AVAILABLE = False

if GOAL_PATH_FINDER_AVAILABLE:
    __all__.extend(["GoalPathFinder", "AchievementPath"])

try:
    from .system_query_translator import SystemQueryTranslator, SystemQueryTranslation
    SYSTEM_QUERY_TRANSLATOR_AVAILABLE = True
except ImportError:
    SYSTEM_QUERY_TRANSLATOR_AVAILABLE = False

if SYSTEM_QUERY_TRANSLATOR_AVAILABLE:
    __all__.extend(["SystemQueryTranslator", "SystemQueryTranslation"])