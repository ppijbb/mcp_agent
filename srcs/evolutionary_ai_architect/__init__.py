"""
Evolutionary AI Architect Module

Advanced AI agent that evolves AI architectures using genetic algorithms,
self-improvement mechanisms, and meta-learning capabilities.

Main Components:
- EvolutionaryAIArchitectAgent: Main evolutionary agent
- ArchitectureGenome: Genetic encoding for AI architectures  
- SelfImprovementEngine: Performance monitoring and improvement
- AIArchitectureDesigner: Architecture design and generation
"""

try:
    from .evolutionary_ai_architect_agent import EvolutionaryAIArchitectMCP, EvolutionaryAIArchitectAgent
except ImportError as e:
    print(f"Warning: Could not import EvolutionaryAIArchitectAgent: {e}")
    EvolutionaryAIArchitectMCP = None
    EvolutionaryAIArchitectAgent = None

try:
    from .genome import ArchitectureGenome, PerformanceMetrics, EvolutionHistory
except ImportError as e:
    print(f"Warning: Could not import genome components: {e}")
    ArchitectureGenome = None
    PerformanceMetrics = None
    EvolutionHistory = None

try:
    from .improvement_engine import SelfImprovementEngine, SelfImprovementEngineMCP
except ImportError as e:
    print(f"Warning: Could not import SelfImprovementEngine: {e}")
    SelfImprovementEngine = None
    SelfImprovementEngineMCP = None

try:
    from .architect import AIArchitectureDesigner, AIArchitectMCP
except ImportError as e:
    print(f"Warning: Could not import AIArchitectureDesigner: {e}")
    AIArchitectureDesigner = None
    AIArchitectMCP = None

__all__ = [
    'EvolutionaryAIArchitectMCP',
    'EvolutionaryAIArchitectAgent', 
    'ArchitectureGenome',
    'PerformanceMetrics',
    'EvolutionHistory',
    'SelfImprovementEngine',
    'SelfImprovementEngineMCP',
    'AIArchitectureDesigner',
    'AIArchitectMCP'
]

__version__ = '1.0.0' 