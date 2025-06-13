"""
Business Strategy Agents - Real MCPAgent Implementation
======================================================

This package contains real MCPAgent implementations for comprehensive 
business strategy analysis and planning.

All agents use the standard mcp_agent library for reliable, 
quality-controlled business intelligence workflows.
"""

# Version and metadata
__version__ = "2.0.0"
__description__ = "Real MCPAgent Business Strategy Suite"
__author__ = "MCPAgent Team"

# Import real MCPAgents only
try:
    from .business_data_scout_agent import (
        BusinessDataScoutMCPAgent,
        run_business_data_scout,
        create_business_data_scout
    )
except ImportError as e:
    print(f"Warning: Could not import BusinessDataScoutMCPAgent: {e}")
    BusinessDataScoutMCPAgent = None

try:
    from .trend_analyzer_agent import (
        TrendAnalyzerMCPAgent,
        run_trend_analysis,
        create_trend_analyzer
    )
except ImportError as e:
    print(f"Warning: Could not import TrendAnalyzerMCPAgent: {e}")
    TrendAnalyzerMCPAgent = None

try:
    from .strategy_planner_agent import (
        StrategyPlannerMCPAgent,
        run_strategy_planning,
        create_strategy_planner
    )
except ImportError as e:
    print(f"Warning: Could not import StrategyPlannerMCPAgent: {e}")
    StrategyPlannerMCPAgent = None

try:
    from .unified_business_strategy_agent import (
        UnifiedBusinessStrategyMCPAgent,
        run_unified_business_strategy,
        create_unified_business_strategy
    )
except ImportError as e:
    print(f"Warning: Could not import UnifiedBusinessStrategyMCPAgent: {e}")
    UnifiedBusinessStrategyMCPAgent = None

try:
    from .run_business_strategy_agents import (
        BusinessStrategyRunner
    )
except ImportError as e:
    print(f"Warning: Could not import BusinessStrategyRunner: {e}")
    BusinessStrategyRunner = None

# Import supporting modules (if compatible with MCPAgent)
try:
    from .config import get_config
except ImportError as e:
    print(f"Warning: Could not import config: {e}")
    get_config = None

try:
    from .architecture import (
        RegionType,
        BusinessOpportunityLevel,
        ContentType,
        DataSource,
        RawContent,
        ProcessedInsight,
        BusinessStrategy
    )
except ImportError as e:
    print(f"Warning: Could not import architecture components: {e}")

try:
    from .notion_integration import get_notion_integration
except ImportError as e:
    print(f"Warning: Could not import notion_integration: {e}")
    get_notion_integration = None

# Public API - Business Strategy MCPAgents
__all__ = [
    # Business Strategy MCPAgent Classes
    "BusinessDataScoutMCPAgent",
    "TrendAnalyzerMCPAgent", 
    "StrategyPlannerMCPAgent",
    "UnifiedBusinessStrategyMCPAgent",
    
    # Runner and execution functions
    "BusinessStrategyRunner",
    "run_business_data_scout",
    "run_trend_analysis",
    "run_strategy_planning", 
    "run_unified_business_strategy",
    
    # Factory functions
    "create_business_data_scout",
    "create_trend_analyzer",
    "create_strategy_planner",
    "create_unified_business_strategy",
    
    # Supporting components (if available)
    "get_config",
    "get_notion_integration",
    
    # Data structures
    "RegionType",
    "BusinessOpportunityLevel",
    "ContentType",
    "DataSource",
    "RawContent",
    "ProcessedInsight",
    "BusinessStrategy",
]

# Convenience functions
def get_available_agents():
    """Get list of available business strategy MCPAgents"""
    agents = []
    
    if BusinessDataScoutMCPAgent:
        agents.append("BusinessDataScoutMCPAgent")
    if TrendAnalyzerMCPAgent:
        agents.append("TrendAnalyzerMCPAgent")
    if StrategyPlannerMCPAgent:
        agents.append("StrategyPlannerMCPAgent")
    if UnifiedBusinessStrategyMCPAgent:
        agents.append("UnifiedBusinessStrategyMCPAgent")
    
    return agents

def get_package_info():
    """Get package information"""
    return {
        "name": "business_strategy_agents",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "agent_type": "Business Strategy MCPAgent",
        "architecture": "mcp_agent.app.MCPApp + mcp_agent.agents.agent.Agent",
        "available_agents": get_available_agents(),
        "total_agents": len(get_available_agents())
    }

# Module initialization
def _initialize_package():
    """Initialize the package and check dependencies"""
    try:
        import mcp_agent
        print(f"✅ Business Strategy MCPAgent package initialized successfully")
        print(f"📊 Available agents: {len(get_available_agents())}")
        return True
    except ImportError:
        print("⚠️  Warning: mcp_agent library not found")
        print("📦 Install with: pip install mcp_agent")
        return False

# Auto-initialize when imported
_package_ready = _initialize_package()

# Simple deprecation warnings for old imports
def __getattr__(name):
    """Handle deprecated imports with helpful error messages"""
    deprecated_agents = [
        "BaseAgent", "DataScoutAgent", "TrendAnalyzerAgent", 
        "HookingDetectorAgent", "StrategyPlannerAgent", "AgentOrchestrator"
    ]
    
    if name in deprecated_agents:
        raise ImportError(
            f"❌ {name} is deprecated.\n"
            f"✅ Use current MCPAgent equivalent instead:\n"
            f"   - BusinessDataScoutMCPAgent\n"
            f"   - TrendAnalyzerMCPAgent\n" 
            f"   - StrategyPlannerMCPAgent\n"
            f"   - UnifiedBusinessStrategyMCPAgent"
        )
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'") 