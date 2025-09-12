"""
Advanced Trading Tools Package
Comprehensive tools for Ethereum trading analysis and execution
"""

from .market_data_tools import AdvancedMarketDataCollector, MarketDataConfig, DataSource
from .technical_analysis_tools import AdvancedTechnicalAnalyzer, TechnicalAnalysisConfig, IndicatorType
from .fundamental_analysis_tools import AdvancedFundamentalAnalyzer, FundamentalAnalysisConfig, AnalysisType
from .sentiment_analysis_tools import AdvancedSentimentAnalyzer, SentimentAnalysisConfig, SentimentSource
from .risk_management_tools import AdvancedRiskManager, RiskManagementConfig, RiskType
from .portfolio_management_tools import AdvancedPortfolioManager, PortfolioConfig, PortfolioAction
from .execution_tools import AdvancedExecutionManager, ExecutionConfig, OrderType, OrderSide, OrderStatus

__all__ = [
    # Market Data Tools
    "AdvancedMarketDataCollector",
    "MarketDataConfig", 
    "DataSource",
    
    # Technical Analysis Tools
    "AdvancedTechnicalAnalyzer",
    "TechnicalAnalysisConfig",
    "IndicatorType",
    
    # Fundamental Analysis Tools
    "AdvancedFundamentalAnalyzer",
    "FundamentalAnalysisConfig",
    "AnalysisType",
    
    # Sentiment Analysis Tools
    "AdvancedSentimentAnalyzer",
    "SentimentAnalysisConfig",
    "SentimentSource",
    
    # Risk Management Tools
    "AdvancedRiskManager",
    "RiskManagementConfig",
    "RiskType",
    
    # Portfolio Management Tools
    "AdvancedPortfolioManager",
    "PortfolioConfig",
    "PortfolioAction",
    
    # Execution Tools
    "AdvancedExecutionManager",
    "ExecutionConfig",
    "OrderType",
    "OrderSide",
    "OrderStatus"
]

# Tool categories for easy access
MARKET_DATA_TOOLS = {
    "collector": AdvancedMarketDataCollector,
    "config": MarketDataConfig,
    "sources": DataSource
}

TECHNICAL_ANALYSIS_TOOLS = {
    "analyzer": AdvancedTechnicalAnalyzer,
    "config": TechnicalAnalysisConfig,
    "indicators": IndicatorType
}

FUNDAMENTAL_ANALYSIS_TOOLS = {
    "analyzer": AdvancedFundamentalAnalyzer,
    "config": FundamentalAnalysisConfig,
    "types": AnalysisType
}

SENTIMENT_ANALYSIS_TOOLS = {
    "analyzer": AdvancedSentimentAnalyzer,
    "config": SentimentAnalysisConfig,
    "sources": SentimentSource
}

RISK_MANAGEMENT_TOOLS = {
    "manager": AdvancedRiskManager,
    "config": RiskManagementConfig,
    "types": RiskType
}

PORTFOLIO_MANAGEMENT_TOOLS = {
    "manager": AdvancedPortfolioManager,
    "config": PortfolioConfig,
    "actions": PortfolioAction
}

EXECUTION_TOOLS = {
    "manager": AdvancedExecutionManager,
    "config": ExecutionConfig,
    "order_types": OrderType,
    "order_sides": OrderSide,
    "order_status": OrderStatus
}

# All tool categories
TOOL_CATEGORIES = {
    "market_data": MARKET_DATA_TOOLS,
    "technical_analysis": TECHNICAL_ANALYSIS_TOOLS,
    "fundamental_analysis": FUNDAMENTAL_ANALYSIS_TOOLS,
    "sentiment_analysis": SENTIMENT_ANALYSIS_TOOLS,
    "risk_management": RISK_MANAGEMENT_TOOLS,
    "portfolio_management": PORTFOLIO_MANAGEMENT_TOOLS,
    "execution": EXECUTION_TOOLS
}

def get_tool_category(category: str):
    """Get tools for a specific category"""
    return TOOL_CATEGORIES.get(category, {})

def get_all_tools():
    """Get all available tools"""
    return TOOL_CATEGORIES

def create_tool_instance(tool_name: str, category: str, config: dict = None):
    """Create an instance of a specific tool"""
    try:
        category_tools = get_tool_category(category)
        tool_class = category_tools.get(tool_name)
        
        if not tool_class:
            raise ValueError(f"Tool {tool_name} not found in category {category}")
        
        if config:
            return tool_class(**config)
        else:
            return tool_class()
            
    except Exception as e:
        raise ValueError(f"Failed to create tool instance: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Ethereum Trading Agent Team"
__description__ = "Advanced trading tools for Ethereum market analysis and execution"
