"""
Most Hooking Business Strategy Agent Package

This package provides a comprehensive business intelligence system that monitors
global digital trends and generates actionable business insights with hooking opportunities.

🌟 Key Features:
- 360-degree global monitoring (News, Social Media, Communities, Trends)
- MCP server integration for diverse data sources
- AI-powered hooking point detection
- Cross-regional analysis (East Asia & North America focus)
- Automated Notion documentation
- Real-time business opportunity scoring

📁 Package Structure:
- architecture.py: Core system architecture and interfaces
- config.py: Configuration management and environment settings
- mcp_layer.py: MCP server communication layer
- ai_engine.py: AI processing and analysis engines
- notion_integration.py: Notion API integration
- main_agent.py: Main orchestration agent
- demo.py: Demonstration and testing scripts
"""

from .architecture import (
    CoreArchitecture,
    RegionType,
    ContentType,
    BusinessOpportunityLevel,
    DataSource,
    RawContent,
    ProcessedInsight,
    BusinessStrategy,
    get_architecture
)

from .config import (
    Config,
    APIConfig,
    NotionConfig,
    MonitoringConfig,
    RegionConfig,
    get_config,
    validate_config,
    setup_environment
)

from .mcp_layer import (
    MCPServerManager,
    MCPRequest,
    MCPResponse,
    MCPServerStatus,
    DataCollectorFactory,
    NewsCollector,
    SocialMediaCollector,
    CommunityCollector,
    get_mcp_manager
)

from .ai_engine import (
    AgentRole,
    BaseAgent,
    DataScoutAgent,
    TrendAnalyzerAgent,
    HookingDetectorAgent,
    StrategyPlannerAgent,
    AgentOrchestrator,
    get_orchestrator
)

from .notion_integration import (
    NotionClient,
    NotionFormatter,
    NotionIntegration,
    get_notion_integration
)

from .main_agent import (
    MostHookingBusinessStrategyAgent,
    AdditionalMCPServers,
    EnhancedDataCollector,
    get_main_agent,
    run_quick_analysis,
    get_agent_status
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Most Hooking Business Strategy Agent - Global Business Intelligence System"

# 패키지 초기화
def initialize_package():
    """패키지 초기화 함수"""
    print(f"🚀 Initializing {__description__} v{__version__}")
    
    # 환경 설정
    setup_environment()
    
    # 설정 검증
    issues = validate_config()
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("💡 See ARCHITECTURE.md for setup instructions")
    else:
        print("✅ Configuration validation passed")
    
    # 추가 MCP 서버 정보
    additional_servers_count = len(AdditionalMCPServers.ADDITIONAL_SERVERS)
    print(f"🌐 {additional_servers_count} additional MCP servers available")
    
    # 에이전트 역할 정보
    agent_roles = len([role for role in AgentRole])
    print(f"🤖 {agent_roles} specialized AI agent roles defined")
    
    print("📊 Package initialized successfully!")
    print("🎯 Ready to discover the most hooking business opportunities!")

# 자동 초기화
initialize_package()

__all__ = [
    # Architecture
    'CoreArchitecture',
    'RegionType',
    'ContentType', 
    'BusinessOpportunityLevel',
    'DataSource',
    'RawContent',
    'ProcessedInsight', 
    'BusinessStrategy',
    'get_architecture',
    
    # Config
    'Config',
    'APIConfig',
    'NotionConfig',
    'MonitoringConfig',
    'RegionConfig',
    'get_config',
    'validate_config',
    'setup_environment',
    
    # MCP Layer
    'MCPServerManager',
    'MCPRequest',
    'MCPResponse',
    'MCPServerStatus',
    'DataCollectorFactory',
    'NewsCollector',
    'SocialMediaCollector',
    'CommunityCollector',
    'get_mcp_manager',
    
    # AI Engine
    'AgentRole',
    'BaseAgent',
    'DataScoutAgent',
    'TrendAnalyzerAgent',
    'HookingDetectorAgent',
    'StrategyPlannerAgent',
    'AgentOrchestrator',
    'get_orchestrator',
    
    # Notion Integration
    'NotionClient',
    'NotionFormatter',
    'NotionIntegration',
    'get_notion_integration',
    
    # Main Agent
    'MostHookingBusinessStrategyAgent',
    'AdditionalMCPServers',
    'EnhancedDataCollector',
    'get_main_agent',
    'run_quick_analysis',
    'get_agent_status'
] 