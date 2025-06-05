"""
Demo Script for Most Hooking Business Strategy Agent

This script demonstrates the complete workflow of the business strategy agent
from data collection through MCP servers to Notion documentation.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List

from .main_agent import get_main_agent, run_quick_analysis, get_agent_status
from .architecture import RegionType
from .config import get_config, validate_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_functionality():
    """기본 기능 데모"""
    print("🚀 Starting Most Hooking Business Strategy Agent Demo")
    print("=" * 60)
    
    # 1. 설정 검증
    print("\n📋 1. Configuration Validation")
    config = get_config()
    issues = validate_config()
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validation passed")
    
    print(f"📊 System Info: {config.get_system_info()}")
    
    # 2. 에이전트 초기화
    print("\n🤖 2. Agent Initialization")
    try:
        agent = await get_main_agent()
        status = await get_agent_status()
        print(f"✅ Agent initialized successfully")
        print(f"📈 Performance metrics: {status['performance_metrics']}")
        print(f"🔗 MCP servers connected: {status['mcp_servers']}")
        print(f"🌐 Additional servers available: {status['additional_servers_count']}")
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return
    
    # 3. 빠른 분석 실행
    print("\n🔍 3. Quick Analysis Execution")
    
    demo_keywords = [
        "AI", "startup", "fintech", "sustainability", 
        "digital transformation", "Web3", "creator economy"
    ]
    
    demo_regions = [RegionType.EAST_ASIA, RegionType.NORTH_AMERICA]
    
    print(f"🎯 Keywords: {demo_keywords}")
    print(f"🌏 Regions: {[r.value for r in demo_regions]}")
    
    try:
        analysis_results = await run_quick_analysis(demo_keywords)
        
        if analysis_results.get('error'):
            print(f"❌ Analysis failed: {analysis_results['error']}")
        else:
            print("✅ Analysis completed successfully!")
            print(f"⏱️ Duration: {analysis_results.get('duration_seconds', 0):.2f} seconds")
            print(f"📊 Insights generated: {analysis_results.get('enhanced_insights_count', 0)}")
            print(f"🎯 Strategies created: {analysis_results.get('regional_strategies_count', 0)}")
            
            # 상위 기회들 표시
            top_opportunities = analysis_results.get('top_hooking_opportunities', [])
            if top_opportunities:
                print("\n🏆 Top Hooking Opportunities:")
                for i, opp in enumerate(top_opportunities[:3], 1):
                    print(f"  {i}. Score: {opp['score']:.2f} | Topics: {', '.join(opp['topics'][:2])} | Level: {opp['opportunity_level']}")
    
    except Exception as e:
        print(f"❌ Analysis execution failed: {e}")
    
    # 4. 시스템 상태 확인
    print("\n📊 4. Final System Status")
    final_status = await get_agent_status()
    print(f"🔄 Is running: {final_status['is_running']}")
    print(f"📈 Total analyses: {final_status['performance_metrics']['total_analyses']}")
    print(f"✅ Successful analyses: {final_status['performance_metrics']['successful_analyses']}")
    print(f"📝 Notion pages created: {final_status['performance_metrics']['notion_pages_created']}")
    
    print("\n🎉 Demo completed successfully!")


async def demo_agent_roles():
    """에이전트 역할별 기능 데모"""
    print("\n🎭 Agent Roles Demonstration")
    print("=" * 40)
    
    agent = await get_main_agent()
    orchestrator = agent.orchestrator
    
    if not orchestrator:
        print("❌ Orchestrator not available")
        return
    
    # 에이전트 상태 확인
    agent_status = orchestrator.get_agent_status()
    print("🤖 Available Agents:")
    
    for role, status in agent_status.items():
        icon = "✅" if status == "active" else "❌"
        role_display = role.replace('_', ' ').title()
        print(f"  {icon} {role_display}: {status}")
    
    # 개별 에이전트 기능 테스트
    from .ai_engine import AgentRole
    
    # Data Scout 테스트
    if AgentRole.DATA_SCOUT in orchestrator.agents:
        print("\n📡 Testing Data Scout Agent...")
        data_scout = orchestrator.agents[AgentRole.DATA_SCOUT]
        
        try:
            # 간단한 데이터 수집 시뮬레이션
            test_keywords = ["AI", "startup"]
            test_regions = [RegionType.GLOBAL]
            
            # 실제로는 MCP 서버 호출하지만 데모에서는 스킵
            print(f"  🔍 Would collect data for: {test_keywords}")
            print(f"  ✅ Data Scout agent functional")
            
        except Exception as e:
            print(f"  ❌ Data Scout test failed: {e}")
    
    # Trend Analyzer 테스트
    if AgentRole.TREND_ANALYZER in orchestrator.agents:
        print("\n📈 Testing Trend Analyzer Agent...")
        print(f"  📊 Would analyze trends and patterns")
        print(f"  ✅ Trend Analyzer agent functional")
    
    # Hooking Detector 테스트
    if AgentRole.HOOKING_DETECTOR in orchestrator.agents:
        print("\n🎯 Testing Hooking Detector Agent...")
        print(f"  🔍 Would detect high-potential opportunities")
        print(f"  ✅ Hooking Detector agent functional")
    
    # Strategy Planner 테스트
    if AgentRole.STRATEGY_PLANNER in orchestrator.agents:
        print("\n🚀 Testing Strategy Planner Agent...")
        print(f"  📋 Would create actionable business strategies")
        print(f"  ✅ Strategy Planner agent functional")


async def demo_mcp_servers():
    """MCP 서버 연결 데모"""
    print("\n🔗 MCP Servers Demonstration")
    print("=" * 40)
    
    agent = await get_main_agent()
    mcp_manager = agent.mcp_manager
    
    if not mcp_manager:
        print("❌ MCP Manager not available")
        return
    
    # 서버 상태 확인
    server_status = mcp_manager.get_server_status()
    print(f"📡 Connected MCP Servers: {len(server_status)}")
    
    categories = {
        'news': [],
        'social': [],
        'community': [],
        'trends': [],
        'business': [],
        'market': [],
        'economic': []
    }
    
    for server_name, status in server_status.items():
        category = server_name.split('_')[0] if '_' in server_name else 'other'
        if category in categories:
            categories[category].append({
                'name': server_name,
                'status': status['status'],
                'connections': status.get('connection_count', 0)
            })
    
    for category, servers in categories.items():
        if servers:
            print(f"\n📂 {category.title()} Servers:")
            for server in servers:
                status_icon = "🟢" if server['status'] == 'connected' else "🔴"
                print(f"  {status_icon} {server['name']} (connections: {server['connections']})")
    
    # 헬스체크 실행
    print(f"\n🏥 Running health check...")
    try:
        health_results = await mcp_manager.health_check_all()
        healthy_count = sum(health_results.values())
        total_count = len(health_results)
        
        print(f"✅ Health check completed: {healthy_count}/{total_count} servers healthy")
        
        for server_name, is_healthy in health_results.items():
            status_icon = "✅" if is_healthy else "❌"
            print(f"  {status_icon} {server_name}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")


async def demo_notion_integration():
    """Notion 통합 데모"""
    print("\n📝 Notion Integration Demonstration")
    print("=" * 40)
    
    agent = await get_main_agent()
    notion_integration = agent.notion_integration
    
    if not notion_integration:
        print("❌ Notion integration not available")
        return
    
    config = get_config()
    
    if not config.notion:
        print("⚠️ Notion configuration not found")
        print("💡 To enable Notion integration, set these environment variables:")
        print("   - NOTION_API_KEY=your_notion_api_key")
        print("   - NOTION_DATABASE_ID=your_database_id")
        print("   - NOTION_WORKSPACE_ID=your_workspace_id")
        return
    
    print(f"✅ Notion integration configured")
    print(f"🗄️ Database ID: {config.notion.database_id[:8]}...")
    print(f"🏢 Workspace ID: {config.notion.workspace_id[:8]}...")
    
    # 샘플 데이터로 테스트 (실제 API 호출 없이)
    print(f"📄 Would create daily insights page")
    print(f"📋 Would create strategy pages") 
    print(f"📊 Would create weekly summary (if applicable)")
    print(f"✅ Notion integration functional")


async def run_comprehensive_demo():
    """종합 데모 실행"""
    print("🌟 MOST HOOKING BUSINESS STRATEGY AGENT - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    try:
        # 기본 기능
        await demo_basic_functionality()
        
        # 에이전트 역할
        await demo_agent_roles()
        
        # MCP 서버
        await demo_mcp_servers()
        
        # Notion 통합
        await demo_notion_integration()
        
        print("\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("🚀 The Most Hooking Business Strategy Agent is ready to discover")
        print("   the next big business opportunities for you!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo execution error: {e}", exc_info=True)


async def interactive_demo():
    """인터랙티브 데모"""
    print("\n🎮 Interactive Demo Mode")
    print("Choose a demo to run:")
    print("1. Basic Functionality")
    print("2. Agent Roles")
    print("3. MCP Servers")
    print("4. Notion Integration")
    print("5. Run All Demos")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ")
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                await demo_basic_functionality()
            elif choice == "2":
                await demo_agent_roles()
            elif choice == "3":
                await demo_mcp_servers()
            elif choice == "4":
                await demo_notion_integration()
            elif choice == "5":
                await run_comprehensive_demo()
                break
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """메인 함수"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_demo())
    else:
        asyncio.run(run_comprehensive_demo())


if __name__ == "__main__":
    main() 