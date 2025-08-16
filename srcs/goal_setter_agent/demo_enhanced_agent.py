#!/usr/bin/env python3
"""
Demo script for the Enhanced Goal Setter Agent with MCP Integration

This script demonstrates the enhanced capabilities with practical business examples.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.goal_setter_agent.goal_setter import MCPGoalSetterAgent

async def demo_business_goals():
    """Demonstrate business goal planning capabilities"""
    print("🏢 Business Goals Demo")
    print("=" * 50)
    
    agent = MCPGoalSetterAgent(enable_mcp=False, output_dir="demo_plans")
    
    business_goals = [
        "Increase annual revenue by 40%",
        "Reduce customer acquisition cost by 25%",
        "Improve employee retention to 90%",
        "Launch 3 new product lines in 12 months"
    ]
    
    for i, goal in enumerate(business_goals, 1):
        print(f"\n🎯 Demo {i}: {goal}")
        print("-" * 40)
        
        try:
            plan = await agent.generate_enhanced_goal_plan(goal, enable_research=False)
            
            # Show key metrics
            sub_goals = plan.get('decomposed_plan', [])
            total_actions = sum(len(sub.get('action_plan', [])) for sub in sub_goals)
            total_kpis = sum(len(sub.get('kpis', [])) for sub in sub_goals)
            
            print(f"📊 Generated {len(sub_goals)} sub-goals")
            print(f"🔧 Total actions: {total_actions}")
            print(f"📈 Total KPIs: {total_kpis}")
            
            # Show first sub-goal details
            if sub_goals:
                first_sub = sub_goals[0]
                print(f"📋 First sub-goal: {first_sub.get('sub_goal')}")
                print(f"   Priority: {first_sub.get('priority')}")
                print(f"   Actions: {len(first_sub.get('action_plan', []))}")
            
            # Save the plan
            filename = await agent.save_goal_plan(plan)
            print(f"💾 Saved to: {Path(filename).name}")
            
        except Exception as e:
            print(f"❌ Failed to process goal: {e}")
    
    print("\n✅ Business goals demo completed!")

async def demo_technical_goals():
    """Demonstrate technical goal planning capabilities"""
    print("\n⚙️ Technical Goals Demo")
    print("=" * 50)
    
    agent = MCPGoalSetterAgent(enable_mcp=False, output_dir="demo_plans")
    
    technical_goals = [
        "Implement comprehensive CI/CD pipeline",
        "Migrate legacy system to microservices architecture",
        "Achieve 99.9% system uptime",
        "Reduce API response time to under 200ms"
    ]
    
    for i, goal in enumerate(technical_goals, 1):
        print(f"\n🎯 Demo {i}: {goal}")
        print("-" * 40)
        
        try:
            plan = await agent.generate_enhanced_goal_plan(goal, enable_research=False)
            
            # Show technical-specific details
            sub_goals = plan.get('decomposed_plan', [])
            
            # Count technical agents used
            agent_usage = {}
            for sub in sub_goals:
                for action in sub.get('action_plan', []):
                    agent_name = action.get('suggested_agent', 'Unknown')
                    agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
            
            print(f"📊 Generated {len(sub_goals)} sub-goals")
            print(f"🤖 Agent distribution:")
            for agent_name, count in agent_usage.items():
                print(f"   - {agent_name}: {count} actions")
            
            # Show MCP tool usage plan
            mcp_plan = plan.get('mcp_tool_usage', {})
            if mcp_plan:
                print(f"🔧 MCP tool usage planned:")
                for tool_type, operations in mcp_plan.items():
                    print(f"   - {tool_type}: {len(operations)} operations")
            
            # Save the plan
            filename = await agent.save_goal_plan(plan)
            print(f"💾 Saved to: {Path(filename).name}")
            
        except Exception as e:
            print(f"❌ Failed to process goal: {e}")
    
    print("\n✅ Technical goals demo completed!")

async def demo_mcp_features():
    """Demonstrate MCP-specific features"""
    print("\n🔧 MCP Features Demo")
    print("=" * 50)
    
    # Mock MCP servers for demonstration
    mcp_servers = {
        "filesystem": {
            "command": "echo",
            "args": ["mock_filesystem"],
            "env": {}
        },
        "search": {
            "command": "echo", 
            "args": ["mock_search"],
            "env": {}
        },
        "browser": {
            "command": "echo",
            "args": ["mock_browser"],
            "env": {}
        }
    }
    
    agent = MCPGoalSetterAgent(
        enable_mcp=True,
        mcp_servers=mcp_servers,
        output_dir="demo_plans"
    )
    
    try:
        print("🔄 Initializing MCP connections...")
        await asyncio.sleep(1)  # Simulate connection time
        
        print(f"📡 MCP Status:")
        print(f"   - Enabled: {agent.enable_mcp}")
        print(f"   - Filesystem: {'✅' if agent.filesystem_session else '❌'}")
        print(f"   - Search: {'✅' if agent.search_session else '❌'}")
        print(f"   - Browser: {'✅' if agent.browser_session else '❌'}")
        
        # Test goal with MCP research
        goal = "Create a data-driven marketing strategy"
        print(f"\n🎯 Testing MCP-enhanced goal: {goal}")
        
        plan = await agent.generate_enhanced_goal_plan(goal, enable_research=True)
        
        # Show MCP-enhanced features
        metadata = plan.get('metadata', {})
        print(f"📊 MCP Enhancement Details:")
        print(f"   - Version: {metadata.get('version', 'N/A')}")
        print(f"   - MCP Enabled: {metadata.get('mcp_enabled', False)}")
        print(f"   - Research Data: {'✅' if metadata.get('research_data') else '❌'}")
        
        # Show MCP tool usage plan
        mcp_tool_usage = plan.get('mcp_tool_usage', {})
        if mcp_tool_usage:
            print(f"🔧 MCP Tool Usage Plan:")
            for tool_type, operations in mcp_tool_usage.items():
                print(f"   - {tool_type}: {operations}")
        
        # Test MCP tool execution
        print(f"\n🔧 Executing MCP tool plan...")
        execution_results = await agent.execute_mcp_tool_plan(plan)
        
        print(f"📊 Execution Results:")
        print(f"   - Start: {execution_results.get('execution_start', 'N/A')}")
        print(f"   - End: {execution_results.get('execution_end', 'N/A')}")
        print(f"   - Results: {len(execution_results.get('results', {}))}")
        print(f"   - Errors: {len(execution_results.get('errors', []))}")
        
        # Save the plan
        filename = await agent.save_goal_plan(plan)
        print(f"💾 Saved to: {Path(filename).name}")
        
        print("\n✅ MCP features demo completed!")
        
    except Exception as e:
        print(f"❌ MCP features demo failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_plan_management():
    """Demonstrate plan management capabilities"""
    print("\n📋 Plan Management Demo")
    print("=" * 50)
    
    agent = MCPGoalSetterAgent(enable_mcp=False, output_dir="demo_plans")
    
    try:
        # List existing plans
        print("📋 Listing existing plans...")
        plans = await agent.list_saved_plans()
        
        if plans:
            print(f"Found {len(plans)} saved plans:")
            for plan in plans:
                print(f"   - {plan['name']}")
                if 'size' in plan:
                    print(f"     Size: {plan['size']} bytes")
                if 'modified' in plan:
                    print(f"     Modified: {plan['modified']}")
            
            # Load and display the first plan
            if plans:
                first_plan = plans[0]
                print(f"\n📖 Loading first plan: {first_plan['name']}")
                
                loaded_plan = await agent.load_goal_plan(first_plan['name'])
                print(f"✅ Loaded plan: {loaded_plan.get('original_goal', 'Unknown goal')}")
                
                # Show plan summary
                sub_goals = loaded_plan.get('decomposed_plan', [])
                total_actions = sum(len(sub.get('action_plan', [])) for sub in sub_goals)
                
                print(f"📊 Plan Summary:")
                print(f"   - Sub-goals: {len(sub_goals)}")
                print(f"   - Total actions: {total_actions}")
                print(f"   - Generated: {loaded_plan.get('metadata', {}).get('generated_at', 'N/A')}")
        else:
            print("No saved plans found.")
        
        print("\n✅ Plan management demo completed!")
        
    except Exception as e:
        print(f"❌ Plan management demo failed: {e}")

async def main():
    """Run the comprehensive demo"""
    print("🎯 Enhanced Goal Setter Agent - Comprehensive Demo")
    print("=" * 70)
    print("This demo showcases the enhanced MCP capabilities of the goal setter agent.")
    print("=" * 70)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY environment variable not set.")
        print("   Set it to run the demo: export OPENAI_API_KEY='your_key_here'")
        print("   Demo will be skipped.")
        return
    
    demos = [
        ("Business Goals", demo_business_goals),
        ("Technical Goals", demo_technical_goals),
        ("MCP Features", demo_mcp_features),
        ("Plan Management", demo_plan_management)
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            await demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed!")
    print("=" * 70)
    
    # Show final results
    demo_dir = Path("demo_plans")
    if demo_dir.exists():
        plans = list(demo_dir.glob("*.json"))
        print(f"📁 Generated {len(plans)} demo plans in 'demo_plans' directory")
        
        # Show file sizes
        total_size = sum(plan.stat().st_size for plan in plans)
        print(f"💾 Total size: {total_size / 1024:.1f} KB")
    
    print("\n🔧 To test with real MCP servers:")
    print("   1. Install MCP servers: npm install -g @modelcontextprotocol/server-filesystem g-search-mcp @modelcontextprotocol/server-puppeteer")
    print("   2. Run: python goal_setter.py --goal 'Your goal here' --enable-mcp")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    asyncio.run(main())
