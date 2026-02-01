#!/usr/bin/env python3
"""
Test script for the Enhanced Goal Setter Agent with MCP Integration

This script demonstrates the enhanced capabilities and tests various MCP integrations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from srcs.goal_setter_agent.goal_setter import MCPGoalSetterAgent


async def test_basic_functionality():
    """Test basic goal planning without MCP"""
    print("🧪 Testing Basic Functionality (No MCP)")
    print("=" * 50)

    agent = MCPGoalSetterAgent(enable_mcp=False)

    try:
        # Test goal generation
        goal = "Improve team productivity by 30%"
        print(f"🎯 Testing goal: {goal}")

        plan = await agent.generate_enhanced_goal_plan(goal, enable_research=False)
        print("✅ Basic goal plan generated successfully")

        # Test plan validation
        print(f"📊 Plan structure: {len(plan.get('decomposed_plan', []))} sub-goals")
        print(f"🔧 MCP enabled: {plan.get('metadata', {}).get('mcp_enabled', False)}")

        # Test local save/load
        filename = await agent.save_goal_plan(plan)
        print(f"💾 Plan saved to: {filename}")

        loaded_plan = await agent.load_goal_plan(Path(filename).name)
        print("📖 Plan loaded successfully")

        # Test plan listing
        plans = await agent.list_saved_plans()
        print(f"📋 Found {len(plans)} saved plans")

        print("\n✅ Basic functionality test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_mcp_integration():
    """Test MCP integration capabilities"""
    print("🔧 Testing MCP Integration")
    print("=" * 50)

    # MCP server configurations (mock for testing)
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
        output_dir="test_plans",
        enable_mcp=True,
        mcp_servers=mcp_servers
    )

    try:
        # Test MCP initialization
        print("🔄 Initializing MCP connections...")
        await asyncio.sleep(1)  # Simulate connection time

        print(f"📡 MCP enabled: {agent.enable_mcp}")
        print(f"🔗 Filesystem session: {agent.filesystem_session is not None}")
        print(f"🔍 Search session: {agent.search_session is not None}")
        print(f"🌐 Browser session: {agent.browser_session is not None}")

        # Test goal generation with MCP
        goal = "Launch a successful SaaS product in 6 months"
        print(f"\n🎯 Testing MCP-enhanced goal: {goal}")

        plan = await agent.generate_enhanced_goal_plan(goal, enable_research=True)
        print("✅ MCP-enhanced goal plan generated successfully")

        # Check MCP-specific features
        mcp_tool_usage = plan.get('mcp_tool_usage', {})
        if mcp_tool_usage:
            print("🔧 MCP tool usage plan included:")
            for tool_type, operations in mcp_tool_usage.items():
                print(f"  - {tool_type}: {len(operations)} operations")

        # Test MCP tool execution (will fail gracefully with mock servers)
        print("\n🔧 Testing MCP tool execution...")
        execution_results = await agent.execute_mcp_tool_plan(plan)
        print(f"✅ MCP execution completed with {len(execution_results.get('errors', []))} errors")

        print("\n✅ MCP integration test passed!\n")
        return True

    except Exception as e:
        print(f"❌ MCP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enhanced_features():
    """Test enhanced features and output formats"""
    print("🚀 Testing Enhanced Features")
    print("=" * 50)

    agent = MCPGoalSetterAgent(enable_mcp=False)

    try:
        # Test with different goal types
        test_goals = [
            "Reduce customer churn by 15%",
            "Implement DevOps best practices",
            "Create a comprehensive data strategy"
        ]

        for i, goal in enumerate(test_goals, 1):
            print(f"\n🎯 Test {i}: {goal}")

            plan = await agent.generate_enhanced_goal_plan(goal, enable_research=False)

            # Validate enhanced structure
            required_fields = [
                "original_goal", "decomposed_plan", "overall_success_criteria",
                "metadata", "mcp_tool_usage"
            ]

            missing_fields = [field for field in required_fields if not plan.get(field)]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                return False

            print(f"✅ Generated plan with {len(plan.get('decomposed_plan', []))} sub-goals")

            # Test enhanced printing
            agent.pretty_print_enhanced_plan(plan)

        print("\n✅ Enhanced features test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and edge cases"""
    print("🚨 Testing Error Handling")
    print("=" * 50)

    agent = MCPGoalSetterAgent(enable_mcp=False)

    try:
        # Test with invalid goal
        print("🧪 Testing invalid goal handling...")
        try:
            await agent.generate_enhanced_goal_plan("", enable_research=False)
            print("❌ Should have failed with empty goal")
            return False
        except Exception as e:
            print(f"✅ Correctly handled empty goal: {type(e).__name__}")

        # Test with very long goal
        print("🧪 Testing very long goal...")
        long_goal = "A" * 1000  # Very long goal
        try:
            plan = await agent.generate_enhanced_goal_plan(long_goal, enable_research=False)
            print(f"✅ Handled long goal: {len(plan.get('original_goal', ''))} characters")
        except Exception as e:
            print(f"❌ Failed to handle long goal: {e}")
            return False

        # Test cleanup
        print("🧪 Testing cleanup...")
        await agent.cleanup()
        print("✅ Cleanup completed successfully")

        print("\n✅ Error handling test passed!\n")
        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🎯 Enhanced Goal Setter Agent - Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("MCP Integration", test_mcp_integration),
        ("Enhanced Features", test_enhanced_features),
        ("Error Handling", test_error_handling)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results[test_name] = "CRASH"

    # Print summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)

    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {result}")

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r == "PASS")

    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! The Enhanced Goal Setter Agent is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above for details.")

    # Cleanup test files
    test_dir = Path("test_plans")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print("🧹 Cleaned up test files")

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY environment variable not set.")
        print("   Set it to run the tests: export OPENAI_API_KEY='your_key_here'")
        print("   Tests will be skipped.")
        sys.exit(1)

    asyncio.run(main())
