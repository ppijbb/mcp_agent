#!/usr/bin/env python3
"""
Test file for the Genome Agent

This file provides basic testing functionality for the genome agent.
"""

import asyncio
import os
from datetime import datetime

from genome_agent import GenomeAgentMCP, GenomeData, GenomeDataType
from config import get_config


async def test_basic_functionality():
    """Test basic genome agent functionality"""
    print("🧪 Testing basic functionality...")

    try:
        # Create agent
        agent = GenomeAgentMCP(output_dir="test_reports")
        print("✅ Agent created successfully")

        # Test configuration
        config = get_config()
        print(f"✅ Configuration loaded: {len(config.get_active_databases())} databases")

        # Test data creation
        test_data = GenomeData(
            data_id="test_sequence",
            data_type=GenomeDataType.DNA_SEQUENCE,
            organism="Homo sapiens",
            sequence="ATGCGATCGATCG",
            metadata={"test": True},
            source="test",
            timestamp=datetime.now()
        )
        print("✅ Test data created successfully")

        # Test save/load
        filepath = await agent.save_genome_data(test_data)
        print(f"✅ Data saved to: {filepath}")

        loaded_data = await agent.load_genome_data(os.path.basename(filepath))
        if loaded_data and loaded_data.data_id == test_data.data_id:
            print("✅ Data loaded successfully")
        else:
            print("❌ Data loading failed")

        # Cleanup
        await agent.cleanup()
        print("✅ Cleanup completed")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


async def test_analysis_planning():
    """Test analysis plan generation"""
    print("\n🧪 Testing analysis planning...")

    try:
        agent = GenomeAgentMCP(output_dir="test_reports")

        # Test plan generation
        plan = await agent.generate_enhanced_analysis_plan(
            analysis_request="Test analysis request",
            enable_research=False
        )

        if "error" not in plan and "plan_id" in plan.get("plan", {}):
            print("✅ Analysis plan generated successfully")
            print(f"   Plan ID: {plan['plan']['plan_id']}")
            print(f"   Steps: {len(plan['plan'].get('analysis_steps', []))}")
        else:
            print("❌ Analysis plan generation failed")
            return False

        await agent.cleanup()
        return True

    except Exception as e:
        print(f"❌ Analysis planning test failed: {e}")
        return False


async def test_mcp_integration():
    """Test MCP integration (mock)"""
    print("\n🧪 Testing MCP integration...")

    try:
        # Test with mock MCP servers
        mock_mcp_servers = {
            "filesystem": "http://localhost:3000",
            "search": "http://localhost:3001",
            "browser": "http://localhost:3002"
        }

        agent = GenomeAgentMCP(
            output_dir="test_reports",
            enable_mcp=True,
            mcp_servers=mock_mcp_servers
        )

        # Initialize MCP connections (will fail but shouldn't crash)
        await agent._initialize_mcp_connections()
        print("✅ MCP initialization handled gracefully")

        await agent.cleanup()
        return True

    except Exception as e:
        print(f"❌ MCP integration test failed: {e}")
        return False


async def test_data_management():
    """Test genome data management"""
    print("\n🧪 Testing data management...")

    try:
        agent = GenomeAgentMCP(output_dir="test_reports")

        # Create multiple test data entries
        test_data_list = []
        for i in range(3):
            data = GenomeData(
                data_id=f"test_sequence_{i}",
                data_type=GenomeDataType.DNA_SEQUENCE,
                organism="Homo sapiens",
                sequence=f"ATGCGATCGATCG{i}",
                metadata={"test": True, "index": i},
                source="test",
                timestamp=datetime.now()
            )
            test_data_list.append(data)

        # Save all data
        for data in test_data_list:
            await agent.save_genome_data(data)

        # List saved data
        saved_files = await agent.list_saved_data()
        if len(saved_files) >= 3:
            print(f"✅ Data management test passed: {len(saved_files)} files found")
        else:
            print(f"❌ Data management test failed: expected 3+ files, got {len(saved_files)}")
            return False

        await agent.cleanup()
        return True

    except Exception as e:
        print(f"❌ Data management test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("🧬 GENOME AGENT - TEST SUITE")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Analysis Planning", test_analysis_planning),
        ("MCP Integration", test_mcp_integration),
        ("Data Management", test_data_management)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} - {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return False


async def main():
    """Main test function"""
    try:
        success = await run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
