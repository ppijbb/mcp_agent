#!/usr/bin/env python3
"""
Test script for GraphRAG Agent

This script demonstrates how to use the agent programmatically
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import GraphRAGAgent


async def test_agent():
    """Test the GraphRAG Agent"""
    print("🧪 Testing GraphRAG Agent...")
    
    # Test 1: Initialize agent
    print("\n1. Testing agent initialization...")
    agent = GraphRAGAgent()
    
    if not agent.initialize():
        print("❌ Agent initialization failed")
        return False
    
    print("✅ Agent initialized successfully")
    
    # Test 2: Create sample data
    print("\n2. Creating sample data...")
    sample_file = agent.create_sample_data()
    
    if not sample_file:
        print("❌ Failed to create sample data")
        return False
    
    print(f"✅ Sample data created: {sample_file}")
    
    # Test 3: Test configuration
    print("\n3. Testing configuration...")
    print(f"   - Mode: {agent.config.mode}")
    print(f"   - Model: {agent.config.agent.model_name}")
    print(f"   - Visualization: {agent.config.visualization.enabled}")
    print(f"   - Optimization: {agent.config.optimization.enabled}")
    
    # Test 4: Test status check
    print("\n4. Testing status check...")
    agent.config.mode = "status"
    
    success = await agent.run()
    if not success:
        print("❌ Status check failed")
        return False
    
    print("✅ Status check completed")
    
    # Cleanup
    if os.path.exists(sample_file):
        os.remove(sample_file)
        print(f"\n🧹 Cleaned up sample file: {sample_file}")
    
    print("\n🎉 All tests passed!")
    return True


async def test_with_environment():
    """Test agent with environment variables"""
    print("\n🔧 Testing with environment variables...")
    
    # Set test environment variables
    os.environ["MODE"] = "status"
    os.environ["VERBOSE"] = "true"
    
    agent = GraphRAGAgent()
    
    if not agent.initialize():
        print("❌ Agent initialization failed")
        return False
    
    print("✅ Agent initialized with environment variables")
    print(f"   - Mode from env: {agent.config.mode}")
    print(f"   - Verbose from env: {agent.config.verbose}")
    
    return True


if __name__ == "__main__":
    print("GraphRAG Agent Test Suite")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some tests may fail.")
        print("   Set it with: export OPENAI_API_KEY='your_key_here'")
        print()
    
    try:
        # Run basic tests
        success1 = asyncio.run(test_agent())
        
        # Run environment variable tests
        success2 = asyncio.run(test_with_environment())
        
        if success1 and success2:
            print("\n🎉 All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
