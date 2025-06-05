#!/usr/bin/env python3
"""
Simple Test Script for Most Hooking Business Strategy Agent

This script performs basic functionality tests without complex dependencies.
"""

import sys
import os
import asyncio
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_imports():
    """모듈 임포트 테스트"""
    print("🧪 Testing imports...")
    
    try:
        from srcs.business_strategy_agents.config import get_config
        print("✅ Config import successful")
        
        from srcs.business_strategy_agents.architecture import RegionType, BusinessOpportunityLevel
        print("✅ Architecture import successful")
        
        from srcs.business_strategy_agents.ai_engine import AgentRole, DataScoutAgent
        print("✅ AI Engine import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """설정 테스트"""
    print("🧪 Testing configuration...")
    
    try:
        from srcs.business_strategy_agents.config import get_config, validate_config
        
        config = get_config()
        print(f"✅ Config loaded: {type(config)}")
        
        issues = validate_config()
        if issues:
            print(f"⚠️ Config issues found: {len(issues)}")
            for issue in issues[:3]:  # 처음 3개만 표시
                print(f"  - {issue}")
        else:
            print("✅ Config validation passed")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_agent_creation():
    """에이전트 생성 테스트"""
    print("🧪 Testing agent creation...")
    
    try:
        from srcs.business_strategy_agents.ai_engine import DataScoutAgent, AgentRole
        
        agent = DataScoutAgent()
        print(f"✅ DataScoutAgent created: {agent.role}")
        
        assert agent.role == AgentRole.DATA_SCOUT
        print("✅ Agent role verification passed")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

async def test_mock_llm():
    """Mock LLM 응답 테스트"""
    print("🧪 Testing mock LLM responses...")
    
    try:
        from srcs.business_strategy_agents.ai_engine import DataScoutAgent
        
        agent = DataScoutAgent()
        
        # Mock 응답 테스트
        response = await agent._mock_llm_response("system", "user")
        print(f"✅ Mock LLM response generated: {len(response)} chars")
        
        return True
    except Exception as e:
        print(f"❌ Mock LLM test failed: {e}")
        return False

async def test_agent_processing():
    """에이전트 처리 테스트"""
    print("🧪 Testing agent processing...")
    
    try:
        from srcs.business_strategy_agents.ai_engine import DataScoutAgent
        from srcs.business_strategy_agents.architecture import RegionType
        
        agent = DataScoutAgent()
        await agent.initialize()
        
        # 간단한 처리 테스트
        input_data = {
            'keywords': ['AI', 'test'],
            'regions': [RegionType.GLOBAL]
        }
        
        response = await agent.process(input_data)
        print(f"✅ Agent processing completed: success={response.success}")
        print(f"✅ Execution time: {response.execution_time:.2f}s")
        print(f"✅ Result type: {type(response.result)}")
        
        return True
    except Exception as e:
        print(f"❌ Agent processing test failed: {e}")
        return False

def test_streamlit_app_exists():
    """Streamlit 앱 파일 존재 확인"""
    print("🧪 Testing Streamlit app file...")
    
    app_file = "srcs/business_strategy_agents/streamlit_app.py"
    
    if os.path.exists(app_file):
        print(f"✅ Streamlit app file found: {app_file}")
        
        # 파일 크기 확인
        size = os.path.getsize(app_file)
        print(f"✅ File size: {size} bytes")
        
        return True
    else:
        print(f"❌ Streamlit app file not found: {app_file}")
        return False

async def run_all_tests():
    """모든 테스트 실행"""
    print("🎯 Most Hooking Business Strategy Agent - Simple Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config),
        ("Agent Creation Tests", test_agent_creation),
        ("Mock LLM Tests", test_mock_llm),
        ("Agent Processing Tests", test_agent_processing),
        ("Streamlit App Tests", test_streamlit_app_exists)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for Streamlit.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 