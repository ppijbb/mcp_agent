#!/usr/bin/env python3
"""
MCP tool 호출 테스트 스크립트 - 디버깅용
"""

import asyncio
import sys
import json
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "sparkleforge"))

from src.core.researcher_config import load_config_from_env
from src.core.mcp_integration import execute_tool, get_mcp_hub

async def test_mcp_tool_call():
    """MCP tool 호출 테스트"""
    print("=" * 80)
    print("MCP Tool 호출 테스트 시작")
    print("=" * 80)
    
    # 설정 로드
    print("\n1. 설정 로드 중...")
    try:
        config = load_config_from_env()
        print("✅ 설정 로드 성공")
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return False
    
    # MCP Hub 초기화
    print("\n2. MCP Hub 초기화 중...")
    try:
        mcp_hub = get_mcp_hub()
        await mcp_hub.initialize_mcp()
        print(f"✅ MCP Hub 초기화 완료")
        print(f"   연결된 서버: {list(mcp_hub.mcp_sessions.keys())}")
        print(f"   사용 가능한 도구:")
        for server_name, tools in mcp_hub.mcp_tools_map.items():
            print(f"     - {server_name}: {len(tools)} tools")
            for tool_name in list(tools.keys())[:3]:  # 처음 3개만 표시
                print(f"       * {tool_name}")
    except Exception as e:
        print(f"❌ MCP Hub 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Tool 호출 테스트
    print("\n3. Tool 호출 테스트 중...")
    test_tools = [
        ("g-search", {"query": "Python programming", "max_results": 2}),
        ("tavily", {"query": "Python programming", "max_results": 2}),
    ]
    
    for tool_name, params in test_tools:
        print(f"\n   테스트 도구: {tool_name}")
        print(f"   파라미터: {params}")
        try:
            result = await execute_tool(tool_name, params)
            print(f"   결과: success={result.get('success')}")
            if result.get('success'):
                data = result.get('data')
                if isinstance(data, dict) and 'results' in data:
                    print(f"   검색 결과: {len(data['results'])}개")
                else:
                    print(f"   데이터 타입: {type(data).__name__}")
            else:
                print(f"   오류: {result.get('error')}")
        except Exception as e:
            print(f"   ❌ 예외 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 정리
    print("\n4. 정리 중...")
    try:
        await mcp_hub.cleanup()
        print("✅ 정리 완료")
    except Exception as e:
        print(f"⚠️ 정리 중 오류: {e}")
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_tool_call())
    sys.exit(0 if success else 1)

