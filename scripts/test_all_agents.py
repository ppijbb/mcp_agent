#!/usr/bin/env python3
"""
Test All Agents Script

Tests all registered agents by running them with standard input to verify functionality.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srcs.common.standard_agent_runner import StandardAgentRunner
from srcs.common.a2a_integration import get_global_registry
from srcs.common.agent_interface import AgentType


async def test_all_agents():
    """모든 등록된 agent 테스트"""
    runner = StandardAgentRunner()
    registry = get_global_registry()
    
    # 모든 agent 목록 조회
    agents = await registry.list_agents()
    
    if not agents:
        print("❌ 등록된 agent가 없습니다.")
        return
    
    print(f"📋 총 {len(agents)}개의 agent를 테스트합니다.\n")
    
    results = []
    
    for agent_info in agents:
        agent_id = agent_info.get("agent_id")
        agent_type = agent_info.get("agent_type")
        metadata = agent_info.get("metadata", {})
        
        print(f"🧪 테스트 중: {agent_id} ({agent_type})")
        
        # 기본 테스트 입력 데이터
        test_input = {
            "task": f"Test task for {agent_id}",
            "query": "This is a test query",
            "context": {}
        }
        
        try:
            # Agent 실행
            result = await runner.run_agent(
                agent_id=agent_id,
                input_data=test_input,
                use_a2a=False
            )
            
            if result.success:
                print(f"  ✅ 성공: {agent_id}")
                results.append({
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "status": "success",
                    "execution_time": result.execution_time,
                    "error": None
                })
            else:
                print(f"  ❌ 실패: {agent_id} - {result.error}")
                results.append({
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "status": "failed",
                    "execution_time": result.execution_time,
                    "error": result.error
                })
        
        except Exception as e:
            print(f"  ❌ 오류: {agent_id} - {str(e)}")
            results.append({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "status": "error",
                "execution_time": 0.0,
                "error": str(e)
            })
        
        print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {failed_count}개")
    print(f"💥 오류: {error_count}개")
    print(f"📈 성공률: {success_count / len(results) * 100:.1f}%")
    
    # 결과를 JSON 파일로 저장
    output_file = project_root / "test_results" / f"agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agents),
            "success_count": success_count,
            "failed_count": failed_count,
            "error_count": error_count,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과가 저장되었습니다: {output_file}")
    
    return results


async def test_agent_by_type(agent_type: str):
    """특정 타입의 agent만 테스트"""
    runner = StandardAgentRunner()
    registry = get_global_registry()
    
    agents = await registry.list_agents(agent_type=agent_type)
    
    if not agents:
        print(f"❌ {agent_type} 타입의 agent가 없습니다.")
        return
    
    print(f"📋 {agent_type} 타입의 {len(agents)}개 agent를 테스트합니다.\n")
    
    for agent_info in agents:
        agent_id = agent_info.get("agent_id")
        print(f"🧪 테스트 중: {agent_id}")
        
        test_input = {
            "task": f"Test task for {agent_id}",
            "query": "This is a test query",
            "context": {}
        }
        
        try:
            result = await runner.run_agent(
                agent_id=agent_id,
                input_data=test_input,
                use_a2a=False
            )
            
            if result.success:
                print(f"  ✅ 성공: {agent_id}")
            else:
                print(f"  ❌ 실패: {agent_id} - {result.error}")
        
        except Exception as e:
            print(f"  ❌ 오류: {agent_id} - {str(e)}")
        
        print()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모든 Agent 테스트")
    parser.add_argument(
        "--type",
        type=str,
        choices=["mcp_agent", "langgraph_agent", "cron_agent", "sparkleforge_agent"],
        help="특정 타입의 agent만 테스트"
    )
    
    args = parser.parse_args()
    
    if args.type:
        asyncio.run(test_agent_by_type(args.type))
    else:
        asyncio.run(test_all_agents())


if __name__ == "__main__":
    main()

