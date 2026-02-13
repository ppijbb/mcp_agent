#!/usr/bin/env python3
"""
A2A 연결 테스트 스크립트

Wrapper 간 메시지 송수신 및 상태 체크를 수행합니다.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
_primary = project_root / "primary"
if _primary.exists():
    sys.path.insert(0, str(_primary))

from srcs.common.a2a_integration import (
    get_global_registry,
    get_global_broker,
    A2AMessage,
    MessagePriority,
)
from srcs.common.a2a_adapter import CommonAgentA2AWrapper
from lang_graph.common.a2a_adapter import LangGraphAgentA2AWrapper
from cron_agents.common.a2a_adapter import CronAgentA2AWrapper
from SparkleForge.common.a2a_adapter import SparkleForgeA2AWrapper


async def test_a2a_message_sending():
    """A2A 메시지 전송 테스트"""
    print("📨 A2A 메시지 전송 테스트")
    print("-" * 60)
    
    registry = get_global_registry()
    broker = get_global_broker()
    
    # 등록된 agent 목록 조회
    agents = await registry.list_agents()
    
    if len(agents) < 2:
        print("❌ 메시지 전송 테스트를 위해서는 최소 2개의 agent가 필요합니다.")
        return
    
    # 첫 번째 agent에서 두 번째 agent로 메시지 전송
    source_agent = agents[0]
    target_agent = agents[1]
    
    source_adapter = source_agent.get("a2a_adapter")
    target_adapter = target_agent.get("a2a_adapter")
    
    if not source_adapter or not target_adapter:
        print("❌ A2A adapter가 설정되지 않은 agent가 있습니다.")
        return
    
    print(f"📤 소스 Agent: {source_agent['agent_id']}")
    print(f"📥 타겟 Agent: {target_agent['agent_id']}")
    
    # 리스너 시작
    await source_adapter.start_listener()
    await target_adapter.start_listener()
    
    # 메시지 전송
    test_message = {
        "test": True,
        "timestamp": datetime.now().isoformat(),
        "content": "This is a test message"
    }
    
    success = await source_adapter.send_message(
        target_agent=target_agent['agent_id'],
        message_type="test_message",
        payload=test_message,
        priority=MessagePriority.HIGH.value
    )
    
    if success:
        print("✅ 메시지 전송 성공")
    else:
        print("❌ 메시지 전송 실패")
    
    # 잠시 대기 (메시지 처리 시간)
    await asyncio.sleep(1)
    
    # 리스너 중지
    await source_adapter.stop_listener()
    await target_adapter.stop_listener()
    
    print()


async def test_a2a_broadcast():
    """A2A 브로드캐스트 테스트"""
    print("📢 A2A 브로드캐스트 테스트")
    print("-" * 60)
    
    registry = get_global_registry()
    broker = get_global_broker()
    
    agents = await registry.list_agents()
    
    if len(agents) < 2:
        print("❌ 브로드캐스트 테스트를 위해서는 최소 2개의 agent가 필요합니다.")
        return
    
    # 모든 agent의 리스너 시작
    adapters = []
    for agent_info in agents:
        adapter = agent_info.get("a2a_adapter")
        if adapter:
            await adapter.start_listener()
            adapters.append(adapter)
    
    print(f"📡 {len(adapters)}개의 agent에 브로드캐스트 전송")
    
    # 브로드캐스트 메시지 전송
    if adapters:
        source_adapter = adapters[0]
        broadcast_message = {
            "test": True,
            "type": "broadcast",
            "timestamp": datetime.now().isoformat(),
            "content": "This is a broadcast message"
        }
        
        success = await source_adapter.send_message(
            target_agent="",  # 빈 문자열 = 브로드캐스트
            message_type="broadcast_test",
            payload=broadcast_message,
            priority=MessagePriority.MEDIUM.value
        )
        
        if success:
            print("✅ 브로드캐스트 전송 성공")
        else:
            print("❌ 브로드캐스트 전송 실패")
        
        # 잠시 대기
        await asyncio.sleep(2)
        
        # 모든 리스너 중지
        for adapter in adapters:
            await adapter.stop_listener()
    
    print()


async def test_a2a_capabilities():
    """A2A 능력 등록 테스트"""
    print("🎯 A2A 능력 등록 테스트")
    print("-" * 60)
    
    registry = get_global_registry()
    
    # 테스트용 wrapper 생성
    test_wrapper = CommonAgentA2AWrapper(
        agent_id="test_agent_001",
        agent_metadata={
            "name": "Test Agent",
            "description": "Test agent for A2A capabilities"
        }
    )
    
    capabilities = ["test_capability_1", "test_capability_2", "test_capability_3"]
    
    await test_wrapper.register_capabilities(capabilities)
    
    # 등록 확인
    agent_info = await registry.get_agent("test_agent_001")
    
    if agent_info:
        registered_capabilities = agent_info.get("metadata", {}).get("capabilities", [])
        if set(capabilities) == set(registered_capabilities):
            print("✅ 능력 등록 성공")
            print(f"   등록된 능력: {registered_capabilities}")
        else:
            print("❌ 능력 등록 실패 - 능력이 일치하지 않습니다.")
            print(f"   예상: {capabilities}")
            print(f"   실제: {registered_capabilities}")
    else:
        print("❌ Agent 등록 실패")
    
    # 정리
    await registry.unregister_agent("test_agent_001")
    
    print()


async def test_a2a_message_history():
    """A2A 메시지 히스토리 테스트"""
    print("📜 A2A 메시지 히스토리 테스트")
    print("-" * 60)
    
    broker = get_global_broker()
    
    # 테스트 메시지 생성 및 라우팅
    test_messages = []
    for i in range(5):
        message = A2AMessage(
            source_agent="test_source",
            target_agent="test_target",
            message_type="test",
            payload={"index": i}
        )
        await broker.route_message(message)
        test_messages.append(message)
    
    # 히스토리 조회
    history = broker.get_message_history(limit=10)
    
    if len(history) >= len(test_messages):
        print(f"✅ 메시지 히스토리 조회 성공 ({len(history)}개 메시지)")
        print(f"   최근 메시지 ID: {history[-1].message_id}")
    else:
        print(f"❌ 메시지 히스토리 조회 실패 (예상: {len(test_messages)}, 실제: {len(history)})")
    
    print()


async def run_all_tests():
    """모든 A2A 테스트 실행"""
    print("=" * 60)
    print("🧪 A2A 연결 테스트 시작")
    print("=" * 60)
    print()
    
    try:
        await test_a2a_capabilities()
        await test_a2a_message_sending()
        await test_a2a_broadcast()
        await test_a2a_message_history()
        
        print("=" * 60)
        print("✅ 모든 A2A 테스트 완료")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()

