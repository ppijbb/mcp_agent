"""
Drone Control Agent - Quick Demo
빠른 데모를 위한 드론 제어 에이전트
"""

import asyncio
import logging
from srcs.enterprise_agents.drone_control_agent_fixed import DroneControlAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_demo():
    """Quick demo with faster updates to show real-time progress"""
    agent = DroneControlAgent()
    
    print("🚁 Drone Control Agent - Quick Demo")
    print("=" * 50)
    
    # Start a crop inspection task
    print("\n📋 농장 A구역 작물 상태 점검 시작")
    task_result = await agent.execute_natural_language_task(
        "농장 A구역 작물 상태 점검해줘",
        {"max_altitude": 150}
    )
    print(f"✅ {task_result}")
    
    # Monitor progress for 30 seconds
    print(f"\n📊 실시간 진행 상황 모니터링 (30초):")
    print("-" * 40)
    
    for i in range(10):  # Monitor for 30 seconds with 3-second intervals
        await asyncio.sleep(3)
        
        active_tasks = await agent.get_all_active_tasks()
        if active_tasks:
            status = active_tasks[0]
            progress = status['current_status']['progress']
            action = status['current_status']['current_action']
            findings = status['findings']
            alerts = status['alerts']
            battery = status['current_status']['battery']
            
            print(f"⏰ {i*3+3}초: {progress:.1f}% | {action}")
            print(f"   🔋 배터리: {battery:.1f}% | 위치: ({status['current_status']['position']['latitude']:.6f}, {status['current_status']['position']['longitude']:.6f})")
            
            if findings:
                print(f"   🔍 발견사항: {', '.join(findings)}")
            
            if alerts:
                print(f"   ⚠️ 알림: {', '.join(alerts)}")
            
            if progress >= 100:
                print(f"   ✅ 작업 완료!")
                break
                
            print()
        else:
            print(f"⏰ {i*3+3}초: 작업 완료됨")
            break
    
    # Start surveillance task
    print("\n📋 회사 주변 보안 감시 시작")
    task_result2 = await agent.execute_natural_language_task(
        "회사 주변 보안 감시 시작해"
    )
    print(f"✅ {task_result2}")
    
    # Brief monitoring
    await asyncio.sleep(5)
    active_tasks = await agent.get_all_active_tasks()
    
    print(f"\n🚁 현재 드론 함대 상태:")
    fleet_status = await agent.get_drone_fleet_status()
    print(f"   총 드론 수: {fleet_status['total_drones']}")
    print(f"   활성 작업: {fleet_status['active_tasks']}")
    print(f"   사용 가능한 프로바이더: {', '.join(fleet_status['providers'])}")
    
    for drone_id, drone_info in fleet_status['drones'].items():
        if 'error' not in drone_info:
            print(f"   🚁 {drone_id}:")
            print(f"      상태: {drone_info['status']} | 배터리: {drone_info['battery']:.1f}%")
            print(f"      모델: {drone_info['capabilities']['model']}")
            print(f"      센서: {', '.join(drone_info['capabilities']['sensors'])}")
    
    print(f"\n🎯 작업별 현황:")
    for task in active_tasks:
        print(f"   📋 {task['task_info']['title']}")
        print(f"      진행률: {task['current_status']['progress']:.1f}%")
        print(f"      현재 동작: {task['current_status']['action']}")
        if task['findings']:
            print(f"      최근 발견: {', '.join(task['findings'])}")
    
    print("\n✅ 빠른 데모 완료!")

if __name__ == "__main__":
    asyncio.run(quick_demo()) 