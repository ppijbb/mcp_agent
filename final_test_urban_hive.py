#!/usr/bin/env python3
"""
Urban Hive Agent 최종 테스트 스크립트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def final_test_urban_hive():
    """Urban Hive Agent 최종 테스트"""
    
    print("🚀 Urban Hive Agent 최종 테스트 시작")
    print("=" * 60)
    
    try:
        from srcs.urban_hive.urban_hive_agent import UrbanHiveMCPAgent, UrbanDataCategory
        print("✅ 모듈 임포트 성공")
        
        agent = UrbanHiveMCPAgent()
        print("✅ Agent 초기화 성공")
        
        print("\n🔍 실제 도시 분석 실행 (최대 3분)...")
        
        result = await agent.analyze_urban_data(
            category=UrbanDataCategory.TRAFFIC_FLOW,
            location="서울 강남구",
            time_range="24h",
            include_predictions=True
        )
        
        print("\n🎉 Urban Hive Agent 실행 완료!")
        
        if result and result.critical_issues and "Analysis failed" in result.critical_issues[0]:
             print("\n💀 그러나, 분석 중 에러가 발생했습니다:")
             for issue in result.critical_issues:
                 print(f"  - {issue}")
        else:
             print("\n✅ 분석 성공! 결과 파일을 확인하세요.")
        
    except Exception as e:
        print(f"\n💥 심각한 예외 발생: {e}")
        print(f"예외 타입: {type(e).__name__}")
        import traceback
        print("\n🔍 상세 에러:")
        traceback.print_exc()

if __name__ == "__main__":
    # 3분 (180초) 타임아웃 설정
    try:
        asyncio.run(asyncio.wait_for(final_test_urban_hive(), timeout=180.0))
    except asyncio.TimeoutError:
        print("\n💀 테스트 시간이 3분을 초과했습니다. 명백한 성능 문제입니다.") 