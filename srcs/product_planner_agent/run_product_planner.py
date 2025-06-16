#!/usr/bin/env python3
"""
Product Planner Agent 실행 스크립트

Figma와 Notion을 연동한 프로덕트 기획 자동화 Agent 실행
"""

import asyncio
import sys
import os
import json
import logging
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Agent 및 설정 import
from srcs.product_planner_agent import ProductPlannerAgent
from srcs.product_planner_agent.config import validate_config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('product_planner_agent.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """메인 실행 함수"""
    print("🚀 Product Planner Agent 시작")
    print("=" * 50)
    
    try:
        # 1. 설정 검증
        print("📋 설정 검증 중...")
        config_status = validate_config()
        print(f"설정 상태: {config_status['status']}")
        
        if config_status['status'] != 'valid':
            print(f"❌ 설정 오류: {config_status.get('error', 'Unknown error')}")
            return
        
        # 2. Agent 초기화
        print("\n🤖 Agent 초기화 중...")
        agent = ProductPlannerAgent(
            company_name="TechCorp Inc.",
            project_name="Sample Product Planning"
        )
        
        # 3. 상태 확인
        print("\n📊 Agent 상태 확인...")
        status = agent.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 4. 사용자 입력
        print("\n" + "=" * 50)
        print("🎨 Figma URL을 입력하세요 (예시용 기본값 제공):")
        print("예시: https://www.figma.com/file/ABC123/Sample-Design")
        
        figma_url = input("Figma URL: ").strip()
        
        if not figma_url:
            figma_url = "https://www.figma.com/file/sample123/Login-Design-Sample"
            print(f"기본값 사용: {figma_url}")
        
        print("\n🔄 워크플로우 실행 중...")
        print("1. Figma 디자인 분석")
        print("2. PRD 생성")  
        print("3. 로드맵 생성")
        
        # 5. 전체 워크플로우 실행
        try:
            result = await agent.run_full_workflow(figma_url)
            
            print("\n✅ 워크플로우 완료!")
            print("=" * 50)
            print("📄 생성된 문서:")
            print(f"  • PRD 페이지: {result.get('prd_page_id')}")
            print(f"  • 로드맵 페이지: {result.get('roadmap_page_id')}")
            print(f"  • 분석 요약: {result.get('analysis_summary')}")
            print(f"  • 타임스탬프: {result.get('timestamp')}")
            
            # 결과를 파일로 저장
            output_file = f"product_planning_result_{result.get('timestamp', 'unknown')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 결과가 저장되었습니다: {output_file}")
            
        except Exception as workflow_error:
            print(f"\n❌ 워크플로우 실행 중 오류: {str(workflow_error)}")
            logger.error(f"워크플로우 오류: {str(workflow_error)}")
            
            # 개별 단계 테스트
            print("\n🔧 개별 단계 테스트 진행...")
            
            try:
                # 디자인 분석만 실행
                print("  - 디자인 분석 테스트...")
                analysis = await agent.analyze_figma_design(figma_url)
                print(f"    ✅ 분석 완료: {len(analysis.get('component_analysis', {}).get('components_detail', []))}개 컴포넌트 발견")
                
                # PRD 생성 테스트
                print("  - PRD 생성 테스트...")
                prd_id = await agent.generate_prd(analysis)
                print(f"    ✅ PRD 생성 완료: {prd_id}")
                
                # 로드맵 생성 테스트
                print("  - 로드맵 생성 테스트...")
                roadmap_id = await agent.create_roadmap()
                print(f"    ✅ 로드맵 생성 완료: {roadmap_id}")
                
                print("\n✅ 모든 개별 단계 테스트 성공!")
                
            except Exception as test_error:
                print(f"    ❌ 개별 테스트 오류: {str(test_error)}")
                logger.error(f"개별 테스트 오류: {str(test_error)}")
        
        print("\n" + "=" * 50)
        print("🎉 Product Planner Agent 실행 완료!")
        print("\n📚 다음 단계:")
        print("1. 생성된 Notion 페이지 확인")
        print("2. 요구사항 리뷰 및 보완")
        print("3. 개발팀과 로드맵 논의")
        print("4. 정기적인 디자인-개발 동기화 설정")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {str(e)}")
        logger.error(f"메인 실행 오류: {str(e)}")
        import traceback
        traceback.print_exc()

def run_interactive_demo():
    """대화형 데모 실행"""
    print("🎮 Product Planner Agent 대화형 데모")
    print("=" * 50)
    
    while True:
        print("\n메뉴를 선택하세요:")
        print("1. 전체 워크플로우 실행")
        print("2. 디자인 분석만 실행")
        print("3. Agent 상태 확인")
        print("4. 설정 검증")
        print("0. 종료")
        
        choice = input("\n선택 (0-4): ").strip()
        
        if choice == "0":
            print("👋 안녕히 가세요!")
            break
        elif choice == "1":
            asyncio.run(main())
        elif choice == "2":
            asyncio.run(run_analysis_only())
        elif choice == "3":
            asyncio.run(show_agent_status())
        elif choice == "4":
            show_config_status()
        else:
            print("❌ 잘못된 선택입니다. 다시 시도해주세요.")

async def run_analysis_only():
    """디자인 분석만 실행"""
    try:
        print("\n🔍 디자인 분석 실행")
        
        agent = ProductPlannerAgent()
        figma_url = input("Figma URL: ").strip()
        
        if not figma_url:
            figma_url = "https://www.figma.com/file/sample123/Sample-Design"
            print(f"기본값 사용: {figma_url}")
        
        analysis = await agent.analyze_figma_design(figma_url)
        
        print("\n📊 분석 결과:")
        print(f"  • 컴포넌트 수: {analysis.get('component_analysis', {}).get('total_components', 0)}")
        print(f"  • 복잡도: {analysis.get('overall_assessment', {}).get('development_complexity', 'unknown')}")
        print(f"  • 신뢰도: {analysis.get('confidence_score', 0):.2f}")
        
        # 상세 결과를 파일로 저장
        output_file = f"design_analysis_{agent.timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"  • 상세 결과 저장: {output_file}")
        
    except Exception as e:
        print(f"❌ 분석 실행 오류: {str(e)}")

async def show_agent_status():
    """Agent 상태 표시"""
    try:
        print("\n📊 Agent 상태 확인")
        
        agent = ProductPlannerAgent()
        status = agent.get_status()
        
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 상태 확인 오류: {str(e)}")

def show_config_status():
    """설정 상태 표시"""
    try:
        print("\n⚙️  설정 상태 확인")
        
        config_status = validate_config()
        print(json.dumps(config_status, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 설정 확인 오류: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_interactive_demo()
    else:
        asyncio.run(main()) 