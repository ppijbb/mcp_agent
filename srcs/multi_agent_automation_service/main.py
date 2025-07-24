"""
Multi-Agent Automation Service - 메인 실행 파일
==============================================

Python mcp_agent 라이브러리 기반 Multi-Agent 시스템
Gemini CLI를 통한 최종 명령 실행
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

from .orchestrator import MultiAgentOrchestrator
from .agents import (
    CodeReviewAgent,
    DocumentationAgent,
    PerformanceTestAgent,
    SecurityDeploymentAgent
)

class MultiAgentAutomationService:
    """Multi-Agent 자동화 서비스 메인 클래스"""
    
    def __init__(self):
        self.orchestrator = MultiAgentOrchestrator()
        
    async def run_full_automation(self, target_paths: list = None):
        """전체 자동화 실행"""
        print("🚀 Multi-Agent 전체 자동화 시작...")
        
        try:
            result = await self.orchestrator.run_full_automation(target_paths)
            
            print("\n" + "="*60)
            print("📊 전체 자동화 결과")
            print("="*60)
            print(result.execution_summary)
            
            if result.overall_status == "success":
                print("✅ 전체 자동화 성공!")
            elif result.overall_status == "partial_success":
                print("⚠️ 부분적 성공 (일부 실패)")
            else:
                print("❌ 전체 자동화 실패")
            
            return result
            
        except Exception as e:
            print(f"❌ 전체 자동화 실패: {e}")
            raise
    
    async def run_code_review_workflow(self, target_paths: list = None):
        """코드 리뷰 워크플로우 실행"""
        print("🔍 코드 리뷰 워크플로우 시작...")
        
        try:
            result = await self.orchestrator.run_code_review_workflow(target_paths)
            
            print("\n" + "="*60)
            print("📋 코드 리뷰 결과")
            print("="*60)
            print(result.execution_summary)
            
            return result
            
        except Exception as e:
            print(f"❌ 코드 리뷰 워크플로우 실패: {e}")
            raise
    
    async def run_deployment_workflow(self, deployment_id: str = None):
        """배포 워크플로우 실행"""
        print("🚀 배포 워크플로우 시작...")
        
        try:
            result = await self.orchestrator.run_deployment_workflow(deployment_id)
            
            print("\n" + "="*60)
            print("📦 배포 결과")
            print("="*60)
            print(result.execution_summary)
            
            return result
            
        except Exception as e:
            print(f"❌ 배포 워크플로우 실패: {e}")
            raise
    
    async def run_individual_agents(self, target_paths: list = None):
        """개별 Agent 실행"""
        print("🤖 개별 Agent 실행...")
        
        results = {}
        
        # 1. 코드 리뷰 Agent
        print("\n1️⃣ 코드 리뷰 Agent 실행 중...")
        try:
            code_review_agent = CodeReviewAgent()
            code_result = await code_review_agent.review_code(target_paths)
            results["code_review"] = code_result
            print(f"✅ 코드 리뷰 완료: {len(code_result.files_reviewed)}개 파일 검토")
        except Exception as e:
            print(f"❌ 코드 리뷰 실패: {e}")
            results["code_review"] = None
        
        # 2. 문서화 Agent
        print("\n2️⃣ 문서화 Agent 실행 중...")
        try:
            doc_agent = DocumentationAgent()
            doc_result = await doc_agent.update_documentation(target_paths)
            results["documentation"] = doc_result
            print(f"✅ 문서화 완료: {len(doc_result.files_updated)}개 파일 업데이트")
        except Exception as e:
            print(f"❌ 문서화 실패: {e}")
            results["documentation"] = None
        
        # 3. 성능 테스트 Agent
        print("\n3️⃣ 성능 테스트 Agent 실행 중...")
        try:
            perf_agent = PerformanceTestAgent()
            perf_result = await perf_agent.analyze_performance(target_paths)
            results["performance_test"] = perf_result
            print(f"✅ 성능 테스트 완료: {len(perf_result.bottlenecks_found)}개 병목 지점 발견")
        except Exception as e:
            print(f"❌ 성능 테스트 실패: {e}")
            results["performance_test"] = None
        
        # 4. 보안/배포 Agent
        print("\n4️⃣ 보안/배포 Agent 실행 중...")
        try:
            security_agent = SecurityDeploymentAgent()
            security_result = await security_agent.security_scan(target_paths)
            results["security_deployment"] = security_result
            print(f"✅ 보안 스캔 완료: {len(security_result.security_vulnerabilities)}개 취약점 발견")
        except Exception as e:
            print(f"❌ 보안 스캔 실패: {e}")
            results["security_deployment"] = None
        
        # 결과 요약
        print("\n" + "="*60)
        print("📊 개별 Agent 실행 결과")
        print("="*60)
        
        for agent_name, result in results.items():
            if result:
                print(f"✅ {agent_name}: 성공")
            else:
                print(f"❌ {agent_name}: 실패")
        
        return results
    
    def start_scheduler(self):
        """스케줄러 시작"""
        print("⏰ Multi-Agent 자동화 스케줄러 시작...")
        print("스케줄:")
        print("- 매일 새벽 2시: 전체 자동화")
        print("- 매주 월요일 오전 9시: 코드 리뷰 워크플로우")
        print("- 매시간: 배포 상태 확인")
        print("\nCtrl+C로 종료할 수 있습니다.")
        
        try:
            self.orchestrator.setup_scheduled_automation()
            self.orchestrator.run_scheduler()
        except KeyboardInterrupt:
            print("\n👋 스케줄러를 종료합니다.")
    
    def show_status(self):
        """현재 상태 표시"""
        print("📊 Multi-Agent 자동화 서비스 상태")
        print("="*50)
        
        # Orchestrator 히스토리
        history_count = len(self.orchestrator.orchestration_history)
        print(f"총 실행 횟수: {history_count}")
        
        if history_count > 0:
            latest = self.orchestrator.orchestration_history[-1]
            print(f"최근 실행: {latest.timestamp}")
            print(f"최근 상태: {latest.overall_status}")
        
        # Gemini CLI 실행 히스토리
        gemini_history = self.orchestrator.gemini_executor.execution_history
        print(f"Gemini CLI 실행 횟수: {len(gemini_history)}")
        
        if gemini_history:
            success_count = sum(1 for r in gemini_history if r.exit_code == 0)
            print(f"Gemini CLI 성공률: {(success_count/len(gemini_history))*100:.1f}%")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Automation Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python -m multi_agent_automation_service full                    # 전체 자동화
  python -m multi_agent_automation_service review                  # 코드 리뷰만
  python -m multi_agent_automation_service deploy                  # 배포 워크플로우
  python -m multi_agent_automation_service individual              # 개별 Agent 실행
  python -m multi_agent_automation_service scheduler               # 스케줄러 시작
  python -m multi_agent_automation_service status                  # 상태 확인
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["full", "review", "deploy", "individual", "scheduler", "status"],
        help="실행 모드"
    )
    
    parser.add_argument(
        "--paths",
        nargs="+",
        help="대상 경로 (여러 개 지정 가능)"
    )
    
    parser.add_argument(
        "--deployment-id",
        help="배포 ID (deploy 모드에서 사용)"
    )
    
    args = parser.parse_args()
    
    # 서비스 초기화
    service = MultiAgentAutomationService()
    
    try:
        if args.mode == "full":
            asyncio.run(service.run_full_automation(args.paths))
            
        elif args.mode == "review":
            asyncio.run(service.run_code_review_workflow(args.paths))
            
        elif args.mode == "deploy":
            asyncio.run(service.run_deployment_workflow(args.deployment_id))
            
        elif args.mode == "individual":
            asyncio.run(service.run_individual_agents(args.paths))
            
        elif args.mode == "scheduler":
            service.start_scheduler()
            
        elif args.mode == "status":
            service.show_status()
            
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 