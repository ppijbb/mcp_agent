#!/usr/bin/env python3
"""
DevOps Assistant Agent Runner
============================
대화형 DevOps Assistant Agent 실행 스크립트

Usage:
    python run_devops_assistant.py

Features:
- 🔍 코드 리뷰 분석
- 🚀 배포 상태 확인  
- 🎯 이슈 우선순위 분석
- 👥 팀 스탠드업 생성
- 📊 성능 분석
- 🔒 보안 스캔

Model: gemini-2.5-flash-lite-preview-0607
"""

import asyncio
import sys
import os
from typing import Dict, Any
import json
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.devops_assistant_agent import (
    DevOpsAssistantMCPAgent,
    DevOpsTaskType,
    create_devops_assistant,
    run_code_review,
    run_deployment_check,
    run_issue_analysis,
    run_team_standup,
    run_performance_analysis,
    run_security_scan
)

class DevOpsAssistantRunner:
    """DevOps Assistant Agent 실행기"""
    
    def __init__(self):
        self.agent = None
        self.session_start = datetime.now()
        
    async def initialize(self):
        """에이전트 초기화"""
        print("🚀 DevOps Assistant Agent 초기화 중...")
        try:
            self.agent = await create_devops_assistant()
            print("✅ 에이전트 초기화 완료!")
            print(f"📅 세션 시작: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🤖 모델: {self.agent.model_name}")
            print()
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            sys.exit(1)
    
    def show_menu(self):
        """메뉴 표시"""
        print("=" * 60)
        print("🛠️  DevOps Assistant Agent - Main Menu")
        print("=" * 60)
        print("1. 🔍 코드 리뷰 분석 (Code Review)")
        print("2. 🚀 배포 상태 확인 (Deployment Check)")
        print("3. 🎯 이슈 우선순위 분석 (Issue Analysis)")
        print("4. 👥 팀 스탠드업 생성 (Team Standup)")
        print("5. 📊 성능 분석 (Performance Analysis)")
        print("6. 🔒 보안 스캔 (Security Scan)")
        print("7. 📋 작업 히스토리 (Task History)")
        print("8. 📈 종합 리포트 (Summary Report)")
        print("9. 🏢 팀 메트릭 (Team Metrics)")
        print("0. 🚪 종료 (Exit)")
        print("=" * 60)
    
    async def handle_code_review(self):
        """코드 리뷰 처리"""
        print("\n🔍 GitHub Pull Request 코드 리뷰 분석")
        print("-" * 40)
        
        try:
            owner = input("GitHub Owner/Organization: ").strip() or "example-org"
            repo = input("Repository 이름: ").strip() or "example-repo"
            pr_number = int(input("PR 번호: ").strip() or "123")
            
            print(f"\n📝 분석 중: {owner}/{repo}#{pr_number}")
            result = await run_code_review(self.agent, owner, repo, pr_number)
            
            self.display_result(result)
            
        except ValueError:
            print("❌ PR 번호는 숫자여야 합니다.")
        except Exception as e:
            print(f"❌ 코드 리뷰 분석 실패: {e}")
    
    async def handle_deployment_check(self):
        """배포 상태 확인 처리"""
        print("\n🚀 서비스 배포 상태 확인")
        print("-" * 40)
        
        try:
            service_name = input("서비스 이름: ").strip() or "web-api"
            environment = input("환경 (production/staging/dev): ").strip() or "production"
            
            print(f"\n📊 확인 중: {service_name} ({environment})")
            result = await run_deployment_check(self.agent, service_name, environment)
            
            self.display_result(result)
            
        except Exception as e:
            print(f"❌ 배포 상태 확인 실패: {e}")
    
    async def handle_issue_analysis(self):
        """이슈 분석 처리"""
        print("\n🎯 GitHub 이슈 우선순위 분석")
        print("-" * 40)
        
        try:
            owner = input("GitHub Owner/Organization: ").strip() or "example-org"
            repo = input("Repository 이름: ").strip() or "example-repo"
            
            print(f"\n🔍 분석 중: {owner}/{repo} 이슈들")
            result = await run_issue_analysis(self.agent, owner, repo)
            
            self.display_result(result)
            
        except Exception as e:
            print(f"❌ 이슈 분석 실패: {e}")
    
    async def handle_team_standup(self):
        """팀 스탠드업 처리"""
        print("\n👥 팀 스탠드업 요약 생성")
        print("-" * 40)
        
        try:
            team_name = input("팀 이름: ").strip() or "Backend Team"
            
            print(f"\n📝 생성 중: {team_name} 스탠드업")
            result = await run_team_standup(self.agent, team_name)
            
            self.display_result(result)
            
        except Exception as e:
            print(f"❌ 팀 스탠드업 생성 실패: {e}")
    
    async def handle_performance_analysis(self):
        """성능 분석 처리"""
        print("\n📊 서비스 성능 분석")
        print("-" * 40)
        
        try:
            service_name = input("서비스 이름: ").strip() or "web-api"
            timeframe = input("분석 기간 (24h/7d/30d): ").strip() or "24h"
            
            print(f"\n🔍 분석 중: {service_name} ({timeframe})")
            result = await run_performance_analysis(self.agent, service_name, timeframe)
            
            self.display_result(result)
            
        except Exception as e:
            print(f"❌ 성능 분석 실패: {e}")
    
    async def handle_security_scan(self):
        """보안 스캔 처리"""
        print("\n🔒 보안 스캔 실행")
        print("-" * 40)
        
        try:
            target = input("스캔 대상 (URL/IP/Service): ").strip() or "https://api.example.com"
            scan_type = input("스캔 유형 (full/quick/specific): ").strip() or "full"
            
            print(f"\n🛡️ 스캔 중: {target} ({scan_type})")
            result = await run_security_scan(self.agent, target, scan_type)
            
            self.display_result(result)
            
        except Exception as e:
            print(f"❌ 보안 스캔 실패: {e}")
    
    def handle_task_history(self):
        """작업 히스토리 표시"""
        print("\n📋 작업 히스토리")
        print("-" * 40)
        
        history = self.agent.get_task_history()
        
        if not history:
            print("📝 아직 수행된 작업이 없습니다.")
            return
        
        for i, task in enumerate(history, 1):
            print(f"\n{i}. {task.task_type.value}")
            print(f"   ⏰ 시간: {task.timestamp}")
            print(f"   ✅ 상태: {task.status}")
            print(f"   🚀 처리시간: {task.processing_time:.2f}초")
            print(f"   💡 권장사항: {len(task.recommendations)}개")
    
    def handle_summary_report(self):
        """종합 리포트 표시"""
        print("\n📈 종합 요약 리포트")
        print("-" * 40)
        
        report = self.agent.get_summary_report()
        
        if "message" in report:
            print(f"📝 {report['message']}")
            return
        
        print(f"📊 총 작업 수: {report['total_tasks']}")
        print(f"⏱️ 총 처리시간: {report['total_processing_time']}")
        print(f"⚡ 평균 처리시간: {report['avg_processing_time']}")
        print(f"🤖 사용 모델: {report['model_used']}")
        print(f"🕐 마지막 업데이트: {report['last_updated']}")
        
        print("\n📋 작업 유형별 분석:")
        for task_type, count in report['task_breakdown'].items():
            print(f"   {task_type}: {count}회")
    
    def handle_team_metrics(self):
        """팀 메트릭 표시"""
        print("\n🏢 팀 메트릭")
        print("-" * 40)
        
        metrics = self.agent.get_team_metrics()
        
        if not metrics:
            print("📝 아직 기록된 팀 메트릭이 없습니다.")
            return
        
        for team_name, activity in metrics.items():
            print(f"\n👥 {team_name}")
            print(f"   📝 오늘 커밋: {activity.commits_today}")
            print(f"   🔄 PR 열림: {activity.prs_opened}")
            print(f"   ✅ PR 머지: {activity.prs_merged}")
            print(f"   🎯 이슈 해결: {activity.issues_resolved}")
            print(f"   🏗️ 빌드 성공률: {activity.build_success_rate}%")
            print(f"   ⏰ 평균 리뷰시간: {activity.avg_review_time}시간")
    
    def display_result(self, result):
        """결과 표시"""
        print(f"\n✅ {result.task_type.value} 완료!")
        print(f"⏰ 처리시간: {result.processing_time:.2f}초")
        print(f"📝 상태: {result.status}")
        
        print(f"\n💡 권장사항:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        
        # 상세 결과 표시 (선택적)
        show_details = input("\n📄 상세 결과를 보시겠습니까? (y/N): ").strip().lower()
        if show_details == 'y':
            print(f"\n📊 상세 결과:")
            print(json.dumps(result.result_data, ensure_ascii=False, indent=2))
    
    async def run(self):
        """메인 실행 루프"""
        await self.initialize()
        
        while True:
            try:
                self.show_menu()
                choice = input("\n선택하세요 (0-9): ").strip()
                
                if choice == '0':
                    print("\n👋 DevOps Assistant Agent를 종료합니다.")
                    break
                elif choice == '1':
                    await self.handle_code_review()
                elif choice == '2':
                    await self.handle_deployment_check()
                elif choice == '3':
                    await self.handle_issue_analysis()
                elif choice == '4':
                    await self.handle_team_standup()
                elif choice == '5':
                    await self.handle_performance_analysis()
                elif choice == '6':
                    await self.handle_security_scan()
                elif choice == '7':
                    self.handle_task_history()
                elif choice == '8':
                    self.handle_summary_report()
                elif choice == '9':
                    self.handle_team_metrics()
                else:
                    print("❌ 잘못된 선택입니다. 0-9 사이의 숫자를 입력하세요.")
                
                input("\n⏸️  계속하려면 Enter를 누르세요...")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 사용자가 중단했습니다. 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {e}")
                input("⏸️  계속하려면 Enter를 누르세요...")

async def main():
    """메인 함수"""
    print("🚀 DevOps Assistant Agent")
    print("=" * 60)
    print("MCP 기반 개발자 생산성 자동화 도구")
    print("Model: gemini-2.5-flash-lite-preview-0607")
    print("=" * 60)
    
    runner = DevOpsAssistantRunner()
    await runner.run()

if __name__ == "__main__":
    # Windows 환경에서의 asyncio 이벤트 루프 설정
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 