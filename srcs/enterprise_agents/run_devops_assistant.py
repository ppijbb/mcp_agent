"""
DevOps Assistant Agent 실행 스크립트
MCP 기반 개발자 생산성 자동화 에이전트 데모
"""

import asyncio
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enterprise_agents.devops_assistant_agent import (
    DevOpsAssistantAgent,
    run_code_review,
    run_deployment_check,
    run_issue_analysis,
    run_team_standup,
    run_performance_analysis
)


class DevOpsAssistantDemo:
    """DevOps Assistant Agent 데모 클래스"""

    def __init__(self):
        self.agent = None
        self.is_running = False

    async def initialize_agent(self):
        """에이전트 초기화"""
        print("🔧 DevOps Assistant Agent 초기화 중...")

        try:
            # 기본 설정으로 에이전트 생성
            self.agent = DevOpsAssistantAgent()
            print("✅ 에이전트 초기화 완료!")
            return True

        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {e}")
            print("💡 GOOGLE_API_KEY 환경변수가 설정되어 있는지 확인해주세요.")
            return False

    def display_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*60)
        print("🚀 DevOps Assistant Agent - Demo Menu")
        print("="*60)
        print("1. 🔍 코드 리뷰 자동화")
        print("2. 🚀 배포 상태 확인")
        print("3. 🎯 이슈 우선순위 분석")
        print("4. 👥 팀 스탠드업 준비")
        print("5. 📊 성능 분석")
        print("6. 🔄 연속 모니터링 모드 (시작)")
        print("7. 🛑 연속 모니터링 모드 (중지)")
        print("8. 📋 에이전트 상태 확인")
        print("9. ❌ 종료")
        print("="*60)

    async def demo_code_review(self):
        """코드 리뷰 데모"""
        print("\n🔍 코드 리뷰 자동화 데모")
        print("-" * 40)

        # 사용자 입력 받기
        owner = input("GitHub 소유자 (기본값: microsoft): ").strip() or "microsoft"
        repo = input("저장소 이름 (기본값: vscode): ").strip() or "vscode"
        pull_number = input("PR 번호 (기본값: 42): ").strip() or "42"

        try:
            pull_number = int(pull_number)
            print(f"\n📋 처리 중: {owner}/{repo}#{pull_number}")

            result = await run_code_review(self.agent, owner, repo, pull_number)

            if result:
                print("✅ 코드 리뷰가 성공적으로 완료되었습니다!")
            else:
                print("❌ 코드 리뷰 처리 중 오류가 발생했습니다.")

        except ValueError:
            print("❌ PR 번호는 숫자여야 합니다.")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    async def demo_deployment_check(self):
        """배포 상태 확인 데모"""
        print("\n🚀 배포 상태 확인 데모")
        print("-" * 40)

        owner = input("GitHub 소유자 (기본값: kubernetes): ").strip() or "kubernetes"
        repo = input("저장소 이름 (기본값: kubernetes): ").strip() or "kubernetes"

        try:
            print(f"\n📋 처리 중: {owner}/{repo}")

            result = await run_deployment_check(self.agent, owner, repo)

            if result:
                print("✅ 배포 상태 확인이 완료되었습니다!")
            else:
                print("❌ 배포 상태 확인 중 오류가 발생했습니다.")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    async def demo_issue_analysis(self):
        """이슈 분석 데모"""
        print("\n🎯 이슈 우선순위 분석 데모")
        print("-" * 40)

        owner = input("GitHub 소유자 (기본값: facebook): ").strip() or "facebook"
        repo = input("저장소 이름 (기본값: react): ").strip() or "react"

        try:
            print(f"\n📋 처리 중: {owner}/{repo}")

            result = await run_issue_analysis(self.agent, owner, repo)

            if result:
                print("✅ 이슈 분석이 완료되었습니다!")
            else:
                print("❌ 이슈 분석 중 오류가 발생했습니다.")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    async def demo_team_standup(self):
        """팀 스탠드업 데모"""
        print("\n👥 팀 스탠드업 준비 데모")
        print("-" * 40)

        team = input("팀 이름 (기본값: development): ").strip() or "development"

        try:
            print(f"\n📋 처리 중: {team} 팀")

            result = await run_team_standup(self.agent, team)

            if result:
                print("✅ 팀 스탠드업 요약이 완료되었습니다!")
            else:
                print("❌ 팀 스탠드업 준비 중 오류가 발생했습니다.")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    async def demo_performance_analysis(self):
        """성능 분석 데모"""
        print("\n📊 성능 분석 데모")
        print("-" * 40)

        service = input("서비스 이름 (기본값: main-api): ").strip() or "main-api"
        timeframe = input("분석 기간 (기본값: 24h): ").strip() or "24h"

        try:
            print(f"\n📋 처리 중: {service} ({timeframe})")

            result = await run_performance_analysis(self.agent, service, timeframe)

            if result:
                print("✅ 성능 분석이 완료되었습니다!")
                print(f"\n📊 분석 결과:")
                print(result[:300] + "..." if len(result) > 300 else result)
            else:
                print("❌ 성능 분석 중 오류가 발생했습니다.")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    async def start_monitoring(self):
        """연속 모니터링 시작"""
        if self.is_running:
            print("⚠️  모니터링이 이미 실행 중입니다.")
            return

        print("\n🔄 연속 모니터링 모드 시작")
        print("💡 이 모드에서는 주기적으로 상태를 확인하고 자동으로 작업을 처리합니다.")
        print("   Ctrl+C를 눌러서 중지하거나 메뉴에서 '7'을 선택하세요.")

        self.is_running = True

        # 백그라운드 작업으로 시작
        asyncio.create_task(self.agent.start())

        print("✅ 연속 모니터링이 시작되었습니다!")

    async def stop_monitoring(self):
        """연속 모니터링 중지"""
        if not self.is_running:
            print("⚠️  모니터링이 실행되고 있지 않습니다.")
            return

        print("\n🛑 연속 모니터링 모드 중지")

        self.is_running = False
        await self.agent.stop()

        print("✅ 연속 모니터링이 중지되었습니다!")

    def show_agent_status(self):
        """에이전트 상태 표시"""
        print("\n📋 에이전트 상태")
        print("-" * 40)

        if self.agent:
            status = self.agent.get_status()

            print(f"실행 상태: {'🟢 실행 중' if status['is_running'] else '🔴 중지됨'}")
            print(f"모니터링: {'🟢 활성화' if status['monitoring_active'] else '🔴 비활성화'}")
            print(f"대기 중인 작업: {status['queue_size']}개")
            print(f"AI 모델: {status['model']}")
            print(f"마지막 업데이트: {status['uptime']}")
        else:
            print("❌ 에이전트가 초기화되지 않았습니다.")

    async def run_demo_mode(self):
        """대화형 데모 실행"""
        print("🎉 DevOps Assistant Agent에 오신 것을 환영합니다!")
        print("💡 이 데모는 AI 기반 개발자 생산성 자동화 기능을 보여줍니다.")

        # 에이전트 초기화
        if not await self.initialize_agent():
            return

        # 메인 루프
        while True:
            try:
                self.display_menu()
                choice = input("\n선택하세요 (1-9): ").strip()

                if choice == "1":
                    await self.demo_code_review()
                elif choice == "2":
                    await self.demo_deployment_check()
                elif choice == "3":
                    await self.demo_issue_analysis()
                elif choice == "4":
                    await self.demo_team_standup()
                elif choice == "5":
                    await self.demo_performance_analysis()
                elif choice == "6":
                    await self.start_monitoring()
                elif choice == "7":
                    await self.stop_monitoring()
                elif choice == "8":
                    self.show_agent_status()
                elif choice == "9":
                    print("\n👋 DevOps Assistant Agent를 종료합니다.")
                    if self.is_running:
                        await self.stop_monitoring()
                    break
                else:
                    print("❌ 잘못된 선택입니다. 1-9 사이의 숫자를 입력하세요.")

                input("\n계속하려면 Enter를 누르세요...")

            except KeyboardInterrupt:
                print("\n\n⚠️  사용자에 의해 중단되었습니다.")
                if self.is_running:
                    await self.stop_monitoring()
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {e}")
                input("계속하려면 Enter를 누르세요...")


async def run_quick_demo():
    """빠른 데모 실행 (모든 기능 자동 실행)"""
    print("🚀 DevOps Assistant Agent - 빠른 데모")
    print("="*50)

    # 에이전트 초기화
    agent = DevOpsAssistantAgent()

    # 모든 기능 순차 실행
    demos = [
        ("코드 리뷰", run_code_review, ["microsoft", "vscode", 42]),
        ("배포 상태 확인", run_deployment_check, ["kubernetes", "kubernetes"]),
        ("이슈 분석", run_issue_analysis, ["facebook", "react"]),
        ("팀 스탠드업", run_team_standup, ["development"]),
        ("성능 분석", run_performance_analysis, ["main-api", "24h"])
    ]

    for name, func, args in demos:
        print(f"\n📋 {name} 실행 중...")
        try:
            result = await func(agent, *args)
            print(f"✅ {name} 완료!")
        except Exception as e:
            print(f"❌ {name} 실패: {e}")

        await asyncio.sleep(2)  # 잠시 대기

    print("\n🎉 모든 데모가 완료되었습니다!")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="DevOps Assistant Agent 실행")
    parser.add_argument(
        "--mode",
        choices=["demo", "quick"],
        default="demo",
        help="실행 모드 선택 (demo: 대화형, quick: 자동 실행)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "quick":
            asyncio.run(run_quick_demo())
        else:
            demo = DevOpsAssistantDemo()
            asyncio.run(demo.run_demo_mode())

    except KeyboardInterrupt:
        print("\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
