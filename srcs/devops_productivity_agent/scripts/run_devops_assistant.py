#!/usr/bin/env python3
"""
Interactive runner for the DevOps Productivity Agent
with MCP server integrations
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.devops_assistant_agent import DevOpsProductivityAgent


class DevOpsAssistantRunner:
    """Interactive DevOps Assistant runner with MCP integration"""

    def __init__(self):
        self.agent = DevOpsProductivityAgent()
        self.commands = {
            "1": ("☁️ AWS 리소스 관리", self.aws_management),
            "2": ("🐙 GitHub 작업", self.github_operations),
            "3": ("⚙️ Kubernetes 관리", self.kubernetes_ops),
            "4": ("📊 인프라 모니터링", self.infrastructure_monitoring),
            "5": ("🌐 멀티클라우드 조정", self.multi_cloud_coordination),
            "6": ("💬 사용자 정의 요청", self.custom_request),
            "7": ("🚪 종료", self.exit_app)
        }

    def display_banner(self):
        """Display application banner"""
        print("\n" + "="*60)
        print("🚀 DEVOPS PRODUCTIVITY AGENT")
        print("MCP 기반 멀티클라우드 DevOps 자동화")
        print("="*60)
        print("\nMCP 서버 통합:")
        print("• AWS Knowledge Base - EC2, S3, Lambda, CloudFormation")
        print("• GitHub Operations - 리포지토리, PR, CI/CD")
        print("• Prometheus Metrics - 인프라 모니터링")
        print("• Kubernetes - 클러스터 및 워크로드 관리")
        print("• GCP/Azure - 멀티클라우드 조정")
        print("\n필수 환경변수:")
        print("• GOOGLE_API_KEY - Gemini API 키")
        print("• GITHUB_TOKEN - GitHub API 토큰")
        print("• AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - AWS 자격증명")
        print("="*60)

    def display_menu(self):
        """Display main menu"""
        print("\n📋 Available Commands:")
        for key, (description, _) in self.commands.items():
            print(f"{key}. {description}")
        print()

    async def aws_management(self):
        """AWS 리소스 관리"""
        print("\n☁️ AWS 리소스 관리")
        request = input("AWS 작업을 설명해주세요 (예: EC2 인스턴스 상태 확인): ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ AWS 작업 실행 중: {request}")
        await self._execute_request(request)

    async def github_operations(self):
        """GitHub 작업"""
        print("\n🐙 GitHub 작업")
        request = input("GitHub 작업을 설명해주세요 (예: microsoft 조직 리포지토리 분석): ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ GitHub 작업 실행 중: {request}")
        await self._execute_request(request)

    async def kubernetes_ops(self):
        """Kubernetes 관리"""
        print("\n⚙️ Kubernetes 관리")
        request = input("Kubernetes 작업을 설명해주세요 (예: 클러스터 리소스 사용률 조회): ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ Kubernetes 작업 실행 중: {request}")
        await self._execute_request(request)

    async def infrastructure_monitoring(self):
        """인프라 모니터링"""
        print("\n📊 인프라 모니터링")
        request = input("모니터링 작업을 설명해주세요 (예: Prometheus 메트릭 확인): ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ 모니터링 작업 실행 중: {request}")
        await self._execute_request(request)

    async def multi_cloud_coordination(self):
        """멀티클라우드 조정"""
        print("\n🌐 멀티클라우드 조정")
        request = input("멀티클라우드 작업을 설명해주세요 (예: AWS와 GCP 리소스 비교): ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ 멀티클라우드 작업 실행 중: {request}")
        await self._execute_request(request)

    async def _execute_request(self, request: str):
        """Execute request using the agent with mcp_agent standard"""
        try:
            print(f"⏳ mcp_agent 표준으로 요청 처리 중...")
            result = await self.agent.run_workflow(request)

            if result['status'] == 'success':
                print(f"\n✅ 작업 완료!")
                print(f"📁 결과 파일: {result['output_file']}")

                # Show result preview (Markdown format)
                if 'result' in result and isinstance(result['result'], str):
                    print(f"\n📄 결과 미리보기:")
                    preview = result['result'][:300] + "..." if len(result['result']) > 300 else result['result']
                    print(f"  {preview}")

                    # Show file info
                    if os.path.exists(result['output_file']):
                        file_size = os.path.getsize(result['output_file'])
                        print(f"  📊 파일 크기: {file_size:,} bytes")
            else:
                print(f"❌ 오류: {result.get('error', '알 수 없는 오류')}")

        except Exception as e:
            print(f"❌ 예외 발생: {str(e)}")
            import traceback
            print(f"🔍 상세 오류: {traceback.format_exc()}")

    async def custom_request(self):
        """사용자 정의 요청 처리"""
        print("\n💬 사용자 정의 DevOps 요청")
        request = input("DevOps 요청을 입력해주세요: ").strip()

        if not request:
            print("❌ 요청 내용이 필요합니다")
            return

        print(f"\n⏳ 요청 처리 중: '{request}'...")
        await self._execute_request(request)

    def exit_app(self):
        """애플리케이션 종료"""
        print("\n👋 DevOps Productivity Agent를 사용해주셔서 감사합니다!")
        sys.exit(0)

    def check_mcp_servers(self):
        """MCP 서버 연결 상태 확인 (mcp_agent 표준)"""
        print("\n🔍 MCP 서버 연결 상태 확인 중...")

        # Check if MCP servers are configured in mcp_agent.config.yaml
        mcp_servers = ["aws-kb", "github", "prometheus", "kubernetes", "gcp-admin", "azure-admin"]
        available_servers = []

        try:
            # Check if agent can access MCP servers through BaseAgent
            if hasattr(self.agent, 'app') and hasattr(self.agent.app, 'settings'):
                configured_servers = self.agent.app.settings.get('mcp', {}).get('servers', {})
                for server in mcp_servers:
                    if server in configured_servers and configured_servers[server].get('enabled', True):
                        available_servers.append(server)

            if available_servers:
                print(f"✅ 사용 가능한 MCP 서버: {', '.join(available_servers)}")
                print(f"📋 서버 유형:")
                for server in available_servers:
                    if server == "aws-kb":
                        print(f"  • {server}: AWS Knowledge Base")
                    elif server == "github":
                        print(f"  • {server}: GitHub Operations")
                    elif server == "prometheus":
                        print(f"  • {server}: Prometheus Metrics")
                    elif server == "kubernetes":
                        print(f"  • {server}: Kubernetes Cluster")
                    elif server in ["gcp-admin", "azure-admin"]:
                        print(f"  • {server}: Multi-cloud Management")
                return True
            else:
                print("⚠️ MCP 서버가 설정되지 않았습니다")
                print("💡 mcp_agent.config.yaml 파일을 확인해주세요")
                return False

        except Exception as e:
            print(f"❌ MCP 서버 확인 중 오류: {str(e)}")
            return False

    def check_configuration(self):
        """필수 환경변수 확인"""
        required_vars = {
            "GOOGLE_API_KEY": "Gemini API 접근",
            "GITHUB_TOKEN": "GitHub API 접근",
            "AWS_ACCESS_KEY_ID": "AWS 리소스 접근",
            "AWS_SECRET_ACCESS_KEY": "AWS 리소스 접근"
        }

        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"  • {var}: {description}")

        if missing_vars:
            print("\n⚠️ 누락된 필수 환경변수:")
            for var in missing_vars:
                print(var)
            print("\n.env.example 파일을 참고하여 환경변수를 설정해주세요.")
            return False

        return True

    async def run(self):
        """메인 애플리케이션 루프"""
        self.display_banner()

        # 환경변수 확인
        if not self.check_configuration():
            return

        # MCP 서버 연결 확인
        if not self.check_mcp_servers():
            print("⚠️ MCP 서버 연결에 문제가 있지만 계속 진행합니다...")

        print("\n✅ 설정 확인 완료")

        while True:
            try:
                self.display_menu()
                choice = input("옵션을 선택하세요 (1-7): ").strip()

                if choice in self.commands:
                    _, action = self.commands[choice]
                    await action()
                else:
                    print("❌ 잘못된 선택입니다. 1-7 중에서 선택해주세요.")

                input("\n계속하려면 Enter를 누르세요...")

            except KeyboardInterrupt:
                print("\n\n👋 안녕히 가세요!")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류: {str(e)}")
                input("계속하려면 Enter를 누르세요...")


async def main():
    """메인 진입점"""
    try:
        runner = DevOpsAssistantRunner()
        await runner.run()
    except Exception as e:
        print(f"\n❌ 치명적 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
