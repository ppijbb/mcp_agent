#!/usr/bin/env python3
"""
Interactive runner for the production DevOps Assistant Agent
with GitHub and Prometheus API integrations
"""

import asyncio
import os
import sys
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.devops_assistant_agent import DevOpsAssistantAgent


class DevOpsAssistantRunner:
    """Interactive DevOps Assistant runner with error handling"""
    
    def __init__(self):
        self.agent = DevOpsAssistantAgent()
        self.commands = {
            "1": ("🔍 Analyze GitHub Repositories", self.analyze_repos),
            "2": ("🚀 Monitor CI/CD Pipelines", self.monitor_pipelines),
            "3": ("📊 Check Infrastructure Health", self.check_health),
            "4": ("💬 Custom Request", self.custom_request),
            "5": ("🚪 Exit", self.exit_app)
        }
    
    def display_banner(self):
        """Display application banner"""
        print("\n" + "="*60)
        print("🚀 DEVOPS ASSISTANT AGENT")
        print("Production-level DevOps automation with GitHub and Prometheus integrations")
        print("="*60)
        print("\nAPI Integrations:")
        print("• GitHub API for repositories and CI/CD")
        print("• Prometheus API for metrics and monitoring")
        print("\nConfiguration required:")
        print("• GITHUB_TOKEN environment variable")
        print("• PROMETHEUS_URL environment variable")
        print("• GOOGLE_API_KEY environment variable")
        print("="*60)
    
    def display_menu(self):
        """Display main menu"""
        print("\n📋 Available Commands:")
        for key, (description, _) in self.commands.items():
            print(f"{key}. {description}")
        print()
    
    async def analyze_repos(self):
        """Analyze GitHub repositories"""
        print("\n🔍 GitHub Repository Analysis")
        org = input("Enter GitHub organization (e.g., microsoft): ").strip()
        
        if not org:
            print("❌ Organization name is required")
            return
        
        print(f"\n⏳ Analyzing repositories in '{org}' organization...")
        
        try:
            result = await self.agent.analyze_github_repositories(org=org)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            print(f"\n✅ Analysis Results:")
            print(f"• Total repositories: {result['total_repositories']}")
            print(f"• Total stars: {result['stars_total']}")
            
            if result.get('languages'):
                print(f"• Languages: {', '.join(result['languages'].keys())}")
            
            if result.get('repositories'):
                print(f"\n🏆 Top repositories:")
                top_repos = sorted(result['repositories'], 
                                 key=lambda x: x['stars'], reverse=True)[:5]
                for repo in top_repos:
                    print(f"  • {repo['name']} ({repo['stars']} ⭐)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    async def monitor_pipelines(self):
        """Monitor CI/CD pipelines"""
        print("\n🚀 CI/CD Pipeline Monitoring")
        owner = input("Enter repository owner: ").strip()
        repo = input("Enter repository name: ").strip()
        branch = input("Enter branch name (optional): ").strip() or None
        
        if not owner or not repo:
            print("❌ Both owner and repository name are required")
            return
        
        print(f"\n⏳ Monitoring pipelines for {owner}/{repo}...")
        
        try:
            result = await self.agent.monitor_ci_cd_pipelines(owner=owner, repo=repo, branch=branch)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            print(f"\n✅ Pipeline Status:")
            print(f"• Total runs: {result['total_runs']}")
            print(f"• Successful: {result['success_count']}")
            print(f"• Failed: {result['failure_count']}")
            
            if 'success_rate' in result:
                print(f"• Success rate: {result['success_rate']}%")
            
            if result.get('recent_runs'):
                print(f"\n📋 Recent runs:")
                for run in result['recent_runs'][:5]:
                    status = run.get('conclusion') or run.get('status') or 'unknown'
                    branch_info = f" ({run['branch']})" if run.get('branch') else ""
                    print(f"  • Run #{run['id']} - {status}{branch_info}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    async def check_health(self):
        """Check infrastructure health"""
        print("\n📊 Infrastructure Health Check")
        print("⏳ Checking system metrics...")
        
        try:
            result = await self.agent.check_infrastructure_health()
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
            
            print(f"\n✅ Health Status:")
            print(f"• Overall status: {result['overall_status']}")
            
            if result.get('metrics'):
                for metric_name, metric_value in result['metrics'].items():
                    if metric_name == 'cpu_usage_percent' and isinstance(metric_value, (int, float)):
                        print(f"• CPU usage: {metric_value}%")
                    elif metric_name == 'error':
                        print(f"• Error: {metric_value}")
                    else:
                        print(f"• {metric_name}: {metric_value}")
            
            print(f"• Timestamp: {result['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    async def custom_request(self):
        """Process custom request"""
        print("\n💬 Custom DevOps Request")
        request = input("Enter your DevOps request: ").strip()
        
        if not request:
            print("❌ Request cannot be empty")
            return
        
        print(f"\n⏳ Processing request: '{request}'...")
        
        try:
            result = await self.agent.process_request(request)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return
                
            print(f"\n✅ Response:")
            print(f"• Action: {result.get('action', 'unknown')}")
            
            if result.get('parameters'):
                print(f"• Parameters: {json.dumps(result['parameters'])}")
                
            print(f"• Timestamp: {result.get('timestamp', 'unknown')}")
            
            if result.get('result'):
                if isinstance(result['result'], dict) and len(result['result']) > 10:
                    print(f"• Result summary: {len(result['result'])} data points")
                    show_details = input("\nShow full result details? (y/N): ").strip().lower() == 'y'
                    if show_details:
                        print(f"\n{json.dumps(result['result'], indent=2)}")
                else:
                    print(f"• Result: {json.dumps(result['result'], indent=2)}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def exit_app(self):
        """Exit the application"""
        print("\n👋 Thanks for using DevOps Assistant!")
        sys.exit(0)
    
    def check_configuration(self):
        """Check if required environment variables are set"""
        required_vars = {
            "GITHUB_TOKEN": "GitHub API access",
            "GOOGLE_API_KEY": "Google Gemini API access"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"  • {var}: {description}")
        
        if missing_vars:
            print("\n⚠️ Missing required environment variables:")
            for var in missing_vars:
                print(var)
            print("\nPlease set these variables before running the agent.")
            return False
        
        return True
    
    async def run(self):
        """Main application loop with error handling"""
        self.display_banner()
        
        if not self.check_configuration():
            return
        
        print("\n✅ Configuration check passed")
        
        while True:
            try:
                self.display_menu()
                choice = input("Select an option (1-5): ").strip()
                
                if choice in self.commands:
                    _, action = self.commands[choice]
                    await action()
                else:
                    print("❌ Invalid choice. Please select 1-5.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                input("Press Enter to continue...")


async def main():
    """Main entry point with error handling"""
    try:
        runner = DevOpsAssistantRunner()
        await runner.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 