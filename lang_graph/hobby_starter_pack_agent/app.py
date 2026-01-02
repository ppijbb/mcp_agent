#!/usr/bin/env python3
"""
Hobby Starter Pack Agent - 메인 애플리케이션 런처
AutoGen + LangGraph 하이브리드 아키텍처 기반 취미 추천 시스템
"""

import asyncio
import uvicorn
import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from api.main import app
from autogen.agents import HSPAutoGenAgents
from langgraph_workflow.workflow import HSPLangGraphWorkflow
from bridge.a2a_bridge import A2AProtocolBridge
from mcp.manager import MCPServerManager

class HSPAgentApplication:
    """HSP Agent 메인 애플리케이션"""
    
    def __init__(self):
        self.autogen_agents = None
        self.langgraph_workflow = None
        self.a2a_bridge = None
        self.mcp_manager = None
        self.api_server = None
        
    async def initialize_components(self):
        """모든 컴포넌트 초기화"""
        print("🚀 HSP Agent 초기화 중...")
        
        try:
            # 1. AutoGen 에이전트 초기화
            print("📋 AutoGen 에이전트 초기화...")
            self.autogen_agents = HSPAutoGenAgents()
            
            # 2. MCP 서버 매니저 초기화
            print("🔌 MCP 서버 매니저 초기화...")
            self.mcp_manager = MCPServerManager()
            
            # 3. A2A 프로토콜 브리지 초기화
            print("🌉 A2A 프로토콜 브리지 초기화...")
            self.a2a_bridge = A2AProtocolBridge()
            
            # 4. 벡터 스토어 초기화
            print("🔍 벡터 스토어 초기화...")
            from langgraph_workflow.vector_store import HSPVectorStore
            vector_store = HSPVectorStore()
            
            # 5. LangGraph 워크플로우 초기화
            print("📊 LangGraph 워크플로우 초기화...")
            self.langgraph_workflow = HSPLangGraphWorkflow(
                autogen_agents=self.autogen_agents,
                mcp_manager=self.mcp_manager,
                vector_store=vector_store
            )
            
            # 6. 컴포넌트 간 연결 설정
            print("🔗 컴포넌트 간 연결 설정...")
            self.langgraph_workflow.a2a_bridge = self.a2a_bridge
            
            # 7. A2A 브리지에 주요 에이전트들 등록
            await self._register_agents()
            
            print("✅ 모든 컴포넌트 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 컴포넌트 초기화 실패: {e}")
            return False
    
    async def _register_agents(self):
        """A2A 브리지에 에이전트들 사전 등록"""
        agents_to_register = [
            ("ProfileAnalyst", "profile_analyst", "autogen"),
            ("HobbyDiscoverer", "hobby_discoverer", "autogen"),
            ("ScheduleIntegrator", "schedule_integrator", "autogen"),
            ("CommunityMatcher", "community_matcher", "autogen"), 
            ("ProgressTracker", "progress_tracker", "autogen"),
            ("DecisionModerator", "decision_moderator", "autogen"),
            ("LangGraphWorkflow", "workflow_orchestrator", "langgraph")
        ]
        
        for agent_id, agent_type, framework in agents_to_register:
            await self.a2a_bridge.register_agent(
                agent_id=agent_id,
                agent_type=agent_type,
                framework=framework
            )
            print(f"✓ {agent_id} 에이전트 등록 완료")
    
    def check_environment(self):
        """환경 설정 확인"""
        print("🔍 환경 설정 확인 중...")
        
        required_env_vars = [
            "GOOGLE_MAPS_API_KEY",
            "OPENWEATHER_API_KEY"
        ]
        
        optional_env_vars = [
            "GOOGLE_CALENDAR_TOKEN",
            "SOCIAL_MEDIA_TOKEN",
            "EDUCATION_PLATFORM_TOKEN",
            "FITNESS_TRACKER_TOKEN",
            "MUSIC_PLATFORM_TOKEN",
            "ECOMMERCE_API_KEY",
            "READING_PLATFORM_API_KEY",
            "RECIPE_API_KEY"
        ]
        
        missing_required = []
        missing_optional = []
        
        for var in required_env_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var in optional_env_vars:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_required:
            print(f"⚠️  필수 환경변수 누락: {', '.join(missing_required)}")
            print("   일부 MCP 서버 기능이 제한될 수 있습니다.")
        
        if missing_optional:
            print(f"ℹ️  선택적 환경변수 누락: {', '.join(missing_optional)}")
            print("   해당 서비스 연동이 제한됩니다.")
        
        print("✅ 환경 설정 확인 완료")
        return len(missing_required) == 0
    
    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """API 서버 시작"""
        print(f"🌐 API 서버 시작 중... http://{host}:{port}")
        
        # FastAPI 앱에 초기화된 컴포넌트들 주입
        app.state.autogen_agents = self.autogen_agents
        app.state.langgraph_workflow = self.langgraph_workflow
        app.state.a2a_bridge = self.a2a_bridge
        app.state.mcp_manager = self.mcp_manager
        app.state.vector_store = self.langgraph_workflow.vector_store
        
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            reload=False
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    
    def print_system_info(self):
        """시스템 정보 출력"""
        print("\n" + "="*60)
        print("🎯 Hobby Starter Pack Agent")
        print("🏗️  AutoGen + LangGraph 하이브리드 아키텍처")
        print("="*60)
        print(f"📦 AutoGen 에이전트: 6개 전문 에이전트")
        print(f"🔄 LangGraph 워크플로우: 7단계 처리 파이프라인")
        print(f"🌉 A2A 프로토콜 브리지: 프레임워크 간 통신")
        print(f"🔌 MCP 서버: 10개 외부 서비스 연동")
        print(f"🌐 FastAPI 서버: REST API 엔드포인트")
        print("="*60)
        print("📍 주요 엔드포인트:")
        print("   POST /api/workflow/run - 워크플로우 실행")
        print("   POST /api/agents/consensus - 에이전트 합의")
        print("   POST /api/mcp/call - MCP 서버 호출")
        print("   POST /api/a2a/send-message - A2A 메시지 전송")
        print("   GET  /api/health - 헬스 체크")
        print("="*60 + "\n")

async def main():
    """메인 실행 함수"""
    app_instance = HSPAgentApplication()
    
    # 시스템 정보 출력
    app_instance.print_system_info()
    
    # 환경 확인
    env_ok = app_instance.check_environment()
    
    # 컴포넌트 초기화
    init_success = await app_instance.initialize_components()
    
    if not init_success:
        print("❌ 애플리케이션 초기화 실패")
        return
    
    # API 서버 시작
    await app_instance.start_api_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 HSP Agent 종료")
    except Exception as e:
        print(f"❌ 애플리케이션 실행 오류: {e}")
        sys.exit(1) 