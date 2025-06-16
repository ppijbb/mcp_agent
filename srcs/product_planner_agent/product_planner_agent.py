#!/usr/bin/env python3
"""
Product Planner Agent - 구조화된 제품 기획 자동화 Agent
모듈화된 구조로 Agent 관리 및 워크플로우 실행
"""

import asyncio
import os
import sys
import re
from datetime import datetime
from urllib.parse import urlparse

from mcp_agent.app import MCPApp
from mcp_agent.config import get_settings
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from .config.agent_config import AgentConfig, AgentFactory


def validate_figma_url(url: str) -> tuple[bool, str]:
    """
    Figma URL 유효성 검증
    
    Args:
        url: 검증할 Figma URL
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not url:
        return False, "URL이 제공되지 않았습니다."
    
    # Figma URL 패턴 검증
    figma_pattern = r'^https://www\.figma\.com/(file|proto)/[A-Za-z0-9]+(/.*)?$'
    if not re.match(figma_pattern, url):
        return False, "유효한 Figma URL 형식이 아닙니다. (예: https://www.figma.com/file/...)"
    
    # URL 구조 검증
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc, parsed.path]):
            return False, "URL 구조가 올바르지 않습니다."
    except Exception as e:
        return False, f"URL 파싱 오류: {str(e)}"
    
    return True, ""


def get_figma_url() -> str:
    """
    Figma URL을 안전하게 가져오기
    
    Returns:
        str: 검증된 Figma URL
    """
    if len(sys.argv) > 1:
        url = sys.argv[1]
        is_valid, error_msg = validate_figma_url(url)
        if not is_valid:
            print(f"❌ 오류: {error_msg}")
            print("사용법: python product_planner_agent.py <figma_url>")
            sys.exit(1)
        return url
    else:
        print("❌ Figma URL이 필요합니다.")
        print("사용법: python product_planner_agent.py <figma_url>")
        print("예시: python product_planner_agent.py https://www.figma.com/file/abc123/project-name")
        sys.exit(1)


def validate_config_file(config_path: str) -> bool:
    """
    설정 파일 존재 여부 및 유효성 검증
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        bool: 설정 파일 유효성
    """
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    try:
        # 설정 파일 읽기 테스트
        settings = get_settings(config_path)
        if not settings:
            print(f"❌ 설정 파일이 올바르지 않습니다: {config_path}")
            return False
        return True
    except Exception as e:
        print(f"❌ 설정 파일 로드 오류: {str(e)}")
        return False


def print_mcp_setup_guide():
    """
    Figma MCP 서버 설정 가이드 출력
    """
    print("\n" + "="*60)
    print("🎨 FIGMA MCP 서버 설정 가이드")
    print("="*60)
    print("현재 Product Planner Agent는 다음 MCP 서버들을 지원합니다:")
    print()
    
    print("📋 1. **공식 Figma Dev Mode MCP** (권장)")
    print("   - 베타 버전 (2025년 6월 출시)")
    print("   - Figma 데스크톱 앱 필요")
    print("   - 설정: Figma > Preferences > Enable Dev Mode MCP Server")
    print("   - URL: http://127.0.0.1:3845/sse")
    print()
    
    print("🎯 2. **Talk to Figma MCP** (디자인 생성 가능)")
    print("   - WebSocket 기반 양방향 통신")
    print("   - Figma에서 직접 그림 그리기 가능")
    print("   - GitHub: https://github.com/yhc984/cursor-talk-to-figma-mcp")
    print()
    
    print("🌉 3. **Framelink Figma MCP** (커뮤니티)")
    print("   - 8.2k+ 스타의 인기 도구")
    print("   - 구조화된 디자인 데이터 제공")
    print("   - 설정: npx figma-developer-mcp --figma-api-key=YOUR-KEY")
    print()
    
    print("🔧 **설정 예시 (Cursor)**:")
    print("""
{
  "mcpServers": {
    "Figma Dev Mode MCP": {
      "type": "sse",
      "url": "http://127.0.0.1:3845/sse"
    },
    "Talk to Figma MCP": {
      "command": "bun",
      "args": ["/path/to/cursor-talk-to-figma-mcp/src/talk_to_figma_mcp/server.ts"]
    },
    "Framelink Figma MCP": {
      "command": "npx",
      "args": ["-y", "figma-developer-mcp", "--figma-api-key=YOUR-KEY", "--stdio"]
    }
  }
}
    """)
    print("="*60)
    print("💡 Figma API 토큰이 필요합니다: https://www.figma.com/settings/tokens")
    print("="*60)


async def validate_and_report_results(output_path: str, logger, workflow_result: str) -> bool:
    """
    결과 파일 검증 및 품질 체크
    
    Args:
        output_path: 출력 파일 경로
        logger: 로거 인스턴스
        workflow_result: 워크플로우 실행 결과
        
    Returns:
        bool: 검증 성공 여부
    """
    from .config.agent_config import AgentFactory
    from .agents.prd_writer_agent import PRDWriterAgent
    
    validation_results = []
    
    # 1. 파일 존재 여부 확인
    if os.path.exists(output_path):
        validation_results.append("✅ PRD 파일 생성됨")
        logger.info(f"PRD file created: {output_path}")
        
        # 2. 파일 크기 확인
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # 최소 1KB 이상
            validation_results.append(f"✅ 파일 크기 적절함 ({file_size} bytes)")
        else:
            validation_results.append(f"⚠️  파일 크기 작음 ({file_size} bytes)")
        
        # 3. 파일 내용 품질 체크
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 기본 섹션 존재 여부 확인
            required_sections = PRDWriterAgent.get_required_sections()
            
            missing_sections = []
            for section in required_sections:
                if section.lower() not in content.lower():
                    missing_sections.append(section)
            
            if not missing_sections:
                validation_results.append("✅ 모든 필수 섹션 포함됨")
            else:
                validation_results.append(f"⚠️  누락된 섹션: {', '.join(missing_sections)}")
            
            # 내용 길이 체크
            if len(content.split()) > 500:  # 최소 500단어 이상
                validation_results.append("✅ 충분한 내용 분량")
            else:
                validation_results.append("⚠️  내용이 부족할 수 있음")
                
        except Exception as e:
            validation_results.append(f"❌ 파일 내용 분석 실패: {str(e)}")
            
    else:
        validation_results.append("❌ PRD 파일 생성 실패")
        logger.error(f"Failed to create PRD at {output_path}")
    
    # 4. 워크플로우 결과 분석
    if workflow_result:
        validation_results.append("✅ 워크플로우 정상 완료")
    else:
        validation_results.append("⚠️  워크플로우 부분 완료")
    
    # 결과 출력
    print("\n" + "="*50)
    print("📋 PRD 생성 결과 검증")
    print("="*50)
    for result in validation_results:
        print(result)
    
    # 성공 기준: 파일 존재 + 기본 품질 충족
    success_count = sum(1 for result in validation_results if result.startswith("✅"))
    total_checks = len(validation_results)
    
    if success_count >= total_checks * 0.6:  # 60% 이상 성공
        print(f"\n🎉 PRD 생성 성공! ({success_count}/{total_checks} 검증 통과)")
        print(f"📄 결과 파일: {output_path}")
        return True
    else:
        print(f"\n⚠️  PRD 생성 부분 성공 ({success_count}/{total_checks} 검증 통과)")
        print("💡 결과를 확인하고 필요시 재실행해주세요.")
        return False


async def main():
    """구조화된 Product Planner Agent 메인 함수"""
    
    print("🚀 Product Planner Agent v2.0")
    print("=" * 50)
    
    # 1. 입력 검증 및 설정
    figma_url = get_figma_url()
    config_path = "configs/mcp_agent.config.yaml"
    
    if not validate_config_file(config_path):
        print("💡 설정 파일을 확인하고 다시 시도해주세요.")
        print_mcp_setup_guide()
        sys.exit(1)
    
    # 2. Agent 설정 및 팩토리 초기화
    agent_config = AgentConfig(figma_url)
    # AgentFactory는 나중에 orchestrator와 함께 초기화됩니다
    
    # 3. MCP App 초기화
    try:
        app = MCPApp(
            name="product_planner_v2",
            settings=get_settings(config_path),
            human_input_callback=None
        )
    except Exception as e:
        print(f"❌ MCP App 초기화 실패: {str(e)}")
        print("💡 설정 파일과 의존성을 확인해주세요.")
        sys.exit(1)
    
    # 4. 워크플로우 실행
    success = False
    try:
        async with app.run() as planner_app:
            context = planner_app.context
            logger = planner_app.logger
            
            print("✅ MCP App 초기화 완료")
            
            # MCP 서버 설정 확인
            if "filesystem" in context.config.mcp.servers:
                context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
                logger.info("Filesystem server configured")
                print("✅ Filesystem 서버 설정 완료")
            else:
                logger.warning("Filesystem server not configured")
                print("⚠️  Filesystem 서버 미설정")
            
            # Orchestrator 초기화 (Agent들보다 먼저)
            from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
            from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
            
            orchestrator = Orchestrator(
                llm_factory=OpenAIAugmentedLLM,
                available_agents=[],  # 초기에는 빈 리스트
                plan_type="full"
            )
            
            # Agent Factory 초기화 (Orchestrator와 함께)
            agent_factory = AgentFactory(agent_config, orchestrator)
            
            # ReAct 패턴 Agent들 생성
            react_agents = agent_factory.create_react_agents_dict()

            # Orchestrator에 에이전트 등록 (리스트 대신 딕셔너리로)
            orchestrator.available_agents = react_agents
            
            # CoordinatorAgent 가져오기
            coordinator = react_agents["coordinator_agent"]
            
            # 워크플로우 정보 출력
            print("\n" + "="*80)
            print("🚀 REACT 패턴 MULTI-AGENT PRODUCT PLANNING SYSTEM v2.1")
            print("="*80)
            print(f"📋 분석 대상: {agent_config.figma_url}")
            print(f"📁 출력 디렉토리: {agent_config.output_dir}")
            print(f"📄 결과 파일: {agent_config.output_path}")
            print(f"⏰ 타임스탬프: {agent_config.timestamp}")
            print("\n🔄 ReAct 패턴 적용:")
            print("   🎯 CoordinatorAgent - THOUGHT → ACTION → OBSERVATION")
            print("   🎨 FigmaAnalyzerAgent - THOUGHT → ACTION → OBSERVATION")
            print("   📋 PRDWriterAgent - THOUGHT → ACTION → OBSERVATION")
            print("="*80)
            
            # ReAct 워크플로우 실행
            print("\n🚀 ReAct 워크플로우 실행 시작...")
            task = f"""
            Please analyze the Figma design at {agent_config.figma_url} and create a comprehensive product plan.
            
            The output should be saved to: {agent_config.output_path}
            
            Use the ReAct pattern (THOUGHT → ACTION → OBSERVATION) to systematically:
            1. Analyze the Figma design
            2. Create detailed product requirements
            3. Generate comprehensive business planning documents
            """
            
            try:
                result = await coordinator.run(task)
                
                # 결과 검증
                success = await validate_and_report_results(
                    agent_config.output_path, logger, result
                )
                
            except Exception as workflow_error:
                print(f"❌ 워크플로우 실행 중 오류: {workflow_error}")
                logger.error(f"Workflow execution error: {workflow_error}")
                success = False
            
            # MCP App이 종료될 때 자동으로 정리됩니다
            logger.info("Workflow completed successfully")
            
    except Exception as e:
        import traceback
        print(f"❌ MCP App 실행 실패: {e}")
        print("--- TRACEBACK ---")
        traceback.print_exc()
        print("-----------------")
        print("💡 설정과 환경을 확인해주세요.")
        success = False
    
    return success


if __name__ == "__main__":
    print("🎯 구조화된 Product Planner Agent 시작")
    
    # mcp-agent 라이브러리의 알려진 asyncio 정리 버그로 인한 경고 메시지를 무시합니다.
    # 이 문제는 우리 코드의 기능에 영향을 주지 않습니다.
    # 출처: https://github.com/lastmile-ai/mcp-agent/issues/35
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine is being awaited but was never created*")

    try:
        success = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        success = False
    except Exception as e:
        print(f"\n❌ 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n🎉 제품 기획 워크플로우 완료!")
        print("📋 분석, 문서화, 디자인 생성이 모두 완료되었습니다.")
    else:
        print("\n💥 워크플로우 실행 실패!")
        print("💡 로그를 확인하고 다시 시도해주세요.")
    
    # 워크플로우의 실제 성공 여부에 따라 종료 코드를 반환합니다.
    sys.exit(0 if success else 1) 