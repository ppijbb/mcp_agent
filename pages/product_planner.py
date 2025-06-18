"""
🚀 Product Planner Agent Test Page

실제 Product Planner Agent를 테스트할 수 있는 인터페이스
"""

import streamlit as st
import sys
from pathlib import Path
import asyncio
import os
import json
from datetime import datetime
import traceback

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# Product Planner Agent 임포트
try:
    from srcs.product_planner_agent.agents.coordinator_agent import CoordinatorAgent
    from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
    from srcs.product_planner_agent.utils.status_logger import StatusLogger
except ImportError as e:
    st.error(f"❌ Product Planner Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: Product Planner Agent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

# 페이지 설정
setup_page("🚀 Product Planner Agent Test", "🚀")

def parse_figma_url(url: str) -> tuple[str | None, str | None]:
    """Figma URL에서 file_id와 node_id를 추출"""
    import re
    from urllib.parse import unquote
    
    # file_id: /file/ 다음에 오는 문자열
    file_id_match = re.search(r'figma\.com/file/([^/]+)', url)
    file_id = file_id_match.group(1) if file_id_match else None
    
    # node-id: 쿼리 파라미터에서 추출
    node_id_match = re.search(r'node-id=([^&]+)', url)
    node_id = unquote(node_id_match.group(1)) if node_id_match else None
    
    return file_id, node_id

async def run_product_planner_agent(figma_api_key: str, figma_file_id: str, figma_node_id: str, task_description: str):
    """Product Planner Agent 실행"""
    try:
        # Orchestrator 및 LLM 팩토리 초기화
        orchestrator = Orchestrator(llm_factory=OpenAIAugmentedLLM)
        
        # CoordinatorAgent 초기화
        coordinator = CoordinatorAgent(orchestrator=orchestrator)
        
        # ReAct 패턴으로 작업 실행
        task = f"""
        Product Planning Task:
        - Figma File ID: {figma_file_id}
        - Figma Node ID: {figma_node_id}
        - Task Description: {task_description}
        - API Key Available: Yes
        
        Please analyze the Figma design, create a comprehensive PRD, and develop a business plan.
        """
        
        result = await coordinator.run_react(task)
        return result, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg

def run_sync_wrapper(coro):
    """비동기 함수를 동기적으로 실행하는 래퍼"""
    try:
        # 기존 이벤트 루프가 있는지 확인
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def main():
    """Product Planner Agent 테스트 페이지"""
    
    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    
    # 헤더 렌더링
    header_html = get_page_header("product", "🚀 Product Planner Agent Test", 
                                 "실제 Product Planner Agent 테스트 및 실행 인터페이스")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    render_home_button()
    
    st.markdown("---")
    
    # 입력 섹션
    st.markdown("### 📋 테스트 설정")
    
    # API Key 입력
    figma_api_key = st.text_input(
        "🔑 Figma API Key",
        type="password",
        help="Figma API Key를 입력하세요. 환경변수 FIGMA_API_KEY가 설정되어 있으면 자동으로 사용됩니다."
    )
    
    # 환경변수에서 API Key 가져오기
    if not figma_api_key:
        figma_api_key = os.getenv("FIGMA_API_KEY")
        if figma_api_key:
            st.success("✅ 환경변수에서 Figma API Key를 가져왔습니다.")
    
    # Figma URL 입력
    figma_url = st.text_input(
        "🎨 Figma URL",
        placeholder="https://www.figma.com/file/FILE_ID/File-Name?node-id=NODE_ID",
        help="분석할 Figma 디자인의 URL을 입력하세요."
    )
    
    # 작업 설명 입력
    task_description = st.text_area(
        "📝 작업 설명",
        placeholder="예: 모바일 앱의 로그인 화면을 분석하고 PRD를 작성해주세요.",
        help="Product Planner Agent가 수행할 작업에 대한 설명을 입력하세요."
    )
    
    # 테스트 모드 선택
    test_mode = st.selectbox(
        "🧪 테스트 모드",
        ["ReAct Pattern (권장)", "Static Workflow", "Agent Method Test"],
        help="테스트할 실행 모드를 선택하세요."
    )
    
    st.markdown("---")
    
    # 실행 버튼
    if st.button("🚀 Product Planner Agent 실행", type="primary"):
        
        # 입력 검증
        if not figma_api_key:
            st.error("❌ Figma API Key가 필요합니다.")
            return
            
        if not figma_url:
            st.error("❌ Figma URL이 필요합니다.")
            return
            
        if not task_description:
            st.error("❌ 작업 설명이 필요합니다.")
            return
        
        # Figma URL 파싱
        file_id, node_id = parse_figma_url(figma_url)
        
        if not file_id or not node_id:
            st.error("❌ 유효하지 않은 Figma URL입니다. file_id와 node-id가 모두 포함되어 있는지 확인하세요.")
            return
        
        # 실행 정보 표시
        with st.expander("📊 실행 정보", expanded=True):
            st.write(f"**Figma File ID**: {file_id}")
            st.write(f"**Figma Node ID**: {node_id}")
            st.write(f"**테스트 모드**: {test_mode}")
            st.write(f"**작업 설명**: {task_description}")
        
        # 실행 시작
        st.markdown("### 🔄 실행 중...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 비동기 실행
            with st.spinner("Product Planner Agent 실행 중..."):
                status_text.text("초기화 중...")
                progress_bar.progress(10)
                
                # 실행
                if test_mode == "ReAct Pattern (권장)":
                    result, error = run_sync_wrapper(run_product_planner_agent(
                        figma_api_key, file_id, node_id, task_description
                    ))
                elif test_mode == "Static Workflow":
                    # Static workflow 실행
                    async def run_static():
                        orchestrator = Orchestrator(llm_factory=OpenAIAugmentedLLM)
                        coordinator = CoordinatorAgent(orchestrator=orchestrator)
                        return await coordinator.run_static_workflow(figma_api_key, file_id, node_id)
                    
                    result = run_sync_wrapper(run_static())
                    error = None
                else:
                    # Agent Method Test
                    st.info("🧪 Agent Method Test 모드는 개발 중입니다.")
                    result = "Agent Method Test - 개발 중"
                    error = None
                
                progress_bar.progress(100)
                status_text.text("완료!")
                
                # 결과 표시
                st.markdown("### 📊 실행 결과")
                
                if error:
                    st.error("❌ 실행 중 오류 발생:")
                    st.code(error, language="python")
                else:
                    st.success("✅ 실행 완료!")
                    
                    # 결과 표시
                    if result:
                        if isinstance(result, str):
                            try:
                                # JSON 형태인지 확인
                                parsed_result = json.loads(result)
                                st.json(parsed_result)
                            except json.JSONDecodeError:
                                st.text_area("결과", result, height=300)
                        else:
                            st.json(result)
                    else:
                        st.warning("⚠️ 결과가 없습니다.")
                
        except Exception as e:
            progress_bar.progress(100)
            status_text.text("오류 발생!")
            st.error(f"❌ 실행 중 오류 발생: {str(e)}")
            st.code(traceback.format_exc(), language="python")
    
    # 디버깅 정보
    with st.expander("🔧 디버깅 정보"):
        st.write("**환경 변수**:")
        st.write(f"- FIGMA_API_KEY: {'설정됨' if os.getenv('FIGMA_API_KEY') else '설정되지 않음'}")
        
        st.write("**시스템 정보**:")
        st.write(f"- Python Path: {sys.path[:3]}...")
        st.write(f"- 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Agent 상태 확인
        try:
            orchestrator = Orchestrator(llm_factory=OpenAIAugmentedLLM)
            coordinator = CoordinatorAgent(orchestrator=orchestrator)
            st.write(f"- 사용 가능한 Agent: {coordinator.available_agents}")
        except Exception as e:
            st.write(f"- Agent 초기화 오류: {str(e)}")

if __name__ == "__main__":
    main() 