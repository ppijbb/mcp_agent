"""
🔄 Workflow Orchestrator Page

복잡한 워크플로우 자동화 및 다중 에이전트 협업
"""

import streamlit as st
import sys
import asyncio
import os
import tempfile
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Workflow Orchestrator 임포트 시도
try:
    from mcp_agent.app import MCPApp
    from mcp_agent.agents.agent import Agent
    from mcp_agent.workflows.llm.augmented_llm import RequestParams
    from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
    from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
    WORKFLOW_AGENT_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AGENT_AVAILABLE = False
    import_error = str(e)

def main():
    """Workflow Orchestrator 메인 페이지"""
    
    # 헤더
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    ">
        <h1>🔄 Workflow Orchestrator</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            복잡한 워크플로우 자동화 및 다중 에이전트 협업 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not WORKFLOW_AGENT_AVAILABLE:
        st.error(f"⚠️ Workflow Orchestrator를 불러올 수 없습니다: {import_error}")
        st.info("MCP Agent 시스템과 필요한 의존성을 설치해주세요.")
        
        with st.expander("🔧 설치 가이드"):
            st.markdown("""
            ### Workflow Orchestrator 설정
            
            1. **MCP Agent 패키지 설치**:
            ```bash
            pip install mcp-agent
            ```
            
            2. **필요한 패키지 설치**:
            ```bash
            pip install asyncio rich openai
            ```
            
            3. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            4. **에이전트 모듈 확인**:
            ```bash
            ls srcs/basic_agents/workflow_orchestration.py
            ```
            """)
        
        # 에이전트 소개만 제공
        render_agent_info()
        return
    else:
        st.success("🤖 Workflow Orchestrator가 성공적으로 연결되었습니다!")
        
        # 에이전트 실행 인터페이스 제공
        render_workflow_interface()

def render_agent_info():
    """에이전트 기능 소개"""
    
    st.markdown("### 🔄 Workflow Orchestrator 소개")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 주요 기능
        - **다중 에이전트 협업**: 여러 AI 에이전트 동시 운영
        - **워크플로우 자동화**: 복잡한 비즈니스 프로세스 자동화
        - **실시간 모니터링**: 작업 진행 상황 추적 및 알림
        - **동적 스케줄링**: 우선순위 기반 작업 배정
        - **오류 복구**: 자동 재시도 및 대안 경로 실행
        """)
    
    with col2:
        st.markdown("""
        #### ✨ 스페셜 기능
        - **적응형 워크플로우**: 실행 결과 기반 자동 최적화
        - **병렬 처리**: 독립적 작업의 동시 실행
        - **조건부 분기**: 상황별 다른 경로 실행
        - **리소스 관리**: 자동 부하 분산 및 리소스 할당
        - **감사 추적**: 모든 실행 과정 기록 및 분석
        """)
    
    st.markdown("""
    #### 🎯 사용 사례
    - 대규모 데이터 처리 파이프라인
    - 고객 서비스 자동화 시스템
    - 콘텐츠 생성 및 배포 워크플로우
    - 비즈니스 프로세스 최적화
    """)

def render_workflow_interface():
    """Workflow Orchestrator 인터페이스"""
    
    st.markdown("### 🚀 Workflow Orchestrator 실행")
    
    # 워크플로우 설정
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 워크플로우 설정")
        
        workflow_type = st.selectbox(
            "워크플로우 타입",
            [
                "문서 검토 및 피드백 생성",
                "콘텐츠 분석 및 요약",
                "커스텀 워크플로우"
            ],
            help="실행할 워크플로우 타입을 선택하세요"
        )
        
        if workflow_type == "문서 검토 및 피드백 생성":
            st.markdown("##### 📄 문서 검토 설정")
            
            input_text = st.text_area(
                "검토할 문서 내용",
                value="The Battle of Glimmerwood was a legendary conflict that took place in the mystical Glimmerwood forest. The battle was fought between the forces of light and darkness, with magical creatures on both sides.",
                height=150,
                help="검토하고 피드백을 받을 문서 내용을 입력하세요"
            )
            
            feedback_types = st.multiselect(
                "피드백 타입",
                ["문법 및 맞춤법", "사실 일관성", "스타일 가이드"],
                default=["문법 및 맞춤법", "사실 일관성", "스타일 가이드"],
                help="생성할 피드백의 종류를 선택하세요"
            )
            
        elif workflow_type == "콘텐츠 분석 및 요약":
            st.markdown("##### 📊 콘텐츠 분석 설정")
            
            input_text = st.text_area(
                "분석할 콘텐츠",
                value="Enter your content here for analysis and summarization...",
                height=150,
                help="분석하고 요약할 콘텐츠를 입력하세요"
            )
            
            analysis_types = st.multiselect(
                "분석 타입",
                ["주요 키워드 추출", "감정 분석", "내용 요약"],
                default=["주요 키워드 추출", "내용 요약"],
                help="수행할 분석의 종류를 선택하세요"
            )
            
        else:  # 커스텀 워크플로우
            st.markdown("##### 🛠️ 커스텀 워크플로우")
            
            task_description = st.text_area(
                "작업 설명",
                value="Analyze the provided text and generate comprehensive insights including key themes, sentiment analysis, and actionable recommendations.",
                height=100,
                help="수행할 작업을 상세히 설명하세요"
            )
            
            input_text = st.text_area(
                "입력 데이터",
                value="Enter your data or content here...",
                height=100,
                help="처리할 입력 데이터를 입력하세요"
            )
        
        # 추가 설정
        st.markdown("##### ⚙️ 실행 설정")
        
        model_type = st.selectbox(
            "LLM 모델",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="사용할 언어 모델을 선택하세요"
        )
        
        plan_type = st.selectbox(
            "플랜 타입",
            ["full", "step"],
            help="full: 전체 계획 수립 후 실행, step: 단계별 계획 수립"
        )
        
        save_results = st.checkbox(
            "결과 파일로 저장",
            value=False,
            help="실행 결과를 임시 파일로 저장합니다"
        )
        
        if st.button("🚀 워크플로우 실행", type="primary", use_container_width=True):
            if input_text.strip():
                execute_workflow(workflow_type, input_text, model_type, plan_type, save_results)
            else:
                st.error("입력 데이터를 입력해주세요.")
    
    with col2:
        if 'workflow_execution_result' in st.session_state:
            result = st.session_state['workflow_execution_result']
            
            if result['success']:
                st.success("✅ 워크플로우 실행 완료!")
                
                # 실행 정보
                st.markdown("#### 📊 실행 결과")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("실행 시간", f"{result['execution_time']:.2f}초")
                with col_r2:
                    st.metric("에이전트 수", result.get('agent_count', 'N/A'))
                with col_r3:
                    st.metric("작업 단계", result.get('step_count', 'N/A'))
                
                # 워크플로우 결과
                if 'output' in result and result['output']:
                    st.markdown("#### 📄 생성된 결과")
                    
                    output = result['output']
                    
                    # 결과가 길면 확장 가능한 형태로 표시
                    if len(output) > 1000:
                        with st.expander("📋 전체 결과 보기", expanded=True):
                            st.markdown(output)
                    else:
                        st.markdown(output)
                    
                    # 결과 다운로드 버튼
                    st.download_button(
                        label="📥 결과 다운로드",
                        data=output,
                        file_name=f"workflow_result_{result['timestamp']}.md",
                        mime="text/markdown"
                    )
                
                # 파일 저장 정보 표시
                if result.get('save_results') and result.get('saved_files'):
                    st.markdown("#### 💾 저장된 파일들")
                    
                    output_dir = result.get('output_directory', 'Unknown')
                    st.success(f"📁 **저장 위치**: `{output_dir}`")
                    
                    saved_files = result.get('saved_files', [])
                    st.info(f"💾 **저장된 파일 수**: {len(saved_files)}개")
                    
                    # 저장된 파일 목록
                    with st.expander("📂 저장된 파일 목록", expanded=False):
                        for i, file_path in enumerate(saved_files, 1):
                            file_name = Path(file_path).name
                            file_type = "📄 입력 파일" if "input" in file_name or "content_to_analyze" in file_name or "custom_input" in file_name else \
                                       "📊 실행 로그" if "execution_log" in file_name else \
                                       "📋 결과 파일"
                            st.markdown(f"{i}. {file_type}: `{file_name}`")
                            st.text(f"   전체 경로: {file_path}")
                    
                    # 디렉토리 열기 안내
                    st.markdown(f"""
                    **💡 파일 확인 방법:**
                    ```bash
                    # 디렉토리로 이동
                    cd {output_dir}
                    
                    # 파일 목록 확인
                    ls -la
                    
                    # 결과 파일 확인 (예시)
                    cat *.md
                    ```
                    """)
                
                # 상세 실행 정보
                with st.expander("🔍 상세 실행 정보"):
                    st.markdown("#### 워크플로우 상세")
                    st.json({
                        'workflow_type': result['workflow_type'],
                        'model_type': result['model_type'],
                        'plan_type': result['plan_type'],
                        'execution_time': result['execution_time'],
                        'success': result['success'],
                        'save_results': result.get('save_results', False),
                        'output_length': len(result.get('output', '')),
                        'saved_files_count': len(result.get('saved_files', [])),
                        'output_directory': result.get('output_directory', None)
                    })
                    
                    if 'error_details' in result:
                        st.markdown("#### 처리 과정 상세")
                        st.text(result['error_details'])
                
            else:
                st.error("❌ 워크플로우 실행 중 오류 발생")
                st.error(f"**오류**: {result['message']}")
                
                with st.expander("🔍 오류 상세"):
                    st.code(result.get('error', 'Unknown error'))
                    
        else:
            st.markdown("""
            #### 🤖 워크플로우 실행 정보
            
            **실행되는 프로세스:**
            1. **MCP App 초기화** - MCP 프레임워크 연결
            2. **에이전트 생성** - 전문화된 AI 에이전트들 생성
            3. **워크플로우 계획** - 동적 실행 계획 수립
            4. **다중 에이전트 협업** - 병렬 및 순차 작업 실행
            5. **결과 통합** - 최종 결과 생성 및 검증
            
            **사용되는 에이전트:**
            - 🔍 **Finder**: 데이터 검색 및 수집
            - ✍️ **Writer**: 콘텐츠 생성 및 파일 작성
            - 📝 **Proofreader**: 문법 및 맞춤법 검토
            - 🔍 **Fact Checker**: 사실 확인 및 일관성 검증
            - 🎨 **Style Enforcer**: 스타일 가이드 준수 검토
            
            **특징:**
            - **동적 계획**: 작업에 따라 자동으로 최적 계획 수립
            - **병렬 처리**: 독립적 작업 동시 실행
            - **오류 복구**: 자동 재시도 및 대안 경로
            - **실시간 모니터링**: 진행 상황 추적
            """)

def execute_workflow(workflow_type, input_text, model_type, plan_type, save_results):
    """워크플로우 실행"""
    
    try:
        with st.spinner("🔄 워크플로우를 실행하는 중..."):
            import time
            
            # 비동기 함수 실행을 위한 래퍼
            def run_async_workflow():
                return asyncio.run(execute_async_workflow(workflow_type, input_text, model_type, plan_type, save_results))
            
            start_time = time.time()
            workflow_result = run_async_workflow()
            execution_time = time.time() - start_time
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # workflow_result에서 데이터 추출
            if isinstance(workflow_result, dict):
                output_text = workflow_result.get('result', '')
                saved_files = workflow_result.get('saved_files', [])
                output_directory = workflow_result.get('output_directory', None)
            else:
                # 이전 버전 호환성을 위해
                output_text = str(workflow_result)
                saved_files = []
                output_directory = None
            
            st.session_state['workflow_execution_result'] = {
                'success': True,
                'workflow_type': workflow_type,
                'model_type': model_type,
                'plan_type': plan_type,
                'execution_time': execution_time,
                'output': output_text,
                'agent_count': 5,  # finder, writer, proofreader, fact_checker, style_enforcer
                'step_count': '다단계',
                'save_results': save_results,
                'saved_files': saved_files,
                'output_directory': output_directory,
                'timestamp': timestamp
            }
            st.rerun()
            
    except Exception as e:
        st.session_state['workflow_execution_result'] = {
            'success': False,
            'message': f'워크플로우 실행 중 오류 발생: {str(e)}',
            'error': str(e),
            'workflow_type': workflow_type
        }
        st.rerun()

async def execute_async_workflow(workflow_type, input_text, model_type, plan_type, save_results):
    """비동기 워크플로우 실행"""
    
    app = MCPApp(name="streamlit_workflow_orchestrator")
    
    async with app.run() as orchestrator_app:
        context = orchestrator_app.context
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        context.config.mcp.servers["filesystem"].args.extend([temp_dir])
        
        # 결과 저장을 위한 영구 디렉토리 설정
        permanent_output_dir = None
        if save_results:
            # workflow_results 디렉토리 생성
            permanent_output_dir = Path("workflow_results")
            permanent_output_dir.mkdir(exist_ok=True)
            
            # 타임스탬프별 하위 디렉토리 생성
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            permanent_output_dir = permanent_output_dir / f"workflow_{timestamp}"
            permanent_output_dir.mkdir(exist_ok=True)
        
        # 에이전트 생성
        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        writer_agent = Agent(
            name="writer",
            instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
            server_names=["filesystem"],
        )

        proofreader = Agent(
            name="proofreader",
            instruction="""Review the text for grammar, spelling, and punctuation errors.
            Identify any awkward phrasing or structural issues that could improve clarity. 
            Provide detailed feedback on corrections.""",
            server_names=["fetch"],
        )

        fact_checker = Agent(
            name="fact_checker",
            instruction="""Verify the factual consistency within the text. Identify any contradictions,
            logical inconsistencies, or inaccuracies. Highlight potential issues with reasoning or coherence.""",
            server_names=["fetch"],
        )

        style_enforcer = Agent(
            name="style_enforcer",
            instruction="""Analyze the text for adherence to style guidelines.
            Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
            enhance readability and engagement.""",
            server_names=["fetch"],
        )
        
        # 워크플로우 타입에 따른 작업 설정
        output_filename = ""
        if workflow_type == "문서 검토 및 피드백 생성":
            # 임시 파일에 입력 텍스트 저장
            input_file = os.path.join(temp_dir, "input_document.md")
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_text)
            
            output_filename = "feedback_report.md"
            task = f"""Load the document from input_document.md in {temp_dir}, 
            and generate a comprehensive feedback report covering proofreading, 
            factual consistency, and style adherence. 
            Write the feedback report to {output_filename} in the same directory."""
            
        elif workflow_type == "콘텐츠 분석 및 요약":
            # 임시 파일에 입력 텍스트 저장
            input_file = os.path.join(temp_dir, "content_to_analyze.md")
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_text)
            
            output_filename = "analysis_report.md"
            task = f"""Load the content from content_to_analyze.md in {temp_dir}, 
            analyze it for key themes, sentiment, and important insights, 
            then create a comprehensive summary report. 
            Write the analysis report to {output_filename} in the same directory."""
            
        else:  # 커스텀 워크플로우
            # 임시 파일에 입력 텍스트 저장
            input_file = os.path.join(temp_dir, "custom_input.md")
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(input_text)
            
            output_filename = "custom_output.md"
            task = f"""Load the data from custom_input.md in {temp_dir}, 
            process it according to the requirements, and generate appropriate output. 
            Write the results to {output_filename} in the same directory."""

        # 오케스트레이터 생성 및 실행
        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                finder_agent,
                writer_agent,
                proofreader,
                fact_checker,
                style_enforcer,
            ],
            plan_type=plan_type,
        )

        # 작업 실행
        result = await orchestrator.generate_str(
            message=task, 
            request_params=RequestParams(model=model_type)
        )
        
        # 결과 파일이 생성되었는지 확인하고 읽기
        output_files = []
        saved_files_info = []
        
        for file_name in ["feedback_report.md", "analysis_report.md", "custom_output.md"]:
            output_path = os.path.join(temp_dir, file_name)
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    output_files.append(f"## {file_name}\n\n{file_content}")
                
                # save_results가 True인 경우 영구 디렉토리에 파일 저장
                if save_results and permanent_output_dir:
                    # 입력 파일도 함께 저장
                    if file_name == output_filename:
                        # 입력 파일 저장
                        input_filename = "input_document.md" if workflow_type == "문서 검토 및 피드백 생성" else \
                                        "content_to_analyze.md" if workflow_type == "콘텐츠 분석 및 요약" else \
                                        "custom_input.md"
                        input_path = os.path.join(temp_dir, input_filename)
                        if os.path.exists(input_path):
                            permanent_input_path = permanent_output_dir / input_filename
                            import shutil
                            shutil.copy2(input_path, permanent_input_path)
                            saved_files_info.append(str(permanent_input_path))
                    
                    # 출력 파일 저장
                    permanent_file_path = permanent_output_dir / file_name
                    import shutil
                    shutil.copy2(output_path, permanent_file_path)
                    saved_files_info.append(str(permanent_file_path))
                    
                    # 실행 로그도 저장
                    log_file_path = permanent_output_dir / "execution_log.md"
                    with open(log_file_path, 'w', encoding='utf-8') as log_file:
                        log_content = f"""# Workflow Execution Log

## 실행 정보
- **워크플로우 타입**: {workflow_type}
- **모델**: {model_type}
- **플랜 타입**: {plan_type}
- **실행 시간**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 오케스트레이터 결과
{result}

## 생성된 파일들
{chr(10).join([f"- {file}" for file in saved_files_info])}
"""
                        log_file.write(log_content)
                    saved_files_info.append(str(log_file_path))
        
        # 최종 결과 조합
        final_result = result
        if output_files:
            final_result += "\n\n---\n\n## 생성된 파일들\n\n" + "\n\n".join(output_files)
        
        # 파일 저장 정보 추가
        if save_results and saved_files_info:
            final_result += f"""

---

## 💾 저장된 파일들

다음 파일들이 `{permanent_output_dir}` 디렉토리에 저장되었습니다:

{chr(10).join([f"- `{file}`" for file in saved_files_info])}

**저장된 파일 구성:**
- 입력 파일: 원본 입력 데이터
- 출력 파일: 워크플로우 실행 결과  
- 실행 로그: 전체 실행 과정 및 결과 요약
"""
        
        # 임시 파일 정리
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 반환 데이터에 저장 정보 포함
        return_data = {
            'result': final_result,
            'saved_files': saved_files_info if save_results else [],
            'output_directory': str(permanent_output_dir) if save_results else None
        }
        
        return return_data

if __name__ == "__main__":
    main() 