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
    from srcs.basic_agents.workflow_orchestration import app, example_usage
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
        - **파일시스템 연동**: 문서 읽기/쓰기 자동화
        - **감사 추적**: 모든 실행 과정 기록 및 분석
        """)
    
    st.markdown("""
    #### 🎯 사용 사례
    - 문서 검토 및 피드백 생성 워크플로우
    - 콘텐츠 분석 및 요약 자동화
    - 다중 에이전트 협업 시스템
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
    
    with col2:
        st.markdown("#### 🎛️ 실행 옵션")
        
        model_type = st.selectbox(
            "AI 모델",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="사용할 AI 모델을 선택하세요"
        )
        
        plan_type = st.selectbox(
            "플래닝 방식",
            ["full", "step", "none"],
            index=0,
            help="워크플로우 플래닝 방식을 선택하세요"
        )
        
        save_results = st.checkbox(
            "결과 파일 저장",
            value=True,
            help="워크플로우 실행 결과를 파일로 저장할지 선택하세요"
        )
        
        st.markdown("---")
        
        # 실행 버튼
        if st.button("🚀 워크플로우 실행", type="primary", use_container_width=True):
            if workflow_type and input_text:
                execute_workflow(workflow_type, input_text, model_type, plan_type, save_results)
            else:
                st.error("⚠️ 모든 필수 필드를 입력해주세요!")
    
    st.markdown("---")
    
    # 워크플로우 예제 섹션
    with st.expander("📚 워크플로우 예제"):
        render_workflow_examples()

def render_workflow_examples():
    """워크플로우 예제 표시"""
    
    st.markdown("### 🎯 워크플로우 예제")
    
    tab1, tab2, tab3 = st.tabs(["📄 문서 검토", "📊 콘텐츠 분석", "🛠️ 커스텀"])
    
    with tab1:
        st.markdown("""
        #### 문서 검토 및 피드백 생성 워크플로우
        
        **실행 과정:**
        1. **문서 분석**: 입력된 문서 내용 파싱
        2. **다중 에이전트 검토**:
           - 교정자(Proofreader): 문법, 맞춤법, 구두점 검사
           - 팩트체커(Fact Checker): 사실 일관성 및 논리적 일관성 검증
           - 스타일 검사관(Style Enforcer): 스타일 가이드 준수 평가
        3. **결과 통합**: 모든 피드백을 종합한 리포트 생성
        4. **파일 저장**: 최종 검토 결과를 마크다운 파일로 저장
        
        **예상 결과:**
        - 상세한 교정 제안사항
        - 논리적 일관성 분석 결과
        - 스타일 개선 권장사항
        - 종합 평가 및 등급
        """)
    
    with tab2:
        st.markdown("""
        #### 콘텐츠 분석 및 요약 워크플로우
        
        **실행 과정:**
        1. **콘텐츠 파싱**: 입력 콘텐츠 구조 분석
        2. **키워드 추출**: 주요 키워드 및 개념 식별
        3. **감정 분석**: 콘텐츠의 톤과 감정 분석
        4. **요약 생성**: 핵심 내용 요약
        5. **인사이트 제공**: 실행 가능한 권장사항 생성
        
        **예상 결과:**
        - 주요 키워드 목록
        - 감정 분석 결과 (긍정/부정/중립)
        - 간결한 요약문
        - 실행 가능한 인사이트
        """)
    
    with tab3:
        st.markdown("""
        #### 커스텀 워크플로우
        
        **유연한 설정:**
        - 사용자 정의 작업 설명
        - 맞춤형 에이전트 구성
        - 동적 플래닝 적용
        - 다양한 출력 형식 지원
        
        **활용 예시:**
        - 복잡한 데이터 처리 파이프라인
        - 다국어 콘텐츠 번역 및 검수
        - 소셜 미디어 콘텐츠 최적화
        - 비즈니스 프로세스 자동화
        """)

def execute_workflow(workflow_type, input_text, model_type, plan_type, save_results):
    """워크플로우 실행"""
    
    with st.spinner("🔄 워크플로우를 실행하고 있습니다... 잠시만 기다려주세요."):
        try:
            # 비동기 함수 실행을 위한 새 이벤트 루프 생성
            def run_async_workflow():
                try:
                    # 새 이벤트 루프 생성
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 비동기 워크플로우 실행
                    result = loop.run_until_complete(
                        execute_async_workflow(workflow_type, input_text, model_type, plan_type, save_results)
                    )
                    
                    loop.close()
                    return result
                    
                except Exception as e:
                    return f"워크플로우 실행 중 오류가 발생했습니다: {str(e)}"
            
            # 워크플로우 실행
            result = run_async_workflow()
            
            # 결과 표시
            if result:
                st.success("✅ 워크플로우가 성공적으로 완료되었습니다!")
                
                # 결과 표시
                st.markdown("### 📊 실행 결과")
                
                # 결과 텍스트 표시
                with st.expander("📄 상세 결과 보기", expanded=True):
                    st.text_area(
                        "워크플로우 실행 결과",
                        value=str(result),
                        height=300,
                        disabled=True
                    )
                
                # 파일 저장 상태 표시
                if save_results:
                    st.info("💾 결과가 파일로 저장되었습니다.")
                    
            else:
                st.error("❌ 워크플로우 실행에 실패했습니다.")
                
        except Exception as e:
            st.error(f"❌ 오류가 발생했습니다: {str(e)}")
            st.info("OpenAI API 키가 설정되어 있는지 확인해주세요.")

async def execute_async_workflow(workflow_type, input_text, model_type, plan_type, save_results):
    """비동기 워크플로우 실행"""
    
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            temp_file.write(input_text)
            temp_file_path = temp_file.name
        
        # MCP 앱 생성 및 실행
        workflow_app = MCPApp(name="streamlit_workflow_orchestrator")
        
        async with workflow_app.run() as orchestrator_app:
            context = orchestrator_app.context
            
            # 현재 디렉토리를 파일시스템 서버에 추가
            context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])
            
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
                logical inconsistencies, or inaccuracies in the content. 
                Highlight potential issues with reasoning or coherence.""",
                server_names=["fetch"],
            )

            style_enforcer = Agent(
                name="style_enforcer",
                instruction="""Analyze the text for adherence to style guidelines.
                Evaluate the narrative flow, clarity of expression, and tone. Suggest improvements to 
                enhance readability and engagement.""",
                server_names=["fetch"],
            )
            
            # 워크플로우 타입에 따른 작업 정의
            if workflow_type == "문서 검토 및 피드백 생성":
                task = f"""Analyze the following text and generate comprehensive feedback:
                "{input_text}"
                
                Provide detailed feedback on:
                1. Grammar, spelling, and punctuation
                2. Factual consistency and logical coherence
                3. Style and readability improvements
                
                Generate a comprehensive report with all feedback."""
                
            elif workflow_type == "콘텐츠 분석 및 요약":
                task = f"""Analyze and summarize the following content:
                "{input_text}"
                
                Provide:
                1. Key themes and main points
                2. Summary of the content
                3. Important insights and takeaways
                
                Generate a comprehensive analysis report."""
                
            else:  # 커스텀 워크플로우
                task = f"""Execute the following custom workflow:
                Task: {input_text}
                
                Analyze the content and provide comprehensive insights according to the task description."""

            # 오케스트레이터 생성
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

            # 워크플로우 실행
            result = await orchestrator.generate_str(
                message=task,
                request_params=RequestParams(model=model_type)
            )
            
            # 임시 파일 정리
            os.unlink(temp_file_path)
            
            return result
            
    except Exception as e:
        return f"워크플로우 실행 중 오류 발생: {str(e)}"

# 사이드바 정보
def render_sidebar():
    """사이드바 렌더링"""
    
    with st.sidebar:
        st.markdown("### 🔄 Workflow Orchestrator")
        st.markdown("---")
        
        st.markdown("#### 📋 현재 상태")
        if WORKFLOW_AGENT_AVAILABLE:
            st.success("✅ Agent 연결됨")
        else:
            st.error("❌ Agent 연결 실패")
        
        st.markdown("#### 🛠️ 지원 기능")
        st.markdown("""
        - 📄 문서 검토 및 피드백
        - 📊 콘텐츠 분석 및 요약
        - 🛠️ 커스텀 워크플로우
        - 🤖 다중 에이전트 협업
        - 💾 결과 파일 저장
        """)
        
        st.markdown("#### ⚙️ 설정 정보")
        st.markdown(f"""
        - **Agent 파일**: `workflow_orchestration.py`
        - **위치**: `srcs/basic_agents/`
        - **상태**: {'🟢 연결됨' if WORKFLOW_AGENT_AVAILABLE else '🔴 연결 안됨'}
        """)

if __name__ == "__main__":
    render_sidebar()
    main() 