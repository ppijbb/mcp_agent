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
import json
from datetime import datetime
import streamlit_process_manager as spm
from srcs.common.standard_a2a_page_helper import execute_standard_agent_via_a2a
from srcs.common.agent_interface import AgentType

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 임포트
from configs.settings import get_reports_path

# Workflow Orchestrator 임포트 시도
try:
    # We only need the app for some info, not execution
    from srcs.basic_agents.workflow_orchestration import app
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
        render_agent_info()
        return
    else:
        st.success("🤖 Workflow Orchestrator가 성공적으로 연결되었습니다!")
        render_workflow_interface()
        render_info_panels()

def render_results(result_data: dict):
    """실행 결과를 메인 패널에 표시합니다."""
    st.markdown("### 📊 실행 결과")
    
    if result_data.get('success'):
        st.success("✅ 워크플로우가 성공적으로 완료되었습니다!")
        with st.expander("📄 상세 결과 보기", expanded=True):
            st.text_area(
                "워크플로우 실행 결과",
                value=str(result_data.get('result', '내용 없음')),
                height=400,
                disabled=True
            )
    else:
        st.error(f"❌ 워크플로우 실행 중 오류가 발생했습니다: {result_data.get('error', '알 수 없는 오류')}")

def render_workflow_interface():
    """Workflow Orchestrator 인터페이스를 메인 화면에 렌더링합니다."""
    st.markdown("### 🚀 Workflow Orchestrator 실행")

    with st.form(key="workflow_form"):
        st.markdown("#### ⚙️ 워크플로우 설정")
        workflow_type = st.selectbox(
            "워크플로우 타입",
            ["문서 검토 및 피드백 생성", "콘텐츠 분석 및 요약", "커스텀 워크플로우"],
            help="실행할 워크플로우 타입을 선택하세요"
        )

        input_text = ""
        task_description = ""

        if workflow_type == "문서 검토 및 피드백 생성":
            input_text = st.text_area(
                "검토할 문서 내용",
                "The Battle of Glimmerwood was a legendary conflict...",
                height=150,
                key="doc_review_text"
            )
            task_description = "Review the provided document and generate feedback."
        elif workflow_type == "콘텐츠 분석 및 요약":
            input_text = st.text_area(
                "분석할 콘텐츠", "Enter your content here...", height=150, key="content_analysis_text"
            )
            task_description = "Analyze and summarize the provided content."
        else:  # 커스텀 워크플로우
            task_description = st.text_area(
                "작업 설명", "Analyze the provided text...", height=100, key="custom_task_desc"
            )
            input_text = st.text_area(
                "입력 데이터", "Enter your data here...", height=100, key="custom_input_data"
            )

        st.markdown("#### 🎛️ 실행 옵션")
        model_name = st.selectbox("실행할 모델:", ["gemini-2.5-flash-lite"], key="model_name")
        plan_type = st.selectbox("플래닝 방식:", ["full", "step", "none"], key="plan_type")

        submitted = st.form_submit_button("🚀 워크플로우 실행", type="primary", width='stretch')

    if submitted:
        final_task = ""
        if workflow_type == "커스텀 워크플로우":
            if task_description and input_text:
                final_task = f"Task: {task_description}\n\nData: {input_text}"
            else:
                st.error("⚠️ 커스텀 워크플로우의 작업 설명과 입력 데이터를 모두 입력해주세요!")
                st.stop()
        else:
            if not input_text:
                st.error("⚠️ 입력 내용을 채워주세요!")
                st.stop()
            final_task = f"Task: {task_description}\n\nData: {input_text}"

        execute_workflow_process(final_task, model_name, plan_type)

    st.markdown("---")
    with st.expander("📚 워크플로우 예제 보기"):
        render_workflow_examples()

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

def execute_workflow_process(task: str, model_name: str, plan_type: str):
    """워크플로우를 별도 프로세스로 실행하고 결과를 표시합니다."""
    
    reports_path = Path(get_reports_path('workflow'))
    reports_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json_path = reports_path / f"workflow_result_{timestamp}.json"
    
                # 표준화된 방식으로 agent 실행
            result = execute_standard_agent_via_a2a(
                placeholder=result_placeholder,
                
        "agent_id": "workflow_orchestrator_agent",
        "agent_name": "Workflow Orchestrator Agent",
        "entry_point": "srcs.basic_agents.run_workflow_agent",
        agent_type=AgentType.MCP_AGENT,
        "capabilities": ["workflow_automation", "multi_agent_coordination", "task_orchestration"],
        "description": "복잡한 워크플로우 자동화 및 다중 에이전트 협업 시스템"
    ,
                input_params=input_data,
                result_json_path=result_json_path,
                use_a2a=True
            )
    
    if result:
        render_results(result)

def render_info_panels():
    """정보 패널들을 메인 화면에 렌더링합니다."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📋 현재 상태")
        if WORKFLOW_AGENT_AVAILABLE:
            st.success("✅ Agent 연결됨")
        else:
            st.error("❌ Agent 연결 실패")

    with col2:
        st.markdown("#### 🎯 사용 사례")
        st.info("""
        - 문서 검토 및 피드백
        - 콘텐츠 분석 및 요약
        - 커스텀 워크플로우
        """)

    with col3:
        st.markdown("#### ⚙️ 설정 정보")
        st.markdown(f"""
        - **Agent 파일**: `workflow_orchestration.py`
        - **위치**: `srcs/basic_agents/`
        - **상태**: {'🟢 연결됨' if WORKFLOW_AGENT_AVAILABLE else '🔴 연결 안됨'}
        """)

if __name__ == "__main__":
    main()

# 최신 Workflow Orchestrator 결과 확인
st.markdown("---")
st.markdown("## 📊 최신 Workflow Orchestrator 결과")

latest_workflow_result = result_reader.get_latest_result("workflow_orchestrator", "workflow_execution")

if latest_workflow_result:
    with st.expander("🔄 최신 워크플로우 실행 결과", expanded=False):
        st.subheader("🤖 최근 워크플로우 실행 결과")
        
        if isinstance(latest_workflow_result, dict):
            # 워크플로우 정보 표시
            task = latest_workflow_result.get('task', 'N/A')
            plan_type = latest_workflow_result.get('plan_type', 'N/A')
            model_name = latest_workflow_result.get('model_name', 'N/A')
            
            st.success(f"**작업: {task}**")
            st.info(f"**플랜 타입: {plan_type}** | **모델: {model_name}**")
            
            # 실행 결과 요약
            col1, col2, col3 = st.columns(3)
            col1.metric("실행 상태", "완료" if latest_workflow_result.get('success', False) else "실패")
            col2.metric("실행 시간", f"{latest_workflow_result.get('execution_time', 0):.1f}초")
            col3.metric("단계 수", len(latest_workflow_result.get('steps', [])))
            
            # 실행 결과 표시
            result = latest_workflow_result.get('result', '')
            if result:
                st.subheader("📄 실행 결과")
                with st.expander("상세 결과", expanded=False):
                    st.text_area("결과 내용", result, height=200, disabled=True)
            
            # 실행 단계 표시
            steps = latest_workflow_result.get('steps', [])
            if steps:
                st.subheader("📋 실행 단계")
                with st.expander("단계별 실행 내역", expanded=False):
                    for i, step in enumerate(steps, 1):
                        st.write(f"{i}. {step}")
            
            # 메타데이터 표시
            if 'timestamp' in latest_workflow_result:
                st.caption(f"⏰ 실행 시간: {latest_workflow_result['timestamp']}")
        else:
else:
    st.info("💡 아직 Workflow Orchestrator의 결과가 없습니다. 위에서 워크플로우를 실행해보세요.") 