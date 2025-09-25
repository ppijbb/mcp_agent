"""
🔍 Research Agent Page - Local Researcher Integration

AI 기반 자율 연구 시스템과 통합된 연구 에이전트
"""

import streamlit as st
import sys
from pathlib import Path
import streamlit_process_manager as spm
from srcs.common.ui_utils import run_agent_process
import tempfile
import json
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 임포트
from configs.settings import get_reports_path

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

# Result Reader 임포트
try:
    from srcs.utils.result_reader import result_reader, result_display
except ImportError as e:
    st.error(f"❌ 결과 읽기 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# Local Researcher Project 임포트 시도
try:
    # Local Researcher 프로젝트 경로 추가
    local_researcher_path = Path(__file__).parent.parent / "local_researcher_project"
    sys.path.insert(0, str(local_researcher_path))
    
    from local_researcher_project.src.core.autonomous_orchestrator import LangGraphOrchestrator
    from local_researcher_project.src.agents.task_analyzer import TaskAnalyzerAgent
    from local_researcher_project.src.agents.task_decomposer import TaskDecomposerAgent
    from local_researcher_project.src.agents.research_agent import ResearchAgent
    from local_researcher_project.src.agents.evaluation_agent import EvaluationAgent
    from local_researcher_project.src.agents.validation_agent import ValidationAgent
    from local_researcher_project.src.agents.synthesis_agent import SynthesisAgent
    from local_researcher_project.src.core.mcp_integration import MCPIntegrationManager
    from local_researcher_project.src.utils.config_manager import ConfigManager
    
    LOCAL_RESEARCHER_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Local Researcher를 불러올 수 없습니다: {e}")
    st.info("Local Researcher 프로젝트를 확인하고 필요한 의존성을 설치해주세요.")
    LOCAL_RESEARCHER_AVAILABLE = False

# 기존 Research Agent 임포트 (fallback)
if not LOCAL_RESEARCHER_AVAILABLE:
    try:
        from srcs.advanced_agents.researcher_v2 import (
            ResearcherAgent,
            load_research_focus_options,
            load_research_templates,
            get_research_agent_status,
            save_research_report
        )
    except ImportError as e:
        st.error(f"⚠️ Research Agent를 불러올 수 없습니다: {e}")
        st.info("에이전트 모듈을 확인하고 필요한 의존성을 설치해주세요.")
        st.stop()


def main():
    """메인 함수 - Local Researcher 통합"""
    st.title("🔍 Research Agent - Local Researcher Integration")
    st.markdown("AI 기반 자율 연구 시스템")
    
    # Local Researcher 사용 가능 여부 확인
    if LOCAL_RESEARCHER_AVAILABLE:
        st.success("✅ Local Researcher 프로젝트가 연결되었습니다!")
        run_local_researcher_interface()
    else:
        st.warning("⚠️ Local Researcher를 사용할 수 없습니다. 기본 Research Agent를 사용합니다.")
        run_fallback_interface()


def run_local_researcher_interface():
    """Local Researcher 통합 인터페이스 실행"""
    try:
        # 탭으로 기능 분리
        tab1, tab2, tab3 = st.tabs(["연구 실행", "데이터 시각화", "시스템 모니터"])
        
        with tab1:
            run_research_interface()
        
        with tab2:
            run_visualization_interface()
        
        with tab3:
            run_monitoring_interface()
    
    except Exception as e:
        st.error(f"❌ Local Researcher 초기화 실패: {e}")
        st.info("기본 Research Agent로 전환합니다.")
        run_fallback_interface()


def run_research_interface():
    """연구 실행 인터페이스"""
    st.subheader("🚀 자율 연구 실행")
    
    # 연구 설정
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_query = st.text_area(
            "연구 주제를 입력하세요:",
            placeholder="예: 인공지능의 최신 동향과 미래 전망",
            height=100
        )
    
    with col2:
        research_depth = st.selectbox(
            "연구 깊이",
            options=["Quick", "Standard", "Deep", "Comprehensive"],
            index=1
        )
        
        research_domain = st.selectbox(
            "연구 도메인",
            options=["General", "Academic", "Business", "Technical", "Scientific"],
            index=0
        )
        
        use_browser = st.checkbox("브라우저 자동화", value=True)
        use_mcp = st.checkbox("MCP 도구 사용", value=True)
    
    # 고급 옵션
    with st.expander("고급 옵션"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_iterations = st.number_input("최대 반복 횟수", min_value=1, max_value=10, value=6)
            quality_threshold = st.slider("품질 임계값", 0.0, 1.0, 0.8)
        
        with col2:
            parallel_execution = st.checkbox("병렬 실행", value=True)
            real_time_monitoring = st.checkbox("실시간 모니터링", value=True)
    
    # 연구 실행
    if st.button("🔍 연구 시작", type="primary"):
        if not research_query.strip():
            st.warning("⚠️ 연구 주제를 입력해주세요.")
            return
        
        # 연구 컨텍스트 구성
        context = {
            "research_depth": research_depth,
            "research_domain": research_domain,
            "use_browser": use_browser,
            "use_mcp": use_mcp,
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
            "parallel_execution": parallel_execution,
            "real_time_monitoring": real_time_monitoring,
            "timestamp": datetime.now().isoformat()
        }
        
        # 연구 실행
        with st.spinner("연구를 진행하는 중입니다..."):
            try:
                # Local Researcher 컴포넌트 초기화
                config_manager = ConfigManager()
                mcp_manager = MCPIntegrationManager()
                
                agents = {
                    'analyzer': TaskAnalyzerAgent(),
                    'decomposer': TaskDecomposerAgent(),
                    'researcher': ResearchAgent(),
                    'evaluator': EvaluationAgent(),
                    'validator': ValidationAgent(),
                    'synthesizer': SynthesisAgent()
                }
                
                orchestrator = LangGraphOrchestrator(
                    config_path=None,
                    agents=agents,
                    mcp_manager=mcp_manager
                )
                
                # 비동기 함수 실행
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                objective_id = loop.run_until_complete(
                    orchestrator.start_autonomous_research(research_query, context)
                )
                loop.close()
                
                st.success(f"✅ 연구가 완료되었습니다! Objective ID: {objective_id}")
                
                # 결과를 세션 상태에 저장
                st.session_state['last_research_id'] = objective_id
                st.session_state['last_orchestrator'] = orchestrator
                
                # 결과 표시
                display_local_researcher_results(objective_id, orchestrator)
                
            except Exception as e:
                st.error(f"❌ 연구 실행 실패: {e}")
                st.exception(e)


def run_visualization_interface():
    """데이터 시각화 인터페이스"""
    st.subheader("📊 데이터 시각화")
    
    try:
        # 시각화 옵션
        viz_type = st.selectbox(
            "시각화 유형",
            ["연구 타임라인", "에이전트 성능", "품질 분포", "연구 트렌드", "도메인 분석", "시스템 상태"]
        )
        
        if st.button("📈 시각화 생성"):
            with st.spinner("시각화를 생성하는 중..."):
                # 샘플 데이터로 시각화 생성
                if viz_type == "연구 타임라인":
                    sample_data = generate_sample_timeline_data()
                    st.write("연구 타임라인 데이터:")
                    st.json(sample_data)
                elif viz_type == "에이전트 성능":
                    sample_data = generate_sample_performance_data()
                    st.write("에이전트 성능 데이터:")
                    st.json(sample_data)
                elif viz_type == "품질 분포":
                    sample_data = generate_sample_quality_data()
                    st.write("품질 분포 데이터:")
                    st.write(f"평균: {sum(sample_data)/len(sample_data):.2f}")
                elif viz_type == "연구 트렌드":
                    sample_data = generate_sample_trends_data()
                    st.write("연구 트렌드 데이터:")
                    st.json(sample_data)
                elif viz_type == "도메인 분석":
                    sample_data = generate_sample_domain_data()
                    st.write("도메인 분석 데이터:")
                    st.json(sample_data)
                elif viz_type == "시스템 상태":
                    sample_data = generate_sample_system_data()
                    st.write("시스템 상태 데이터:")
                    st.json(sample_data)
    
    except Exception as e:
        st.error(f"시각화 생성 실패: {e}")


def run_monitoring_interface():
    """시스템 모니터링 인터페이스"""
    st.subheader("🔍 시스템 모니터링")
    
    try:
        # 현재 시스템 상태 (시뮬레이션)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU 사용률", "45.2%")
        with col2:
            st.metric("메모리 사용률", "67.8%")
        with col3:
            st.metric("디스크 사용률", "23.1%")
        with col4:
            st.metric("활성 프로세스", "156")
        
        # 시스템 건강 점수
        st.metric("시스템 건강 점수", "87.5/100")
        
        # 최근 알림
        st.subheader("최근 알림")
        alerts = [
            {"level": "info", "message": "연구 작업이 완료되었습니다", "time": "10:30 AM"},
            {"level": "warning", "message": "메모리 사용률이 높습니다", "time": "10:25 AM"},
            {"level": "info", "message": "새로운 에이전트가 시작되었습니다", "time": "10:20 AM"},
        ]
        
        for alert in alerts:
            alert_color = {
                "info": "blue",
                "warning": "orange", 
                "error": "red",
                "critical": "darkred"
            }.get(alert["level"], "gray")
            
            st.markdown(f"<div style='color: {alert_color}'>[{alert['level'].upper()}] {alert['message']} - {alert['time']}</div>", 
                       unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"모니터링 실패: {e}")


def run_fallback_interface():
    """기본 Research Agent 인터페이스 (fallback)"""
    st.info("기본 Research Agent를 사용합니다.")
    
    # 기존 Research Agent 로직
    research_focus_options = load_research_focus_options()
    research_templates = load_research_templates()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_query = st.text_area(
            "연구 주제를 입력하세요:",
            placeholder="예: 인공지능의 최신 동향과 미래 전망",
            height=100
        )
    
    with col2:
        research_focus = st.selectbox(
            "연구 초점",
            options=list(research_focus_options.keys()),
            help="연구의 주요 초점 영역을 선택하세요"
        )
        
        research_depth = st.selectbox(
            "연구 깊이",
            options=["표면적", "중간", "깊이있게", "매우 깊이있게"],
            index=1
        )
    
    if st.button("🔍 연구 시작", type="primary"):
        if not research_query.strip():
            st.warning("⚠️ 연구 주제를 입력해주세요.")
            return
        
        research_config = {
            "query": research_query,
            "focus": research_focus,
            "depth": research_depth
        }
        
        with st.spinner("연구를 진행하는 중입니다..."):
            try:
                result = run_agent_process("researcher_v2", research_config, timeout=300)
                
                if result and result.get("success"):
                    st.success("✅ 연구가 완료되었습니다!")
                    display_research_results(result)
                else:
                    st.error("❌ 연구 실행 중 오류가 발생했습니다.")
            
            except Exception as e:
                st.error(f"❌ 연구 실행 실패: {e}")


def display_local_researcher_results(objective_id: str, orchestrator: LangGraphOrchestrator):
    """Local Researcher 결과 표시"""
    try:
        # 연구 상태 가져오기
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(orchestrator.get_research_status(objective_id))
        loop.close()
        
        if status:
            st.subheader("📊 연구 결과")
            
            # 기본 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("상태", status.get('status', 'Unknown'))
            with col2:
                st.metric("목표 수", len(status.get('analyzed_objectives', [])))
            with col3:
                st.metric("작업 수", len(status.get('decomposed_tasks', [])))
            
            # 상세 결과
            if status.get('final_synthesis'):
                st.subheader("📝 최종 보고서")
                synthesis = status['final_synthesis']
                
                if synthesis.get('summary'):
                    st.write("**요약:**")
                    st.write(synthesis['summary'])
                
                if synthesis.get('key_findings'):
                    st.write("**주요 발견사항:**")
                    for finding in synthesis['key_findings']:
                        st.write(f"• {finding}")
                
                if synthesis.get('recommendations'):
                    st.write("**권장사항:**")
                    for rec in synthesis['recommendations']:
                        st.write(f"• {rec}")
            
            # 보고서 다운로드
            if status.get('final_synthesis', {}).get('deliverable_path'):
                st.subheader("📄 보고서 다운로드")
                report_path = status['final_synthesis']['deliverable_path']
                if os.path.exists(report_path):
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            label="보고서 다운로드",
                            data=f.read(),
                            file_name=os.path.basename(report_path),
                            mime="application/octet-stream"
                        )
        
    except Exception as e:
        st.error(f"결과 표시 실패: {e}")


def display_research_results(result: dict):
    """기존 Research Agent 결과 표시"""
    try:
        if result.get("research_summary"):
            st.subheader("📝 연구 요약")
            st.write(result["research_summary"])
        
        if result.get("key_findings"):
            st.subheader("🔍 주요 발견사항")
            for finding in result["key_findings"]:
                st.write(f"• {finding}")
        
        if result.get("sources"):
            st.subheader("📚 참고 자료")
            for source in result["sources"]:
                st.write(f"• {source}")
    
    except Exception as e:
        st.error(f"결과 표시 실패: {e}")


# 샘플 데이터 생성 함수들
def generate_sample_timeline_data():
    """샘플 타임라인 데이터 생성"""
    return [
        {
            "objective_id": "obj_001",
            "start_time": "2024-01-01 09:00:00",
            "end_time": "2024-01-01 10:30:00",
            "status": "completed",
            "agent_count": 3,
            "quality_score": 0.85
        },
        {
            "objective_id": "obj_002", 
            "start_time": "2024-01-01 11:00:00",
            "end_time": "2024-01-01 12:15:00",
            "status": "completed",
            "agent_count": 4,
            "quality_score": 0.92
        }
    ]


def generate_sample_performance_data():
    """샘플 성능 데이터 생성"""
    return {
        "Task Analyzer": {"tasks_completed": 45, "success_rate": 0.95, "avg_quality": 0.88},
        "Research Agent": {"tasks_completed": 38, "success_rate": 0.88, "avg_quality": 0.85},
        "Evaluation Agent": {"tasks_completed": 42, "success_rate": 0.92, "avg_quality": 0.90},
        "Validation Agent": {"tasks_completed": 40, "success_rate": 0.90, "avg_quality": 0.87},
        "Synthesis Agent": {"tasks_completed": 35, "success_rate": 0.87, "avg_quality": 0.89}
    }


def generate_sample_quality_data():
    """샘플 품질 데이터 생성"""
    import random
    return [random.uniform(0.6, 1.0) for _ in range(100)]


def generate_sample_trends_data():
    """샘플 트렌드 데이터 생성"""
    import random
    from datetime import datetime, timedelta
    
    data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "research_count": random.randint(1, 5),
            "avg_quality": random.uniform(0.7, 0.95),
            "agent_utilization": random.uniform(0.6, 0.9),
            "success_rate": random.uniform(0.8, 0.95)
        })
    
    return data


def generate_sample_domain_data():
    """샘플 도메인 데이터 생성"""
    return {
        "AI/ML": {"research_count": 25, "avg_quality": 0.88, "success_rate": 0.92},
        "Business": {"research_count": 18, "avg_quality": 0.85, "success_rate": 0.89},
        "Science": {"research_count": 22, "avg_quality": 0.91, "success_rate": 0.94},
        "Technology": {"research_count": 30, "avg_quality": 0.87, "success_rate": 0.90}
    }


def generate_sample_system_data():
    """샘플 시스템 데이터 생성"""
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "research_status": {"running": 3, "completed": 15, "failed": 1},
        "error_rates": {"network": 2, "processing": 1, "validation": 0}
    }


if __name__ == "__main__":
    main()
