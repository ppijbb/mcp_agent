"""
👥 HR Recruitment Agent Page

인재 채용 및 관리 최적화 AI
"""

import streamlit as st
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 시스템 import
from configs.settings import get_reports_path

# HR Recruitment Agent 임포트 시도
try:
    from srcs.enterprise_agents.hr_recruitment_agent import HRRecruitmentAgent
except ImportError as e:
    st.error(f"HR Recruitment Agent를 사용하려면 필요한 의존성을 설치해야 합니다: {e}")
    st.error("시스템 관리자에게 문의하여 HR Recruitment Agent 모듈을 설정하세요.")
    st.stop()

def load_workflow_options():
    """워크플로우 옵션 동적 로딩"""
    # 실제 시스템에서 지원하는 워크플로우 로드 (환경변수 또는 설정 파일에서)
    default_workflows = [
        "채용공고 생성", 
        "이력서 스크리닝 가이드", 
        "면접 질문 세트", 
        "레퍼런스 체크 프로세스",
        "오퍼레터 템플릿",
        "온보딩 프로그램",
        "성과 평가 기준",
        "급여 협상 가이드",
        "팀 문화 적합성 평가",
        "기술 역량 테스트"
    ]
    
    # 환경변수에서 커스텀 워크플로우 로드
    custom_workflows = os.getenv("HR_CUSTOM_WORKFLOWS", "").split(",")
    if custom_workflows[0]:  # 빈 문자열이 아닌 경우
        return custom_workflows
    
    return default_workflows

def load_default_workflows():
    """기본 워크플로우 동적 로딩"""
    # 실제 사용자 설정에서 기본 워크플로우 로드
    default_selection = os.getenv("HR_DEFAULT_WORKFLOWS", "채용공고 생성,면접 질문 세트").split(",")
    return [w.strip() for w in default_selection if w.strip()]

def get_user_company_info():
    """사용자 회사 정보 조회"""
    # 실제 사용자 프로필에서 회사 정보 로드 (환경변수 또는 세션에서)
    
    # 세션에서 회사 정보 확인
    if 'company_info' in st.session_state:
        return st.session_state.company_info
    
    # 환경변수에서 기본 회사 정보 로드
    company_info = {
        "company_name": os.getenv("COMPANY_NAME", ""),
        "industry": os.getenv("COMPANY_INDUSTRY", ""),
        "size": os.getenv("COMPANY_SIZE", ""),
        "location": os.getenv("COMPANY_LOCATION", ""),
        "default_positions": os.getenv("COMPANY_DEFAULT_POSITIONS", "").split(","),
        "hr_contact": os.getenv("HR_CONTACT_EMAIL", ""),
        "company_culture": os.getenv("COMPANY_CULTURE", ""),
        "benefits": os.getenv("COMPANY_BENEFITS", "").split(",")
    }
    
    # 빈 값들 정리
    company_info["default_positions"] = [p.strip() for p in company_info["default_positions"] if p.strip()]
    company_info["benefits"] = [b.strip() for b in company_info["benefits"] if b.strip()]
    
    # 세션에 저장
    st.session_state.company_info = company_info
    
    return company_info

def get_position_templates():
    """직책별 템플릿 조회"""
    # 실제 구현에서는 데이터베이스나 파일에서 로드
    templates = {
        "소프트웨어 엔지니어": {
            "required_skills": ["Python", "JavaScript", "Git", "SQL"],
            "preferred_skills": ["React", "Docker", "AWS", "Kubernetes"],
            "experience_years": "2-5년",
            "education": "컴퓨터공학 또는 관련 분야 학사 이상",
            "responsibilities": [
                "웹 애플리케이션 개발 및 유지보수",
                "코드 리뷰 및 품질 관리",
                "기술 문서 작성",
                "팀 협업 및 커뮤니케이션"
            ]
        },
        "데이터 사이언티스트": {
            "required_skills": ["Python", "R", "SQL", "Machine Learning"],
            "preferred_skills": ["TensorFlow", "PyTorch", "Tableau", "Spark"],
            "experience_years": "3-7년",
            "education": "통계학, 수학, 컴퓨터공학 또는 관련 분야 석사 이상",
            "responsibilities": [
                "데이터 분석 및 모델링",
                "비즈니스 인사이트 도출",
                "머신러닝 모델 개발",
                "데이터 시각화 및 보고서 작성"
            ]
        },
        "프로덕트 매니저": {
            "required_skills": ["Product Strategy", "Data Analysis", "Communication"],
            "preferred_skills": ["Agile", "Scrum", "Figma", "Analytics Tools"],
            "experience_years": "3-8년",
            "education": "경영학, 공학 또는 관련 분야 학사 이상",
            "responsibilities": [
                "제품 전략 수립 및 실행",
                "크로스 펑셔널 팀 리드",
                "시장 조사 및 경쟁 분석",
                "제품 로드맵 관리"
            ]
        }
    }
    
    return templates

def save_hr_result_to_file(position_name: str, company_name: str, workflows: List[str], result: Dict[str, Any]):
    """HR 결과를 파일로 저장"""
    
    try:
        # 저장 경로 생성
        output_dir = get_reports_path('hr_recruitment')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성 (안전한 파일명으로 변환)
        safe_position = "".join(c for c in position_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_company = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hr_recruitment_{safe_position}_{safe_company}_{timestamp}.json"
        filepath = output_dir / filename
        
        # 저장할 데이터 구성
        save_data = {
            "position_name": position_name,
            "company_name": company_name,
            "workflows_executed": workflows,
            "execution_timestamp": datetime.now().isoformat(),
            "result": result,
            "file_generated_at": datetime.now().isoformat()
        }
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        st.success(f"✅ HR 채용 결과가 저장되었습니다: {filepath}")
        
        # 추가로 텍스트 요약 파일도 생성
        txt_filename = f"hr_recruitment_summary_{safe_position}_{timestamp}.txt"
        txt_filepath = output_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("HR Recruitment Agent 채용 보고서\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"채용 포지션: {position_name}\n")
            f.write(f"회사명: {company_name}\n")
            f.write(f"실행된 워크플로우: {', '.join(workflows)}\n\n")
            
            if 'content' in result:
                f.write("생성된 콘텐츠:\n")
                f.write("-" * 40 + "\n")
                f.write(result['content'])
                f.write("\n" + "-" * 40 + "\n\n")
            
            f.write("*본 보고서는 HR Recruitment Agent에 의해 자동 생성되었습니다.*\n")
        
        return filepath
        
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")
        return None

def main():
    """HR Recruitment Agent 메인 페이지"""
    
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
        <h1>👥 HR Recruitment Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 인재 채용 및 관리 최적화 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    save_to_file = st.checkbox(
        "채용 결과를 파일로 저장", 
        value=False,
        help=f"체크하면 {get_reports_path('hr_recruitment')} 디렉토리에 결과를 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    st.success("🤖 HR Recruitment Agent가 성공적으로 연결되었습니다!")
        
    # 에이전트 실행 인터페이스 제공
    render_hr_agent_interface(save_to_file)

def render_hr_agent_interface(save_to_file=False):
    """HR Agent 실행 인터페이스"""
    
    st.markdown("### 🚀 HR Recruitment Agent 실행")
    
    # 에이전트 초기화
    try:
        if 'hr_agent' not in st.session_state:
            st.session_state.hr_agent = HRRecruitmentAgent()
        
        agent = st.session_state.hr_agent
        
        # 실행 설정
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ 채용 프로젝트 설정")
            
            # 사용자 회사 정보 로딩
            company_info = get_user_company_info()
            
            position_name = st.text_input(
                "채용 포지션", 
                value=None,
                placeholder="채용하려는 직책명을 입력하세요",
                help="채용하려는 직책명을 입력하세요"
            )
            
            company_name = st.text_input(
                "회사명", 
                value=company_info.get("company_name"),
                placeholder="회사명을 입력하세요",
                help="회사명을 입력하세요"
            )
            
            # 워크플로우 옵션 동적 로딩
            workflow_options = load_workflow_options()
            default_workflows = load_default_workflows()
            
            workflow_scope = st.multiselect(
                "실행할 워크플로우",
                workflow_options,
                default=default_workflows,
                help="실행할 채용 워크플로우를 선택하세요"
            )
            
            # 필수 입력값 검증
            if all([position_name, company_name, workflow_scope]):
                if st.button("🚀 HR Agent 실행", type="primary", use_container_width=True):
                    execute_hr_agent(agent, position_name, company_name, workflow_scope, save_to_file)
            else:
                st.warning("모든 필수 정보를 입력해주세요.")
                if st.button("🚀 HR Agent 실행", type="primary", use_container_width=True, disabled=True):
                    pass
        
        with col2:
            if 'hr_execution_result' in st.session_state:
                result = st.session_state['hr_execution_result']
                
                if result.get('success', False):
                    st.success("✅ HR Recruitment Agent 실행 완료!")
                    
                    # 결과 검증
                    if not result:
                        st.error("HR Agent 실행 결과를 받을 수 없습니다.")
                        return
                    
                    # 결과 정보 표시
                    display_hr_results(result, position_name if 'position_name' in locals() else 'unknown')
                        
                else:
                    st.error("❌ 실행 중 오류 발생")
                    st.error(f"**오류**: {result.get('message', 'Unknown error')}")
                    
                    with st.expander("🔍 오류 상세"):
                        st.code(result.get('error', 'Unknown error'))
                        
            else:
                st.markdown("""
                #### 🤖 Agent 실행 정보
                
                **실행되는 프로세스:**
                1. **MCP App 초기화** - MCP 프레임워크 연결
                2. **다중 에이전트 생성** - 채용 전문 AI 에이전트들
                3. **오케스트레이터 실행** - 통합 워크플로우 관리
                4. **콘텐츠 생성** - 실시간 결과 표시 또는 파일 저장
                
                **생성되는 콘텐츠:**
                - 📝 채용공고 (Job Description)
                - 📋 이력서 스크리닝 가이드
                - ❓ 면접 질문 세트
                - 📞 레퍼런스 체크 프로세스
                - 📄 오퍼레터 템플릿
                - 🎯 온보딩 프로그램
                
                **출력 옵션:**
                - 🖥️ **화면 표시**: 즉시 결과 확인 (기본값)
                - 💾 **파일 저장**: 동적 경로에 저장
                """)
                
    except Exception as e:
        st.error(f"Agent 초기화 중 오류: {e}")
        st.info("에이전트 클래스를 확인해주세요.")

def display_hr_results(result, position_name):
    """HR 실행 결과 표시"""
    
    st.markdown("#### 📊 실행 결과")
    
    # 기본 정보 표시
    if 'message' in result:
        st.info(f"**메시지**: {result['message']}")
    
    if result.get('save_to_file') and result.get('output_dir'):
        st.info(f"**출력 디렉토리**: {result['output_dir']}")
    
    if 'workflows_executed' in result:
        st.info(f"**실행된 워크플로우**: {', '.join(result['workflows_executed'])}")
    
    # 생성된 콘텐츠 표시
    if 'content' in result and result['content']:
        st.markdown("#### 📄 생성된 콘텐츠")
        
        content = result['content']
        
        # 콘텐츠 길이에 따른 표시 방식
        if len(content) > 1000:
            with st.expander("📋 전체 콘텐츠 보기", expanded=True):
                st.markdown(content)
        else:
            st.markdown(content)
        
        # 콘텐츠 다운로드 버튼
        safe_filename = position_name.replace(' ', '_').lower() if position_name else 'hr_result'
        st.download_button(
            label="📥 콘텐츠 다운로드",
            data=content,
            file_name=f"hr_recruitment_result_{safe_filename}.md",
            mime="text/markdown"
        )
    
    # 상세 결과 (디버그용)
    with st.expander("🔍 상세 실행 정보"):
        debug_info = {
            'success': result.get('success', False),
            'message': result.get('message', 'N/A'),
            'workflows_executed': result.get('workflows_executed', []),
            'save_to_file': result.get('save_to_file', False),
            'output_dir': result.get('output_dir', 'N/A'),
            'content_length': len(result.get('content', '')) if result.get('content') else 0
        }
        st.json(debug_info)

def execute_hr_agent(agent, position, company, workflows, save_to_file):
    """HR Agent 실행"""
    
    try:
        with st.spinner("🔄 HR Recruitment Agent를 실행하는 중..."):
            # 에이전트 실행
            result = agent.run_recruitment_workflow(
                position=position,
                company=company,
                workflows=workflows,
                save_to_file=save_to_file
            )
            
            # 결과 검증
            if not result:
                st.session_state['hr_execution_result'] = {
                    'success': False,
                    'message': 'HR Agent가 결과를 반환하지 않았습니다.',
                    'error': 'Empty result from agent',
                    'save_to_file': save_to_file
                }
            else:
                st.session_state['hr_execution_result'] = result
            
            st.rerun()
            
    except Exception as e:
        st.session_state['hr_execution_result'] = {
            'success': False,
            'message': f'Agent 실행 중 오류 발생: {str(e)}',
            'error': str(e),
            'save_to_file': save_to_file
        }
        st.rerun()

if __name__ == "__main__":
    main() 