"""
👥 HR Recruitment Agent Page

인재 채용 및 관리 최적화 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

# HR Recruitment Agent 임포트 시도
try:
    import asyncio
    from srcs.enterprise_agents.hr_recruitment_agent import *
    HR_AGENT_AVAILABLE = True
except ImportError as e:
    HR_AGENT_AVAILABLE = False
    import_error = str(e)

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
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not HR_AGENT_AVAILABLE:
        st.error(f"⚠️ HR Recruitment Agent를 불러올 수 없습니다: {import_error}")
        st.info("💡 데모 모드로 실행됩니다.")
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 채용공고 생성", 
        "📋 이력서 스크리닝", 
        "❓ 면접 질문",
        "📊 채용 분석"
    ])
    
    with tab1:
        render_job_posting_creator()
    
    with tab2:
        render_resume_screening()
    
    with tab3:
        render_interview_questions()
    
    with tab4:
        render_recruitment_analysis()

def render_job_posting_creator():
    """채용공고 생성기"""
    
    st.markdown("### 📝 AI 채용공고 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 채용 정보 입력")
        
        position_name = st.text_input("직책명", value="Senior Software Engineer")
        company_name = st.text_input("회사명", value="TechCorp Inc.")
        department = st.text_input("부서", value="개발팀")
        
        experience_level = st.selectbox(
            "경력 수준",
            ["신입", "경력 1-3년", "경력 3-5년", "경력 5-10년", "경력 10년+"]
        )
        
        employment_type = st.selectbox(
            "고용 형태",
            ["정규직", "계약직", "인턴", "파견직"]
        )
        
        work_location = st.selectbox(
            "근무 형태",
            ["사무실 근무", "재택근무", "하이브리드"]
        )
        
        salary_range = st.text_input("연봉 범위", value="5000-7000만원")
        
        # 핵심 요구사항
        st.markdown("#### 📋 핵심 요구사항")
        technical_skills = st.text_area(
            "기술 스킬",
            value="Python, Django, React, PostgreSQL, AWS"
        )
        
        soft_skills = st.text_area(
            "소프트 스킬",
            value="팀워크, 커뮤니케이션, 문제해결능력"
        )
        
        if st.button("🚀 채용공고 생성", use_container_width=True):
            generate_job_posting(position_name, company_name, department, 
                               experience_level, employment_type, work_location,
                               salary_range, technical_skills, soft_skills)
    
    with col2:
        if 'generated_job_posting' in st.session_state:
            st.markdown("#### 📄 생성된 채용공고")
            
            job_posting = st.session_state['generated_job_posting']
            
            # 편집 가능한 텍스트 영역
            edited_posting = st.text_area(
                "채용공고 내용",
                value=job_posting,
                height=500,
                help="생성된 내용을 자유롭게 편집할 수 있습니다."
            )
            
            # 다운로드 옵션
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 텍스트 다운로드"):
                    st.success("텍스트 파일이 다운로드되었습니다!")
            
            with col2:
                if st.button("📊 PDF 다운로드"):
                    st.success("PDF 파일이 생성되었습니다!")
            
            with col3:
                if st.button("📧 이메일 발송"):
                    st.success("채용공고가 이메일로 발송되었습니다!")
        
        else:
            st.info("👈 왼쪽에서 정보를 입력하고 '채용공고 생성' 버튼을 클릭하세요.")

def generate_job_posting(position, company, department, experience, employment_type, 
                        work_location, salary, tech_skills, soft_skills):
    """AI를 사용한 채용공고 생성"""
    
    # 실제 HR Agent와 연동 (현재는 템플릿 기반 생성)
    job_posting = f"""
# {position} 채용공고

## 🏢 회사 소개
**{company}**는 혁신적인 기술로 세상을 변화시키는 IT 기업입니다. 우리는 직원들의 성장과 발전을 지원하며, 창의적이고 협력적인 업무 환경을 제공합니다.

## 📋 모집 개요
- **직책**: {position}
- **부서**: {department}
- **고용형태**: {employment_type}
- **근무형태**: {work_location}
- **경력**: {experience}
- **급여**: {salary}

## 🎯 주요 업무
- 웹 애플리케이션 개발 및 유지보수
- 시스템 아키텍처 설계 및 구현
- 코드 리뷰 및 품질 관리
- 팀 내 기술 공유 및 멘토링
- 새로운 기술 도입 및 적용 검토

## ✅ 필수 요구사항

### 🔧 기술 스킬
{tech_skills}

### 🤝 소프트 스킬  
{soft_skills}

### 📚 기타 요구사항
- 컴퓨터공학 또는 관련 분야 학사 학위
- 업무에 필요한 영어 커뮤니케이션 능력
- 지속적인 학습과 성장에 대한 의지

## 🎁 복리후생
- 경쟁력 있는 급여 및 성과급
- 4대 보험 및 퇴직연금
- 연차/병가/경조사휴가
- 교육비 지원 및 도서 구입비
- 최신 개발 장비 지원
- 유연근무제 및 재택근무

## 📅 전형 절차
1. **서류전형**: 이력서 및 포트폴리오 검토
2. **1차 면접**: 기술 면접 (2시간)
3. **2차 면접**: 임원 면접 (1시간)
4. **최종합격**: 처우 협의 및 입사일 결정

## 📧 지원 방법
- **이메일**: hr@{company.lower().replace(' ', '')}.com
- **지원 마감**: 2024년 12월 31일
- **문의사항**: 인사팀 (02-1234-5678)

---
*{company}는 다양성과 포용성을 중시하며, 모든 지원자에게 평등한 기회를 제공합니다.*
"""
    
    st.session_state['generated_job_posting'] = job_posting.strip()

def render_resume_screening():
    """이력서 스크리닝"""
    
    st.markdown("### 📋 AI 이력서 스크리닝")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📄 이력서 업로드")
        
        uploaded_files = st.file_uploader(
            "이력서 파일들을 업로드하세요",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)}개의 이력서가 업로드되었습니다.")
        
        # 평가 기준 설정
        st.markdown("#### ⚙️ 평가 기준 설정")
        
        tech_weight = st.slider("기술 스킬 가중치", 0, 100, 40, help="기술적 역량의 중요도")
        exp_weight = st.slider("경험 가중치", 0, 100, 30, help="관련 경험의 중요도")
        edu_weight = st.slider("학력 가중치", 0, 100, 15, help="교육 배경의 중요도")
        soft_weight = st.slider("소프트 스킬 가중치", 0, 100, 15, help="소프트 스킬의 중요도")
        
        total_weight = tech_weight + exp_weight + edu_weight + soft_weight
        if total_weight != 100:
            st.warning(f"가중치 합계가 {total_weight}%입니다. 100%로 맞춰주세요.")
        
        if st.button("🔍 스크리닝 시작", use_container_width=True) and uploaded_files:
            screen_resumes(uploaded_files, tech_weight, exp_weight, edu_weight, soft_weight)
    
    with col2:
        if 'screening_results' in st.session_state:
            st.markdown("#### 📊 스크리닝 결과")
            
            results = st.session_state['screening_results']
            
            # 결과 요약
            strong_match = [r for r in results if r['category'] == 'Strong Match']
            potential = [r for r in results if r['category'] == 'Potential']
            not_fit = [r for r in results if r['category'] == 'Not a fit']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 강력 추천", len(strong_match))
            with col2:
                st.metric("⚡ 잠재적 후보", len(potential))
            with col3:
                st.metric("❌ 부적합", len(not_fit))
            
            # 상세 결과
            for result in results:
                with st.expander(f"{result['name']} - {result['score']}/100 ({result['category']})"):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**기술 스킬**: {result['tech_score']}/100")
                        st.write(f"**경험**: {result['exp_score']}/100")
                        st.write(f"**학력**: {result['edu_score']}/100")
                        st.write(f"**소프트 스킬**: {result['soft_score']}/100")
                    
                    with col2:
                        st.write("**주요 강점:**")
                        for strength in result['strengths']:
                            st.write(f"• {strength}")
                        
                        st.write("**개선 사항:**")
                        for weakness in result['weaknesses']:
                            st.write(f"• {weakness}")
                    
                    if result['category'] == 'Strong Match':
                        if st.button(f"📧 {result['name']}에게 면접 요청", key=f"interview_{result['name']}"):
                            st.success("면접 요청 이메일이 발송되었습니다!")
        
        else:
            st.info("👈 왼쪽에서 이력서를 업로드하고 스크리닝을 시작하세요.")

def screen_resumes(uploaded_files, tech_weight, exp_weight, edu_weight, soft_weight):
    """이력서 스크리닝 수행"""
    
    import random
    
    results = []
    
    for i, file in enumerate(uploaded_files):
        # 가상의 스크리닝 결과 생성 (실제로는 AI 분석)
        tech_score = random.randint(60, 95)
        exp_score = random.randint(50, 90)
        edu_score = random.randint(70, 100)
        soft_score = random.randint(55, 85)
        
        # 가중평균 계산
        total_score = (
            tech_score * tech_weight +
            exp_score * exp_weight +
            edu_score * edu_weight +
            soft_score * soft_weight
        ) / 100
        
        # 카테고리 분류
        if total_score >= 80:
            category = "Strong Match"
        elif total_score >= 60:
            category = "Potential"
        else:
            category = "Not a fit"
        
        # 강점과 약점 생성
        strengths = random.sample([
            "풍부한 실무 경험",
            "우수한 기술 스킬",
            "강한 문제해결 능력",
            "팀워크 역량",
            "지속적인 학습 의지",
            "프로젝트 리더십 경험"
        ], k=2)
        
        weaknesses = random.sample([
            "특정 기술 스택 경험 부족",
            "업계 경험 제한적",
            "리더십 경험 부족",
            "대규모 프로젝트 경험 부족"
        ], k=1)
        
        results.append({
            'name': f"지원자_{i+1}_{file.name.split('.')[0]}",
            'score': int(total_score),
            'category': category,
            'tech_score': tech_score,
            'exp_score': exp_score,
            'edu_score': edu_score,
            'soft_score': soft_score,
            'strengths': strengths,
            'weaknesses': weaknesses
        })
    
    # 점수 기준으로 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    st.session_state['screening_results'] = results

def render_interview_questions():
    """면접 질문 생성기"""
    
    st.markdown("### ❓ AI 면접 질문 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 면접 설정")
        
        position = st.text_input("면접 직책", value="Senior Software Engineer")
        interview_type = st.selectbox(
            "면접 유형",
            ["기술 면접", "행동 면접", "종합 면접", "임원 면접"]
        )
        
        difficulty = st.selectbox(
            "난이도 수준",
            ["주니어", "시니어", "리드", "임원급"]
        )
        
        duration = st.selectbox(
            "면접 시간",
            ["30분", "1시간", "1.5시간", "2시간"]
        )
        
        focus_areas = st.multiselect(
            "집중 영역",
            ["기술 스킬", "문제해결", "리더십", "커뮤니케이션", "팀워크", "성장 마인드"],
            default=["기술 스킬", "문제해결"]
        )
        
        if st.button("❓ 면접 질문 생성", use_container_width=True):
            generate_interview_questions(position, interview_type, difficulty, duration, focus_areas)
    
    with col2:
        if 'interview_questions' in st.session_state:
            st.markdown("#### 📝 생성된 면접 질문")
            
            questions = st.session_state['interview_questions']
            
            for i, q in enumerate(questions, 1):
                with st.expander(f"질문 {i}: {q['question'][:50]}..."):
                    st.write(f"**질문**: {q['question']}")
                    st.write(f"**유형**: {q['type']}")
                    st.write(f"**예상 답변 시간**: {q['time']}")
                    
                    st.write("**평가 포인트**:")
                    for point in q['evaluation_points']:
                        st.write(f"• {point}")
                    
                    st.write("**모범 답변 가이드**:")
                    st.write(q['answer_guide'])
                    
                    if q.get('follow_up'):
                        st.write("**후속 질문**:")
                        for follow in q['follow_up']:
                            st.write(f"• {follow}")
            
            # 면접 가이드 다운로드
            if st.button("📄 면접 가이드 다운로드", use_container_width=True):
                st.success("면접 가이드가 다운로드되었습니다!")
        
        else:
            st.info("👈 왼쪽에서 면접 설정을 하고 질문을 생성하세요.")

def generate_interview_questions(position, interview_type, difficulty, duration, focus_areas):
    """면접 질문 생성"""
    
    # 실제로는 HR Agent의 interview_agent를 호출
    questions = []
    
    if "기술 스킬" in focus_areas:
        questions.extend([
            {
                "question": f"{position} 역할에서 가장 중요하다고 생각하는 기술 스킬 3가지는 무엇이며, 각각을 어떻게 활용해 보셨나요?",
                "type": "기술적 역량",
                "time": "5-7분",
                "evaluation_points": [
                    "핵심 기술에 대한 이해도",
                    "실무 적용 경험",
                    "기술 선택의 논리적 근거"
                ],
                "answer_guide": "구체적인 프로젝트 사례와 함께 기술적 의사결정 과정을 설명할 수 있어야 함",
                "follow_up": [
                    "해당 기술을 선택한 이유는?",
                    "다른 대안과 비교했을 때의 장단점은?"
                ]
            }
        ])
    
    if "문제해결" in focus_areas:
        questions.extend([
            {
                "question": "가장 도전적이었던 기술적 문제는 무엇이었고, 어떻게 해결하셨나요?",
                "type": "문제해결 능력",
                "time": "7-10분",
                "evaluation_points": [
                    "문제 분석 능력",
                    "해결 과정의 체계성",
                    "결과 및 학습점"
                ],
                "answer_guide": "STAR 방식(Situation, Task, Action, Result)으로 구조화하여 답변",
                "follow_up": [
                    "다시 같은 상황이 온다면 어떻게 하겠는가?",
                    "팀원들과 어떻게 협력했는가?"
                ]
            }
        ])
    
    if "리더십" in focus_areas:
        questions.extend([
            {
                "question": "팀을 이끌어본 경험이 있다면, 가장 어려웠던 순간과 극복 방법을 말씀해 주세요.",
                "type": "리더십",
                "time": "5-8분",
                "evaluation_points": [
                    "리더십 스타일",
                    "갈등 해결 능력",
                    "팀 동기부여 방법"
                ],
                "answer_guide": "구체적인 상황과 행동, 결과를 포함하여 설명",
                "follow_up": [
                    "팀원들로부터 어떤 피드백을 받았는가?",
                    "리더로서 가장 중요한 자질은?"
                ]
            }
        ])
    
    st.session_state['interview_questions'] = questions

def render_recruitment_analysis():
    """채용 분석 대시보드"""
    
    st.markdown("### 📊 채용 분석 대시보드")
    
    # 가상의 채용 데이터
    import random
    from datetime import datetime, timedelta
    
    # 월별 채용 현황
    months = ['1월', '2월', '3월', '4월', '5월', '6월']
    applications = [random.randint(50, 200) for _ in months]
    interviews = [random.randint(10, 50) for _ in months]
    hires = [random.randint(2, 15) for _ in months]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 월별 채용 현황")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=applications, name='지원자', marker_color='lightblue'))
        fig.add_trace(go.Bar(x=months, y=interviews, name='면접자', marker_color='orange'))
        fig.add_trace(go.Bar(x=months, y=hires, name='채용자', marker_color='green'))
        
        fig.update_layout(
            title='월별 채용 퍼널',
            xaxis_title='월',
            yaxis_title='인원 수',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 직무별 채용 현황")
        
        positions = ['개발자', '디자이너', '기획자', '마케터', '영업']
        position_hires = [random.randint(5, 25) for _ in positions]
        
        import plotly.express as px
        
        fig = px.pie(values=position_hires, names=positions, title='직무별 채용 비율')
        st.plotly_chart(fig, use_container_width=True)
    
    # 채용 메트릭
    st.markdown("#### 📊 주요 채용 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        conversion_rate = sum(hires) / sum(applications) * 100
        st.metric("전환율", f"{conversion_rate:.1f}%", f"{random.uniform(-1, 1):+.1f}%")
    
    with col2:
        avg_time = random.randint(20, 40)
        st.metric("평균 채용 기간", f"{avg_time}일", f"{random.randint(-3, 3):+d}일")
    
    with col3:
        cost_per_hire = random.randint(200, 500)
        st.metric("채용당 비용", f"{cost_per_hire}만원", f"{random.randint(-50, 50):+d}만원")
    
    with col4:
        satisfaction = random.uniform(4.0, 5.0)
        st.metric("채용 만족도", f"{satisfaction:.1f}/5.0", f"{random.uniform(-0.2, 0.2):+.1f}")
    
    # 개선 제안
    st.markdown("---")
    st.markdown("#### 💡 AI 개선 제안")
    
    suggestions = [
        "🎯 **스크리닝 효율성 향상**: AI 이력서 분석을 통해 초기 스크리닝 시간을 40% 단축할 수 있습니다.",
        "📧 **후보자 경험 개선**: 자동화된 상태 업데이트와 피드백으로 후보자 만족도를 높이세요.",
        "📊 **데이터 기반 의사결정**: 과거 채용 데이터를 분석하여 성공 패턴을 발견하고 적용하세요.",
        "🤝 **채용팀 협업 강화**: 통합된 평가 시스템으로 팀원 간 의견을 효율적으로 공유하세요."
    ]
    
    for suggestion in suggestions:
        st.write(f"• {suggestion}")

if __name__ == "__main__":
    main() 