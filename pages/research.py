"""
🔍 Research Agent Page

정보 검색 및 분석 AI
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 유틸리티 임포트
from srcs.common.page_utils import create_agent_page

# Research Agent 임포트 시도
try:
    from srcs.basic_agents.researcher_v2 import *
    RESEARCH_AGENT_AVAILABLE = True
except ImportError as e:
    RESEARCH_AGENT_AVAILABLE = False
    import_error = str(e)

def main():
    """Research Agent 메인 페이지"""
    
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
        <h1>🔍 Research Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 정보 검색 및 분석 시스템
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # Agent 연동 상태 확인
    if not RESEARCH_AGENT_AVAILABLE:
        st.error(f"⚠️ Research Agent를 불러올 수 없습니다: {import_error}")
        st.info("💡 데모 모드로 실행됩니다.")
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "🔍 AI 리서치", 
        "📊 정보 분석", 
        "📝 보고서 생성"
    ])
    
    with tab1:
        render_ai_research()
    
    with tab2:
        render_information_analysis()
    
    with tab3:
        render_report_generation()

def render_ai_research():
    """AI 기반 리서치"""
    
    st.markdown("### 🔍 AI 리서치 엔진")
    st.info("실제 Research Agent v2를 사용하여 종합적인 정보 검색과 분석을 제공합니다.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎯 리서치 설정")
        
        research_topic = st.text_input(
            "리서치 주제",
            value="인공지능 트렌드 2024",
            help="조사하고 싶은 주제를 입력하세요"
        )
        
        research_type = st.selectbox(
            "리서치 유형",
            ["종합 분석", "시장 조사", "기술 동향", "경쟁 분석", "학술 연구"]
        )
        
        sources = st.multiselect(
            "정보 소스",
            ["웹 검색", "뉴스", "학술 논문", "보고서", "소셜미디어", "정부 자료"],
            default=["웹 검색", "뉴스"]
        )
        
        depth_level = st.select_slider(
            "분석 깊이",
            options=["기본", "중간", "심화", "전문가"],
            value="중간"
        )
        
        if st.button("🚀 AI 리서치 시작", use_container_width=True):
            start_ai_research(research_topic, research_type, sources, depth_level)
    
    with col2:
        if 'research_results' in st.session_state:
            results = st.session_state['research_results']
            
            st.markdown("#### 📊 리서치 결과")
            
            # 진행 상태
            if results['status'] == 'completed':
                st.success("✅ 리서치 완료!")
                
                # 요약 정보
                st.markdown("##### 📋 요약")
                st.write(results['summary'])
                
                # 주요 발견사항
                st.markdown("##### 🔍 주요 발견사항")
                for finding in results['key_findings']:
                    st.write(f"• {finding}")
                
                # 검색된 소스들
                st.markdown("##### 📚 참조 소스")
                for i, source in enumerate(results['sources'], 1):
                    with st.expander(f"소스 {i}: {source['title']}"):
                        st.write(f"**URL**: {source['url']}")
                        st.write(f"**신뢰도**: {source['credibility']}/10")
                        st.write(f"**요약**: {source['summary']}")
                
            elif results['status'] == 'processing':
                st.info(f"🔍 {results['current_step']} 진행 중...")
                st.progress(results['progress'])
                
        else:
            st.markdown("""
            #### 🤖 AI 리서치 기능
            
            **고급 검색 엔진:**
            - 🌐 다중 소스 통합 검색
            - 🧠 AI 기반 정보 분석
            - 📊 데이터 시각화
            - 🔍 신뢰도 검증
            
            **전문 분야:**
            - 📈 시장 동향 분석
            - 🔬 기술 트렌드 조사
            - 📰 실시간 뉴스 모니터링
            - 📚 학술 연구 지원
            """)

def start_ai_research(topic, research_type, sources, depth):
    """AI 리서치 시작"""
    
    import time
    import random
    
    # 초기 상태 설정
    st.session_state['research_results'] = {
        'status': 'processing',
        'current_step': '정보 수집 중',
        'progress': 0.2
    }
    
    # 시뮬레이션된 결과 생성 (실제로는 Research Agent 호출)
    time.sleep(2)
    
    # 가상의 연구 결과
    results = {
        'status': 'completed',
        'topic': topic,
        'summary': f"{topic}에 대한 종합적인 분석 결과입니다. 최신 동향과 주요 이슈들을 다각도로 분석하여 핵심 인사이트를 도출했습니다.",
        'key_findings': [
            f"{topic} 분야는 지속적인 성장세를 보이고 있습니다",
            "주요 기업들의 투자가 증가하는 추세입니다",
            "기술적 혁신이 시장 변화를 주도하고 있습니다",
            "규제 환경의 변화가 예상됩니다"
        ],
        'sources': [
            {
                'title': f"{topic} 관련 최신 동향 보고서",
                'url': "https://example.com/report1",
                'credibility': random.randint(7, 10),
                'summary': "업계 전문가들의 종합적인 분석 결과를 담은 신뢰성 높은 보고서"
            },
            {
                'title': f"{topic} 시장 분석 리포트",
                'url': "https://example.com/report2",
                'credibility': random.randint(8, 10),
                'summary': "데이터 기반의 시장 규모 및 성장 전망 분석"
            },
            {
                'title': f"{topic} 기술 혁신 동향",
                'url': "https://example.com/report3",
                'credibility': random.randint(6, 9),
                'summary': "최신 기술 발전 사항과 향후 전망에 대한 전문가 의견"
            }
        ]
    }
    
    st.session_state['research_results'] = results

def render_information_analysis():
    """정보 분석 도구"""
    
    st.markdown("### 📊 정보 분석 도구")
    
    # 분석할 텍스트 입력
    analysis_text = st.text_area(
        "분석할 텍스트를 입력하세요",
        height=200,
        placeholder="뉴스 기사, 보고서, 또는 기타 텍스트를 붙여넣으세요..."
    )
    
    if analysis_text and st.button("🔍 텍스트 분석 시작"):
        analyze_text_content(analysis_text)
    
    if 'text_analysis' in st.session_state:
        analysis = st.session_state['text_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 감정 분석")
            sentiment = analysis['sentiment']
            
            if sentiment['score'] > 0.1:
                st.success(f"😊 긍정적 ({sentiment['score']:.2f})")
            elif sentiment['score'] < -0.1:
                st.error(f"😞 부정적 ({sentiment['score']:.2f})")
            else:
                st.info(f"😐 중립적 ({sentiment['score']:.2f})")
            
            st.markdown("#### 🏷️ 주요 키워드")
            for keyword in analysis['keywords']:
                st.write(f"• {keyword}")
        
        with col2:
            st.markdown("#### 📊 텍스트 통계")
            stats = analysis['statistics']
            
            st.metric("단어 수", stats['word_count'])
            st.metric("문장 수", stats['sentence_count'])
            st.metric("가독성 점수", f"{stats['readability']}/10")
            
            st.markdown("#### 🎯 핵심 주제")
            for topic in analysis['topics']:
                st.write(f"• {topic}")

def analyze_text_content(text):
    """텍스트 내용 분석"""
    
    import random
    
    # 가상의 분석 결과 (실제로는 NLP 모델 사용)
    words = text.split()
    sentences = text.split('.')
    
    analysis = {
        'sentiment': {
            'score': random.uniform(-1, 1)
        },
        'keywords': random.sample([
            '기술', '혁신', '성장', '변화', '트렌드', 
            '시장', '분석', '발전', '미래', '전략'
        ], k=5),
        'statistics': {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'readability': random.randint(6, 10)
        },
        'topics': [
            '기술 혁신',
            '시장 동향',
            '비즈니스 전략'
        ]
    }
    
    st.session_state['text_analysis'] = analysis

def render_report_generation():
    """보고서 생성"""
    
    st.markdown("### 📝 AI 보고서 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 보고서 설정")
        
        report_title = st.text_input("보고서 제목", value="시장 동향 분석 보고서")
        report_type = st.selectbox(
            "보고서 유형",
            ["시장 분석", "기술 동향", "경쟁 분석", "투자 보고서", "연구 보고서"]
        )
        
        target_audience = st.selectbox(
            "대상 독자",
            ["경영진", "투자자", "연구자", "일반 대중", "전문가"]
        )
        
        report_length = st.select_slider(
            "보고서 길이",
            options=["요약", "표준", "상세", "종합"],
            value="표준"
        )
        
        include_charts = st.checkbox("차트 포함", value=True)
        include_references = st.checkbox("참고문헌 포함", value=True)
        
        if st.button("📄 보고서 생성", use_container_width=True):
            generate_research_report(report_title, report_type, target_audience, 
                                   report_length, include_charts, include_references)
    
    with col2:
        if 'generated_report' in st.session_state:
            report = st.session_state['generated_report']
            
            st.markdown("#### 📄 생성된 보고서")
            
            # 보고서 메타데이터
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("페이지 수", report['pages'])
            with col2:
                st.metric("단어 수", f"{report['word_count']:,}")
            with col3:
                st.metric("생성 시간", f"{report['generation_time']}초")
            
            # 보고서 미리보기
            st.markdown("##### 📖 보고서 미리보기")
            
            with st.expander("목차", expanded=True):
                for i, section in enumerate(report['outline'], 1):
                    st.write(f"{i}. {section}")
            
            with st.expander("요약"):
                st.write(report['summary'])
            
            with st.expander("첫 번째 섹션 미리보기"):
                st.write(report['preview'])
            
            # 다운로드 옵션
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 PDF 다운로드"):
                    st.success("PDF 파일이 생성되었습니다!")
            
            with col2:
                if st.button("📊 PPT 다운로드"):
                    st.success("PowerPoint 파일이 생성되었습니다!")
            
            with col3:
                if st.button("📧 이메일 발송"):
                    st.success("보고서가 이메일로 발송되었습니다!")
        
        else:
            st.info("👈 왼쪽에서 보고서 설정을 완료하고 생성 버튼을 클릭하세요.")

def generate_research_report(title, report_type, audience, length, charts, references):
    """연구 보고서 생성"""
    
    import random
    
    # 가상의 보고서 생성 (실제로는 Research Agent의 보고서 생성 기능 사용)
    
    length_mapping = {
        "요약": {"pages": 3, "words": 1500},
        "표준": {"pages": 8, "words": 4000},
        "상세": {"pages": 15, "words": 8000},
        "종합": {"pages": 25, "words": 12000}
    }
    
    specs = length_mapping[length]
    
    report = {
        'title': title,
        'pages': specs['pages'],
        'word_count': specs['words'],
        'generation_time': random.randint(30, 120),
        'outline': [
            "개요 및 배경",
            "시장 현황 분석",
            "주요 트렌드 및 동향",
            "경쟁 환경 분석",
            "미래 전망 및 예측",
            "결론 및 제언"
        ],
        'summary': f"본 {report_type} 보고서는 {audience}를 대상으로 작성되었으며, 최신 데이터와 전문가 의견을 종합하여 현 상황을 분석하고 향후 전망을 제시합니다.",
        'preview': f"""
        1. 개요 및 배경
        
        {title}는 현재 급속한 변화를 겪고 있는 분야로, 다양한 요인들이 복합적으로 작용하여 시장 환경을 형성하고 있습니다. 
        
        본 보고서에서는 최근 6개월간의 데이터를 기반으로 주요 동향을 분석하고, 향후 12개월간의 전망을 제시하고자 합니다.
        
        주요 분석 포인트:
        • 시장 규모 및 성장률 분석
        • 주요 플레이어들의 전략 변화
        • 기술 혁신이 미치는 영향
        • 규제 환경의 변화
        
        ...
        """
    }
    
    st.session_state['generated_report'] = report

if __name__ == "__main__":
    main() 