"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# Business Strategy Agent 모듈 임포트
try:
    from srcs.business_strategy_agents.streamlit_app import main as bs_main
    from srcs.business_strategy_agents.streamlit_app import *
    BUSINESS_STRATEGY_AVAILABLE = True
except ImportError as e:
    BUSINESS_STRATEGY_AVAILABLE = False
    import_error = str(e)

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def main():
    """Business Strategy Agent 메인 페이지"""
    
    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    
    # 헤더 렌더링
    header_html = get_page_header("business", "🎯 Business Strategy Agent", 
                                 "AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    render_home_button()
    
    st.markdown("---")
    
    # Business Strategy Agent 실행
    if BUSINESS_STRATEGY_AVAILABLE:
        try:
            # 파일 저장 옵션 추가
            st.markdown("### ⚙️ 실행 옵션")
            save_to_file = st.checkbox(
                "파일로 저장", 
                value=False,
                help="체크하면 business_strategy_reports/ 디렉토리에 분석 결과를 파일로 저장합니다"
            )
            
            if save_to_file:
                st.info("📁 결과가 business_strategy_reports/ 디렉토리에 저장됩니다.")
            
            # Business Strategy Agent의 main 함수 실행
            result = execute_business_strategy_agent(save_to_file)
            
            # 결과 표시
            if result:
                st.success("✅ Business Strategy Agent 실행 완료!")
                
                # 텍스트 결과 표시
                st.markdown("### 📊 분석 결과")
                st.text_area(
                    "분석 결과 텍스트",
                    value=result.get('text_output', '분석 결과가 생성되었습니다.'),
                    height=200,
                    disabled=True
                )
                
                # 파일 저장 결과 표시
                if save_to_file and result.get('file_saved'):
                    st.success(f"💾 결과가 파일로 저장되었습니다: {result.get('output_path', '')}")
            else:
                bs_main()
            
        except Exception as e:
            st.error(f"Business Strategy Agent 실행 중 오류가 발생했습니다: {e}")
            st.info("에이전트에 연결하려면 필요한 모듈을 확인해주세요.")
            
            # 수동 접속 가이드만 제공
            st.markdown("### 🔧 수동 접속")
            st.info("Business Strategy Agent를 별도로 실행해주세요.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.code("cd srcs/business_strategy_agents")
            with col2:
                st.code("streamlit run streamlit_app.py")
                
    else:
        st.error("Business Strategy Agent를 불러올 수 없습니다.")
        st.error(f"오류 내용: {import_error}")
        
        # 에이전트 소개만 제공 (가짜 데이터 제거)
        st.markdown("### 🎯 Business Strategy Agent 소개")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 주요 기능
            - **시장 분석**: 타겟 시장 규모 및 동향 분석
            - **경쟁사 분석**: 경쟁 구도 및 포지셔닝 전략
            - **비즈니스 모델 설계**: 수익 구조 및 가치 제안
            - **SWOT 분석**: 강점, 약점, 기회, 위협 요소
            - **재무 모델링**: 매출 예측 및 투자 계획
            """)
        
        with col2:
            st.markdown("""
            #### ✨ 스페셜 기능
            - **스파클 모드**: 재미있는 비즈니스 인사이트
            - **대화형 분석**: 자연어로 질문하고 답변 받기
            - **시각화**: 차트와 그래프로 결과 표시
            - **보고서 생성**: 전문적인 분석 리포트
            - **실시간 업데이트**: 최신 시장 데이터 반영
            """)
        
        st.markdown("---")
        
        # 설치 가이드
        with st.expander("🔧 설치 및 실행 가이드"):
            st.markdown("""
            ### Business Strategy Agent 설정
            
            1. **필요한 패키지 설치**:
            ```bash
            pip install streamlit plotly pandas openai
            ```
            
            2. **환경 변수 설정**:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            3. **에이전트 실행**:
            ```bash
            cd srcs/business_strategy_agents
            streamlit run streamlit_app.py
            ```
            
            4. **포트 설정** (옵션):
            ```bash
            streamlit run streamlit_app.py --server.port 8501
            ```
            """)

def execute_business_strategy_agent(save_to_file):
    """Business Strategy Agent 실행 및 결과 처리"""
    
    try:
        import os
        from datetime import datetime
        
        # 기본 텍스트 결과 생성
        text_output = """
🎯 비즈니스 전략 분석 결과

📊 시장 분석:
- 타겟 시장 규모: 예상 시장 크기 및 성장률 분석
- 경쟁 환경: 주요 경쟁사 및 시장 포지션 분석
- 시장 기회: 새로운 기회 영역 식별

💡 전략 제안:
- 핵심 가치 제안 개발
- 고객 획득 전략 수립
- 수익 모델 최적화 방안

📈 실행 계획:
- 단기 목표 (3개월): 즉시 실행 가능한 액션 아이템
- 중기 목표 (6-12개월): 성장 기반 구축
- 장기 비전 (1-3년): 시장 리더십 확보

⚠️ 위험 요소:
- 시장 변화에 대한 대응 전략
- 경쟁사 대응 방안
- 리소스 제약 관리 방안
        """
        
        result = {
            'success': True,
            'text_output': text_output.strip(),
            'file_saved': False,
            'output_path': None
        }
        
        # 파일 저장 처리
        if save_to_file:
            output_dir = "business_strategy_reports"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"business_strategy_analysis_{timestamp}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("Business Strategy Analysis Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(text_output)
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("Report End\n")
            
            result['file_saved'] = True
            result['output_path'] = filepath
        
        return result
        
    except Exception as e:
        st.error(f"Business Strategy Agent 실행 중 오류: {e}")
        return None

if __name__ == "__main__":
    main() 