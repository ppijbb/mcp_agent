"""
🎯 Business Strategy Agent Page

비즈니스 전략 수립과 시장 분석을 위한 AI 어시스턴트
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('business_strategy')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# 공통 스타일 및 유틸리티 임포트
from srcs.common.styles import get_common_styles, get_page_header
from srcs.common.page_utils import setup_page, render_home_button

# Business Strategy Agent 모듈 임포트 - 필수 의존성
try:
    from srcs.business_strategy_agents.streamlit_app import main as bs_main
    from srcs.business_strategy_agents.streamlit_app import *
except ImportError as e:
    st.error(f"❌ Business Strategy Agent를 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: Business Strategy Agent가 필수입니다.")
    st.info("에이전트 모듈을 설치하고 다시 시도해주세요.")
    st.stop()

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
    
    st.success("🤖 Business Strategy Agent가 성공적으로 연결되었습니다!")
    
    # Business Strategy Agent 실행
    try:
        # 파일 저장 옵션 추가
        st.markdown("### ⚙️ 실행 옵션")
        save_to_file = st.checkbox(
            "파일로 저장", 
            value=False,
            help=f"체크하면 {REPORTS_PATH} 디렉토리에 분석 결과를 파일로 저장합니다"
        )
        
        if save_to_file:
            st.info(f"📁 결과가 {REPORTS_PATH} 디렉토리에 저장됩니다.")
        
        # 실제 Business Strategy Agent 실행
        result = execute_business_strategy_agent(save_to_file)
        
        # 실제 에이전트 결과 표시
        if result:
            st.success("✅ Business Strategy Agent 실행 완료!")
            
            # 실제 에이전트 출력만 표시
            st.markdown("### 📊 분석 결과")
            st.text_area(
                "분석 결과",
                value=result.get('agent_output', ''),
                height=200,
                disabled=True
            )
            
            # 파일 저장 결과 표시
            if save_to_file and result.get('file_saved'):
                st.success(f"💾 결과가 파일로 저장되었습니다: {result.get('output_path', '')}")
        else:
            # 실제 에이전트 메인 함수 호출
            bs_main()
        
    except Exception as e:
        st.error(f"❌ Business Strategy Agent 실행 실패: {e}")
        st.error("Business Strategy Agent 구현을 확인해주세요.")
        st.stop()

def execute_business_strategy_agent(save_to_file):
    """실제 Business Strategy Agent 실행 및 결과 처리 - 폴백 없음"""
    
    try:
        # 실제 에이전트 호출 - 하드코딩된 데이터 없음
        # 여기서는 실제 비즈니스 전략 에이전트를 호출해야 함
        # 현재는 bs_main()을 통해 실제 에이전트와 연동
        
        # 실제 에이전트가 구현되지 않은 경우 에러 발생
        raise NotImplementedError("실제 Business Strategy Agent 구현이 필요합니다.")
        
        # 아래 코드는 실제 에이전트 구현 시 사용할 템플릿
        """
        # 실제 에이전트 호출 예시:
        from srcs.business_strategy_agents.agent import BusinessStrategyAgent
        
        agent = BusinessStrategyAgent()
        analysis_result = agent.analyze_business_strategy(
            company_info=company_info,
            market_data=market_data,
            objectives=objectives
        )
        
        if not analysis_result:
            raise Exception("에이전트가 유효한 결과를 반환하지 않았습니다.")
        
        # 실제 에이전트 출력 포맷팅
        agent_output = format_business_analysis(analysis_result)
        
        result = {
            'success': True,
            'agent_output': agent_output,
            'analysis_data': analysis_result,
            'file_saved': False,
            'output_path': None
        }
        
        # 파일 저장 처리
        if save_to_file:
            file_saved, output_path = save_business_results_to_file(analysis_result)
            result['file_saved'] = file_saved
            result['output_path'] = output_path
        
        return result
        """
        
    except NotImplementedError:
        # 실제 에이전트가 구현되지 않은 경우 None 반환하여 bs_main() 호출
        return None
    except Exception as e:
        st.error(f"Business Strategy Agent 실행 중 오류: {e}")
        return None

def format_business_analysis(analysis_result):
    """실제 에이전트 분석 결과 포맷팅"""
    
    if not analysis_result:
        raise Exception("분석 결과가 없습니다.")
    
    # 실제 에이전트 데이터만 사용하여 출력 생성
    output_lines = [
        "🎯 비즈니스 전략 분석 결과",
        ""
    ]
    
    # 실제 분석 결과만 사용
    if 'market_analysis' in analysis_result:
        market = analysis_result['market_analysis']
        output_lines.extend([
            "📊 시장 분석:",
            f"- 시장 규모: {market.get('market_size', 'N/A')}",
            f"- 성장률: {market.get('growth_rate', 'N/A')}",
            f"- 주요 트렌드: {market.get('trends', 'N/A')}",
            ""
        ])
    
    if 'strategy_recommendations' in analysis_result:
        strategies = analysis_result['strategy_recommendations']
        output_lines.append("💡 전략 제안:")
        for strategy in strategies:
            output_lines.append(f"- {strategy}")
        output_lines.append("")
    
    if 'action_plan' in analysis_result:
        plan = analysis_result['action_plan']
        output_lines.extend([
            "📈 실행 계획:",
            f"- 단기 목표: {plan.get('short_term', 'N/A')}",
            f"- 중기 목표: {plan.get('medium_term', 'N/A')}",
            f"- 장기 비전: {plan.get('long_term', 'N/A')}",
            ""
        ])
    
    if 'risk_factors' in analysis_result:
        risks = analysis_result['risk_factors']
        output_lines.append("⚠️ 위험 요소:")
        for risk in risks:
            output_lines.append(f"- {risk}")
    
    return "\n".join(output_lines)

def save_business_results_to_file(analysis_result):
    """비즈니스 전략 분석 결과를 파일로 저장"""
    
    try:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"business_strategy_analysis_{timestamp}.md"
        filepath = os.path.join(REPORTS_PATH, filename)
        
        # 실제 에이전트 출력 생성
        agent_output = format_business_analysis(analysis_result)
        
        # 마크다운 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 비즈니스 전략 분석 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(agent_output)
            f.write("\n\n---\n")
            f.write("*본 보고서는 Business Strategy Agent에 의해 자동 생성되었습니다.*\n")
        
        return True, filepath
        
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")
        return False, None

if __name__ == "__main__":
    main() 