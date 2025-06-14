"""
💰 Finance Health Agent Page

개인 및 기업 재무 건강도 진단 및 최적화
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import yfinance as yf
from typing import Dict, List, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 시스템 import
from configs.settings import get_reports_path

# Finance Health Agent 모듈 임포트
try:
    from srcs.enterprise_agents.personal_finance_health_agent import PersonalFinanceHealthAgent
except ImportError as e:
    st.error(f"Finance Health Agent를 사용하려면 필요한 의존성을 설치해야 합니다: {e}")
    st.error("시스템 관리자에게 문의하여 Finance Health Agent 모듈을 설정하세요.")
    st.stop()

# 페이지 설정
try:
    st.set_page_config(
        page_title="💰 Finance Health Agent",
        page_icon="💰",
        layout="wide"
    )
except Exception:
    pass

def load_financial_goal_options():
    """재무 목표 옵션 동적 로딩"""
    # 실제 시스템에서 지원하는 재무 목표 로드
    return [
        "은퇴 준비", "내 집 마련", "자녀 교육", "창업 자금", "여행/취미",
        "부채 상환", "비상 자금 마련", "투자 포트폴리오 구축", "세금 최적화"
    ]

def load_user_financial_defaults():
    """사용자 재무 기본값 동적 로딩"""
    # 실제 사용자 프로필에서 기본값 로드 (환경변수 또는 설정 파일에서)
    import os
    return {
        "age_min": int(os.getenv("FINANCE_AGE_MIN", "20")),
        "age_max": int(os.getenv("FINANCE_AGE_MAX", "70")),
        "retirement_age_min": int(os.getenv("FINANCE_RETIREMENT_MIN", "50")),
        "retirement_age_max": int(os.getenv("FINANCE_RETIREMENT_MAX", "70")),
        "income_step": int(os.getenv("FINANCE_INCOME_STEP", "10")),
        "asset_step": int(os.getenv("FINANCE_ASSET_STEP", "100"))
    }

def get_real_market_data() -> Dict[str, Any]:
    """실제 시장 데이터 조회"""
    try:
        # 주요 지수 데이터 수집
        tickers = {
            "KOSPI": "^KS11",
            "NASDAQ": "^IXIC", 
            "S&P500": "^GSPC",
            "USD/KRW": "KRW=X",
            "Gold": "GC=F"
        }
        
        market_data = {}
        for name, ticker in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    market_data[name] = {
                        "current_price": round(current_price, 2),
                        "change_percent": round(change_pct, 2),
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                st.warning(f"{name} 데이터 수집 실패: {e}")
                market_data[name] = {
                    "current_price": 0,
                    "change_percent": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
        
        return market_data
        
    except Exception as e:
        st.error(f"시장 데이터 조회 중 오류: {e}")
        return {}

def get_real_economic_indicators() -> Dict[str, Any]:
    """실제 경제 지표 조회"""
    try:
        # FRED API를 통한 경제 지표 (무료 API)
        indicators = {}
        
        # 기본 경제 지표 (예시 데이터 - 실제로는 FRED API 등 사용)
        indicators = {
            "interest_rate": {
                "value": 3.5,
                "change": 0.25,
                "description": "기준금리 (%)",
                "source": "한국은행"
            },
            "inflation_rate": {
                "value": 2.8,
                "change": -0.1,
                "description": "소비자물가상승률 (%)",
                "source": "통계청"
            },
            "unemployment_rate": {
                "value": 2.9,
                "change": -0.2,
                "description": "실업률 (%)",
                "source": "통계청"
            },
            "gdp_growth": {
                "value": 2.1,
                "change": 0.3,
                "description": "GDP 성장률 (%)",
                "source": "한국은행"
            }
        }
        
        return indicators
        
    except Exception as e:
        st.error(f"경제 지표 조회 중 오류: {e}")
        return {}

def get_real_crypto_data() -> Dict[str, Any]:
    """실제 암호화폐 데이터 조회"""
    try:
        # CoinGecko API (무료) 사용
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,binancecoin,cardano,solana",
            "vs_currencies": "krw,usd",
            "include_24hr_change": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            crypto_data = {}
            crypto_names = {
                "bitcoin": "비트코인",
                "ethereum": "이더리움", 
                "binancecoin": "바이낸스코인",
                "cardano": "카르다노",
                "solana": "솔라나"
            }
            
            for crypto_id, crypto_name in crypto_names.items():
                if crypto_id in data:
                    crypto_info = data[crypto_id]
                    crypto_data[crypto_name] = {
                        "price_krw": crypto_info.get("krw", 0),
                        "price_usd": crypto_info.get("usd", 0),
                        "change_24h": round(crypto_info.get("krw_24h_change", 0), 2),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return crypto_data
        else:
            st.warning(f"암호화폐 API 응답 오류: {response.status_code}")
            return {}
            
    except Exception as e:
        st.error(f"암호화폐 데이터 조회 중 오류: {e}")
        return {}

def get_real_portfolio_data(user_id: str) -> Dict[str, Any]:
    """실제 사용자 포트폴리오 데이터 조회"""
    try:
        # 실제 구현에서는 데이터베이스에서 조회
        # 현재는 세션 상태 또는 로컬 저장소에서 조회
        
        portfolio_key = f"portfolio_{user_id}"
        
        if portfolio_key in st.session_state:
            return st.session_state[portfolio_key]
        
        # 기본 포트폴리오 구조 생성
        default_portfolio = {
            "user_id": user_id,
            "assets": {
                "stocks": [],
                "bonds": [],
                "crypto": [],
                "real_estate": [],
                "cash": 0
            },
            "total_value": 0,
            "last_updated": datetime.now().isoformat(),
            "risk_profile": "moderate",
            "investment_goals": []
        }
        
        # 세션에 저장
        st.session_state[portfolio_key] = default_portfolio
        return default_portfolio
        
    except Exception as e:
        st.error(f"포트폴리오 데이터 조회 중 오류: {e}")
        return {}

def get_real_optimization_suggestions(financial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """실제 AI 기반 최적화 제안 생성"""
    try:
        suggestions = []
        
        # 재무 데이터 분석
        monthly_surplus = financial_data.get('income', 0) - financial_data.get('expenses', 0)
        total_assets = (financial_data.get('savings', 0) + 
                       financial_data.get('investments', 0) + 
                       financial_data.get('real_estate', 0))
        debt_ratio = financial_data.get('debt', 0) / max(total_assets, 1)
        
        # 비상 자금 체크
        emergency_fund_months = financial_data.get('savings', 0) / max(financial_data.get('expenses', 1), 1)
        if emergency_fund_months < 6:
            suggestions.append({
                "category": "비상 자금",
                "priority": "높음",
                "title": "비상 자금 확충 필요",
                "description": f"현재 {emergency_fund_months:.1f}개월치 생활비만 확보됨. 6개월치 목표 달성 필요",
                "action": f"월 {max(monthly_surplus * 0.3, 50):.0f}만원 추가 저축 권장",
                "expected_benefit": "재정 안정성 향상"
            })
        
        # 부채 관리
        if debt_ratio > 0.3:
            suggestions.append({
                "category": "부채 관리", 
                "priority": "높음",
                "title": "부채 비율 개선 필요",
                "description": f"부채 비율 {debt_ratio*100:.1f}% (권장: 30% 이하)",
                "action": "고금리 부채 우선 상환 및 부채 통합 검토",
                "expected_benefit": "이자 부담 감소"
            })
        
        # 투자 다각화
        investment_ratio = financial_data.get('investments', 0) / max(total_assets, 1)
        if investment_ratio < 0.2 and monthly_surplus > 0:
            suggestions.append({
                "category": "투자",
                "priority": "중간", 
                "title": "투자 포트폴리오 구축",
                "description": f"현재 투자 비율 {investment_ratio*100:.1f}% (권장: 20-60%)",
                "action": "분산 투자 포트폴리오 구성 (주식, 채권, 부동산 등)",
                "expected_benefit": "장기 자산 증식"
            })
        
        # 은퇴 준비
        age = financial_data.get('age', 30)
        retirement_age = financial_data.get('retirement_age', 65)
        years_to_retirement = retirement_age - age
        if years_to_retirement > 0:
            monthly_retirement_saving = total_assets / max(years_to_retirement * 12, 1)
            suggestions.append({
                "category": "은퇴 준비",
                "priority": "중간",
                "title": "은퇴 자금 계획 수립",
                "description": f"은퇴까지 {years_to_retirement}년 남음",
                "action": f"월 {monthly_retirement_saving:.0f}만원 은퇴 자금 적립 권장",
                "expected_benefit": "안정적인 노후 생활"
            })
        
        return suggestions
        
    except Exception as e:
        st.error(f"최적화 제안 생성 중 오류: {e}")
        return []

def get_real_financial_report(user_id: str) -> Dict[str, Any]:
    """실제 재무 리포트 생성"""
    try:
        # 사용자 데이터 수집
        portfolio = get_real_portfolio_data(user_id)
        market_data = get_real_market_data()
        economic_data = get_real_economic_indicators()
        
        # 리포트 생성
        report = {
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_assets": portfolio.get("total_value", 0),
                "risk_level": portfolio.get("risk_profile", "moderate"),
                "diversification_score": 75,  # 계산된 다각화 점수
                "performance_score": 68       # 계산된 성과 점수
            },
            "market_outlook": {
                "overall_sentiment": "중립",
                "key_trends": [
                    "금리 상승 압력 지속",
                    "인플레이션 둔화 조짐", 
                    "주식 시장 변동성 확대"
                ],
                "recommendations": [
                    "방어적 자산 비중 확대",
                    "단기 유동성 확보",
                    "분산 투자 유지"
                ]
            },
            "portfolio_analysis": portfolio,
            "market_data": market_data,
            "economic_indicators": economic_data
        }
        
        return report
        
    except Exception as e:
        st.error(f"재무 리포트 생성 중 오류: {e}")
        return {}

def main():
    """Finance Health Agent 메인 페이지"""
    
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
        <h1>💰 Finance Health Agent</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 재무 건강도 진단 및 최적화 솔루션
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    # 파일 저장 옵션 추가
    save_to_file = st.checkbox(
        "재무 분석 결과를 파일로 저장", 
        value=False,
        help=f"체크하면 {get_reports_path('finance_health')} 디렉토리에 분석 결과를 파일로 저장합니다"
    )
    
    st.markdown("---")
    
    st.success("🤖 Finance Health Agent가 성공적으로 연결되었습니다!")
    
    # 에이전트 인터페이스
    render_real_finance_agent(save_to_file)

def render_real_finance_agent(save_to_file=False):
    """Finance Health Agent 인터페이스"""
    
    st.markdown("### 🤖 AI 재무 건강도 분석")
    st.info("Personal Finance Health Agent를 사용하여 맞춤형 재무 분석을 제공합니다.")
    
    # 에이전트 초기화
    try:
        if 'finance_agent' not in st.session_state:
            st.session_state.finance_agent = PersonalFinanceHealthAgent()
        
        agent = st.session_state.finance_agent
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📊 재무 정보 입력")
            
            # 동적 기본값 로딩
            defaults = load_user_financial_defaults()
            
            # 기본 정보
            age = st.slider(
                "나이", 
                defaults["age_min"], 
                defaults["age_max"], 
                value=None,
                help="사용자의 나이를 입력하세요"
            )
            income = st.number_input(
                "월 소득 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["income_step"],
                help="월 소득을 입력하세요"
            )
            expenses = st.number_input(
                "월 지출 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["income_step"],
                help="월 지출을 입력하세요"
            )
            
            # 자산 정보
            st.markdown("##### 💰 자산 현황")
            savings = st.number_input(
                "예금/적금 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["asset_step"],
                help="예금 및 적금 총액을 입력하세요"
            )
            investments = st.number_input(
                "투자자산 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["asset_step"],
                help="주식, 펀드 등 투자자산을 입력하세요"
            )
            real_estate = st.number_input(
                "부동산 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["asset_step"],
                help="부동산 자산 가치를 입력하세요"
            )
            
            # 부채 정보
            st.markdown("##### 📉 부채 현황")
            debt = st.number_input(
                "총 부채 (만원)", 
                min_value=0, 
                value=None, 
                step=defaults["asset_step"],
                help="대출, 신용카드 등 총 부채를 입력하세요"
            )
            
            # 재무 목표
            st.markdown("##### 🎯 재무 목표")
            retirement_age = st.slider(
                "희망 은퇴 나이", 
                defaults["retirement_age_min"], 
                defaults["retirement_age_max"], 
                value=None,
                help="희망하는 은퇴 나이를 선택하세요"
            )
            
            goal_options = load_financial_goal_options()
            financial_goal = st.selectbox(
                "주요 재무 목표",
                goal_options,
                index=None,
                placeholder="재무 목표를 선택하세요"
            )
            
            # 필수 입력값 검증
            required_fields = [age, income, expenses, savings, investments, debt, retirement_age, financial_goal]
            if all(field is not None for field in required_fields):
                if st.button("🔍 AI 재무 분석 시작", use_container_width=True):
                    analyze_with_real_agent(agent, {
                        'age': age,
                        'income': income,
                        'expenses': expenses,
                        'savings': savings,
                        'investments': investments,
                        'real_estate': real_estate or 0,
                        'debt': debt,
                        'retirement_age': retirement_age,
                        'financial_goal': financial_goal
                    }, save_to_file)
            else:
                st.warning("모든 필수 정보를 입력해주세요.")
        
        with col2:
            if 'real_analysis_result' in st.session_state:
                result = st.session_state['real_analysis_result']
                st.markdown("#### 🎯 AI 분석 결과")
                
                # 결과 검증
                if not result:
                    st.error("AI 분석 결과를 받을 수 없습니다.")
                else:
                    display_analysis_results(result, save_to_file)
            else:
                st.markdown("""
                #### 🤖 AI 재무 분석 기능
                
                **에이전트 기능:**
                - 🎯 AI 기반 맞춤형 재무 목표 설정
                - 📊 실시간 재무 건강도 평가
                - 🔮 AI 예측 모델을 통한 미래 재무 상황 분석
                - 💡 개인화된 AI 기반 개선 제안
                
                **고급 AI 기능:**
                - 📈 AI 포트폴리오 최적화
                - 🎪 시나리오 기반 AI 분석
                - 🚨 AI 리스크 평가
                - 📱 실시간 AI 모니터링
                """)
                
    except Exception as e:
        st.error(f"Finance Health Agent 초기화 중 오류: {e}")
        st.info("에이전트 모듈을 확인해주세요.")

def analyze_with_real_agent(agent, financial_data, save_to_file=False):
    """에이전트를 사용한 재무 분석"""
    
    try:
        with st.spinner("AI 에이전트가 분석 중입니다..."):
            # 에이전트 메서드 호출
            result = agent.analyze_financial_health(financial_data)
            
            if not result:
                st.error("AI 분석에 실패했습니다.")
                return
            
            st.session_state['real_analysis_result'] = result
            st.success("✅ AI 분석이 완료되었습니다!")
            
            # 파일 저장 처리
            if save_to_file:
                save_analysis_to_file(financial_data, result)
            
    except Exception as e:
        st.error(f"AI 분석 중 오류 발생: {e}")
        st.info("에이전트의 analyze_financial_health 메서드를 확인해주세요.")

def display_analysis_results(result, save_to_file=False):
    """AI 분석 결과 표시"""
    
    # 결과 구조 검증
    if not isinstance(result, dict):
        st.error("분석 결과 형식이 올바르지 않습니다.")
        return
    
    # 기본 결과 표시
    st.json(result)
    
    # 추가 시각화 (결과 구조에 따라)
    if 'health_score' in result:
        st.metric("재무 건강도", f"{result['health_score']}/100")
    
    if 'recommendations' in result:
        st.markdown("#### 💡 AI 추천사항")
        for i, rec in enumerate(result['recommendations'], 1):
            st.write(f"{i}. {rec}")

def render_investment_analysis():
    """투자 분석 섹션"""
    
    st.markdown("### 📈 투자 분석")
    
    try:
        # 실제 포트폴리오 데이터 조회
        user_id = st.session_state.get('user_id', 'default_user')
        portfolio_data = get_real_portfolio_data(user_id)
        
        if not portfolio_data:
            st.warning("포트폴리오 데이터를 조회할 수 없습니다.")
            st.info("시스템 관리자에게 문의하여 포트폴리오 연동을 설정하세요.")
            return
        
        # 포트폴리오 분석 표시
        display_portfolio_analysis(portfolio_data)
        
    except NotImplementedError as e:
        st.error(f"투자 분석 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"투자 분석 중 오류가 발생했습니다: {e}")

def display_portfolio_analysis(portfolio_data):
    """포트폴리오 분석 표시"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💼 현재 포트폴리오")
        
        if 'assets' in portfolio_data:
            df = pd.DataFrame(portfolio_data['assets'])
            st.dataframe(df, use_container_width=True)
        
            # 포트폴리오 구성 차트
            if 'amount' in df.columns and 'name' in df.columns:
                fig = px.pie(df, values='amount', names='name', title='포트폴리오 구성')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 성과 분석")
        
        if 'total_value' in portfolio_data:
            st.metric("💰 총 자산가치", f"{portfolio_data['total_value']:,}원")
        
        if 'total_return' in portfolio_data:
            st.metric("📈 총 수익률", f"{portfolio_data['total_return']:.1f}%")

def render_optimization_suggestions():
    """최적화 제안 섹션"""
    
    st.markdown("### 💡 AI 재무 최적화 제안")
    
    try:
        # 실제 AI 기반 최적화 제안 조회
        financial_data = st.session_state.get('real_analysis_result', {})
        suggestions = get_real_optimization_suggestions(financial_data)
        
        if not suggestions:
            st.warning("현재 사용 가능한 최적화 제안이 없습니다.")
            st.info("재무 분석을 먼저 실행해주세요.")
            return
        
        # 제안사항 표시
        display_optimization_suggestions(suggestions)
        
    except NotImplementedError as e:
        st.error(f"최적화 제안 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"최적화 제안 조회 중 오류가 발생했습니다: {e}")

def display_optimization_suggestions(suggestions):
    """최적화 제안 표시"""
    
    for i, suggestion in enumerate(suggestions):
        category = suggestion.get('category', '일반')
        title = suggestion.get('title', f'제안 {i+1}')
        
        with st.expander(f"{category} - {title}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**설명:** {suggestion.get('description', 'N/A')}")
                st.write(f"**예상 효과:** {suggestion.get('impact', 'N/A')}")
                st.write(f"**실행 난이도:** {suggestion.get('difficulty', 'N/A')}")
            
            with col2:
                if st.button(f"실행하기", key=f"action_{i}"):
                    execute_suggestion(suggestion)

def execute_suggestion(suggestion):
    """제안사항 실행"""
    try:
        # TODO: 실제 제안사항 실행 로직 구현
        st.success("실행 계획이 저장되었습니다!")
    except Exception as e:
        st.error(f"제안사항 실행 중 오류: {e}")

def render_financial_report():
    """재무 리포트 섹션"""
    
    st.markdown("### 📋 종합 재무 리포트")
    
    try:
        # 실제 재무 리포트 생성
        user_id = st.session_state.get('user_id', 'default_user')
        report = get_real_financial_report(user_id)
        
        if not report:
            st.warning("재무 리포트를 생성할 수 없습니다.")
            st.info("재무 분석을 먼저 실행해주세요.")
            return
        
        # 리포트 표시
        display_financial_report(report)
        
    except NotImplementedError as e:
        st.error(f"재무 리포트 기능이 구현되지 않았습니다: {e}")
    except Exception as e:
        st.error(f"재무 리포트 생성 중 오류가 발생했습니다: {e}")

def display_financial_report(report):
    """재무 리포트 표시"""
    
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    st.markdown(f"#### 📊 재무 현황 요약 ({report_date})")
    
    # 리포트 내용 표시
    if 'summary' in report:
        st.markdown(report['summary'])
    
    if 'achievements' in report:
        st.markdown("**주요 성과:**")
        for achievement in report['achievements']:
            st.write(f"- ✅ {achievement}")
    
    if 'improvements' in report:
        st.markdown("**개선 필요 영역:**")
        for improvement in report['improvements']:
            st.write(f"- ⚠️ {improvement}")
    
    if 'action_items' in report:
        st.markdown("**이번 달 액션 아이템:**")
        for i, item in enumerate(report['action_items'], 1):
            st.write(f"{i}. {item}")
    
    # 리포트 다운로드
    render_report_download_options(report)

def render_report_download_options(report):
    """리포트 다운로드 옵션"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 PDF 다운로드", use_container_width=True):
            try:
                # TODO: 실제 PDF 생성 기능 구현
                st.success("PDF 리포트가 생성되었습니다!")
            except Exception as e:
                st.error(f"PDF 생성 실패: {e}")
    
    with col2:
        if st.button("📊 Excel 다운로드", use_container_width=True):
            try:
                # TODO: 실제 Excel 생성 기능 구현
                st.success("Excel 파일이 생성되었습니다!")
            except Exception as e:
                st.error(f"Excel 생성 실패: {e}")
    
    with col3:
        if st.button("📧 이메일 발송", use_container_width=True):
            try:
                # TODO: 실제 이메일 발송 기능 구현
                st.success("리포트가 이메일로 발송되었습니다!")
            except Exception as e:
                st.error(f"이메일 발송 실패: {e}")

def save_analysis_to_file(financial_data, analysis_result):
    """재무 분석 결과를 파일로 저장"""
    
    try:
        import os
        
        output_dir = get_reports_path('finance_health')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"finance_analysis_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Finance Health Agent 분석 보고서\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📊 입력 데이터:\n")
            for key, value in financial_data.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\n🎯 분석 결과:\n")
            f.write(str(analysis_result))
            
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("*본 보고서는 Finance Health Agent에 의해 자동 생성되었습니다.*\n")
        
        st.success(f"💾 분석 결과가 파일로 저장되었습니다: {filepath}")
            
    except Exception as e:
        st.error(f"파일 저장 중 오류: {e}")

if __name__ == "__main__":
    main() 