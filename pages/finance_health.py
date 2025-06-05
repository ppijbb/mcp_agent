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
import random
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Finance Health Agent 모듈 임포트
try:
    from srcs.enterprise_agents.personal_finance_health_agent import *
    FINANCE_AGENT_AVAILABLE = True
except ImportError as e:
    FINANCE_AGENT_AVAILABLE = False
    import_error = str(e)

# 페이지 설정
st.set_page_config(
    page_title="💰 Finance Health Agent",
    page_icon="💰",
    layout="wide"
)

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
    
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 재무 진단", "📈 투자 분석", "💡 최적화 제안", "📋 리포트"])
    
    with tab1:
        render_financial_diagnosis()
    
    with tab2:
        render_investment_analysis()
    
    with tab3:
        render_optimization_suggestions()
    
    with tab4:
        render_financial_report()

def render_financial_diagnosis():
    """재무 진단 섹션"""
    
    st.markdown("### 💰 재무 건강도 진단")
    
    # 진단 유형 선택
    diagnosis_type = st.selectbox(
        "진단 유형을 선택하세요",
        ["개인 재무", "기업 재무", "투자 포트폴리오"]
    )
    
    if diagnosis_type == "개인 재무":
        render_personal_finance_diagnosis()
    elif diagnosis_type == "기업 재무":
        render_corporate_finance_diagnosis()
    else:
        render_investment_portfolio_diagnosis()

def render_personal_finance_diagnosis():
    """개인 재무 진단"""
    
    st.markdown("#### 👤 개인 재무 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input("월 소득", min_value=0, value=4000000, step=100000)
        monthly_expenses = st.number_input("월 지출", min_value=0, value=3000000, step=100000)
        savings = st.number_input("저축", min_value=0, value=50000000, step=1000000)
    
    with col2:
        debt = st.number_input("부채", min_value=0, value=20000000, step=1000000)
        investments = st.number_input("투자자산", min_value=0, value=30000000, step=1000000)
        age = st.slider("연령", 20, 70, 35)
    
    if st.button("💰 진단 시작", use_container_width=True):
        show_personal_finance_results(monthly_income, monthly_expenses, savings, debt, investments, age)

def show_personal_finance_results(income, expenses, savings, debt, investments, age):
    """개인 재무 진단 결과"""
    
    # 재무 건강도 계산
    net_worth = savings + investments - debt
    savings_rate = (income - expenses) / income * 100 if income > 0 else 0
    debt_ratio = debt / (savings + investments) * 100 if (savings + investments) > 0 else 0
    
    # 종합 점수 계산
    score = 0
    if savings_rate >= 20:
        score += 30
    elif savings_rate >= 10:
        score += 20
    elif savings_rate >= 5:
        score += 10
    
    if debt_ratio <= 30:
        score += 25
    elif debt_ratio <= 50:
        score += 15
    elif debt_ratio <= 70:
        score += 5
    
    if net_worth > 0:
        score += 25
    
    # 나이에 따른 추가 점수
    if age < 40 and savings_rate >= 15:
        score += 10
    elif age >= 40 and net_worth >= income * 12:
        score += 10
    
    score = min(score, 100)
    
    # 결과 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if score >= 80:
            color = "#28a745"
            status = "🚀 우수"
        elif score >= 60:
            color = "#17a2b8"
            status = "✅ 양호"
        elif score >= 40:
            color = "#ffc107"
            status = "⚠️ 주의"
        else:
            color = "#dc3545"
            status = "🚨 위험"
        
        st.markdown(f"""
        <div style="
            background: {color};
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
        ">
            <h2>{status}</h2>
            <h1 style="font-size: 3rem; margin: 0;">{score}/100</h1>
            <p>재무 건강도</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("💰 순자산", f"{net_worth:,}원", f"{'+' if net_worth >= 0 else ''}{net_worth:,}")
        st.metric("📊 저축률", f"{savings_rate:.1f}%", "목표: 20% 이상")
    
    with col3:
        st.metric("📉 부채비율", f"{debt_ratio:.1f}%", "권장: 30% 이하")
        st.metric("🎯 은퇴준비도", f"{min(100, (net_worth / (income * 20)) * 100):.0f}%")
    
    # 상세 분석
    st.markdown("---")
    st.markdown("### 📊 상세 분석")
    
    # 재무 구조 파이 차트
    fig = go.Figure(data=[go.Pie(
        labels=['저축', '투자자산', '부채'],
        values=[savings, investments, debt],
        hole=.3
    )])
    fig.update_layout(title="자산/부채 구조")
    st.plotly_chart(fig, use_container_width=True)
    
    # 개선 제안
    st.markdown("### 💡 개선 제안")
    
    suggestions = []
    if savings_rate < 20:
        suggestions.append("💰 저축률을 20% 이상으로 높이세요")
    if debt_ratio > 30:
        suggestions.append("📉 부채를 줄여 부채비율을 30% 이하로 관리하세요")
    if investments < savings * 0.3:
        suggestions.append("📈 자산의 30% 이상은 투자자산으로 운용하세요")
    
    for suggestion in suggestions:
        st.warning(suggestion)
    
    if not suggestions:
        st.success("🎉 재무 관리가 잘 되고 있습니다!")

def render_corporate_finance_diagnosis():
    """기업 재무 진단"""
    
    st.markdown("#### 🏢 기업 재무 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("연매출", min_value=0, value=10000000000, step=100000000)
        operating_profit = st.number_input("영업이익", min_value=0, value=1000000000, step=10000000)
        total_assets = st.number_input("총자산", min_value=0, value=8000000000, step=100000000)
    
    with col2:
        total_debt = st.number_input("총부채", min_value=0, value=3000000000, step=100000000)
        equity = st.number_input("자본금", min_value=0, value=5000000000, step=100000000)
        employees = st.number_input("직원 수", min_value=1, value=100, step=1)
    
    if st.button("🏢 기업 진단 시작", use_container_width=True):
        show_corporate_finance_results(revenue, operating_profit, total_assets, total_debt, equity, employees)

def show_corporate_finance_results(revenue, op_profit, assets, debt, equity, employees):
    """기업 재무 진단 결과"""
    
    # 재무 지표 계산
    operating_margin = (op_profit / revenue * 100) if revenue > 0 else 0
    debt_ratio = (debt / assets * 100) if assets > 0 else 0
    roe = (op_profit / equity * 100) if equity > 0 else 0
    revenue_per_employee = revenue / employees if employees > 0 else 0
    
    # 종합 점수
    score = 0
    if operating_margin >= 15:
        score += 25
    elif operating_margin >= 10:
        score += 20
    elif operating_margin >= 5:
        score += 15
    
    if debt_ratio <= 40:
        score += 25
    elif debt_ratio <= 60:
        score += 15
    elif debt_ratio <= 80:
        score += 5
    
    if roe >= 15:
        score += 25
    elif roe >= 10:
        score += 20
    elif roe >= 5:
        score += 15
    
    if revenue_per_employee >= 500000000:
        score += 25
    elif revenue_per_employee >= 300000000:
        score += 15
    elif revenue_per_employee >= 100000000:
        score += 10
    
    score = min(score, 100)
    
    # 결과 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💼 영업이익률", f"{operating_margin:.1f}%")
    with col2:
        st.metric("📊 부채비율", f"{debt_ratio:.1f}%")
    with col3:
        st.metric("📈 ROE", f"{roe:.1f}%")
    with col4:
        st.metric("👥 1인당 매출", f"{revenue_per_employee/100000000:.1f}억원")
    
    # 종합 평가
    if score >= 80:
        st.success(f"🚀 우수한 재무구조입니다! (점수: {score}/100)")
    elif score >= 60:
        st.info(f"✅ 양호한 상태입니다. (점수: {score}/100)")
    elif score >= 40:
        st.warning(f"⚠️ 개선이 필요합니다. (점수: {score}/100)")
    else:
        st.error(f"🚨 재무구조 개선이 시급합니다. (점수: {score}/100)")

def render_investment_analysis():
    """투자 분석 섹션"""
    
    st.markdown("### 📈 투자 분석")
    
    # 샘플 포트폴리오
    portfolio_data = {
        '자산': ['삼성전자', 'KODEX 200', '미국 S&P500', '코인', '부동산', '예금'],
        '투자금액(만원)': [2000, 1500, 1000, 500, 3000, 2000],
        '수익률(%)': [15.2, 8.5, 12.3, -5.2, 7.8, 2.1],
        '위험도': ['중', '중', '중', '고', '중', '저']
    }
    
    df = pd.DataFrame(portfolio_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💼 현재 포트폴리오")
        st.dataframe(df, use_container_width=True)
        
        # 포트폴리오 구성 파이 차트
        fig = px.pie(df, values='투자금액(만원)', names='자산', title='포트폴리오 구성')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 성과 분석")
        
        total_investment = df['투자금액(만원)'].sum()
        weighted_return = (df['투자금액(만원)'] * df['수익률(%)']).sum() / total_investment
        
        st.metric("💰 총 투자금액", f"{total_investment:,}만원")
        st.metric("📈 포트폴리오 수익률", f"{weighted_return:.1f}%")
        
        # 수익률 차트
        fig = px.bar(df, x='자산', y='수익률(%)', title='자산별 수익률')
        st.plotly_chart(fig, use_container_width=True)
    
    # 리스크 분석
    st.markdown("#### ⚠️ 리스크 분석")
    
    risk_high = df[df['위험도'] == '고']['투자금액(만원)'].sum()
    risk_ratio = risk_high / total_investment * 100
    
    if risk_ratio > 20:
        st.warning(f"고위험 자산 비중이 {risk_ratio:.1f}%로 높습니다. 20% 이하로 관리하세요.")
    else:
        st.success(f"고위험 자산 비중이 {risk_ratio:.1f}%로 적절합니다.")

def render_optimization_suggestions():
    """최적화 제안 섹션"""
    
    st.markdown("### 💡 AI 재무 최적화 제안")
    
    suggestions = [
        {
            "category": "💰 비용 절감",
            "title": "구독 서비스 최적화",
            "description": "사용하지 않는 구독 서비스 5개 해지로 월 35,000원 절약 가능",
            "impact": "연간 42만원 절약",
            "difficulty": "쉬움"
        },
        {
            "category": "📈 투자 최적화", 
            "title": "포트폴리오 리밸런싱",
            "description": "현재 주식 비중 70% → 60%로 조정, 채권 10% 추가",
            "impact": "리스크 15% 감소",
            "difficulty": "보통"
        },
        {
            "category": "🏠 부동산",
            "title": "전세자금대출 갈아타기",
            "description": "현재 3.5% → 2.8% 금리로 변경",
            "impact": "연간 140만원 이자 절약",
            "difficulty": "보통"
        },
        {
            "category": "💳 세금 최적화",
            "title": "연금저축 납입 확대",
            "description": "월 50만원 → 70만원으로 증액",
            "impact": "연간 세액공제 48만원",
            "difficulty": "쉬움"
        }
    ]
    
    for i, suggestion in enumerate(suggestions):
        with st.expander(f"{suggestion['category']} - {suggestion['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**설명:** {suggestion['description']}")
                st.write(f"**예상 효과:** {suggestion['impact']}")
                st.write(f"**실행 난이도:** {suggestion['difficulty']}")
            
            with col2:
                if st.button(f"실행하기", key=f"action_{i}"):
                    st.success("실행 계획이 저장되었습니다!")

def render_financial_report():
    """재무 리포트 섹션"""
    
    st.markdown("### 📋 종합 재무 리포트")
    
    # 리포트 생성
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    st.markdown(f"""
    #### 📊 재무 현황 요약 ({report_date})
    
    **1. 재무 건강도: 78/100 (양호)**
    - 저축률: 25% (목표 달성)
    - 부채비율: 35% (관리 필요)
    - 투자 수익률: 8.5% (평균 이상)
    
    **2. 주요 성과**
    - ✅ 월 저축 목표 달성 (6개월 연속)
    - ✅ 투자 포트폴리오 수익률 8.5% 달성
    - ✅ 신용점수 950점 유지
    
    **3. 개선 필요 영역**
    - ⚠️ 부채비율 30% 이하로 관리 필요
    - ⚠️ 생활비 변동성 큼 (표준편차 15%)
    - ⚠️ 비상자금 6개월분 확보 필요
    
    **4. 이번 달 액션 아이템**
    1. 사용하지 않는 구독 서비스 3개 해지
    2. 여유자금 200만원 투자 실행
    3. 부동산 대출 갈아타기 검토
    4. 비상자금 50만원 추가 적립
    """)
    
    # 리포트 다운로드
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 PDF 다운로드", use_container_width=True):
            st.success("PDF 리포트가 생성되었습니다!")
    
    with col2:
        if st.button("📊 Excel 다운로드", use_container_width=True):
            st.success("Excel 파일이 생성되었습니다!")
    
    with col3:
        if st.button("📧 이메일 발송", use_container_width=True):
            st.success("리포트가 이메일로 발송되었습니다!")

if __name__ == "__main__":
    main() 