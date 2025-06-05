"""
📊 Data Generator Page

다양한 형태의 데이터 생성 및 분석 도구
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import plotly.express as px

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="📊 Data Generator",
    page_icon="📊",
    layout="wide"
)

def main():
    """Data Generator 메인 페이지"""
    
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
        <h1>📊 Data Generator</h1>
        <p style="font-size: 1.2rem; margin: 0;">
            AI 기반 다양한 형태의 데이터 생성 및 분석 도구
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 다크모드 대응 CSS
    st.markdown("""
    <style>
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    if st.button("🏠 홈으로 돌아가기", key="home"):
        st.switch_page("main.py")
    
    st.markdown("---")
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs([
        "🎲 랜덤 데이터", 
        "👥 고객 데이터", 
        "📈 시계열 데이터"
    ])
    
    with tab1:
        render_random_data_generator()
    
    with tab2:
        render_customer_data_generator()
    
    with tab3:
        render_timeseries_data_generator()

def render_random_data_generator():
    """랜덤 데이터 생성기"""
    
    st.markdown("### 🎲 랜덤 데이터 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 설정")
        
        data_type = st.selectbox(
            "데이터 타입",
            ["숫자 데이터", "텍스트 데이터", "날짜 데이터", "혼합 데이터"]
        )
        
        rows = st.number_input("행 수", min_value=10, max_value=10000, value=100)
        cols = st.number_input("열 수", min_value=1, max_value=20, value=5)
        
        if st.button("🎲 데이터 생성", use_container_width=True):
            data = generate_random_data(data_type, rows, cols)
            st.session_state['generated_data'] = data
    
    with col2:
        if 'generated_data' in st.session_state:
            st.markdown("#### 📊 생성된 데이터")
            
            data = st.session_state['generated_data']
            st.dataframe(data, use_container_width=True)
            
            # 데이터 요약
            st.markdown("#### 📈 데이터 요약")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("행 수", len(data))
            with col2:
                st.metric("열 수", len(data.columns))
            with col3:
                st.metric("크기", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")

def generate_random_data(data_type, rows, cols):
    """랜덤 데이터 생성"""
    
    if data_type == "숫자 데이터":
        data = {}
        for i in range(cols):
            if i % 3 == 0:
                data[f'정수_{i+1}'] = np.random.randint(1, 1000, rows)
            elif i % 3 == 1:
                data[f'실수_{i+1}'] = np.random.normal(100, 25, rows)
            else:
                data[f'확률_{i+1}'] = np.random.uniform(0, 1, rows)
        
    elif data_type == "텍스트 데이터":
        names = ['김철수', '이영희', '박민수', '최지영', '정성호', '이미경', '김영수', '박지은']
        cities = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기']
        companies = ['삼성', 'LG', 'SK', '현대', '롯데', 'CJ', 'GS', '한화']
        
        data = {}
        for i in range(cols):
            if i % 3 == 0:
                data[f'이름_{i+1}'] = [random.choice(names) for _ in range(rows)]
            elif i % 3 == 1:
                data[f'도시_{i+1}'] = [random.choice(cities) for _ in range(rows)]
            else:
                data[f'회사_{i+1}'] = [random.choice(companies) for _ in range(rows)]
    
    elif data_type == "날짜 데이터":
        start_date = datetime.now() - timedelta(days=365)
        data = {}
        for i in range(cols):
            dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(rows)]
            data[f'날짜_{i+1}'] = dates
    
    else:  # 혼합 데이터
        data = {
            '이름': [f'User_{i:04d}' for i in range(1, rows+1)],
            '나이': np.random.randint(20, 65, rows),
            '점수': np.random.normal(75, 15, rows),
            '등급': [random.choice(['A', 'B', 'C', 'D']) for _ in range(rows)],
            '가입일': [datetime.now() - timedelta(days=random.randint(0, 730)) for _ in range(rows)]
        }
    
    return pd.DataFrame(data)

def render_customer_data_generator():
    """고객 데이터 생성기"""
    
    st.markdown("### 👥 고객 데이터 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 고객 프로필 설정")
        
        customer_count = st.number_input("고객 수", min_value=10, max_value=5000, value=500)
        
        age_range = st.select_slider(
            "연령대",
            options=["20-30", "30-40", "40-50", "50-60", "전체"],
            value="전체"
        )
        
        include_purchase = st.checkbox("구매 이력 포함", value=True)
        include_behavior = st.checkbox("행동 데이터 포함", value=True)
        
        if st.button("👥 고객 데이터 생성", use_container_width=True):
            customers = generate_customer_data(customer_count, age_range, include_purchase, include_behavior)
            st.session_state['customer_data'] = customers
    
    with col2:
        if 'customer_data' in st.session_state:
            data = st.session_state['customer_data']
            
            st.markdown("#### 👥 생성된 고객 데이터")
            st.dataframe(data.head(20), use_container_width=True)
            
            # 고객 분석
            st.markdown("#### 📊 고객 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 연령 분포
                fig = px.histogram(data, x='나이', title='연령 분포', nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 지역별 분포
                region_counts = data['지역'].value_counts()
                fig = px.pie(values=region_counts.values, names=region_counts.index, title='지역별 분포')
                st.plotly_chart(fig, use_container_width=True)

def generate_customer_data(count, age_range, include_purchase, include_behavior):
    """고객 데이터 생성"""
    
    # 기본 정보
    names = ['김' + random.choice(['철수', '영희', '민수', '지영', '성호', '미경', '영수', '지은']) 
             for _ in range(count)]
    
    if age_range == "전체":
        ages = np.random.randint(20, 65, count)
    else:
        start, end = map(int, age_range.split('-'))
        ages = np.random.randint(start, end, count)
    
    regions = ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산']
    
    data = {
        '고객ID': [f'C{i:06d}' for i in range(1, count+1)],
        '이름': names,
        '나이': ages,
        '성별': [random.choice(['남', '여']) for _ in range(count)],
        '지역': [random.choice(regions) for _ in range(count)],
        '가입일': [datetime.now() - timedelta(days=random.randint(0, 730)) for _ in range(count)],
        '등급': [random.choice(['브론즈', '실버', '골드', '플래티넘']) for _ in range(count)]
    }
    
    if include_purchase:
        data.update({
            '총구매금액': np.random.exponential(200000, count).astype(int),
            '구매횟수': np.random.poisson(8, count),
            '평균구매금액': np.random.normal(50000, 20000, count).astype(int),
            '마지막구매일': [datetime.now() - timedelta(days=random.randint(0, 90)) for _ in range(count)]
        })
    
    if include_behavior:
        data.update({
            '웹사이트방문수': np.random.poisson(15, count),
            '평균체류시간': np.random.exponential(300, count).astype(int),  # 초
            '모바일사용률': np.random.uniform(0.3, 0.9, count),
            '이메일구독': [random.choice([True, False]) for _ in range(count)]
        })
    
    return pd.DataFrame(data)

def render_timeseries_data_generator():
    """시계열 데이터 생성기"""
    
    st.markdown("### 📈 시계열 데이터 생성")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ 시계열 설정")
        
        series_type = st.selectbox(
            "시계열 타입",
            ["매출 데이터", "주가 데이터", "방문자 수", "센서 데이터"]
        )
        
        duration = st.selectbox(
            "기간",
            ["1개월", "3개월", "6개월", "1년", "2년"]
        )
        
        frequency = st.selectbox(
            "주기",
            ["일별", "주별", "월별", "시간별"]
        )
        
        trend = st.selectbox(
            "트렌드",
            ["상승", "하락", "안정", "계절성"]
        )
        
        if st.button("📈 시계열 생성", use_container_width=True):
            ts_data = generate_timeseries_data(series_type, duration, frequency, trend)
            st.session_state['timeseries_data'] = ts_data
    
    with col2:
        if 'timeseries_data' in st.session_state:
            data = st.session_state['timeseries_data']
            
            st.markdown("#### 📈 생성된 시계열 데이터")
            
            # 시계열 차트
            fig = px.line(data, x='날짜', y='값', title='시계열 데이터')
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            st.markdown("#### 📊 통계 요약")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("평균", f"{data['값'].mean():.2f}")
            with col2:
                st.metric("표준편차", f"{data['값'].std():.2f}")
            with col3:
                st.metric("최댓값", f"{data['값'].max():.2f}")
            with col4:
                st.metric("최솟값", f"{data['값'].min():.2f}")
            
            # 데이터 테이블
            st.dataframe(data.head(20), use_container_width=True)

def generate_timeseries_data(series_type, duration, frequency, trend):
    """시계열 데이터 생성"""
    
    # 기간 설정
    duration_days = {
        "1개월": 30,
        "3개월": 90,
        "6개월": 180,
        "1년": 365,
        "2년": 730
    }
    
    # 주기 설정
    if frequency == "시간별":
        periods = duration_days[duration] * 24
    elif frequency == "일별":
        periods = duration_days[duration]
    elif frequency == "주별":
        periods = duration_days[duration] // 7
    else:  # 월별
        periods = duration_days[duration] // 30
    
    # 날짜 범위 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=duration_days[duration])
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # 기본 값 생성
    if series_type == "매출 데이터":
        base_value = 1000000
        noise_level = 0.2
    elif series_type == "주가 데이터":
        base_value = 50000
        noise_level = 0.3
    elif series_type == "방문자 수":
        base_value = 1000
        noise_level = 0.4
    else:  # 센서 데이터
        base_value = 25
        noise_level = 0.1
    
    # 트렌드 적용
    t = np.arange(len(dates))
    
    if trend == "상승":
        trend_component = base_value * (1 + 0.001 * t)
    elif trend == "하락":
        trend_component = base_value * (1 - 0.001 * t)
    elif trend == "계절성":
        trend_component = base_value * (1 + 0.3 * np.sin(2 * np.pi * t / (periods / 4)))
    else:  # 안정
        trend_component = base_value * np.ones(len(dates))
    
    # 노이즈 추가
    noise = np.random.normal(0, base_value * noise_level, len(dates))
    values = trend_component + noise
    
    # 음수 값 제거
    values = np.maximum(values, base_value * 0.1)
    
    return pd.DataFrame({
        '날짜': dates,
        '값': values
    })

if __name__ == "__main__":
    main() 