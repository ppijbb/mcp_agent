"""
Streamlit Web UI for Most Hooking Business Strategy Agent

This provides a user-friendly web interface to test and interact with 
the business strategy agent system without modifying core functionality.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import time
from typing import List, Dict, Any

# 기존 기능들 임포트
from .main_agent import get_main_agent, run_quick_analysis, get_agent_status
from .ai_engine import get_orchestrator, AgentRole
from .architecture import RegionType, BusinessOpportunityLevel
from .config import get_config, validate_config

# Streamlit 설정
st.set_page_config(
    page_title="Most Hooking Business Strategy Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 전역 CSS 스타일
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}
.agent-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
}
.warning-card {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
}
.error-card {
    background: #f8d7da;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """세션 상태 초기화"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []


def render_header():
    """헤더 렌더링"""
    st.markdown('<h1 class="main-header">🎯 Most Hooking Business Strategy Agent</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-Powered Global Business Intelligence & Opportunity Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # 시스템 상태 확인
        st.subheader("📊 System Status")
        
        with st.spinner("Checking system status..."):
            try:
                config = get_config()
                issues = validate_config()
                
                if issues:
                    st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                    st.warning("Configuration Issues Found:")
                    for issue in issues:
                        st.write(f"• {issue}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ System Configuration OK")
                
            except Exception as e:
                st.markdown('<div class="error-card">', unsafe_allow_html=True)
                st.error(f"System Check Failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 에이전트 상태
        st.subheader("🤖 Agent Status")
        if st.button("🔄 Refresh Agent Status"):
            try:
                status = asyncio.run(get_agent_status())
                st.session_state.agent_status = status
            except Exception as e:
                st.error(f"Failed to get agent status: {e}")
        
        if st.session_state.agent_status:
            status = st.session_state.agent_status
            st.write(f"**Running:** {'✅' if status['is_running'] else '❌'}")
            st.write(f"**MCP Servers:** {status['mcp_servers']}")
            st.write(f"**Additional Servers:** {status['additional_servers_count']}")
            
            # 성과 지표
            metrics = status['performance_metrics']
            st.write("**Performance Metrics:**")
            st.write(f"• Total Analyses: {metrics['total_analyses']}")
            st.write(f"• Success Rate: {metrics['successful_analyses']}/{metrics['total_analyses']}")
            st.write(f"• Insights Generated: {metrics['insights_generated']}")
            st.write(f"• Strategies Created: {metrics['strategies_created']}")


def render_analysis_input():
    """분석 입력 섹션"""
    st.header("🔍 Run Business Strategy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Keywords")
        default_keywords = ["AI", "startup", "fintech", "sustainability", "digital transformation"]
        
        # 사전 정의된 키워드 세트
        keyword_preset = st.selectbox(
            "Keyword Preset",
            ["Custom", "AI & Tech", "Fintech", "Sustainability", "E-commerce", "Healthcare"]
        )
        
        preset_keywords = {
            "AI & Tech": ["AI", "machine learning", "automation", "chatbot", "robotics"],
            "Fintech": ["fintech", "digital payment", "cryptocurrency", "blockchain", "neobank"],
            "Sustainability": ["sustainability", "green tech", "renewable energy", "ESG", "carbon neutral"],
            "E-commerce": ["e-commerce", "online retail", "marketplace", "logistics", "digital marketplace"],
            "Healthcare": ["digital health", "telemedicine", "health tech", "medical AI", "biotech"]
        }
        
        if keyword_preset != "Custom":
            selected_keywords = preset_keywords[keyword_preset]
        else:
            keywords_input = st.text_area(
                "Enter keywords (one per line)",
                value="\n".join(default_keywords),
                height=150
            )
            selected_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
        
        st.write(f"**Selected Keywords:** {', '.join(selected_keywords)}")
    
    with col2:
        st.subheader("🌏 Regions")
        regions = st.multiselect(
            "Select regions to analyze",
            options=[region.value for region in RegionType],
            default=[RegionType.EAST_ASIA.value, RegionType.NORTH_AMERICA.value]
        )
        
        selected_regions = [RegionType(region) for region in regions]
        
        st.subheader("⚙️ Analysis Options")
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Quick Analysis", "Deep Analysis", "Mock Test"]
        )
        
        include_trends = st.checkbox("Include Trend Analysis", value=True)
        include_strategies = st.checkbox("Generate Strategies", value=True)
    
    # 분석 실행
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not selected_keywords:
            st.error("Please enter at least one keyword")
            return
        
        if not selected_regions:
            st.error("Please select at least one region")
            return
        
        run_analysis(selected_keywords, selected_regions, analysis_mode)


@st.cache_data(ttl=300)  # 5분 캐시
def run_analysis_cached(keywords: List[str], regions: List[str], mode: str):
    """캐시된 분석 실행"""
    return run_analysis_internal(keywords, regions, mode)


def run_analysis_internal(keywords: List[str], regions: List[str], mode: str):
    """내부 분석 실행 함수"""
    try:
        region_enums = [RegionType(region) for region in regions]
        
        if mode == "Mock Test":
            # Mock 테스트 결과 생성
            return create_mock_results(keywords, region_enums)
        else:
            # 실제 분석 실행
            return asyncio.run(run_quick_analysis(keywords))
    except Exception as e:
        return {"error": str(e)}


def run_analysis(keywords: List[str], regions: List[RegionType], mode: str):
    """분석 실행 및 결과 표시"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 진행 상황 시뮬레이션
        progress_steps = [
            "Initializing agents...",
            "Collecting data from MCP servers...",
            "Analyzing trends and patterns...",
            "Detecting hooking opportunities...",
            "Generating business strategies...",
            "Finalizing results..."
        ]
        
        for i, step in enumerate(progress_steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(progress_steps))
            time.sleep(0.5)  # UI 반응성을 위한 지연
        
        # 실제 분석 실행
        region_strings = [region.value for region in regions]
        results = run_analysis_cached(keywords, region_strings, mode)
        
        progress_bar.progress(1.0)
        status_text.text("Analysis completed!")
        
        # 결과 저장
        st.session_state.analysis_results = results
        st.session_state.analysis_history.append({
            'timestamp': datetime.now(),
            'keywords': keywords,
            'regions': region_strings,
            'mode': mode,
            'success': 'error' not in results
        })
        
        # 성공 메시지
        if 'error' not in results:
            st.success(f"✅ Analysis completed! Generated {results.get('enhanced_insights_count', 0)} insights")
        else:
            st.error(f"❌ Analysis failed: {results['error']}")
        
    except Exception as e:
        st.error(f"Analysis execution failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()


def create_mock_results(keywords: List[str], regions: List[RegionType]) -> Dict[str, Any]:
    """Mock 테스트 결과 생성"""
    from .architecture import ProcessedInsight, BusinessStrategy
    
    # Mock insights 생성
    mock_insights = []
    for i, keyword in enumerate(keywords[:5]):
        insight = ProcessedInsight(
            content_id=f"mock_{keyword}_{i}",
            hooking_score=0.8 - (i * 0.1),
            business_opportunity=BusinessOpportunityLevel.HIGH if i < 2 else BusinessOpportunityLevel.MEDIUM,
            region=regions[i % len(regions)],
            category="mock_analysis",
            key_topics=[keyword, "innovation", "market_growth"],
            sentiment_score=0.75,
            trend_direction="rising",
            market_size_estimate=f"${(100 + i * 50)}B market opportunity",
            competitive_landscape=[],
            actionable_insights=[
                f"{keyword} market showing strong growth indicators",
                f"New opportunities in {keyword} sector",
                f"Competitive advantage possible in {keyword}"
            ],
            timestamp=datetime.now(timezone.utc)
        )
        mock_insights.append(insight)
    
    # Mock strategies 생성
    mock_strategies = []
    for insight in mock_insights[:3]:
        strategy = BusinessStrategy(
            strategy_id=f"strategy_{insight.content_id}",
            title=f"{insight.key_topics[0].title()} Innovation Strategy",
            opportunity_level=insight.business_opportunity,
            region=insight.region,
            category=insight.category,
            description=f"Strategic approach to capitalize on {insight.key_topics[0]} market opportunities",
            key_insights=insight.actionable_insights,
            action_items=[
                {"task": "Market research", "timeline": "2 weeks", "resources": "Research team"},
                {"task": "MVP development", "timeline": "6 weeks", "resources": "Dev team + $100K"}
            ],
            timeline="8 weeks",
            resource_requirements={"budget": "$200K", "team": "5 people"},
            roi_prediction={"expected_revenue": "$1M", "roi_percentage": "400%"},
            risk_factors=["Market competition", "Technology adoption"],
            success_metrics=["User acquisition", "Revenue growth"],
            related_trends=insight.key_topics,
            created_at=datetime.now(timezone.utc)
        )
        mock_strategies.append(strategy)
    
    return {
        'analysis_id': f"mock_{int(time.time())}",
        'success': True,
        'enhanced_insights_count': len(mock_insights),
        'regional_strategies_count': len(mock_strategies),
        'enhanced_insights': mock_insights,
        'strategies': mock_strategies,
        'duration_seconds': 2.5,
        'top_hooking_opportunities': [
            {
                'score': insight.hooking_score,
                'topics': insight.key_topics,
                'region': insight.region.value,
                'opportunity_level': insight.business_opportunity.value
            } for insight in mock_insights
        ]
    }


def render_results():
    """분석 결과 렌더링"""
    if not st.session_state.analysis_results:
        st.info("👆 Run an analysis to see results here")
        return
    
    results = st.session_state.analysis_results
    
    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    st.header("📊 Analysis Results")
    
    # 메트릭 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔍 Insights Generated",
            results.get('enhanced_insights_count', 0)
        )
    
    with col2:
        st.metric(
            "🎯 Strategies Created", 
            results.get('regional_strategies_count', 0)
        )
    
    with col3:
        duration = results.get('duration_seconds', 0)
        st.metric(
            "⏱️ Execution Time",
            f"{duration:.1f}s"
        )
    
    with col4:
        top_score = max(results.get('top_hooking_scores', [0]), default=0)
        st.metric(
            "🏆 Top Hooking Score",
            f"{top_score:.2f}"
        )
    
    # 탭으로 결과 구분
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Top Opportunities", "📈 Insights Detail", "🚀 Strategies", "📊 Analytics"])
    
    with tab1:
        render_top_opportunities(results)
    
    with tab2:
        render_insights_detail(results)
    
    with tab3:
        render_strategies(results)
    
    with tab4:
        render_analytics(results)


def render_top_opportunities(results: Dict[str, Any]):
    """상위 기회 렌더링"""
    st.subheader("🏆 Top Hooking Opportunities")
    
    opportunities = results.get('top_hooking_opportunities', [])
    
    if not opportunities:
        st.info("No opportunities found in this analysis")
        return
    
    # 기회들을 카드 형태로 표시
    for i, opp in enumerate(opportunities[:5], 1):
        score = opp['score']
        topics = ', '.join(opp['topics'][:3])
        region = opp['region']
        level = opp['opportunity_level']
        
        # 점수에 따른 색상
        if score >= 0.8:
            color = "#28a745"  # Green
        elif score >= 0.6:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
            border-left: 4px solid {color};
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
        ">
            <h4>#{i} Score: {score:.2f} | {level}</h4>
            <p><strong>Topics:</strong> {topics}</p>
            <p><strong>Region:</strong> {region}</p>
        </div>
        """, unsafe_allow_html=True)


def render_insights_detail(results: Dict[str, Any]):
    """인사이트 상세 렌더링"""
    st.subheader("📈 Detailed Insights")
    
    insights = results.get('enhanced_insights', [])
    
    if not insights:
        st.info("No detailed insights available")
        return
    
    # 인사이트 필터링
    col1, col2 = st.columns(2)
    
    with col1:
        min_score = st.slider("Minimum Hooking Score", 0.0, 1.0, 0.5, 0.1)
    
    with col2:
        region_filter = st.selectbox(
            "Filter by Region",
            ["All"] + [insight.region.value for insight in insights]
        )
    
    # 필터 적용
    filtered_insights = [
        insight for insight in insights
        if insight.hooking_score >= min_score and 
        (region_filter == "All" or insight.region.value == region_filter)
    ]
    
    st.write(f"Showing {len(filtered_insights)} of {len(insights)} insights")
    
    # 인사이트 데이터프레임
    if filtered_insights:
        insight_data = []
        for insight in filtered_insights:
            insight_data.append({
                'Content ID': insight.content_id[:20] + "...",
                'Hooking Score': insight.hooking_score,
                'Opportunity Level': insight.business_opportunity.value,
                'Region': insight.region.value,
                'Topics': ', '.join(insight.key_topics[:3]),
                'Trend Direction': insight.trend_direction,
                'Sentiment': insight.sentiment_score
            })
        
        df = pd.DataFrame(insight_data)
        st.dataframe(df, use_container_width=True)
        
        # 후킹 점수 분포
        fig = px.histogram(
            df, 
            x='Hooking Score', 
            title="Hooking Score Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)


def render_strategies(results: Dict[str, Any]):
    """전략 렌더링"""
    st.subheader("🚀 Generated Strategies")
    
    strategies = results.get('strategies', [])
    
    if not strategies:
        st.info("No strategies generated in this analysis")
        return
    
    for i, strategy in enumerate(strategies, 1):
        with st.expander(f"Strategy #{i}: {strategy.title}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Opportunity Level:** {strategy.opportunity_level.value}")
                st.write(f"**Region:** {strategy.region.value}")
                st.write(f"**Timeline:** {strategy.timeline}")
                
                st.write("**Description:**")
                st.write(strategy.description)
            
            with col2:
                st.write("**Action Items:**")
                for item in strategy.action_items:
                    st.write(f"• **{item.get('task', 'N/A')}** ({item.get('timeline', 'N/A')})")
                
                if strategy.roi_prediction:
                    st.write("**ROI Prediction:**")
                    for key, value in strategy.roi_prediction.items():
                        st.write(f"• {key.replace('_', ' ').title()}: {value}")
                
                if strategy.risk_factors:
                    st.write("**Risk Factors:**")
                    for risk in strategy.risk_factors:
                        st.write(f"⚠️ {risk}")


def render_analytics(results: Dict[str, Any]):
    """분석 통계 렌더링"""
    st.subheader("📊 Analysis Analytics")
    
    # 성과 지표
    if 'agent_performance' in results:
        st.write("**Agent Performance:**")
        
        perf_data = []
        for agent, stats in results['agent_performance'].items():
            perf_data.append({
                'Agent': agent.replace('_', ' ').title(),
                'Execution Time (s)': stats['time'],
                'Success': '✅' if stats['success'] else '❌'
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # 실행 시간 차트
        fig = px.bar(
            perf_df, 
            x='Agent', 
            y='Execution Time (s)',
            title="Agent Execution Times"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 지역별 분석
    insights = results.get('enhanced_insights', [])
    if insights:
        region_data = {}
        for insight in insights:
            region = insight.region.value
            if region not in region_data:
                region_data[region] = {'count': 0, 'avg_score': 0, 'total_score': 0}
            
            region_data[region]['count'] += 1
            region_data[region]['total_score'] += insight.hooking_score
        
        for region in region_data:
            region_data[region]['avg_score'] = region_data[region]['total_score'] / region_data[region]['count']
        
        region_df = pd.DataFrame([
            {
                'Region': region,
                'Insights Count': data['count'],
                'Average Hooking Score': data['avg_score']
            }
            for region, data in region_data.items()
        ])
        
        st.write("**Regional Analysis:**")
        st.dataframe(region_df, use_container_width=True)
    
    # 분석 히스토리
    if st.session_state.analysis_history:
        st.write("**Analysis History:**")
        
        history_df = pd.DataFrame([
            {
                'Timestamp': analysis['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Keywords': ', '.join(analysis['keywords'][:3]) + ('...' if len(analysis['keywords']) > 3 else ''),
                'Regions': ', '.join(analysis['regions']),
                'Mode': analysis['mode'],
                'Success': '✅' if analysis['success'] else '❌'
            }
            for analysis in st.session_state.analysis_history[-10:]  # 최근 10개
        ])
        
        st.dataframe(history_df, use_container_width=True)


def main():
    """메인 함수"""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    # 메인 컨텐츠
    render_analysis_input()
    
    st.divider()
    
    render_results()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🎯 Most Hooking Business Strategy Agent v1.0 | 
        Built with ❤️ using Streamlit & AI Agents
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()