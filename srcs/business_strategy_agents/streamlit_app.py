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
import sys
from pathlib import Path
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 설정 파일에서 경로 가져오기
try:
    from configs.settings import get_reports_path
    REPORTS_PATH = get_reports_path('business_strategy')
except ImportError:
    st.error("❌ 설정 파일을 찾을 수 없습니다. configs/settings.py를 확인해주세요.")
    st.stop()

# 공통 스타일 및 유틸리티 임포트
try:
    from srcs.common.styles import get_common_styles, get_page_header
    from srcs.common.page_utils import setup_page, render_home_button
except ImportError:
    st.error("❌ 공통 스타일 모듈을 찾을 수 없습니다.")
    st.stop()

# 기존 기능들 임포트
try:
    from .main_agent import get_main_agent, run_quick_analysis, get_agent_status
    from .ai_engine import get_orchestrator, AgentRole
    from .architecture import RegionType, BusinessOpportunityLevel
    from .config import get_config, validate_config
except ImportError as e:
    st.error(f"❌ Business Strategy Agent 모듈을 불러올 수 없습니다: {e}")
    st.error("**시스템 요구사항**: Business Strategy Agent가 필수입니다.")
    st.stop()

# 페이지 설정
setup_page("🎯 Business Strategy Agent", "🎯")

def initialize_session_state():
    """세션 상태 초기화"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def render_header():
    """헤더 렌더링 - pages 스타일 적용"""
    # 공통 스타일 적용
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    
    # 헤더 렌더링
    header_html = get_page_header("business", "🎯 Business Strategy Agent", 
                                 "AI 기반 비즈니스 전략 수립 및 시장 분석 플랫폼")
    st.markdown(header_html, unsafe_allow_html=True)
    
    # 홈으로 돌아가기 버튼
    render_home_button()
    
    st.markdown("---")

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
                    st.warning("Configuration Issues Found:")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ System Configuration OK")
                
            except Exception as e:
                st.error(f"System Check Failed: {e}")
        
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
        
        keywords_input = st.text_area(
            "Enter keywords (one per line)",
            placeholder="AI\nstartup\nfintech\nsustainability\ndigital transformation",
            height=150
        )
        selected_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
        
        if selected_keywords:
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
        include_trends = st.checkbox("Include Trend Analysis", value=True)
        include_strategies = st.checkbox("Generate Strategies", value=True)
        
        # 파일 저장 옵션
        save_to_file = st.checkbox(
            "파일로 저장", 
            value=False,
            help=f"체크하면 {REPORTS_PATH} 디렉토리에 분석 결과를 파일로 저장합니다"
        )
    
    # 분석 실행
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not selected_keywords:
            st.error("Please enter at least one keyword")
            return
        
        if not selected_regions:
            st.error("Please select at least one region")
            return
        
        run_analysis(selected_keywords, selected_regions, include_trends, include_strategies, save_to_file)

def run_analysis(keywords: List[str], regions: List[RegionType], include_trends: bool, include_strategies: bool, save_to_file: bool):
    """분석 실행 및 결과 표시"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 진행 상황 표시
        status_text.text("Initializing analysis...")
        progress_bar.progress(0.1)
        
        status_text.text("Running business strategy analysis...")
        progress_bar.progress(0.5)
        
        # 실제 분석 실행
        results = asyncio.run(run_quick_analysis(keywords))
        
        progress_bar.progress(1.0)
        status_text.text("Analysis completed!")
        
        # 결과 저장
        st.session_state.analysis_results = results
        st.session_state.analysis_history.append({
            'timestamp': datetime.now(),
            'keywords': keywords,
            'regions': [region.value for region in regions],
            'success': 'error' not in results
        })
        
        # 성공 메시지
        if 'error' not in results:
            st.success(f"✅ Analysis completed! Generated {results.get('enhanced_insights_count', 0)} insights")
            
            # 파일 저장 처리
            if save_to_file:
                file_saved, output_path = save_business_results_to_file(results)
                if file_saved:
                    st.success(f"💾 결과가 파일로 저장되었습니다: {output_path}")
                else:
                    st.warning("파일 저장에 실패했습니다.")
        else:
            st.error(f"❌ Analysis failed: {results['error']}")
        
    except Exception as e:
        st.error(f"Analysis execution failed: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Top Opportunities", "📈 Insights Detail", "🚀 Strategies", "📊 Analytics"
    ])
    
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
                'Success': '✅' if analysis['success'] else '❌'
            }
            for analysis in st.session_state.analysis_history[-10:]  # 최근 10개
        ])
        
        st.dataframe(history_df, use_container_width=True)

def save_business_results_to_file(analysis_result):
    """비즈니스 전략 분석 결과를 파일로 저장"""
    try:
        os.makedirs(REPORTS_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"business_strategy_analysis_{timestamp}.md"
        filepath = os.path.join(REPORTS_PATH, filename)
        
        # 분석 결과 포맷팅
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

def format_business_analysis(analysis_result):
    """실제 에이전트 분석 결과 포맷팅"""
    if not analysis_result:
        raise Exception("분석 결과가 없습니다.")
    
    output_lines = [
        "🎯 비즈니스 전략 분석 결과",
        ""
    ]
    
    # 인사이트 정보
    if 'enhanced_insights' in analysis_result:
        insights = analysis_result['enhanced_insights']
        output_lines.extend([
            f"📊 총 인사이트 수: {len(insights)}",
            f"🔍 평균 후킹 점수: {sum(i.hooking_score for i in insights) / len(insights):.2f}",
            ""
        ])
        
        output_lines.append("💡 주요 인사이트:")
        for insight in insights[:5]:
            output_lines.append(f"- {', '.join(insight.key_topics[:3])} (점수: {insight.hooking_score:.2f})")
        output_lines.append("")
    
    # 전략 정보
    if 'strategies' in analysis_result:
        strategies = analysis_result['strategies']
        output_lines.append("🚀 생성된 전략:")
        for strategy in strategies:
            output_lines.extend([
                f"### {strategy.title}",
                f"- 지역: {strategy.region.value}",
                f"- 기회 수준: {strategy.opportunity_level.value}",
                f"- 타임라인: {strategy.timeline}",
                f"- 설명: {strategy.description}",
                ""
            ])
    
    return "\n".join(output_lines)

def main():
    """메인 함수"""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    st.success("🤖 Business Strategy Agent가 성공적으로 연결되었습니다!")
    
    # 메인 컨텐츠
    render_analysis_input()
    
    st.divider()
    
    render_results()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🎯 Most Hooking Business Strategy Agent | 
        Built with ❤️ using Streamlit & AI Agents
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()