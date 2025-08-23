"""
통합 모니터링 대시보드
Table Game Mate의 모든 시스템을 한 곳에서 모니터링
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import asyncio

# 프로젝트 루트를 Python 경로에 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.core_system import get_core_system
from core.plugin_system import get_plugin_manager


class MonitorDashboard:
    """통합 모니터링 대시보드"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Table Game Mate - Monitor",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("🖥️ Table Game Mate System Monitor")
        st.markdown("---")
        
        # 사이드바
        self.setup_sidebar()
        
        # 메인 대시보드
        self.render_dashboard()
    
    def setup_sidebar(self):
        """사이드바 설정"""
        st.sidebar.title("🔧 Controls")
        
        if st.sidebar.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # 시스템 제어
        core_system = get_core_system()
        
        if st.sidebar.checkbox("Performance Monitoring", value=core_system.performance_monitor.monitoring_active):
            if not core_system.performance_monitor.monitoring_active:
                core_system.performance_monitor.start_monitoring()
        else:
            if core_system.performance_monitor.monitoring_active:
                core_system.performance_monitor.stop_monitoring()
        
        if st.sidebar.button("Clear Cache", use_container_width=True):
            core_system.cache_manager.memory_cache.clear()
            st.sidebar.success("Cache cleared!")
    
    def render_dashboard(self):
        """대시보드 렌더링"""
        # 시스템 상태 요약
        self.render_system_summary()
        
        st.markdown("---")
        
        # 상세 모니터링
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "⚠️ Errors", "🔌 Plugins"])
        
        with tab1:
            self.render_performance_tab()
        
        with tab2:
            self.render_errors_tab()
        
        with tab3:
            self.render_plugins_tab()
    
    def render_system_summary(self):
        """시스템 상태 요약"""
        core_system = get_core_system()
        system_status = core_system.get_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 성능 모니터링 상태
            monitoring_status = "🟢 Active" if system_status["performance_report"]["monitoring_active"] else "🔴 Inactive"
            st.metric("Performance Monitor", monitoring_status)
        
        with col2:
            # 에러 수
            error_summary = system_status["error_summary"]
            st.metric("Total Errors", error_summary["total_errors"])
        
        with col3:
            # 캐시 히트율
            cache_stats = system_status["cache_stats"]
            st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1f}%")
        
        with col4:
            # 플러그인 수
            plugin_manager = get_plugin_manager()
            plugin_stats = plugin_manager.get_plugin_stats()
            st.metric("Total Plugins", plugin_stats["total_plugins"])
    
    def render_performance_tab(self):
        """성능 모니터링 탭"""
        st.header("📈 Performance Monitoring")
        
        core_system = get_core_system()
        performance_report = core_system.performance_monitor.get_performance_report()
        
        # 메트릭 차트
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU 사용률 (시뮬레이션)
            cpu_data = pd.DataFrame({
                "Time": pd.date_range(start="2024-01-01", periods=20, freq="5min"),
                "CPU": [30 + i * 2 + (i % 3) * 5 for i in range(20)]
            })
            
            fig = px.line(cpu_data, x="Time", y="CPU", title="CPU Usage Trend")
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Warning")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 메모리 사용률 (시뮬레이션)
            memory_data = pd.DataFrame({
                "Time": pd.date_range(start="2024-01-01", periods=20, freq="5min"),
                "Memory": [45 + i * 1.5 + (i % 2) * 3 for i in range(20)]
            })
            
            fig = px.line(memory_data, x="Time", y="Memory", title="Memory Usage Trend")
            fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Warning")
            st.plotly_chart(fig, use_container_width=True)
        
        # 실시간 메트릭
        st.subheader("📊 Real-time Metrics")
        
        metrics = performance_report["metrics"]
        if metrics:
            metric_data = []
            for name, data in metrics.items():
                metric_data.append({
                    "Metric": name,
                    "Latest": f"{data['latest']:.2f}" if data['latest'] else "N/A",
                    "Average (5min)": f"{data['average_5min']:.2f}" if data['average_5min'] else "N/A"
                })
            
            df = pd.DataFrame(metric_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No performance metrics available")
    
    def render_errors_tab(self):
        """에러 모니터링 탭"""
        st.header("⚠️ Error Monitoring")
        
        core_system = get_core_system()
        error_summary = core_system.error_handler.get_error_summary()
        
        # 에러 통계
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Errors", error_summary["total_errors"])
        
        with col2:
            st.metric("Unresolved", error_summary["unresolved_errors"])
        
        with col3:
            resolution_rate = ((error_summary["total_errors"] - error_summary["unresolved_errors"]) / 
                             max(error_summary["total_errors"], 1)) * 100
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        # 에러 분포 차트
        if error_summary["total_errors"] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # 심각도별 에러 분포 (시뮬레이션)
                severity_data = pd.DataFrame({
                    "Severity": ["Low", "Medium", "High", "Critical"],
                    "Count": [2, 5, 3, 1]
                })
                
                fig = px.pie(severity_data, values="Count", names="Severity", title="Errors by Severity")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 카테고리별 에러 분포 (시뮬레이션)
                category_data = pd.DataFrame({
                    "Category": ["LLM", "MCP", "Agent", "System"],
                    "Count": [3, 2, 4, 2]
                })
                
                fig = px.bar(category_data, x="Category", y="Count", title="Errors by Category")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No errors detected! System is running smoothly.")
    
    def render_plugins_tab(self):
        """플러그인 모니터링 탭"""
        st.header("🔌 Plugin Management")
        
        plugin_manager = get_plugin_manager()
        plugin_stats = plugin_manager.get_plugin_stats()
        
        # 플러그인 통계
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Plugins", plugin_stats["total_plugins"])
        
        with col2:
            active_plugins = plugin_stats["status_distribution"].get("active", 0)
            st.metric("Active", active_plugins)
        
        with col3:
            error_plugins = plugin_stats["status_distribution"].get("error", 0)
            st.metric("Errors", error_plugins)
        
        # 플러그인 상태 분포
        if plugin_stats["total_plugins"] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                status_data = pd.DataFrame([
                    {"Status": status, "Count": count}
                    for status, count in plugin_stats["status_distribution"].items()
                    if count > 0
                ])
                
                if not status_data.empty:
                    fig = px.pie(status_data, values="Count", names="Status", title="Plugins by Status")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                type_data = pd.DataFrame([
                    {"Type": ptype, "Count": count}
                    for ptype, count in plugin_stats["type_distribution"].items()
                    if count > 0
                ])
                
                if not type_data.empty:
                    fig = px.bar(type_data, x="Type", y="Count", title="Plugins by Type")
                    st.plotly_chart(fig, use_container_width=True)
        
        # 플러그인 목록
        st.subheader("📋 Plugin List")
        
        all_plugins = list(plugin_manager.plugins.values())
        if all_plugins:
            plugin_data = []
            for plugin in all_plugins:
                plugin_data.append({
                    "Name": plugin.metadata.name,
                    "Type": plugin.metadata.plugin_type.value,
                    "Version": plugin.metadata.version,
                    "Status": plugin.status.value,
                    "Author": plugin.metadata.author
                })
            
            df = pd.DataFrame(plugin_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No plugins registered")
    
    def render_system_info(self):
        """시스템 정보"""
        st.header("📋 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Table Game Mate** - LangGraph 기반 멀티 에이전트 보드게임 플랫폼")
            st.info("**Version:** 0.1.0")
            st.info("**Architecture:** Multi-Agent System")
        
        with col2:
            st.info("**Core Components:** 6 Specialized Agents")
            st.info("**UI Framework:** Streamlit")
            st.info("**AI Integration:** Gemini 2.0 Flash")


def main():
    """메인 함수"""
    dashboard = MonitorDashboard()


if __name__ == "__main__":
    main()
