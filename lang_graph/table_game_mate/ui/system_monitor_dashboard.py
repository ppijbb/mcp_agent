"""
시스템 모니터링 대시보드
Table Game Mate의 안정성과 성능을 실시간으로 모니터링
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Any

# 프로젝트 루트를 Python 경로에 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.error_handler import get_error_handler
from core.cache_manager import get_cache_manager
from core.plugin_manager import get_plugin_manager
from core.performance_monitor import get_performance_monitor


class SystemMonitorDashboard:
    """시스템 모니터링 대시보드"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Table Game Mate - System Monitor",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 페이지 제목
        st.title("🖥️ Table Game Mate System Monitor")
        st.markdown("---")
        
        # 사이드바 설정
        self.setup_sidebar()
        
        # 메인 대시보드
        self.render_main_dashboard()
    
    def setup_sidebar(self):
        """사이드바 설정"""
        st.sidebar.title("🔧 System Controls")
        
        # 새로고침 버튼
        if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # 모니터링 설정
        st.sidebar.subheader("📊 Monitoring Settings")
        
        # 성능 모니터링 활성화/비활성화
        performance_monitor = get_performance_monitor()
        if st.sidebar.checkbox("Enable Performance Monitoring", value=performance_monitor.monitoring_active):
            if not performance_monitor.monitoring_active:
                performance_monitor.start_monitoring()
        else:
            if performance_monitor.monitoring_active:
                performance_monitor.stop_monitoring()
        
        # 에러 핸들러 설정
        error_handler = get_error_handler()
        st.sidebar.subheader("⚠️ Error Handling")
        
        # 에러 로그 정리
        if st.sidebar.button("Clear Old Errors", use_container_width=True):
            error_handler.clear_old_errors(days=7)
            st.sidebar.success("Old errors cleared!")
        
        # 캐시 관리
        cache_manager = get_cache_manager()
        st.sidebar.subheader("💾 Cache Management")
        
        if st.sidebar.button("Clear All Caches", use_container_width=True):
            asyncio.run(cache_manager.clear())
            st.sidebar.success("All caches cleared!")
        
        # 플러그인 관리
        plugin_manager = get_plugin_manager()
        st.sidebar.subheader("🔌 Plugin Management")
        
        if st.sidebar.button("Discover Plugins", use_container_width=True):
            asyncio.run(plugin_manager.auto_discover_and_load())
            st.sidebar.success("Plugin discovery completed!")
    
    def render_main_dashboard(self):
        """메인 대시보드 렌더링"""
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance", "⚠️ Errors", "💾 Cache", "🔌 Plugins", "📋 System Info"
        ])
        
        with tab1:
            self.render_performance_tab()
        
        with tab2:
            self.render_errors_tab()
        
        with tab3:
            self.render_cache_tab()
        
        with tab4:
            self.render_plugins_tab()
        
        with tab5:
            self.render_system_info_tab()
    
    def render_performance_tab(self):
        """성능 모니터링 탭"""
        st.header("📈 Performance Monitoring")
        
        performance_monitor = get_performance_monitor()
        
        # 시스템 건강도
        health_report = performance_monitor.get_performance_report()
        system_health = health_report["system_health"]
        
        # 건강도 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            health_color = {
                "healthy": "🟢",
                "warning": "🟡", 
                "critical": "🔴"
            }.get(system_health["status"], "⚪")
            
            st.metric(
                label="System Health",
                value=f"{health_color} {system_health['status'].title()}",
                delta=f"Score: {system_health['score']}/100"
            )
        
        with col2:
            cpu_metric = performance_monitor.get_metric("cpu_usage")
            cpu_value = cpu_metric.get_latest_value() if cpu_metric else 0
            
            st.metric(
                label="CPU Usage",
                value=f"{cpu_value:.1f}%",
                delta="Current"
            )
        
        with col3:
            memory_metric = performance_monitor.get_metric("memory_usage")
            memory_value = memory_metric.get_latest_value() if memory_metric else 0
            
            st.metric(
                label="Memory Usage",
                value=f"{memory_value:.1f}%",
                delta="Current"
            )
        
        st.markdown("---")
        
        # 성능 메트릭 차트
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_cpu_chart(performance_monitor)
        
        with col2:
            self.render_memory_chart(performance_monitor)
        
        # 권장사항
        if system_health["issues"]:
            st.subheader("⚠️ Performance Issues")
            for issue in system_health["issues"]:
                st.warning(issue)
        
        if health_report["recommendations"]:
            st.subheader("💡 Recommendations")
            for rec in health_report["recommendations"]:
                st.info(rec)
    
    def render_cpu_chart(self, performance_monitor):
        """CPU 사용률 차트"""
        cpu_metric = performance_monitor.get_metric("cpu_usage")
        if not cpu_metric or not cpu_metric.values:
            st.info("No CPU data available")
            return
        
        # 최근 30개 데이터 포인트
        recent_values = list(cpu_metric.values)[-30:]
        
        df = pd.DataFrame([
            {
                "timestamp": mv.timestamp,
                "cpu_usage": mv.value
            }
            for mv in recent_values
        ])
        
        fig = px.line(
            df, 
            x="timestamp", 
            y="cpu_usage",
            title="CPU Usage Trend",
            labels={"cpu_usage": "CPU Usage (%)", "timestamp": "Time"}
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_memory_chart(self, performance_monitor):
        """메모리 사용률 차트"""
        memory_metric = performance_monitor.get_metric("memory_usage")
        if not memory_metric or not memory_metric.values:
            st.info("No memory data available")
            return
        
        # 최근 30개 데이터 포인트
        recent_values = list(memory_metric.values)[-30:]
        
        df = pd.DataFrame([
            {
                "timestamp": mv.timestamp,
                "memory_usage": mv.value
            }
            for mv in recent_values
        ])
        
        fig = px.line(
            df, 
            x="timestamp", 
            y="memory_usage",
            title="Memory Usage Trend",
            labels={"memory_usage": "Memory Usage (%)", "timestamp": "Time"}
        )
        
        fig.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Warning Threshold")
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_errors_tab(self):
        """에러 모니터링 탭"""
        st.header("⚠️ Error Monitoring")
        
        error_handler = get_error_handler()
        error_summary = error_handler.get_error_summary()
        
        # 에러 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Errors",
                value=error_summary["total_errors"]
            )
        
        with col2:
            st.metric(
                label="Recent Errors (24h)",
                value=error_summary["recent_errors_24h"]
            )
        
        with col3:
            st.metric(
                label="Unresolved Errors",
                value=error_summary["unresolved_errors"]
            )
        
        with col4:
            st.metric(
                label="Resolution Rate",
                value=f"{error_summary['resolution_rate']:.1f}%"
            )
        
        st.markdown("---")
        
        # 에러 분포 차트
        col1, col2 = st.columns(2)
        
        with col1:
            # 심각도별 에러 분포
            severity_data = error_summary["errors_by_severity"]
            if severity_data:
                fig = px.pie(
                    values=list(severity_data.values()),
                    names=list(severity_data.keys()),
                    title="Errors by Severity"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 카테고리별 에러 분포
            category_data = error_summary["errors_by_category"]
            if category_data:
                fig = px.bar(
                    x=list(category_data.keys()),
                    y=list(category_data.values()),
                    title="Errors by Category"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # 최근 에러 목록
        st.subheader("📋 Recent Errors")
        
        # 에러 필터링
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                ["All"] + list(error_summary["errors_by_severity"].keys())
            )
        
        with col2:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All"] + list(error_summary["errors_by_category"].keys())
            )
        
        # 에러 목록 표시
        self.render_error_list(error_handler, severity_filter, category_filter)
    
    def render_error_list(self, error_handler, severity_filter, category_filter):
        """에러 목록 렌더링"""
        # 모든 에러 가져오기
        all_errors = error_handler.error_records
        
        # 필터링
        filtered_errors = []
        for error in all_errors:
            if severity_filter != "All" and error.severity.value != severity_filter:
                continue
            if category_filter != "All" and error.category.value != category_filter:
                continue
            filtered_errors.append(error)
        
        # 최근 20개만 표시
        recent_errors = sorted(filtered_errors, key=lambda x: x.timestamp, reverse=True)[:20]
        
        if not recent_errors:
            st.info("No errors found with current filters")
            return
        
        # 에러 테이블
        error_data = []
        for error in recent_errors:
            error_data.append({
                "Time": error.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Severity": error.severity.value,
                "Category": error.category.value,
                "Agent": error.agent_id or "System",
                "Type": error.error_type,
                "Message": error.error_message[:50] + "..." if len(error.error_message) > 50 else error.error_message,
                "Status": "✅ Resolved" if error.resolved else "❌ Unresolved"
            })
        
        df = pd.DataFrame(error_data)
        st.dataframe(df, use_container_width=True)
    
    def render_cache_tab(self):
        """캐시 모니터링 탭"""
        st.header("💾 Cache Monitoring")
        
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        
        # 캐시 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Requests",
                value=cache_stats["total_requests"]
            )
        
        with col2:
            memory_stats = cache_stats["memory_cache"]
            st.metric(
                label="Memory Cache Size",
                value=f"{memory_stats['size']}/{memory_stats['max_size']}"
            )
        
        with col3:
            disk_stats = cache_stats["disk_cache"]
            st.metric(
                label="Disk Cache Files",
                value=disk_stats["total_files"]
            )
        
        with col4:
            hit_rates = cache_stats["hit_rates"]
            avg_hit_rate = (hit_rates["memory"] + hit_rates["disk"]) / 2
            st.metric(
                label="Average Hit Rate",
                value=f"{avg_hit_rate:.1f}%"
            )
        
        st.markdown("---")
        
        # 히트율 차트
        col1, col2 = st.columns(2)
        
        with col1:
            hit_rates = cache_stats["hit_rates"]
            fig = px.bar(
                x=["Memory", "Disk"],
                y=[hit_rates["memory"], hit_rates["disk"]],
                title="Cache Hit Rates",
                labels={"y": "Hit Rate (%)", "x": "Cache Level"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 메모리 캐시 정책
            memory_stats = cache_stats["memory_cache"]
            st.subheader("Memory Cache Policy")
            st.info(f"**Policy:** {memory_stats['policy']}")
            st.info(f"**Current Size:** {memory_stats['size']} items")
            st.info(f"**Total Size:** {memory_stats['total_size_bytes'] / 1024:.1f} KB")
        
        # 캐시 관리 옵션
        st.subheader("🔧 Cache Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Memory Cache", use_container_width=True):
                cache_manager.memory_cache.clear()
                st.success("Memory cache cleared!")
        
        with col2:
            if st.button("Clear Disk Cache", use_container_width=True):
                cache_manager.disk_cache.clear()
                st.success("Disk cache cleared!")
    
    def render_plugins_tab(self):
        """플러그인 모니터링 탭"""
        st.header("🔌 Plugin Management")
        
        plugin_manager = get_plugin_manager()
        plugin_stats = plugin_manager.get_plugin_stats()
        
        # 플러그인 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Plugins",
                value=plugin_stats["total_plugins"]
            )
        
        with col2:
            active_plugins = plugin_stats["status_distribution"].get("active", 0)
            st.metric(
                label="Active Plugins",
                value=active_plugins
            )
        
        with col3:
            error_plugins = plugin_stats["status_distribution"].get("error", 0)
            st.metric(
                label="Error Plugins",
                value=error_plugins
            )
        
        with col4:
            st.metric(
                label="Auto Discover",
                value="✅ On" if plugin_stats["auto_discover"] else "❌ Off"
            )
        
        st.markdown("---")
        
        # 플러그인 상태 분포
        col1, col2 = st.columns(2)
        
        with col1:
            status_data = plugin_stats["status_distribution"]
            if status_data:
                fig = px.pie(
                    values=list(status_data.values()),
                    names=list(status_data.keys()),
                    title="Plugins by Status"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_data = plugin_stats["type_distribution"]
            if type_data:
                fig = px.bar(
                    x=list(type_data.keys()),
                    y=list(type_data.values()),
                    title="Plugins by Type"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # 플러그인 목록
        st.subheader("📋 Plugin List")
        
        all_plugins = plugin_manager.get_all_plugins()
        if not all_plugins:
            st.info("No plugins found")
            return
        
        # 플러그인 테이블
        plugin_data = []
        for plugin in all_plugins:
            plugin_data.append({
                "Name": plugin.metadata.name,
                "Type": plugin.metadata.plugin_type.value,
                "Version": plugin.metadata.version,
                "Status": plugin.status.value,
                "Author": plugin.metadata.author,
                "Load Time": plugin.load_time.strftime("%Y-%m-%d %H:%M:%S") if plugin.load_time else "N/A"
            })
        
        df = pd.DataFrame(plugin_data)
        st.dataframe(df, use_container_width=True)
    
    def render_system_info_tab(self):
        """시스템 정보 탭"""
        st.header("📋 System Information")
        
        # 시스템 개요
        st.subheader("🏗️ System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Table Game Mate** is a LangGraph-based multi-agent board game platform")
            st.info("**Version:** 0.1.0")
            st.info("**Architecture:** Multi-Agent System with Plugin Support")
        
        with col2:
            st.info("**Core Components:** 6 Specialized Agents")
            st.info("**UI Framework:** Streamlit + Plotly")
            st.info("**AI Integration:** Gemini 2.0 Flash")
        
        st.markdown("---")
        
        # 에이전트 상태
        st.subheader("🤖 Agent Status")
        
        # 에이전트별 상태 표시 (실제로는 에이전트 매니저에서 가져와야 함)
        agent_status = {
            "Game Analyzer": "🟢 Active",
            "Rule Parser": "🟢 Active", 
            "Persona Generator": "🟢 Active",
            "Player Manager": "🟢 Active",
            "Game Referee": "🟢 Active",
            "Score Calculator": "🟢 Active"
        }
        
        for agent, status in agent_status.items():
            st.info(f"**{agent}:** {status}")
        
        st.markdown("---")
        
        # 시스템 리소스
        st.subheader("💻 System Resources")
        
        try:
            import psutil
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpu_percent = psutil.cpu_percent(interval=1)
                st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            
            with col2:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            with col3:
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                st.metric("Disk Usage", f"{disk_percent:.1f}%")
            
        except ImportError:
            st.warning("psutil not available for system resource monitoring")
        
        st.markdown("---")
        
        # 로그 정보
        st.subheader("📝 Logging Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Log Level:** INFO")
            st.info("**Log Directory:** ./logs")
            st.info("**Rotation:** Daily")
        
        with col2:
            st.info("**Performance Logging:** Enabled")
            st.info("**Error Logging:** Enabled")
            st.info("**Agent Logging:** Enabled")


def main():
    """메인 함수"""
    dashboard = SystemMonitorDashboard()


if __name__ == "__main__":
    main()
