#!/usr/bin/env python3
"""
Real-time System Monitor for Local Researcher

This module provides comprehensive system monitoring capabilities including
performance metrics, health checks, and real-time alerts.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import threading

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("system_monitor", log_level="INFO")


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int
    research_tasks: int
    agent_status: Dict[str, str]
    error_count: int
    warning_count: int


@dataclass
class Alert:
    """System alert."""
    timestamp: datetime
    level: str  # info, warning, error, critical
    category: str  # performance, memory, disk, network, research
    message: str
    details: Dict[str, Any]


class SystemMonitor:
    """Real-time system monitoring and alerting."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Monitoring settings
        self.monitoring_interval = self.config_manager.get('monitoring.interval', 5)  # seconds
        self.metrics_history_size = self.config_manager.get('monitoring.history_size', 1000)
        self.alert_thresholds = self.config_manager.get('monitoring.thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 10.0
        })
        
        # Data storage
        self.metrics_history = deque(maxlen=self.metrics_history_size)
        self.alerts = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        logger.info("System Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Research-specific metrics (would be provided by orchestrator)
            research_tasks = self._get_research_task_count()
            agent_status = self._get_agent_status()
            
            # Error counts (would be collected from logs)
            error_count, warning_count = self._get_log_counts()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_processes=active_processes,
                research_tasks=research_tasks,
                agent_status=agent_status,
                error_count=error_count,
                warning_count=warning_count
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_processes=0,
                research_tasks=0,
                agent_status={},
                error_count=0,
                warning_count=0
            )
    
    def _get_research_task_count(self) -> int:
        """Get current research task count."""
        # This would be implemented to get actual task count from orchestrator
        return 0
    
    def _get_agent_status(self) -> Dict[str, str]:
        """Get current agent status."""
        # This would be implemented to get actual agent status
        return {
            "analyzer": "running",
            "decomposer": "running",
            "researcher": "running",
            "evaluator": "running",
            "validator": "running",
            "synthesizer": "running"
        }
    
    def _get_log_counts(self) -> Tuple[int, int]:
        """Get error and warning counts from logs."""
        # This would be implemented to parse actual log files
        return 0, 0
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds and generate alerts."""
        try:
            # CPU usage alert
            if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                self._create_alert(
                    level="warning",
                    category="performance",
                    message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                    details={"cpu_usage": metrics.cpu_usage, "threshold": self.alert_thresholds['cpu_usage']}
                )
            
            # Memory usage alert
            if metrics.memory_usage > self.alert_thresholds['memory_usage']:
                self._create_alert(
                    level="warning",
                    category="memory",
                    message=f"High memory usage: {metrics.memory_usage:.1f}%",
                    details={"memory_usage": metrics.memory_usage, "threshold": self.alert_thresholds['memory_usage']}
                )
            
            # Disk usage alert
            if metrics.disk_usage > self.alert_thresholds['disk_usage']:
                self._create_alert(
                    level="error",
                    category="disk",
                    message=f"High disk usage: {metrics.disk_usage:.1f}%",
                    details={"disk_usage": metrics.disk_usage, "threshold": self.alert_thresholds['disk_usage']}
                )
            
            # Error rate alert
            if metrics.error_count > self.alert_thresholds['error_rate']:
                self._create_alert(
                    level="error",
                    category="research",
                    message=f"High error rate: {metrics.error_count} errors",
                    details={"error_count": metrics.error_count, "threshold": self.alert_thresholds['error_rate']}
                )
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    def _create_alert(self, level: str, category: str, message: str, details: Dict[str, Any]):
        """Create and process a new alert."""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{level.upper()}] {category}: {message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp >= cutoff_time]
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            if not self.metrics_history:
                return 100.0
            
            recent_metrics = self.get_metrics_history(hours=1)
            if not recent_metrics:
                return 100.0
            
            # Calculate average metrics
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
            
            # Calculate health score (lower usage = higher score)
            cpu_score = max(0, 100 - avg_cpu)
            memory_score = max(0, 100 - avg_memory)
            disk_score = max(0, 100 - avg_disk)
            
            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            
            return min(100.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 50.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard."""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return {}
            
            recent_alerts = self.get_recent_alerts(hours=1)
            health_score = self.get_system_health_score()
            
            return {
                "timestamp": current_metrics.timestamp.isoformat(),
                "health_score": health_score,
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "disk_usage": current_metrics.disk_usage,
                "active_processes": current_metrics.active_processes,
                "research_tasks": current_metrics.research_tasks,
                "agent_status": current_metrics.agent_status,
                "recent_alerts": len(recent_alerts),
                "error_count": current_metrics.error_count,
                "warning_count": current_metrics.warning_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def export_metrics(self, file_path: str, format: str = "json"):
        """Export metrics to file."""
        try:
            metrics_data = [asdict(m) for m in self.metrics_history]
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(metrics_data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def export_alerts(self, file_path: str, format: str = "json"):
        """Export alerts to file."""
        try:
            alerts_data = [asdict(a) for a in self.alerts]
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(alerts_data, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(alerts_data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Alerts exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export alerts: {e}")
            raise
