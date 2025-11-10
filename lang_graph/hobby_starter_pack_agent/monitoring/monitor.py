"""
시스템 모니터링
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class SystemMonitor:
    """시스템 모니터"""
    
    def __init__(self, check_interval: int = 60):
        """
        SystemMonitor 초기화
        
        Args:
            check_interval: 체크 간격 (초)
        """
        self.check_interval = check_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """
        시스템 메트릭 수집
        
        Returns:
            수집된 메트릭
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """현재 메트릭 반환"""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """메트릭 히스토리 반환"""
        return list(self.metrics_history)[-limit:]
    
    def check_health(self) -> Dict[str, Any]:
        """
        시스템 건강 상태 확인
        
        Returns:
            건강 상태 정보
        """
        if not self.metrics_history:
            return {
                "status": "unknown",
                "message": "No metrics available",
            }
        
        latest = self.metrics_history[-1]
        
        health_status = "healthy"
        issues = []
        
        # CPU 체크
        if latest.get("cpu", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High CPU usage")
        
        # 메모리 체크
        if latest.get("memory", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High memory usage")
        
        # 디스크 체크
        if latest.get("disk", {}).get("percent", 0) > 90:
            health_status = "degraded"
            issues.append("High disk usage")
        
        return {
            "status": health_status,
            "issues": issues,
            "metrics": latest,
        }


class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        """MetricsCollector 초기화"""
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
    
    def increment_counter(self, name: str, value: int = 1):
        """카운터 증가"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float):
        """게이지 설정"""
        self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """히스토그램 기록"""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        # 최대 1000개만 유지
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """수집된 메트릭 반환"""
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                histogram_stats[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if values else 0,
                    "p99": sorted(values)[int(len(values) * 0.99)] if values else 0,
                }
        
        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": histogram_stats,
            "timestamp": datetime.now().isoformat(),
        }
    
    def reset(self):
        """메트릭 리셋"""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()

