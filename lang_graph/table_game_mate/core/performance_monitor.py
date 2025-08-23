"""
성능 모니터링 시스템
Table Game Mate의 성능 최적화를 위한 실시간 모니터링
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from ..utils.logger import get_logger


class MetricType(Enum):
    """메트릭 타입"""
    COUNTER = "counter"           # 누적 값 (요청 수, 에러 수)
    GAUGE = "gauge"               # 현재 값 (메모리 사용량, CPU 사용률)
    HISTOGRAM = "histogram"       # 분포 값 (응답 시간, 처리 시간)
    TIMER = "timer"               # 시간 측정


@dataclass
class MetricValue:
    """메트릭 값"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


@dataclass
class Metric:
    """메트릭 정의"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """값 추가"""
        metric_value = MetricValue(
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self.values.append(metric_value)
    
    def get_latest_value(self) -> Optional[float]:
        """최신 값 조회"""
        if self.values:
            return self.values[-1].value
        return None
    
    def get_average(self, minutes: int = 5) -> Optional[float]:
        """평균값 계산"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_values = [
            mv.value for mv in self.values 
            if mv.timestamp > cutoff_time
        ]
        
        if recent_values:
            return sum(recent_values) / len(recent_values)
        return None
    
    def get_percentile(self, percentile: float, minutes: int = 5) -> Optional[float]:
        """백분위 값 계산"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_values = [
            mv.value for mv in self.values 
            if mv.timestamp > cutoff_time
        ]
        
        if not recent_values:
            return None
        
        recent_values.sort()
        index = int(len(recent_values) * percentile / 100)
        return recent_values[index]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "latest_value": self.get_latest_value(),
            "average_5min": self.get_average(5),
            "p95_5min": self.get_percentile(95, 5),
            "total_values": len(self.values)
        }


class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.logger = get_logger("performance_monitor")
        self.metrics: Dict[str, Metric] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 시스템 메트릭 수집 주기 (초)
        self.system_metrics_interval = 5
        
        # 기본 메트릭 등록
        self._register_default_metrics()
        
        # 성능 임계값
        self.thresholds = {
            "cpu_usage": 80.0,      # CPU 사용률 80% 초과 시 경고
            "memory_usage": 85.0,   # 메모리 사용률 85% 초과 시 경고
            "response_time": 2.0,   # 응답 시간 2초 초과 시 경고
            "error_rate": 5.0       # 에러율 5% 초과 시 경고
        }
        
        # 임계값 초과 시 콜백
        self.threshold_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def _register_default_metrics(self):
        """기본 메트릭 등록"""
        # 시스템 메트릭
        self.register_metric("cpu_usage", MetricType.GAUGE, "CPU 사용률", "%")
        self.register_metric("memory_usage", MetricType.GAUGE, "메모리 사용률", "%")
        self.register_metric("disk_io", MetricType.GAUGE, "디스크 I/O", "MB/s")
        self.register_metric("network_io", MetricType.GAUGE, "네트워크 I/O", "MB/s")
        
        # 애플리케이션 메트릭
        self.register_metric("request_count", MetricType.COUNTER, "요청 수", "requests")
        self.register_metric("response_time", MetricType.HISTOGRAM, "응답 시간", "seconds")
        self.register_metric("error_count", MetricType.COUNTER, "에러 수", "errors")
        self.register_metric("active_connections", MetricType.GAUGE, "활성 연결 수", "connections")
        
        # 에이전트별 메트릭
        self.register_metric("agent_response_time", MetricType.HISTOGRAM, "에이전트 응답 시간", "seconds")
        self.register_metric("agent_error_rate", MetricType.GAUGE, "에이전트 에러율", "%")
        self.register_metric("agent_throughput", MetricType.GAUGE, "에이전트 처리량", "requests/sec")
    
    def register_metric(self, name: str, metric_type: MetricType, description: str, unit: str = "") -> Metric:
        """메트릭 등록"""
        if name in self.metrics:
            self.logger.warning(f"Metric {name} already exists, overwriting")
        
        metric = Metric(name=name, metric_type=metric_type, description=description, unit=unit)
        self.metrics[name] = metric
        
        self.logger.info(f"Registered metric: {name} ({metric_type.value})")
        return metric
    
    def record_value(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """메트릭 값 기록"""
        if metric_name not in self.metrics:
            self.logger.warning(f"Metric {metric_name} not found, creating default")
            self.register_metric(metric_name, MetricType.GAUGE, f"Auto-created metric: {metric_name}")
        
        metric = self.metrics[metric_name]
        metric.add_value(value, labels)
        
        # 임계값 확인
        self._check_thresholds(metric_name, value)
    
    def record_timing(self, metric_name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """시간 측정 메트릭 기록"""
        self.record_value(metric_name, duration, labels)
    
    def increment_counter(self, metric_name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """카운터 증가"""
        current_value = self.metrics.get(metric_name, Metric(metric_name, MetricType.COUNTER, "", ""))
        latest_value = current_value.get_latest_value() or 0
        self.record_value(metric_name, latest_value + increment, labels)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """임계값 확인"""
        threshold = self.thresholds.get(metric_name)
        if threshold is None:
            return
        
        if value > threshold:
            self._trigger_threshold_alert(metric_name, value, threshold)
    
    def _trigger_threshold_alert(self, metric_name: str, value: float, threshold: float):
        """임계값 초과 알림"""
        alert_message = f"Performance threshold exceeded: {metric_name} = {value} (threshold: {threshold})"
        self.logger.warning(alert_message)
        
        # 콜백 실행
        callbacks = self.threshold_callbacks.get(metric_name, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(metric_name, value, threshold))
                else:
                    callback(metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Threshold callback execution failed: {e}")
    
    def add_threshold_callback(self, metric_name: str, callback: Callable):
        """임계값 콜백 등록"""
        self.threshold_callbacks[metric_name].append(callback)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.system_metrics_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_value("cpu_usage", cpu_percent)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.record_value("memory_usage", memory_percent)
            
            # 디스크 I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_mb_per_sec = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
                self.record_value("disk_io", disk_mb_per_sec)
            
            # 네트워크 I/O
            network_io = psutil.net_io_counters()
            if network_io:
                network_mb_per_sec = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
                self.record_value("network_io", network_mb_per_sec)
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def get_metric(self, metric_name: str) -> Optional[Metric]:
        """메트릭 조회"""
        return self.metrics.get(metric_name)
    
    def get_metric_summary(self, metric_name: str, minutes: int = 5) -> Optional[Dict[str, Any]]:
        """메트릭 요약 정보"""
        metric = self.metrics.get(metric_name)
        if not metric:
            return None
        
        return {
            "name": metric.name,
            "type": metric.metric_type.value,
            "latest_value": metric.get_latest_value(),
            "average": metric.get_average(minutes),
            "min": min(mv.value for mv in metric.values) if metric.values else None,
            "max": max(mv.value for mv in metric.values) if metric.values else None,
            "p95": metric.get_percentile(95, minutes),
            "p99": metric.get_percentile(99, minutes),
            "total_values": len(metric.values)
        }
    
    def get_all_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """모든 메트릭 요약"""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = self.get_metric_summary(metric_name, minutes)
        
        return summary
    
    def get_performance_report(self, minutes: int = 15) -> Dict[str, Any]:
        """성능 리포트 생성"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "metrics_summary": self.get_all_metrics_summary(minutes),
            "system_health": self._assess_system_health(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """시스템 건강도 평가"""
        health_score = 100.0
        issues = []
        
        # CPU 사용률 확인
        cpu_metric = self.metrics.get("cpu_usage")
        if cpu_metric:
            cpu_value = cpu_metric.get_latest_value()
            if cpu_value and cpu_value > 80:
                health_score -= 20
                issues.append(f"High CPU usage: {cpu_value}%")
        
        # 메모리 사용률 확인
        memory_metric = self.metrics.get("memory_usage")
        if memory_metric:
            memory_value = memory_metric.get_latest_value()
            if memory_value and memory_value > 85:
                health_score -= 20
                issues.append(f"High memory usage: {memory_value}%")
        
        # 응답 시간 확인
        response_metric = self.metrics.get("response_time")
        if response_metric:
            avg_response = response_metric.get_average(5)
            if avg_response and avg_response > 2.0:
                health_score -= 15
                issues.append(f"Slow response time: {avg_response:.2f}s")
        
        # 에러율 확인
        error_metric = self.metrics.get("error_count")
        request_metric = self.metrics.get("request_count")
        if error_metric and request_metric:
            error_count = error_metric.get_latest_value() or 0
            request_count = request_metric.get_latest_value() or 1
            error_rate = (error_count / request_count) * 100 if request_count > 0 else 0
            
            if error_rate > 5.0:
                health_score -= 25
                issues.append(f"High error rate: {error_rate:.1f}%")
        
        return {
            "score": max(0, health_score),
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "issues": issues
        }
    
    def _generate_recommendations(self) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []
        
        # CPU 사용률 권장사항
        cpu_metric = self.metrics.get("cpu_usage")
        if cpu_metric:
            cpu_value = cpu_metric.get_latest_value()
            if cpu_value and cpu_value > 80:
                recommendations.append("CPU 사용률이 높습니다. 작업 부하를 줄이거나 스케일링을 고려하세요.")
        
        # 메모리 사용률 권장사항
        memory_metric = self.metrics.get("memory_usage")
        if memory_metric:
            memory_value = memory_metric.get_latest_value()
            if memory_value and memory_value > 85:
                recommendations.append("메모리 사용률이 높습니다. 메모리 누수를 확인하거나 메모리를 증설하세요.")
        
        # 응답 시간 권장사항
        response_metric = self.metrics.get("response_time")
        if response_metric:
            avg_response = response_metric.get_average(5)
            if avg_response and avg_response > 2.0:
                recommendations.append("응답 시간이 느립니다. 데이터베이스 쿼리나 외부 API 호출을 최적화하세요.")
        
        # 에러율 권장사항
        error_metric = self.metrics.get("error_count")
        request_metric = self.metrics.get("request_count")
        if error_metric and request_metric:
            error_count = error_metric.get_latest_value() or 0
            request_count = request_metric.get_latest_value() or 1
            error_rate = (error_count / request_count) * 100 if request_count > 0 else 0
            
            if error_rate > 5.0:
                recommendations.append("에러율이 높습니다. 로그를 확인하여 에러 원인을 파악하세요.")
        
        if not recommendations:
            recommendations.append("시스템 성능이 양호합니다.")
        
        return recommendations


# 성능 측정 데코레이터
def monitor_performance(metric_name: str, metric_type: MetricType = MetricType.TIMER):
    """함수 성능 모니터링 데코레이터"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if metric_type == MetricType.TIMER:
                    monitor.record_timing(metric_name, duration)
                else:
                    monitor.record_value(metric_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.increment_counter(f"{metric_name}_errors")
                raise e
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if metric_type == MetricType.TIMER:
                    monitor.record_timing(metric_name, duration)
                else:
                    monitor.record_value(metric_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.increment_counter(f"{metric_name}_errors")
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 전역 성능 모니터 인스턴스
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """전역 성능 모니터 인스턴스 반환"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
