"""
Infrastructure Metrics Simulator

시스템 메트릭 시뮬레이션 (CPU, 메모리, 디스크, 네트워크).
실제 모니터링 데이터 패턴을 모방하여 시계열 데이터를 생성합니다.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from srcs.common.simulation_utils import (
    TimeSeriesGenerator, NoiseGenerator, PatternGenerator, ProbabilityDistributions
)


class InfrastructureMetricsSimulator:
    """인프라 메트릭 시뮬레이터"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.time_series_gen = TimeSeriesGenerator(seed)
        self.noise_gen = NoiseGenerator(seed)
        self.pattern_gen = PatternGenerator(seed)
        self.prob_dist = ProbabilityDistributions(seed)
    
    def generate_cpu_metrics(
        self,
        start_time: datetime,
        duration_minutes: int,
        base_load: float = 30.0,
        peak_load: float = 80.0,
        interval_seconds: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        CPU 사용률 메트릭 생성 (부하 패턴 모방)
        
        Args:
            start_time: 시작 시간
            duration_minutes: 지속 시간 (분)
            base_load: 기본 부하 (%)
            peak_load: 피크 부하 (%)
            interval_seconds: 측정 간격 (초)
        """
        duration_seconds = duration_minutes * 60
        
        # 워크로드 패턴 생성 (시간대별 변화)
        peak_hours = [9, 10, 11, 14, 15, 16]  # 업무 시간대
        workload_pattern = self.pattern_gen.generate_workload_pattern(
            duration_hours=duration_minutes // 60 + 1,
            peak_hours=peak_hours,
            base_load=base_load,
            peak_load=peak_load
        )
        
        # 버스트 패턴 추가
        burst_pattern = self.pattern_gen.generate_burst_pattern(
            base_rate=base_load,
            burst_probability=0.05,
            burst_multiplier=1.5,
            duration=int(duration_seconds / interval_seconds)
        )
        
        # 시계열 데이터 생성
        cpu_data = self.time_series_gen.generate(
            start_time=start_time,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            base_value=base_load,
            trend=0.0,
            seasonality_amplitude=(peak_load - base_load) / 2,
            seasonality_period=3600.0,  # 1시간 주기
            noise_std=5.0,
            min_value=0.0,
            max_value=100.0
        )
        
        # 버스트 패턴 적용
        for i, data_point in enumerate(cpu_data):
            if i < len(burst_pattern):
                burst_value = burst_pattern[i]
                data_point["value"] = min(100, max(0, (data_point["value"] + burst_value) / 2))
        
        # CPU 코어별 메트릭 추가
        num_cores = self.rng.randint(4, 16)
        for data_point in cpu_data:
            core_metrics = []
            for core_id in range(num_cores):
                # 각 코어는 약간 다른 부하 패턴
                core_load = data_point["value"] + self.np_rng.normal(0, 5)
                core_load = max(0, min(100, core_load))
                core_metrics.append({
                    "core_id": core_id,
                    "usage_percent": round(core_load, 2)
                })
            data_point["cores"] = core_metrics
            data_point["num_cores"] = num_cores
        
        return cpu_data
    
    def generate_memory_metrics(
        self,
        start_time: datetime,
        duration_minutes: int,
        total_memory_gb: float = 32.0,
        base_usage_percent: float = 40.0,
        interval_seconds: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        메모리 사용 메트릭 생성 (누수 시뮬레이션 포함)
        
        Args:
            start_time: 시작 시간
            duration_minutes: 지속 시간 (분)
            total_memory_gb: 총 메모리 (GB)
            base_usage_percent: 기본 사용률 (%)
            interval_seconds: 측정 간격 (초)
        """
        duration_seconds = duration_minutes * 60
        
        # 메모리 누수 시뮬레이션 (점진적 증가)
        leak_rate = 0.01  # 분당 1% 증가
        trend = (leak_rate * duration_minutes) / (duration_seconds / 3600)
        
        # 시계열 데이터 생성
        memory_data = self.time_series_gen.generate(
            start_time=start_time,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            base_value=base_usage_percent,
            trend=trend,
            seasonality_amplitude=5.0,
            seasonality_period=1800.0,  # 30분 주기
            noise_std=2.0,
            min_value=0.0,
            max_value=95.0  # 95% 이상은 위험
        )
        
        # 메모리 세부 정보 추가
        for data_point in memory_data:
            usage_percent = data_point["value"]
            used_gb = (usage_percent / 100) * total_memory_gb
            free_gb = total_memory_gb - used_gb
            cached_gb = used_gb * 0.2  # 캐시 메모리 추정
            buffers_gb = used_gb * 0.05  # 버퍼 메모리 추정
            
            data_point["used_gb"] = round(used_gb, 2)
            data_point["free_gb"] = round(free_gb, 2)
            data_point["cached_gb"] = round(cached_gb, 2)
            data_point["buffers_gb"] = round(buffers_gb, 2)
            data_point["total_gb"] = total_memory_gb
            data_point["swap_used_gb"] = round(max(0, (usage_percent - 80) * total_memory_gb / 100), 2)
        
        return memory_data
    
    def generate_disk_metrics(
        self,
        start_time: datetime,
        duration_minutes: int,
        total_disk_gb: float = 500.0,
        base_usage_percent: float = 60.0,
        interval_seconds: float = 60.0
    ) -> List[Dict[str, Any]]:
        """
        디스크 사용 메트릭 생성
        
        Args:
            start_time: 시작 시간
            duration_minutes: 지속 시간 (분)
            total_disk_gb: 총 디스크 용량 (GB)
            base_usage_percent: 기본 사용률 (%)
            interval_seconds: 측정 간격 (초)
        """
        duration_seconds = duration_minutes * 60
        
        # 디스크 사용량은 점진적으로 증가
        growth_rate = 0.001  # 시간당 0.1% 증가
        trend = (growth_rate * duration_minutes / 60) / (duration_seconds / 3600)
        
        disk_data = self.time_series_gen.generate(
            start_time=start_time,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            base_value=base_usage_percent,
            trend=trend,
            noise_std=0.5,
            min_value=0.0,
            max_value=100.0
        )
        
        # I/O 메트릭 추가
        for data_point in disk_data:
            usage_percent = data_point["value"]
            used_gb = (usage_percent / 100) * total_disk_gb
            free_gb = total_disk_gb - used_gb
            
            # I/O 속도 (랜덤)
            read_iops = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.POISSON,
                lam=100
            )
            write_iops = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.POISSON,
                lam=50
            )
            
            read_mbps = read_iops * 0.1  # 대략적 변환
            write_mbps = write_iops * 0.1
            
            data_point["used_gb"] = round(used_gb, 2)
            data_point["free_gb"] = round(free_gb, 2)
            data_point["total_gb"] = total_disk_gb
            data_point["read_iops"] = int(read_iops)
            data_point["write_iops"] = int(write_iops)
            data_point["read_mbps"] = round(read_mbps, 2)
            data_point["write_mbps"] = round(write_mbps, 2)
        
        return disk_data
    
    def generate_network_metrics(
        self,
        start_time: datetime,
        duration_minutes: int,
        base_bandwidth_mbps: float = 1000.0,
        interval_seconds: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        네트워크 메트릭 생성
        
        Args:
            start_time: 시작 시간
            duration_minutes: 지속 시간 (분)
            base_bandwidth_mbps: 기본 대역폭 (Mbps)
            interval_seconds: 측정 간격 (초)
        """
        duration_seconds = duration_minutes * 60
        
        # 네트워크 트래픽 패턴 (버스트 포함)
        traffic_pattern = self.pattern_gen.generate_burst_pattern(
            base_rate=base_bandwidth_mbps * 0.3,
            burst_probability=0.1,
            burst_multiplier=3.0,
            duration=int(duration_seconds / interval_seconds)
        )
        
        network_data = []
        num_points = int(duration_seconds / interval_seconds)
        
        for i in range(num_points):
            elapsed = i * interval_seconds
            timestamp = start_time + timedelta(seconds=elapsed)
            
            # 기본 트래픽
            base_traffic = traffic_pattern[i] if i < len(traffic_pattern) else base_bandwidth_mbps * 0.3
            
            # 업로드/다운로드 분리
            download_mbps = base_traffic * 0.7  # 다운로드가 더 많음
            upload_mbps = base_traffic * 0.3
            
            # 패킷 수
            packets_per_sec = base_traffic * 1000  # 대략적 변환
            packets_sent = int(packets_per_sec * interval_seconds * 0.3)
            packets_received = int(packets_per_sec * interval_seconds * 0.7)
            
            # 에러율
            error_rate = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.EXPONENTIAL,
                scale=0.001
            )
            error_rate = min(0.01, error_rate)  # 최대 1%
            
            # 지연 시간
            latency_ms = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.LOG_NORMAL,
                mean=math.log(10), std=0.5
            )
            latency_ms = max(1, min(100, latency_ms))
            
            network_data.append({
                "timestamp": timestamp.isoformat(),
                "download_mbps": round(download_mbps, 2),
                "upload_mbps": round(upload_mbps, 2),
                "total_mbps": round(download_mbps + upload_mbps, 2),
                "packets_sent": packets_sent,
                "packets_received": packets_received,
                "error_rate": round(error_rate, 4),
                "latency_ms": round(latency_ms, 2),
                "elapsed_seconds": elapsed
            })
        
        return network_data
    
    def generate_log_entries(
        self,
        start_time: datetime,
        duration_minutes: int,
        log_levels: List[str] = None,
        num_services: int = 5
    ) -> List[Dict[str, Any]]:
        """
        로그 엔트리 생성 (실제 로그 포맷 모방)
        
        Args:
            start_time: 시작 시간
            duration_minutes: 지속 시간 (분)
            log_levels: 로그 레벨 리스트
            num_services: 서비스 수
        """
        if log_levels is None:
            log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
        
        log_entries = []
        duration_seconds = duration_minutes * 60
        
        # 로그 발생 빈도 (레벨별)
        log_rates = {
            "INFO": 10,  # 초당 10개
            "WARNING": 1,  # 초당 1개
            "ERROR": 0.1,  # 초당 0.1개
            "DEBUG": 5
        }
        
        service_names = [f"service_{i}" for i in range(num_services)]
        log_messages = {
            "INFO": [
                "Request processed successfully",
                "Connection established",
                "Cache updated",
                "Task completed",
                "Health check passed"
            ],
            "WARNING": [
                "High memory usage detected",
                "Slow query detected",
                "Connection timeout",
                "Retry attempt",
                "Rate limit approaching"
            ],
            "ERROR": [
                "Database connection failed",
                "Out of memory",
                "Service unavailable",
                "Authentication failed",
                "Internal server error"
            ],
            "DEBUG": [
                "Processing request",
                "Cache miss",
                "Query executed",
                "Validation passed",
                "State updated"
            ]
        }
        
        current_time = start_time
        elapsed = 0
        
        while elapsed < duration_seconds:
            for level in log_levels:
                rate = log_rates.get(level, 1)
                if self.rng.random() < (rate / 10.0):  # 간격 조정
                    service = self.rng.choice(service_names)
                    message = self.rng.choice(log_messages.get(level, ["Log entry"]))
                    
                    # 로그 포맷 (일반적인 형식)
                    log_entry = {
                        "timestamp": current_time.isoformat(),
                        "level": level,
                        "service": service,
                        "message": message,
                        "pid": self.rng.randint(1000, 9999),
                        "thread": f"thread-{self.rng.randint(1, 10)}",
                        "source": f"{service}.py:{self.rng.randint(1, 500)}"
                    }
                    
                    log_entries.append(log_entry)
            
            # 시간 진행
            elapsed += 0.1
            current_time = start_time + timedelta(seconds=elapsed)
        
        return log_entries
    
    def simulate_alert_conditions(
        self,
        metrics: Dict[str, List[Dict[str, Any]]],
        thresholds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        임계값 기반 알림 생성
        
        Args:
            metrics: 메트릭 데이터 {"cpu": [...], "memory": [...], ...}
            thresholds: 임계값 {"cpu": 80.0, "memory": 90.0, ...}
        """
        alerts = []
        
        for metric_name, metric_data in metrics.items():
            threshold = thresholds.get(metric_name, 80.0)
            
            for data_point in metric_data:
                value = data_point.get("value") or data_point.get("usage_percent") or data_point.get("total_mbps", 0)
                
                if value >= threshold:
                    alert = {
                        "timestamp": data_point["timestamp"],
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "CRITICAL" if value >= threshold * 1.2 else "WARNING",
                        "message": f"{metric_name.upper()} usage exceeded threshold: {value:.2f} >= {threshold:.2f}"
                    }
                    alerts.append(alert)
        
        return alerts








