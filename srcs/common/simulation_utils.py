"""
공통 시뮬레이션 유틸리티

실제 운영 환경과 유사한 시뮬레이션 데이터를 생성하기 위한 공통 유틸리티 모듈.
하드코딩 없이 물리 법칙, 통계적 패턴, 확률 분포를 기반으로 데이터를 생성합니다.
"""

import random
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class DistributionType(Enum):
    """확률 분포 타입"""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    LOG_NORMAL = "log_normal"


class TimeSeriesGenerator:
    """시계열 데이터 생성기 - 트렌드, 계절성, 노이즈 포함"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    def generate(
        self,
        start_time: datetime,
        duration_seconds: int,
        interval_seconds: float = 1.0,
        base_value: float = 0.0,
        trend: float = 0.0,
        seasonality_amplitude: float = 0.0,
        seasonality_period: float = 3600.0,
        noise_std: float = 0.1,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        시계열 데이터 생성
        
        Args:
            start_time: 시작 시간
            duration_seconds: 생성할 데이터의 지속 시간 (초)
            interval_seconds: 데이터 포인트 간격 (초)
            base_value: 기본값
            trend: 시간당 트렌드 (선형 증가/감소)
            seasonality_amplitude: 계절성 진폭
            seasonality_period: 계절성 주기 (초)
            noise_std: 노이즈 표준편차
            min_value: 최소값 제한
            max_value: 최대값 제한
            
        Returns:
            시계열 데이터 리스트 [{"timestamp": ..., "value": ...}, ...]
        """
        data_points = []
        num_points = int(duration_seconds / interval_seconds)
        
        for i in range(num_points):
            elapsed = i * interval_seconds
            
            # 트렌드 계산
            trend_value = base_value + (trend * elapsed / 3600.0)
            
            # 계절성 계산 (사인파)
            if seasonality_amplitude > 0:
                seasonal = seasonality_amplitude * math.sin(2 * math.pi * elapsed / seasonality_period)
            else:
                seasonal = 0.0
            
            # 노이즈 추가
            noise = self.np_rng.normal(0, noise_std)
            
            # 최종 값 계산
            value = trend_value + seasonal + noise
            
            # 범위 제한
            if min_value is not None:
                value = max(value, min_value)
            if max_value is not None:
                value = min(value, max_value)
            
            timestamp = start_time + timedelta(seconds=elapsed)
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "value": round(value, 4),
                "elapsed_seconds": elapsed
            })
        
        return data_points
    
    def generate_with_pattern(
        self,
        start_time: datetime,
        duration_seconds: int,
        interval_seconds: float,
        pattern_func: Callable[[float], float],
        noise_std: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        커스텀 패턴 함수를 사용한 시계열 데이터 생성
        
        Args:
            start_time: 시작 시간
            duration_seconds: 지속 시간
            interval_seconds: 간격
            pattern_func: 패턴 함수 (elapsed_seconds -> value)
            noise_std: 노이즈 표준편차
        """
        data_points = []
        num_points = int(duration_seconds / interval_seconds)
        
        for i in range(num_points):
            elapsed = i * interval_seconds
            base_value = pattern_func(elapsed)
            noise = self.np_rng.normal(0, noise_std)
            value = base_value + noise
            
            timestamp = start_time + timedelta(seconds=elapsed)
            data_points.append({
                "timestamp": timestamp.isoformat(),
                "value": round(value, 4),
                "elapsed_seconds": elapsed
            })
        
        return data_points


class NoiseGenerator:
    """센서 노이즈 및 측정 오차 생성기"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    def add_gaussian_noise(
        self,
        value: float,
        std_dev: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """가우시안 노이즈 추가"""
        noise = self.np_rng.normal(0, std_dev)
        result = value + noise
        
        if min_value is not None:
            result = max(result, min_value)
        if max_value is not None:
            result = min(result, max_value)
        
        return round(result, 4)
    
    def add_percentage_noise(
        self,
        value: float,
        percentage: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> float:
        """백분율 기반 노이즈 추가"""
        noise_range = value * (percentage / 100.0)
        noise = self.rng.uniform(-noise_range, noise_range)
        result = value + noise
        
        if min_value is not None:
            result = max(result, min_value)
        if max_value is not None:
            result = min(result, max_value)
        
        return round(result, 4)
    
    def add_quantization_error(
        self,
        value: float,
        resolution: float
    ) -> float:
        """양자화 오차 추가 (센서 해상도 모방)"""
        quantized = round(value / resolution) * resolution
        return quantized


class ProbabilityDistributions:
    """확률 분포 기반 데이터 생성"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    def sample(
        self,
        distribution: DistributionType,
        **params
    ) -> float:
        """
        확률 분포에서 샘플 추출
        
        Args:
            distribution: 분포 타입
            **params: 분포 파라미터
                - normal: mean, std
                - uniform: min, max
                - exponential: scale
                - poisson: lam
                - log_normal: mean, std
        """
        if distribution == DistributionType.NORMAL:
            return float(self.np_rng.normal(params['mean'], params['std']))
        elif distribution == DistributionType.UNIFORM:
            return self.rng.uniform(params['min'], params['max'])
        elif distribution == DistributionType.EXPONENTIAL:
            return float(self.np_rng.exponential(params['scale']))
        elif distribution == DistributionType.POISSON:
            return float(self.np_rng.poisson(params['lam']))
        elif distribution == DistributionType.LOG_NORMAL:
            return float(self.np_rng.lognormal(params['mean'], params['std']))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def sample_bounded(
        self,
        distribution: DistributionType,
        min_value: float,
        max_value: float,
        **params
    ) -> float:
        """범위 제한된 샘플 추출"""
        value = self.sample(distribution, **params)
        return max(min_value, min(max_value, value))


@dataclass
class State:
    """상태 머신의 상태"""
    name: str
    transitions: Dict[str, str]  # {event: next_state}
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None


class StateMachine:
    """상태 전이 시뮬레이션"""
    
    def __init__(self, initial_state: str, states: Dict[str, State], seed: Optional[int] = None):
        """
        초기화
        
        Args:
            initial_state: 초기 상태 이름
            states: 상태 딕셔너리 {state_name: State}
            seed: 랜덤 시드
        """
        self.current_state = initial_state
        self.states = states
        self.rng = random.Random(seed) if seed else random.Random()
        self.history: List[Tuple[datetime, str, str]] = []  # (timestamp, from_state, to_state)
        
        # 초기 상태 진입
        if initial_state in states and states[initial_state].on_enter:
            states[initial_state].on_enter()
    
    def transition(self, event: str) -> bool:
        """
        상태 전이 시도
        
        Args:
            event: 이벤트 이름
            
        Returns:
            전이 성공 여부
        """
        if self.current_state not in self.states:
            return False
        
        current_state_obj = self.states[self.current_state]
        if event not in current_state_obj.transitions:
            return False
        
        next_state_name = current_state_obj.transitions[event]
        if next_state_name not in self.states:
            return False
        
        # 현재 상태 종료
        if current_state_obj.on_exit:
            current_state_obj.on_exit()
        
        # 상태 전이
        prev_state = self.current_state
        self.current_state = next_state_name
        
        # 이력 기록
        self.history.append((datetime.now(), prev_state, next_state_name))
        
        # 새 상태 진입
        next_state_obj = self.states[next_state_name]
        if next_state_obj.on_enter:
            next_state_obj.on_enter()
        
        return True
    
    def get_state(self) -> str:
        """현재 상태 반환"""
        return self.current_state
    
    def get_history(self) -> List[Tuple[datetime, str, str]]:
        """상태 전이 이력 반환"""
        return self.history.copy()


class PatternGenerator:
    """실제 데이터 패턴 모방 생성기"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
    
    def generate_burst_pattern(
        self,
        base_rate: float,
        burst_probability: float,
        burst_multiplier: float,
        duration: int
    ) -> List[float]:
        """
        버스트 패턴 생성 (트래픽 버스트 등)
        
        Args:
            base_rate: 기본 비율
            burst_probability: 버스트 발생 확률
            burst_multiplier: 버스트 시 배수
            duration: 생성할 데이터 포인트 수
        """
        pattern = []
        in_burst = False
        
        for _ in range(duration):
            if not in_burst:
                if self.rng.random() < burst_probability:
                    in_burst = True
                    value = base_rate * burst_multiplier
                else:
                    value = base_rate
            else:
                if self.rng.random() < 0.3:  # 버스트 종료 확률
                    in_burst = False
                    value = base_rate
                else:
                    value = base_rate * burst_multiplier
            
            # 노이즈 추가
            value += self.np_rng.normal(0, value * 0.1)
            pattern.append(max(0, value))
        
        return pattern
    
    def generate_workload_pattern(
        self,
        duration_hours: int,
        peak_hours: List[int],
        base_load: float,
        peak_load: float
    ) -> List[Dict[str, Any]]:
        """
        워크로드 패턴 생성 (시간대별 부하 변화)
        
        Args:
            duration_hours: 지속 시간 (시간)
            peak_hours: 피크 시간대 리스트 (0-23)
            base_load: 기본 부하
            peak_load: 피크 부하
        """
        pattern = []
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for hour in range(duration_hours):
            current_hour = hour % 24
            if current_hour in peak_hours:
                load = peak_load
            else:
                load = base_load
            
            # 시간대별 부드러운 전환
            if current_hour in peak_hours or (current_hour - 1) % 24 in peak_hours:
                transition_factor = 0.5 + 0.5 * math.sin(math.pi * (current_hour % 1))
                load = base_load + (peak_load - base_load) * transition_factor
            
            # 노이즈 추가
            load += self.np_rng.normal(0, load * 0.05)
            load = max(0, load)
            
            timestamp = start_time + timedelta(hours=hour)
            pattern.append({
                "timestamp": timestamp.isoformat(),
                "load": round(load, 2),
                "hour": current_hour
            })
        
        return pattern
    
    def generate_cyclic_pattern(
        self,
        duration: int,
        period: float,
        amplitude: float,
        phase: float = 0.0,
        offset: float = 0.0
    ) -> List[float]:
        """
        주기적 패턴 생성 (사인/코사인 기반)
        
        Args:
            duration: 데이터 포인트 수
            period: 주기
            amplitude: 진폭
            phase: 위상
            offset: 오프셋
        """
        pattern = []
        for i in range(duration):
            value = offset + amplitude * math.sin(2 * math.pi * i / period + phase)
            pattern.append(round(value, 4))
        return pattern







