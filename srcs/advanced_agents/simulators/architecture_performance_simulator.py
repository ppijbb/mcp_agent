"""
Architecture Performance Simulator

아키텍처 구성요소별 성능 모델링 및 시뮬레이션.
벤치마크 데이터 기반 성능 추정, 리소스 사용량 계산, 비용 모델링을 수행합니다.
"""

import math
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from srcs.common.simulation_utils import (
    TimeSeriesGenerator, ProbabilityDistributions, PatternGenerator
)


class ArchitecturePerformanceSimulator:
    """아키텍처 성능 시뮬레이터"""
    
    def __init__(self, seed: Optional[int] = None):
        """초기화"""
        self.rng = random.Random(seed) if seed else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed else np.random.RandomState()
        self.time_series_gen = TimeSeriesGenerator(seed)
        self.prob_dist = ProbabilityDistributions(seed)
        self.pattern_gen = PatternGenerator(seed)
    
    def estimate_latency(
        self,
        architecture_config: Dict[str, Any],
        input_size: int,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        지연 시간 추정 (네트워크 지연, 처리 시간)
        
        Args:
            architecture_config: 아키텍처 구성 (layers, connections 등)
            input_size: 입력 크기
            batch_size: 배치 크기
        """
        layers = architecture_config.get("layers", [])
        num_layers = len(layers)
        
        # 기본 처리 시간 (레이어당)
        base_processing_time = 0.001  # 초
        
        # 레이어별 처리 시간 계산
        total_processing = 0.0
        layer_latencies = []
        
        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "dense")
            units = layer.get("units", 128)
            
            # 레이어 타입별 처리 시간
            if layer_type == "conv2d":
                # 컨볼루션 레이어: 커널 크기, 채널 수에 비례
                kernel_size = layer.get("kernel_size", 3)
                filters = layer.get("filters", 32)
                processing = base_processing_time * kernel_size * filters * (input_size / 1000)
            elif layer_type == "lstm" or layer_type == "gru":
                # 순환 레이어: 유닛 수와 시퀀스 길이에 비례
                sequence_length = layer.get("sequence_length", 100)
                processing = base_processing_time * units * sequence_length * 2
            elif layer_type == "transformer":
                # 트랜스포머: 어텐션 메커니즘 복잡도
                num_heads = layer.get("num_heads", 8)
                d_model = layer.get("d_model", 512)
                processing = base_processing_time * num_heads * d_model * math.log2(input_size)
            else:
                # 일반 레이어 (dense 등)
                processing = base_processing_time * units * (input_size / 1000)
            
            # 배치 크기 영향
            processing = processing / math.sqrt(batch_size)
            
            # 노이즈 추가
            processing = self.prob_dist.sample(
                ProbabilityDistributions.DistributionType.NORMAL,
                mean=processing, std=processing * 0.1
            )
            processing = max(0.0001, processing)
            
            layer_latencies.append({
                "layer_index": i,
                "layer_type": layer_type,
                "latency_ms": processing * 1000
            })
            total_processing += processing
        
        # 네트워크 지연 (레이어 간 통신)
        network_latency = 0.0005 * (num_layers - 1)  # 레이어 간 평균 지연
        
        # 총 지연 시간
        total_latency = total_processing + network_latency
        
        return {
            "total_latency_ms": round(total_latency * 1000, 2),
            "processing_latency_ms": round(total_processing * 1000, 2),
            "network_latency_ms": round(network_latency * 1000, 2),
            "layer_latencies": layer_latencies,
            "num_layers": num_layers
        }
    
    def calculate_throughput(
        self,
        architecture_config: Dict[str, Any],
        batch_size: int,
        latency_ms: float
    ) -> Dict[str, float]:
        """
        처리량 계산 (큐잉 이론 기반)
        
        Args:
            architecture_config: 아키텍처 구성
            batch_size: 배치 크기
            latency_ms: 지연 시간 (밀리초)
        """
        # 기본 처리량 (배치/초)
        if latency_ms > 0:
            base_throughput = (1000.0 / latency_ms) * batch_size
        else:
            base_throughput = 0.0
        
        # 병렬 처리 고려
        num_gpus = architecture_config.get("num_gpus", 1)
        parallel_efficiency = 0.85  # 병렬 처리 효율 (85%)
        effective_throughput = base_throughput * num_gpus * parallel_efficiency
        
        # 큐잉 이론: M/M/1 큐 모델
        arrival_rate = effective_throughput * 0.8  # 80% 부하
        service_rate = effective_throughput
        
        if service_rate > arrival_rate:
            utilization = arrival_rate / service_rate
            queue_length = utilization / (1 - utilization)
            waiting_time = queue_length / service_rate
        else:
            utilization = 1.0
            queue_length = float('inf')
            waiting_time = float('inf')
        
        # 노이즈 추가
        effective_throughput = self.prob_dist.sample(
            ProbabilityDistributions.DistributionType.NORMAL,
            mean=effective_throughput, std=effective_throughput * 0.05
        )
        effective_throughput = max(0, effective_throughput)
        
        return {
            "throughput_samples_per_sec": round(effective_throughput, 2),
            "throughput_batches_per_sec": round(effective_throughput / batch_size, 2),
            "utilization": round(utilization, 3),
            "queue_length": round(queue_length, 2) if queue_length != float('inf') else None,
            "waiting_time_ms": round(waiting_time * 1000, 2) if waiting_time != float('inf') else None
        }
    
    def simulate_training_time(
        self,
        architecture_config: Dict[str, Any],
        dataset_size: int,
        epochs: int,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        학습 시간 추정 (데이터셋 크기, 모델 복잡도 기반)
        
        Args:
            architecture_config: 아키텍처 구성
            dataset_size: 데이터셋 크기 (샘플 수)
            epochs: 에포크 수
            batch_size: 배치 크기
        """
        layers = architecture_config.get("layers", [])
        
        # 모델 파라미터 수 추정
        total_params = 0
        for layer in layers:
            layer_type = layer.get("type", "dense")
            units = layer.get("units", 128)
            
            if layer_type == "dense":
                # Dense: input_units * output_units + bias
                input_units = layer.get("input_units", 128)
                total_params += input_units * units + units
            elif layer_type == "conv2d":
                # Conv2D: kernel_size^2 * input_channels * filters + bias
                kernel_size = layer.get("kernel_size", 3)
                input_channels = layer.get("input_channels", 3)
                filters = layer.get("filters", 32)
                total_params += kernel_size * kernel_size * input_channels * filters + filters
            elif layer_type == "lstm" or layer_type == "gru":
                # LSTM/GRU: 4 * (input_units * units + units^2 + units)
                input_units = layer.get("input_units", 128)
                total_params += 4 * (input_units * units + units * units + units)
        
        # 학습 시간 계산 (파라미터 수, 데이터셋 크기, 배치 크기 기반)
        batches_per_epoch = math.ceil(dataset_size / batch_size)
        
        # 배치당 처리 시간 (파라미터 수에 비례)
        time_per_batch = 0.001 * (total_params / 1000000)  # 100만 파라미터당 1ms
        
        # 에포크당 시간
        time_per_epoch = time_per_batch * batches_per_epoch
        
        # 총 학습 시간
        total_training_time = time_per_epoch * epochs
        
        # GPU 가속 고려
        num_gpus = architecture_config.get("num_gpus", 1)
        gpu_speedup = 1.0 + (num_gpus - 1) * 0.7  # GPU당 70% 추가 속도
        total_training_time = total_training_time / gpu_speedup
        
        # 노이즈 추가
        total_training_time = self.prob_dist.sample(
            ProbabilityDistributions.DistributionType.NORMAL,
            mean=total_training_time, std=total_training_time * 0.1
        )
        total_training_time = max(0, total_training_time)
        
        return {
            "total_training_time_seconds": round(total_training_time, 2),
            "total_training_time_hours": round(total_training_time / 3600, 2),
            "time_per_epoch_seconds": round(time_per_epoch / gpu_speedup, 2),
            "time_per_batch_ms": round(time_per_batch * 1000, 2),
            "total_parameters": total_params,
            "batches_per_epoch": batches_per_epoch,
            "gpu_speedup": round(gpu_speedup, 2)
        }
    
    def calculate_resource_usage(
        self,
        architecture_config: Dict[str, Any],
        batch_size: int,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        리소스 사용량 계산 (GPU, CPU, 메모리)
        
        Args:
            architecture_config: 아키텍처 구성
            batch_size: 배치 크기
            input_shape: 입력 형태
        """
        layers = architecture_config.get("layers", [])
        
        # 메모리 사용량 계산
        memory_per_layer = []
        total_memory_mb = 0.0
        
        for layer in layers:
            layer_type = layer.get("type", "dense")
            units = layer.get("units", 128)
            
            if layer_type == "dense":
                # Dense: weights + activations
                input_units = layer.get("input_units", 128)
                weights_mb = (input_units * units * 4) / (1024 * 1024)  # float32
                activations_mb = (batch_size * units * 4) / (1024 * 1024)
                layer_memory = weights_mb + activations_mb
            elif layer_type == "conv2d":
                # Conv2D: weights + feature maps
                kernel_size = layer.get("kernel_size", 3)
                filters = layer.get("filters", 32)
                input_channels = layer.get("input_channels", 3)
                weights_mb = (kernel_size * kernel_size * input_channels * filters * 4) / (1024 * 1024)
                # Feature map 크기 추정
                feature_map_size = input_shape[0] * input_shape[1] if len(input_shape) >= 2 else 224 * 224
                activations_mb = (batch_size * feature_map_size * filters * 4) / (1024 * 1024)
                layer_memory = weights_mb + activations_mb
            else:
                # 기본 추정
                layer_memory = (units * 4) / (1024 * 1024) * batch_size
            
            memory_per_layer.append({
                "layer_type": layer_type,
                "memory_mb": round(layer_memory, 2)
            })
            total_memory_mb += layer_memory
        
        # GPU 메모리 (추가 오버헤드)
        gpu_overhead = total_memory_mb * 0.2  # 20% 오버헤드
        total_gpu_memory = total_memory_mb + gpu_overhead
        
        # CPU 사용률 추정 (데이터 로딩, 전처리 등)
        cpu_usage_percent = 20.0 + (batch_size / 32) * 10.0  # 배치 크기에 비례
        cpu_usage_percent = min(100, cpu_usage_percent)
        
        # GPU 사용률 (학습 중)
        gpu_usage_percent = 85.0 + self.rng.uniform(-5, 5)
        gpu_usage_percent = max(70, min(100, gpu_usage_percent))
        
        return {
            "total_memory_mb": round(total_memory_mb, 2),
            "total_gpu_memory_mb": round(total_gpu_memory, 2),
            "cpu_usage_percent": round(cpu_usage_percent, 1),
            "gpu_usage_percent": round(gpu_usage_percent, 1),
            "memory_per_layer": memory_per_layer
        }
    
    def estimate_cost(
        self,
        architecture_config: Dict[str, Any],
        training_time_hours: float,
        inference_requests_per_month: int = 0,
        cloud_provider: str = "aws"
    ) -> Dict[str, Any]:
        """
        비용 추정 (클라우드 가격표 기반)
        
        Args:
            architecture_config: 아키텍처 구성
            training_time_hours: 학습 시간 (시간)
            inference_requests_per_month: 월 추론 요청 수
            cloud_provider: 클라우드 제공자
        """
        num_gpus = architecture_config.get("num_gpus", 1)
        gpu_type = architecture_config.get("gpu_type", "V100")
        
        # GPU 가격 (시간당 USD)
        gpu_prices = {
            "V100": 3.06,
            "A100": 11.00,
            "T4": 0.35,
            "P100": 1.46
        }
        gpu_price_per_hour = gpu_prices.get(gpu_type, 3.06)
        
        # 학습 비용
        training_cost = gpu_price_per_hour * num_gpus * training_time_hours
        
        # 추론 비용 (요청당)
        if cloud_provider == "aws":
            inference_cost_per_1k = 0.0001  # $0.0001 per 1k requests
        elif cloud_provider == "gcp":
            inference_cost_per_1k = 0.00008
        else:
            inference_cost_per_1k = 0.0001
        
        inference_cost = (inference_requests_per_month / 1000) * inference_cost_per_1k
        
        # 스토리지 비용 (모델 크기 기반)
        model_size_mb = architecture_config.get("model_size_mb", 100)
        storage_cost_per_gb_month = 0.023  # AWS S3 standard
        storage_cost = (model_size_mb / 1024) * storage_cost_per_gb_month
        
        # 총 비용
        total_cost = training_cost + inference_cost + storage_cost
        
        return {
            "training_cost_usd": round(training_cost, 2),
            "inference_cost_usd": round(inference_cost, 2),
            "storage_cost_usd": round(storage_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "cost_per_hour": round(gpu_price_per_hour * num_gpus, 2),
            "gpu_type": gpu_type,
            "num_gpus": num_gpus
        }
    
    def evolve_architecture(
        self,
        parent_architectures: List[Dict[str, Any]],
        fitness_scores: List[float],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        유전 알고리즘을 사용한 아키텍처 진화
        
        Args:
            parent_architectures: 부모 아키텍처 리스트
            fitness_scores: 적합도 점수 리스트
            mutation_rate: 돌연변이율
            crossover_rate: 교차율
        """
        if len(parent_architectures) == 0:
            return []
        
        # 적합도 기반 선택 확률 계산
        min_fitness = min(fitness_scores)
        adjusted_scores = [score - min_fitness + 0.1 for score in fitness_scores]
        total_fitness = sum(adjusted_scores)
        selection_probs = [score / total_fitness for score in adjusted_scores]
        
        offspring = []
        
        # 자손 생성
        for _ in range(len(parent_architectures)):
            # 선택 (룰렛 휠)
            parent1_idx = self._roulette_wheel_selection(selection_probs)
            parent2_idx = self._roulette_wheel_selection(selection_probs)
            
            parent1 = parent_architectures[parent1_idx].copy()
            parent2 = parent_architectures[parent2_idx].copy()
            
            # 교차
            if self.rng.random() < crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # 돌연변이
            if self.rng.random() < mutation_rate:
                child = self._mutate(child)
            
            offspring.append(child)
        
        return offspring
    
    def _roulette_wheel_selection(self, probabilities: List[float]) -> int:
        """룰렛 휠 선택"""
        r = self.rng.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        return len(probabilities) - 1
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """교차 연산"""
        child = parent1.copy()
        
        # 레이어 교차
        if "layers" in parent1 and "layers" in parent2:
            layers1 = parent1["layers"]
            layers2 = parent2["layers"]
            
            if len(layers1) > 0 and len(layers2) > 0:
                crossover_point = self.rng.randint(1, min(len(layers1), len(layers2)))
                child["layers"] = layers1[:crossover_point] + layers2[crossover_point:]
        
        return child
    
    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """돌연변이 연산"""
        mutated = architecture.copy()
        
        # 레이어 수 변경
        if "layers" in mutated and len(mutated["layers"]) > 0:
            mutation_type = self.rng.choice(["add", "remove", "modify"])
            
            if mutation_type == "add" and len(mutated["layers"]) < 10:
                # 레이어 추가
                new_layer = {
                    "type": self.rng.choice(["dense", "conv2d", "lstm"]),
                    "units": self.rng.randint(32, 512)
                }
                insert_pos = self.rng.randint(0, len(mutated["layers"]))
                mutated["layers"].insert(insert_pos, new_layer)
            
            elif mutation_type == "remove" and len(mutated["layers"]) > 1:
                # 레이어 제거
                remove_pos = self.rng.randint(0, len(mutated["layers"]) - 1)
                mutated["layers"].pop(remove_pos)
            
            elif mutation_type == "modify":
                # 레이어 수정
                modify_pos = self.rng.randint(0, len(mutated["layers"]) - 1)
                layer = mutated["layers"][modify_pos]
                layer["units"] = max(16, layer.get("units", 128) + self.rng.randint(-32, 32))
        
        return mutated






