"""
Genome and Performance Metrics Data Structures

This module defines the core data structures for evolutionary AI:
- ArchitectureGenome: Genetic encoding of AI architectures
- PerformanceMetrics: Performance tracking and evaluation metrics
"""

import json
import hashlib
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ArchitectureGenome:
    """
    Represents a genetic encoding of an AI architecture
    
    This class stores the complete specification of an AI model architecture
    including layers, connections, and hyperparameters, along with metadata
    for evolutionary tracking and scaling laws integration.
    """
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    hyperparameters: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    unique_id: str = None
    
    # Scaling Laws 관련 필드
    estimated_parameters: int = 0
    training_tokens: int = 0
    predicted_loss: float = 0.0
    compute_budget: float = 0.0
    scaling_efficiency: float = 0.0
    
    def __post_init__(self):
        if self.unique_id is None:
            # Generate unique ID based on architecture content
            content_str = json.dumps(asdict(self), sort_keys=True)
            self.unique_id = hashlib.md5(content_str.encode()).hexdigest()[:12]
        
        if self.parent_ids is None:
            self.parent_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary representation"""
        return asdict(self)
    
    def get_complexity_score(self) -> float:
        """Calculate architecture complexity score"""
        base_score = len(self.layers) * 0.1
        
        # Add complexity based on layer types
        for layer in self.layers:
            if layer['type'] == 'transformer_block':
                base_score += 0.3
            elif layer['type'] == 'conv2d':
                base_score += 0.2
            elif layer['type'] == 'lstm':
                base_score += 0.25
        
        return min(base_score, 1.0)
    
    def get_layer_types(self) -> List[str]:
        """Get unique layer types in this architecture"""
        return list(set(layer['type'] for layer in self.layers))
    
    def calculate_parameter_count(self) -> int:
        """Calculate total parameter count from layers"""
        if self.estimated_parameters > 0:
            return self.estimated_parameters
        
        total_params = 0
        for layer in self.layers:
            layer_params = layer.get('parameters', 0)
            if isinstance(layer_params, (int, float)):
                total_params += int(layer_params)
            elif isinstance(layer_params, dict):
                # 복잡한 파라미터 구조의 경우 추정
                total_params += sum(int(v) for v in layer_params.values() if isinstance(v, (int, float)))
        
        return total_params
    
    def calculate_training_tokens(self, dataset_size: int = None) -> int:
        """Calculate required training tokens"""
        if self.training_tokens > 0:
            return self.training_tokens
        
        if dataset_size is not None:
            return dataset_size
        
        # 기본 추정: 파라미터 수의 10배
        return self.calculate_parameter_count() * 10
    
    def calculate_compute_requirements(self) -> float:
        """Calculate FLOPs requirements using scaling laws"""
        if self.compute_budget > 0:
            return self.compute_budget
        
        n_params = self.calculate_parameter_count()
        n_tokens = self.calculate_training_tokens()
        
        # 6 * N * D 공식 (forward + backward pass)
        return 6.0 * n_params * n_tokens
    
    def update_scaling_metrics(self, scaling_data: Dict[str, Any]) -> None:
        """Update scaling-related metrics from external calculations"""
        self.estimated_parameters = scaling_data.get('optimal_parameters', self.estimated_parameters)
        self.training_tokens = scaling_data.get('optimal_tokens', self.training_tokens)
        self.predicted_loss = scaling_data.get('predicted_loss', self.predicted_loss)
        self.compute_budget = scaling_data.get('required_compute', self.compute_budget)
        self.scaling_efficiency = scaling_data.get('efficiency_ratio', self.scaling_efficiency)
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling information"""
        return {
            'estimated_parameters': self.estimated_parameters,
            'training_tokens': self.training_tokens,
            'predicted_loss': self.predicted_loss,
            'compute_budget': self.compute_budget,
            'scaling_efficiency': self.scaling_efficiency,
            'parameter_count': self.calculate_parameter_count(),
            'compute_requirements': self.calculate_compute_requirements()
        }


@dataclass
class PerformanceMetrics:
    """
    Tracks various performance metrics for self-improvement
    
    This class stores comprehensive performance measurements across
    multiple dimensions to enable holistic evaluation and improvement.
    """
    accuracy: float = 0.0
    efficiency: float = 0.0
    adaptability: float = 0.0
    creativity_score: float = 0.0
    problem_solving_time: float = 0.0
    resource_usage: float = 0.0
    success_rate: float = 0.0
    learning_speed: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall performance score"""
        weights = {
            'accuracy': 0.25,
            'efficiency': 0.15,
            'adaptability': 0.20,
            'creativity_score': 0.15,
            'problem_solving_time': 0.10,
            'resource_usage': 0.05,
            'success_rate': 0.20,
            'learning_speed': 0.10
        }
        
        score = sum(getattr(self, metric) * weight for metric, weight in weights.items())
        return min(score, 1.0)  # Cap at 1.0
    
    def get_strengths(self) -> List[str]:
        """Identify performance strengths (metrics above 0.7)"""
        strengths = []
        metrics = {
            'accuracy': self.accuracy,
            'efficiency': self.efficiency,
            'adaptability': self.adaptability,
            'creativity_score': self.creativity_score,
            'success_rate': self.success_rate,
            'learning_speed': self.learning_speed
        }
        
        for metric, value in metrics.items():
            if value > 0.7:
                strengths.append(metric)
        
        return strengths
    
    def get_weaknesses(self) -> List[str]:
        """Identify performance weaknesses (metrics below 0.5)"""
        weaknesses = []
        metrics = {
            'accuracy': self.accuracy,
            'efficiency': self.efficiency,
            'adaptability': self.adaptability,
            'creativity_score': self.creativity_score,
            'success_rate': self.success_rate,
            'learning_speed': self.learning_speed
        }
        
        for metric, value in metrics.items():
            if value < 0.5:
                weaknesses.append(metric)
        
        return weaknesses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation"""
        return asdict(self)


@dataclass
class EvolutionHistory:
    """
    Tracks the evolutionary history of architectures
    """
    generation: int
    best_fitness: float
    average_fitness: float
    diversity_score: float
    improvements: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary representation"""
        return asdict(self) 