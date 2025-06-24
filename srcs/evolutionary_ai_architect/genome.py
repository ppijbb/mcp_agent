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
    for evolutionary tracking.
    """
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    hyperparameters: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    unique_id: str = None
    
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