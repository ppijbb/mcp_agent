"""
AI Architecture Designer and Evolution Engine

This module handles the design, generation, mutation, and crossover
of AI architectures using evolutionary algorithms.
"""

import random
import logging
from typing import Dict, List, Any, Optional
from .genome import ArchitectureGenome

logger = logging.getLogger(__name__)


class AIArchitectureDesigner:
    """
    Designs and generates AI architectures using evolutionary principles
    
    This class provides methods to:
    - Generate random architectures
    - Mutate existing architectures  
    - Crossover between parent architectures
    - Evaluate architecture fitness
    """
    
    def __init__(self):
        self.architecture_templates = {
            'transformer': {
                'attention_heads': [4, 8, 12, 16],
                'hidden_size': [256, 512, 768, 1024],
                'num_layers': [4, 6, 8, 12],
                'dropout': [0.1, 0.2, 0.3],
                'activation': ['relu', 'gelu', 'swish']
            },
            'cnn': {
                'conv_layers': [2, 3, 4, 5],
                'filters': [32, 64, 128, 256],
                'kernel_size': [3, 5, 7],
                'pooling': ['max', 'avg', 'global'],
                'activation': ['relu', 'leaky_relu', 'elu']
            },
            'rnn': {
                'units': [64, 128, 256, 512],
                'layers': [1, 2, 3],
                'cell_type': ['LSTM', 'GRU', 'RNN'],
                'bidirectional': [True, False],
                'dropout': [0.0, 0.1, 0.2]
            },
            'hybrid': {
                'components': ['transformer', 'cnn', 'rnn'],
                'fusion_method': ['concat', 'attention', 'gate', 'residual']
            }
        }
        
        self.generation_stats = {
            'total_generated': 0,
            'successful_mutations': 0,
            'successful_crossovers': 0
        }
    
    def generate_random_architecture(self, architecture_type: str = None, 
                                   complexity_target: float = 0.5) -> ArchitectureGenome:
        """
        Generate a random AI architecture
        
        Args:
            architecture_type: Type of architecture ('transformer', 'cnn', 'rnn', 'hybrid')
            complexity_target: Target complexity level (0.0 to 1.0)
            
        Returns:
            ArchitectureGenome: Generated architecture
        """
        if architecture_type is None:
            architecture_type = random.choice(list(self.architecture_templates.keys()))
        
        template = self.architecture_templates[architecture_type]
        layers = []
        connections = []
        
        if architecture_type == 'transformer':
            layers, connections = self._generate_transformer_layers(template, complexity_target)
        elif architecture_type == 'cnn':
            layers, connections = self._generate_cnn_layers(template, complexity_target)
        elif architecture_type == 'rnn':
            layers, connections = self._generate_rnn_layers(template, complexity_target)
        elif architecture_type == 'hybrid':
            layers, connections = self._generate_hybrid_layers(template, complexity_target)
        
        hyperparameters = self._generate_hyperparameters(architecture_type)
        
        genome = ArchitectureGenome(
            layers=layers,
            connections=connections,
            hyperparameters=hyperparameters
        )
        
        self.generation_stats['total_generated'] += 1
        logger.debug(f"Generated {architecture_type} architecture with ID: {genome.unique_id}")
        
        return genome
    
    def _generate_transformer_layers(self, template: Dict, complexity: float) -> tuple:
        """Generate transformer architecture layers"""
        num_layers = int(complexity * max(template['num_layers']))
        num_layers = max(2, min(num_layers, max(template['num_layers'])))
        
        layers = []
        connections = []
        
        for i in range(num_layers):
            layer = {
                'type': 'transformer_block',
                'attention_heads': random.choice(template['attention_heads']),
                'hidden_size': random.choice(template['hidden_size']),
                'dropout': random.choice(template['dropout']),
                'activation': random.choice(template['activation'])
            }
            layers.append(layer)
            
            if i > 0:
                connections.append((i-1, i))
        
        # Add input/output layers
        input_layer = {'type': 'embedding', 'size': layers[0]['hidden_size']}
        output_layer = {'type': 'classification', 'units': random.choice([2, 10, 100, 1000])}
        
        layers.insert(0, input_layer)
        layers.append(output_layer)
        connections.insert(0, (0, 1))
        connections.append((len(layers)-2, len(layers)-1))
        
        return layers, connections
    
    def _generate_cnn_layers(self, template: Dict, complexity: float) -> tuple:
        """Generate CNN architecture layers"""
        num_conv = int(complexity * max(template['conv_layers']))
        num_conv = max(1, min(num_conv, max(template['conv_layers'])))
        
        layers = []
        connections = []
        
        for i in range(num_conv):
            layer = {
                'type': 'conv2d',
                'filters': random.choice(template['filters']),
                'kernel_size': random.choice(template['kernel_size']),
                'activation': random.choice(template['activation']),
                'padding': 'same'
            }
            layers.append(layer)
            
            # Add pooling layer occasionally
            if random.random() < 0.6:
                pool_layer = {
                    'type': 'pooling',
                    'method': random.choice(template['pooling']),
                    'size': 2
                }
                layers.append(pool_layer)
        
        # Generate connections
        for i in range(len(layers) - 1):
            connections.append((i, i + 1))
        
        # Add final layers
        flatten_layer = {'type': 'flatten'}
        dense_layer = {'type': 'dense', 'units': random.choice([64, 128, 256, 512])}
        output_layer = {'type': 'classification', 'units': random.choice([2, 10, 100, 1000])}
        
        layers.extend([flatten_layer, dense_layer, output_layer])
        connections.extend([
            (len(layers)-4, len(layers)-3),
            (len(layers)-3, len(layers)-2),
            (len(layers)-2, len(layers)-1)
        ])
        
        return layers, connections
    
    def _generate_rnn_layers(self, template: Dict, complexity: float) -> tuple:
        """Generate RNN architecture layers"""
        num_layers = int(complexity * max(template['layers']))
        num_layers = max(1, min(num_layers, max(template['layers'])))
        
        layers = []
        connections = []
        
        for i in range(num_layers):
            layer = {
                'type': random.choice(template['cell_type']).lower(),
                'units': random.choice(template['units']),
                'bidirectional': random.choice(template['bidirectional']),
                'dropout': random.choice(template['dropout']),
                'return_sequences': i < num_layers - 1
            }
            layers.append(layer)
            
            if i > 0:
                connections.append((i-1, i))
        
        # Add output layer
        output_layer = {'type': 'dense', 'units': random.choice([2, 10, 100, 1000])}
        layers.append(output_layer)
        connections.append((len(layers)-2, len(layers)-1))
        
        return layers, connections
    
    def _generate_hybrid_layers(self, template: Dict, complexity: float) -> tuple:
        """Generate hybrid architecture combining multiple types"""
        num_components = int(complexity * 3) + 2  # 2-4 components
        selected_components = random.sample(template['components'], min(num_components, len(template['components'])))
        
        layers = []
        connections = []
        current_idx = 0
        
        for component in selected_components:
            if component == 'transformer':
                comp_layers, comp_connections = self._generate_transformer_layers(
                    self.architecture_templates['transformer'], 0.3
                )
            elif component == 'cnn':
                comp_layers, comp_connections = self._generate_cnn_layers(
                    self.architecture_templates['cnn'], 0.3
                )
            else:  # rnn
                comp_layers, comp_connections = self._generate_rnn_layers(
                    self.architecture_templates['rnn'], 0.3
                )
            
            # Adjust connection indices
            adjusted_connections = [(c[0] + current_idx, c[1] + current_idx) for c in comp_connections]
            
            layers.extend(comp_layers)
            connections.extend(adjusted_connections)
            current_idx = len(layers)
        
        # Add fusion layer
        fusion_layer = {
            'type': 'fusion',
            'method': random.choice(template['fusion_method']),
            'inputs': len(selected_components)
        }
        layers.append(fusion_layer)
        
        return layers, connections
    
    def _generate_hyperparameters(self, architecture_type: str) -> Dict[str, Any]:
        """Generate hyperparameters for the architecture"""
        base_params = {
            'learning_rate': random.uniform(0.0001, 0.01),
            'batch_size': random.choice([16, 32, 64, 128, 256]),
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop', 'adamw']),
            'epochs': random.randint(10, 200),
            'weight_decay': random.uniform(0.0, 0.01)
        }
        
        # Add architecture-specific parameters
        if architecture_type == 'transformer':
            base_params.update({
                'warmup_steps': random.randint(1000, 10000),
                'gradient_clip': random.uniform(0.5, 2.0)
            })
        elif architecture_type == 'cnn':
            base_params.update({
                'data_augmentation': random.choice([True, False]),
                'batch_norm': random.choice([True, False])
            })
        elif architecture_type == 'rnn':
            base_params.update({
                'sequence_length': random.choice([50, 100, 200, 500]),
                'gradient_clip': random.uniform(1.0, 5.0)
            })
        
        return base_params
    
    def mutate_architecture(self, genome: ArchitectureGenome, 
                          mutation_rate: float = 0.1,
                          mutation_strength: float = 0.5) -> ArchitectureGenome:
        """
        Apply mutations to an architecture genome
        
        Args:
            genome: Architecture to mutate
            mutation_rate: Probability of mutation per component
            mutation_strength: Strength of mutations (0.0 to 1.0)
            
        Returns:
            ArchitectureGenome: Mutated architecture
        """
        new_layers = [layer.copy() for layer in genome.layers]
        new_connections = genome.connections.copy()
        new_hyperparams = genome.hyperparameters.copy()
        
        mutations_applied = []
        
        # Mutate layers
        for i, layer in enumerate(new_layers):
            if random.random() < mutation_rate:
                mutations_applied.extend(self._mutate_layer(layer, mutation_strength))
        
        # Mutate hyperparameters
        if random.random() < mutation_rate:
            mutations_applied.extend(self._mutate_hyperparameters(new_hyperparams, mutation_strength))
        
        # Structural mutations (add/remove layers)
        if random.random() < mutation_rate * 0.5:  # Lower probability for structural changes
            if random.random() < 0.5 and len(new_layers) > 2:
                # Remove a layer
                remove_idx = random.randint(1, len(new_layers) - 2)  # Don't remove input/output
                new_layers.pop(remove_idx)
                new_connections = [(c[0], c[1]) for c in new_connections if c[0] != remove_idx and c[1] != remove_idx]
                # Adjust connection indices
                new_connections = [(c[0] if c[0] < remove_idx else c[0] - 1,
                                  c[1] if c[1] < remove_idx else c[1] - 1) for c in new_connections]
                mutations_applied.append("removed_layer")
            else:
                # Add a layer (simplified)
                if len(new_layers) < 20:  # Prevent excessive growth
                    new_layer = self._generate_compatible_layer(new_layers)
                    insert_idx = random.randint(1, len(new_layers) - 1)
                    new_layers.insert(insert_idx, new_layer)
                    # Update connections
                    new_connections = [(c[0] if c[0] < insert_idx else c[0] + 1,
                                      c[1] if c[1] < insert_idx else c[1] + 1) for c in new_connections]
                    mutations_applied.append("added_layer")
        
        mutated_genome = ArchitectureGenome(
            layers=new_layers,
            connections=new_connections,
            hyperparameters=new_hyperparams,
            generation=genome.generation + 1,
            parent_ids=[genome.unique_id]
        )
        
        if mutations_applied:
            self.generation_stats['successful_mutations'] += 1
            logger.debug(f"Applied mutations: {mutations_applied} to {genome.unique_id}")
        
        return mutated_genome
    
    def _mutate_layer(self, layer: Dict[str, Any], strength: float) -> List[str]:
        """Mutate a single layer"""
        mutations = []
        
        if layer['type'] == 'transformer_block':
            if 'attention_heads' in layer and random.random() < strength:
                layer['attention_heads'] = random.choice([4, 8, 12, 16])
                mutations.append("attention_heads")
            if 'hidden_size' in layer and random.random() < strength:
                layer['hidden_size'] = random.choice([256, 512, 768, 1024])
                mutations.append("hidden_size")
                
        elif layer['type'] == 'conv2d':
            if 'filters' in layer and random.random() < strength:
                layer['filters'] = random.choice([32, 64, 128, 256])
                mutations.append("filters")
            if 'kernel_size' in layer and random.random() < strength:
                layer['kernel_size'] = random.choice([3, 5, 7])
                mutations.append("kernel_size")
                
        elif layer['type'] in ['lstm', 'gru', 'rnn']:
            if 'units' in layer and random.random() < strength:
                layer['units'] = random.choice([64, 128, 256, 512])
                mutations.append("units")
        
        return mutations
    
    def _mutate_hyperparameters(self, hyperparams: Dict[str, Any], strength: float) -> List[str]:
        """Mutate hyperparameters"""
        mutations = []
        
        if random.random() < strength:
            hyperparams['learning_rate'] = random.uniform(0.0001, 0.01)
            mutations.append("learning_rate")
        
        if random.random() < strength:
            hyperparams['batch_size'] = random.choice([16, 32, 64, 128, 256])
            mutations.append("batch_size")
            
        if random.random() < strength:
            hyperparams['optimizer'] = random.choice(['adam', 'sgd', 'rmsprop', 'adamw'])
            mutations.append("optimizer")
        
        return mutations
    
    def _generate_compatible_layer(self, existing_layers: List[Dict]) -> Dict[str, Any]:
        """Generate a layer compatible with existing architecture"""
        layer_types = [layer['type'] for layer in existing_layers]
        
        if 'transformer_block' in layer_types:
            return {
                'type': 'transformer_block',
                'attention_heads': random.choice([4, 8, 12]),
                'hidden_size': random.choice([256, 512]),
                'dropout': 0.1
            }
        elif 'conv2d' in layer_types:
            return {
                'type': 'conv2d',
                'filters': random.choice([32, 64, 128]),
                'kernel_size': 3,
                'activation': 'relu'
            }
        else:
            return {
                'type': 'dense',
                'units': random.choice([64, 128, 256]),
                'activation': 'relu'
            }
    
    def crossover_architectures(self, parent1: ArchitectureGenome, 
                              parent2: ArchitectureGenome) -> ArchitectureGenome:
        """
        Create offspring by combining two parent architectures
        
        Args:
            parent1: First parent architecture
            parent2: Second parent architecture
            
        Returns:
            ArchitectureGenome: Offspring architecture
        """
        # Choose crossover strategy
        crossover_methods = ['layer_crossover', 'hyperparameter_crossover', 'hybrid_crossover']
        method = random.choice(crossover_methods)
        
        if method == 'layer_crossover':
            offspring = self._layer_crossover(parent1, parent2)
        elif method == 'hyperparameter_crossover':
            offspring = self._hyperparameter_crossover(parent1, parent2)
        else:
            offspring = self._hybrid_crossover(parent1, parent2)
        
        self.generation_stats['successful_crossovers'] += 1
        logger.debug(f"Crossover between {parent1.unique_id} and {parent2.unique_id} using {method}")
        
        return offspring
    
    def _layer_crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Crossover based on layer exchange"""
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_layers - 1)
        
        new_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        new_connections = parent1.connections[:crossover_point] + parent2.connections[crossover_point:]
        
        # Combine hyperparameters randomly
        new_hyperparams = {}
        for key in set(parent1.hyperparameters.keys()) | set(parent2.hyperparameters.keys()):
            if key in parent1.hyperparameters and key in parent2.hyperparameters:
                new_hyperparams[key] = random.choice([parent1.hyperparameters[key], parent2.hyperparameters[key]])
            elif key in parent1.hyperparameters:
                new_hyperparams[key] = parent1.hyperparameters[key]
            else:
                new_hyperparams[key] = parent2.hyperparameters[key]
        
        return ArchitectureGenome(
            layers=new_layers,
            connections=new_connections,
            hyperparameters=new_hyperparams,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.unique_id, parent2.unique_id]
        )
    
    def _hyperparameter_crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Crossover focusing on hyperparameters"""
        # Choose the better parent's architecture
        if parent1.fitness_score > parent2.fitness_score:
            new_layers = parent1.layers.copy()
            new_connections = parent1.connections.copy()
        else:
            new_layers = parent2.layers.copy()
            new_connections = parent2.connections.copy()
        
        # Mix hyperparameters
        new_hyperparams = {}
        for key in set(parent1.hyperparameters.keys()) | set(parent2.hyperparameters.keys()):
            if key in parent1.hyperparameters and key in parent2.hyperparameters:
                new_hyperparams[key] = random.choice([parent1.hyperparameters[key], parent2.hyperparameters[key]])
            elif key in parent1.hyperparameters:
                new_hyperparams[key] = parent1.hyperparameters[key]
            else:
                new_hyperparams[key] = parent2.hyperparameters[key]
        
        return ArchitectureGenome(
            layers=new_layers,
            connections=new_connections,
            hyperparameters=new_hyperparams,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.unique_id, parent2.unique_id]
        )
    
    def _hybrid_crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Advanced hybrid crossover combining multiple strategies"""
        # Combine layers from both parents in alternating fashion
        new_layers = []
        max_layers = max(len(parent1.layers), len(parent2.layers))
        
        for i in range(max_layers):
            if i < len(parent1.layers) and i < len(parent2.layers):
                # Choose layer based on fitness-weighted probability
                prob = parent1.fitness_score / (parent1.fitness_score + parent2.fitness_score + 1e-8)
                chosen_layer = parent1.layers[i] if random.random() < prob else parent2.layers[i]
            elif i < len(parent1.layers):
                chosen_layer = parent1.layers[i]
            else:
                chosen_layer = parent2.layers[i]
            
            new_layers.append(chosen_layer.copy())
        
        # Generate new connections based on new layer structure
        new_connections = []
        for i in range(len(new_layers) - 1):
            new_connections.append((i, i + 1))
        
        # Blend hyperparameters using weighted averaging for numeric values
        new_hyperparams = {}
        p1_weight = parent1.fitness_score / (parent1.fitness_score + parent2.fitness_score + 1e-8)
        p2_weight = 1.0 - p1_weight
        
        for key in set(parent1.hyperparameters.keys()) | set(parent2.hyperparameters.keys()):
            if key in parent1.hyperparameters and key in parent2.hyperparameters:
                val1, val2 = parent1.hyperparameters[key], parent2.hyperparameters[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    new_hyperparams[key] = val1 * p1_weight + val2 * p2_weight
                else:
                    new_hyperparams[key] = random.choice([val1, val2])
            elif key in parent1.hyperparameters:
                new_hyperparams[key] = parent1.hyperparameters[key]
            else:
                new_hyperparams[key] = parent2.hyperparameters[key]
        
        return ArchitectureGenome(
            layers=new_layers,
            connections=new_connections,
            hyperparameters=new_hyperparams,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.unique_id, parent2.unique_id]
        )
    
    def evaluate_architecture(self, genome: ArchitectureGenome, task_context: Dict[str, Any] = None) -> float:
        """Evaluate an architecture's fitness for a given task"""
        base_score = 0.5
        
        # Reward certain architecture characteristics
        num_layers = len(genome.layers)
        if 3 <= num_layers <= 8:  # Sweet spot for many tasks
            base_score += 0.1
        
        # Check for diversity in layer types
        layer_types = genome.get_layer_types()
        if len(layer_types) > 1:
            base_score += 0.1
        
        # Evaluate hyperparameters
        lr = genome.hyperparameters.get('learning_rate', 0.001)
        if 0.0001 <= lr <= 0.01:
            base_score += 0.1
        
        # Task-specific evaluation
        if task_context:
            if task_context.get('problem_type') == 'computer_vision' and 'conv2d' in layer_types:
                base_score += 0.15
            elif task_context.get('problem_type') == 'natural_language' and 'transformer_block' in layer_types:
                base_score += 0.15
            elif task_context.get('problem_type') == 'sequence_modeling' and any(t in layer_types for t in ['lstm', 'gru', 'rnn']):
                base_score += 0.15
        
        # Add some randomness to simulate real-world variability
        noise = random.uniform(-0.1, 0.1)
        fitness = max(0.0, min(1.0, base_score + noise))
        
        genome.fitness_score = fitness
        return fitness

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats.copy() 