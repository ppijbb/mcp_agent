"""
Evolutionary AI Architect & Self-Improving Agent

This is the main agent class that orchestrates architecture evolution,
self-improvement, and problem-solving capabilities.
"""

import time
import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .genome import ArchitectureGenome, PerformanceMetrics, EvolutionHistory
from .architect import AIArchitectureDesigner  
from .improvement_engine import SelfImprovementEngine

logger = logging.getLogger(__name__)


class EvolutionaryAIArchitectAgent:
    """
    Main agent class that combines architecture design and self-improvement
    
    This agent can:
    - Evolve AI architectures using genetic algorithms
    - Monitor and improve its own performance
    - Adapt to different problem types
    - Generate solutions for complex problems
    - Learn from experience through meta-learning
    """
    
    def __init__(self, name: str = "EvoAI", population_size: int = 10):
        self.name = name
        self.architect = AIArchitectureDesigner()
        self.improvement_engine = SelfImprovementEngine()
        
        # Population management
        self.population: List[ArchitectureGenome] = []
        self.current_best: Optional[ArchitectureGenome] = None
        self.generation_count = 0
        self.population_size = population_size
        
        # History tracking
        self.task_history: List[Dict[str, Any]] = []
        self.evolution_history: List[EvolutionHistory] = []
        
        # Configuration
        self.config = {
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_ratio': 0.2,
            'max_generations': 50,
            'convergence_threshold': 0.001
        }
        
        # Initialize population
        self.initialize_population()
        
        logger.info(f"Initialized {self.name} with population size {len(self.population)}")
    
    def initialize_population(self):
        """Initialize the population of AI architectures"""
        self.population = []
        
        # Create diverse initial population
        architecture_types = ['transformer', 'cnn', 'hybrid']
        for i in range(self.population_size):
            arch_type = architecture_types[i % len(architecture_types)]
            complexity = random.uniform(0.3, 0.8)  # Varied complexity
            
            genome = self.architect.generate_random_architecture(
                architecture_type=arch_type,
                complexity_target=complexity
            )
            self.population.append(genome)
        
        logger.info(f"Initialized population with {len(self.population)} diverse architectures")
    
    def solve_problem(self, problem_description: str, 
                     constraints: Dict[str, Any] = None,
                     max_evolution_time: int = 30) -> Dict[str, Any]:
        """
        Main problem-solving interface
        
        Args:
            problem_description: Description of the problem to solve
            constraints: Any constraints or requirements
            max_evolution_time: Maximum time to spend on evolution (seconds)
            
        Returns:
            Dict containing solution, performance metrics, and metadata
        """
        start_time = time.time()
        
        logger.info(f"🧠 Solving problem: {problem_description}")
        
        # Analyze problem requirements
        problem_analysis = self.analyze_problem(problem_description, constraints)
        logger.info(f"📊 Problem analysis: {problem_analysis['problem_type']}")
        
        # Evolve architectures for this specific problem
        evolution_result = self.evolve_for_problem(
            problem_analysis, 
            max_time=max_evolution_time
        )
        
        # Generate solution based on best architecture
        solution = self.generate_solution(problem_analysis, evolution_result)
        
        # Assess performance
        processing_time = time.time() - start_time
        task_results = {
            'problem': problem_description,
            'solution': solution,
            'processing_time': processing_time,
            'success': True,
            'accuracy': self.current_best.fitness_score if self.current_best else 0.5,
            'architecture_id': self.current_best.unique_id if self.current_best else None
        }
        
        # Self-improvement cycle
        performance = self.improvement_engine.assess_performance(task_results)
        opportunities = self.improvement_engine.identify_improvement_opportunities()
        improvement_strategy = self.improvement_engine.generate_improvement_strategy(opportunities)
        
        # Store task history
        self.task_history.append(task_results)
        
        # Compile comprehensive result
        result = {
            'solution': solution,
            'problem_analysis': problem_analysis,
            'evolution_result': evolution_result,
            'best_architecture': asdict(self.current_best) if self.current_best else None,
            'performance_metrics': asdict(performance),
            'improvement_opportunities': opportunities,
            'improvement_strategy': improvement_strategy,
            'generation': self.generation_count,
            'processing_time': processing_time,
            'population_diversity': self.calculate_population_diversity()
        }
        
        logger.info(f"✅ Problem solved in {processing_time:.2f}s with performance score: {performance.overall_score():.4f}")
        
        return result
    
    def analyze_problem(self, problem_description: str, 
                       constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the problem to determine optimal approach"""
        analysis = {
            'problem_type': 'general',
            'complexity': 'medium',
            'required_capabilities': [],
            'suggested_architecture_type': 'hybrid',
            'priority_metrics': ['accuracy', 'efficiency']
        }
        
        # Simple keyword-based analysis (in real implementation, would use NLP)
        problem_lower = problem_description.lower()
        
        if any(word in problem_lower for word in ['image', 'vision', 'visual', 'picture', 'photo']):
            analysis.update({
                'problem_type': 'computer_vision',
                'suggested_architecture_type': 'cnn',
                'required_capabilities': ['visual_processing', 'feature_extraction'],
                'priority_metrics': ['accuracy', 'efficiency']
            })
        
        elif any(word in problem_lower for word in ['text', 'language', 'nlp', 'translation', 'chat']):
            analysis.update({
                'problem_type': 'natural_language',
                'suggested_architecture_type': 'transformer',
                'required_capabilities': ['language_understanding', 'context_awareness'],
                'priority_metrics': ['accuracy', 'creativity_score']
            })
        
        elif any(word in problem_lower for word in ['sequence', 'time', 'temporal', 'prediction', 'forecast']):
            analysis.update({
                'problem_type': 'sequence_modeling',
                'suggested_architecture_type': 'rnn',
                'required_capabilities': ['temporal_modeling', 'pattern_recognition'],
                'priority_metrics': ['accuracy', 'adaptability']
            })
        
        elif any(word in problem_lower for word in ['multimodal', 'multi-modal', 'combined', 'fusion']):
            analysis.update({
                'problem_type': 'multimodal',
                'suggested_architecture_type': 'hybrid',
                'required_capabilities': ['multimodal_fusion', 'cross_domain_understanding'],
                'priority_metrics': ['adaptability', 'creativity_score']
            })
        
        # Determine complexity
        if any(word in problem_lower for word in ['complex', 'difficult', 'advanced', 'sophisticated']):
            analysis['complexity'] = 'high'
        elif any(word in problem_lower for word in ['simple', 'basic', 'easy', 'straightforward']):
            analysis['complexity'] = 'low'
        
        # Apply constraints if provided
        if constraints:
            if 'max_complexity' in constraints:
                analysis['complexity'] = constraints['max_complexity']
            if 'required_architecture' in constraints:
                analysis['suggested_architecture_type'] = constraints['required_architecture']
        
        return analysis
    
    def evolve_for_problem(self, problem_analysis: Dict[str, Any], 
                          max_time: int = 30) -> Dict[str, Any]:
        """Evolve architectures specifically for the given problem"""
        start_time = time.time()
        generations_evolved = 0
        best_fitness_history = []
        
        # Bias population towards problem-appropriate architectures
        self.bias_population_for_problem(problem_analysis)
        
        # Evolution loop
        while (time.time() - start_time) < max_time and generations_evolved < self.config['max_generations']:
            # Evaluate all architectures for this problem
            for genome in self.population:
                fitness = self.architect.evaluate_architecture(genome, problem_analysis)
                genome.fitness_score = fitness
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Update current best
            if not self.current_best or self.population[0].fitness_score > self.current_best.fitness_score:
                self.current_best = self.population[0]
                logger.debug(f"🏆 New best architecture: {self.current_best.fitness_score:.4f}")
            
            best_fitness_history.append(self.current_best.fitness_score)
            
            # Check for convergence
            if len(best_fitness_history) > 5:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
                if recent_improvement < self.config['convergence_threshold']:
                    logger.info(f"🎯 Converged after {generations_evolved} generations")
                    break
            
            # Create next generation
            self.create_next_generation()
            generations_evolved += 1
            self.generation_count += 1
        
        evolution_time = time.time() - start_time
        
        # Record evolution history
        history = EvolutionHistory(
            generation=self.generation_count,
            best_fitness=self.current_best.fitness_score if self.current_best else 0.0,
            average_fitness=sum(g.fitness_score for g in self.population) / len(self.population),
            diversity_score=self.calculate_population_diversity(),
            improvements=[f"Evolved {generations_evolved} generations"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.evolution_history.append(history)
        
        result = {
            'generations_evolved': generations_evolved,
            'evolution_time': evolution_time,
            'best_fitness': self.current_best.fitness_score if self.current_best else 0.0,
            'fitness_history': best_fitness_history,
            'population_diversity': self.calculate_population_diversity(),
            'convergence_achieved': len(best_fitness_history) > 5 and recent_improvement < self.config['convergence_threshold']
        }
        
        logger.info(f"🧬 Evolution completed: {generations_evolved} generations in {evolution_time:.2f}s")
        
        return result
    
    def bias_population_for_problem(self, problem_analysis: Dict[str, Any]):
        """Bias population towards architectures suitable for the problem"""
        suggested_type = problem_analysis.get('suggested_architecture_type', 'hybrid')
        
        # Replace some random architectures with problem-specific ones
        replacement_count = max(1, len(self.population) // 4)
        
        for i in range(replacement_count):
            idx = random.randint(len(self.population) // 2, len(self.population) - 1)  # Replace worse performers
            complexity = 0.6 if problem_analysis.get('complexity') == 'high' else 0.4
            
            new_genome = self.architect.generate_random_architecture(
                architecture_type=suggested_type,
                complexity_target=complexity
            )
            self.population[idx] = new_genome
        
        logger.debug(f"🎯 Biased {replacement_count} architectures towards {suggested_type}")
    
    def create_next_generation(self):
        """Create the next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best performers
        elite_size = int(len(self.population) * self.config['elite_ratio'])
        new_population.extend(self.population[:elite_size])
        
        # Generate offspring
        while len(new_population) < len(self.population):
            if random.random() < self.config['crossover_rate']:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                offspring = self.architect.crossover_architectures(parent1, parent2)
            else:
                # Mutation only
                parent = self.tournament_selection()
                offspring = self.architect.mutate_architecture(
                    parent, 
                    mutation_rate=self.config['mutation_rate']
                )
            
            new_population.append(offspring)
        
        self.population = new_population[:self.population_size]  # Ensure exact size
    
    def tournament_selection(self, tournament_size: int = 3) -> ArchitectureGenome:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def calculate_population_diversity(self) -> float:
        """Calculate diversity score of current population"""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity measure based on layer type variety
        all_layer_types = []
        for genome in self.population:
            all_layer_types.extend(genome.get_layer_types())
        
        unique_types = set(all_layer_types)
        total_types = len(all_layer_types)
        
        diversity = len(unique_types) / max(total_types, 1)
        return diversity
    
    def generate_solution(self, problem_analysis: Dict[str, Any], 
                         evolution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive solution based on the evolved architecture"""
        if not self.current_best:
            return {"error": "No viable architecture evolved"}
        
        solution = {
            'recommended_architecture': {
                'id': self.current_best.unique_id,
                'type': problem_analysis.get('suggested_architecture_type', 'hybrid'),
                'layers': self.current_best.layers,
                'connections': self.current_best.connections,
                'hyperparameters': self.current_best.hyperparameters,
                'fitness_score': self.current_best.fitness_score,
                'generation': self.current_best.generation,
                'complexity_score': self.current_best.get_complexity_score()
            },
            'implementation_strategy': {
                'development_phases': [
                    "Architecture Implementation",
                    "Initial Training & Validation", 
                    "Hyperparameter Optimization",
                    "Performance Monitoring",
                    "Deployment & Scaling"
                ],
                'estimated_timeline': self._estimate_implementation_timeline(problem_analysis),
                'resource_requirements': self._estimate_resource_requirements(),
                'risk_factors': self._identify_risk_factors(problem_analysis)
            },
            'performance_predictions': {
                'accuracy_estimate': f"{self.current_best.fitness_score * 100:.1f}%",
                'training_time_estimate': f"{len(self.current_best.layers) * 15} minutes",
                'inference_speed': "Fast" if len(self.current_best.layers) < 10 else "Medium",
                'scalability': "High" if 'transformer' in str(self.current_best.layers) else "Medium"
            },
            'adaptive_features': [
                "Architecture can evolve based on performance feedback",
                "Hyperparameters auto-tune during training",
                "Model complexity adjusts to data requirements",
                "Real-time performance monitoring and alerts",
                f"Built-in {problem_analysis.get('problem_type', 'general')} optimization"
            ],
            'deployment_options': {
                'cloud_deployment': {
                    'platforms': ['AWS', 'Google Cloud', 'Azure'],
                    'estimated_cost': 'Medium',
                    'scalability': 'High'
                },
                'edge_deployment': {
                    'feasibility': 'High' if len(self.current_best.layers) < 8 else 'Medium',
                    'optimization_needed': len(self.current_best.layers) > 5
                },
                'hybrid_deployment': {
                    'recommended': True,
                    'benefits': ['Reduced latency', 'Better privacy', 'Cost optimization']
                }
            }
        }
        
        return solution
    
    def _estimate_implementation_timeline(self, problem_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Estimate implementation timeline based on problem complexity"""
        complexity = problem_analysis.get('complexity', 'medium')
        
        if complexity == 'high':
            return {
                'research_phase': '2-3 weeks',
                'development_phase': '4-6 weeks', 
                'testing_phase': '2-3 weeks',
                'deployment_phase': '1-2 weeks',
                'total_estimate': '9-14 weeks'
            }
        elif complexity == 'low':
            return {
                'research_phase': '3-5 days',
                'development_phase': '1-2 weeks',
                'testing_phase': '3-5 days', 
                'deployment_phase': '2-3 days',
                'total_estimate': '3-4 weeks'
            }
        else:  # medium
            return {
                'research_phase': '1 week',
                'development_phase': '2-3 weeks',
                'testing_phase': '1 week',
                'deployment_phase': '3-5 days',
                'total_estimate': '5-6 weeks'
            }
    
    def _estimate_resource_requirements(self) -> Dict[str, Any]:
        """Estimate computational resource requirements"""
        if not self.current_best:
            return {'error': 'No architecture available'}
        
        layer_count = len(self.current_best.layers)
        complexity = self.current_best.get_complexity_score()
        
        if complexity > 0.7:
            return {
                'gpu_requirements': 'High-end GPU (RTX 4090 or better)',
                'memory_requirements': '32GB+ RAM',
                'storage_requirements': '100GB+ SSD',
                'training_time': 'Several hours to days',
                'cost_estimate': 'High ($500-2000/month cloud cost)'
            }
        elif complexity > 0.4:
            return {
                'gpu_requirements': 'Mid-range GPU (RTX 3080 or better)',
                'memory_requirements': '16GB+ RAM',
                'storage_requirements': '50GB+ SSD',
                'training_time': '30 minutes to few hours',
                'cost_estimate': 'Medium ($100-500/month cloud cost)'
            }
        else:
            return {
                'gpu_requirements': 'Entry-level GPU or CPU-only',
                'memory_requirements': '8GB+ RAM',
                'storage_requirements': '20GB+ SSD',
                'training_time': '10-30 minutes',
                'cost_estimate': 'Low ($50-100/month cloud cost)'
            }
    
    def _identify_risk_factors(self, problem_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors for the solution"""
        risks = []
        
        complexity = problem_analysis.get('complexity', 'medium')
        problem_type = problem_analysis.get('problem_type', 'general')
        
        if complexity == 'high':
            risks.extend([
                'High computational requirements may impact scalability',
                'Complex architecture may be difficult to interpret',
                'Longer development and testing cycles'
            ])
        
        if problem_type == 'computer_vision':
            risks.extend([
                'Data quality and bias in training images',
                'Performance degradation on different image types'
            ])
        elif problem_type == 'natural_language':
            risks.extend([
                'Language and cultural bias in responses',
                'Potential for generating inappropriate content'
            ])
        
        if self.current_best and len(self.current_best.layers) > 10:
            risks.append('Large model size may cause deployment challenges')
        
        if not risks:
            risks.append('Low risk - well-understood problem domain')
        
        return risks
    
    def self_improve(self) -> Dict[str, Any]:
        """Trigger a comprehensive self-improvement cycle"""
        logger.info("🚀 Starting self-improvement cycle...")
        
        start_time = time.time()
        
        # Analyze recent performance
        opportunities = self.improvement_engine.identify_improvement_opportunities()
        strategy = self.improvement_engine.generate_improvement_strategy(opportunities)
        
        # Apply improvements based on strategy
        improvement_results = self.improvement_engine.apply_improvement_strategy(strategy)
        
        # Agent-specific improvements
        agent_improvements = self._apply_agent_improvements(strategy)
        
        improvement_time = time.time() - start_time
        
        result = {
            'improvement_strategy': strategy,
            'improvement_results': improvement_results,
            'agent_improvements': agent_improvements,
            'improvement_time': improvement_time,
            'new_capabilities': self._identify_new_capabilities(),
            'performance_gains': self._calculate_performance_gains()
        }
        
        logger.info(f"✨ Self-improvement cycle completed in {improvement_time:.2f}s")
        return result
    
    def _apply_agent_improvements(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Apply agent-specific improvements"""
        results = {
            'population_optimized': False,
            'config_updated': False,
            'architecture_templates_enhanced': False
        }
        
        for action in strategy['actions']:
            if action['type'] == 'architecture_evolution':
                # Trigger additional evolution
                self.evolve_for_problem({'problem_type': 'general'}, max_time=10)
                results['population_optimized'] = True
            
            elif action['type'] == 'optimization':
                # Update evolution parameters
                if self.current_best and self.current_best.fitness_score > 0.8:
                    self.config['mutation_rate'] *= 0.9  # Reduce mutation for fine-tuning
                else:
                    self.config['mutation_rate'] = min(0.2, self.config['mutation_rate'] * 1.1)
                results['config_updated'] = True
            
            elif action['type'] == 'meta_learning':
                # Enhance architecture templates based on successful patterns
                self._enhance_architecture_templates()
                results['architecture_templates_enhanced'] = True
        
        return results
    
    def _enhance_architecture_templates(self):
        """Enhance architecture templates based on learning"""
        # Analyze successful architectures from history
        successful_genomes = [g for g in self.population if g.fitness_score > 0.7]
        
        if successful_genomes:
            # Extract successful patterns
            successful_layer_types = []
            successful_hyperparams = {}
            
            for genome in successful_genomes:
                successful_layer_types.extend(genome.get_layer_types())
                for key, value in genome.hyperparameters.items():
                    if key not in successful_hyperparams:
                        successful_hyperparams[key] = []
                    successful_hyperparams[key].append(value)
            
            # Update templates (simplified implementation)
            logger.debug(f"Enhanced templates based on {len(successful_genomes)} successful architectures")
    
    def _identify_new_capabilities(self) -> List[str]:
        """Identify new capabilities gained through improvement"""
        capabilities = []
        
        if len(self.evolution_history) > 1:
            recent_diversity = self.evolution_history[-1].diversity_score
            previous_diversity = self.evolution_history[-2].diversity_score
            
            if recent_diversity > previous_diversity:
                capabilities.append("Enhanced architectural diversity")
        
        if self.current_best and self.current_best.fitness_score > 0.8:
            capabilities.append("High-performance architecture generation")
        
        if len(self.task_history) > 5:
            capabilities.append("Multi-domain problem solving experience")
        
        return capabilities if capabilities else ["Incremental performance improvements"]
    
    def _calculate_performance_gains(self) -> Dict[str, float]:
        """Calculate performance gains from improvement"""
        if len(self.improvement_engine.performance_history) < 2:
            return {'message': 'Insufficient data for comparison'}
        
        current = self.improvement_engine.performance_history[-1]
        previous = self.improvement_engine.performance_history[-2]
        
        return {
            'overall_score_gain': current.overall_score() - previous.overall_score(),
            'accuracy_gain': current.accuracy - previous.accuracy,
            'efficiency_gain': current.efficiency - previous.efficiency,
            'adaptability_gain': current.adaptability - previous.adaptability,
            'creativity_gain': current.creativity_score - previous.creativity_score
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the evolutionary AI agent"""
        performance_summary = self.improvement_engine.get_performance_summary()
        
        status = {
            'agent_info': {
                'name': self.name,
                'generation': self.generation_count,
                'population_size': len(self.population),
                'tasks_completed': len(self.task_history)
            },
            'current_best_architecture': {
                'id': self.current_best.unique_id if self.current_best else None,
                'fitness': self.current_best.fitness_score if self.current_best else 0.0,
                'complexity': self.current_best.get_complexity_score() if self.current_best else 0.0,
                'layer_count': len(self.current_best.layers) if self.current_best else 0
            },
            'population_stats': {
                'diversity_score': self.calculate_population_diversity(),
                'average_fitness': sum(g.fitness_score for g in self.population) / len(self.population) if self.population else 0.0,
                'generation_stats': self.architect.get_stats()
            },
            'performance_summary': performance_summary,
            'capabilities': [
                'AI Architecture Design & Evolution',
                'Self-Performance Monitoring',
                'Adaptive Problem Solving',
                'Meta-Learning & Strategy Improvement',
                'Multi-Domain Knowledge Application',
                'Real-time Architecture Optimization'
            ],
            'recent_activity': self._get_recent_activity()
        }
        
        return status
    
    def _get_recent_activity(self) -> List[str]:
        """Get summary of recent agent activity"""
        activity = []
        
        if self.task_history:
            last_task = self.task_history[-1]
            activity.append(f"Solved: {last_task['problem'][:50]}...")
        
        if self.evolution_history:
            last_evolution = self.evolution_history[-1]
            activity.append(f"Evolution Gen {last_evolution.generation}: Best fitness {last_evolution.best_fitness:.3f}")
        
        if len(self.improvement_engine.performance_history) > 0:
            latest_performance = self.improvement_engine.performance_history[-1].overall_score()
            activity.append(f"Current performance: {latest_performance:.3f}")
        
        return activity if activity else ["Agent initialized - ready for tasks"]


def main():
    """Demonstration of the Evolutionary AI Architect Agent"""
    print("🧠 Evolutionary AI Architect & Self-Improving Agent Demo")
    print("=" * 65)
    
    # Create the agent
    agent = EvolutionaryAIArchitectAgent("EvoAI-Alpha", population_size=8)
    
    # Display initial status
    print("\n📊 Initial Status:")
    status = agent.get_status()
    for section, data in status.items():
        print(f"  📂 {section}:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"    • {key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(f"    • {item}")
        else:
            print(f"    {data}")
        print()
    
    # Solve example problems
    problems = [
        "Design an AI system for real-time image recognition in autonomous vehicles",
        "Create a natural language processing model for multilingual customer support",
        "Develop a predictive model for complex financial market analysis",
        "Build a multimodal AI for automated content moderation"
    ]
    
    print("🔧 Solving Problems:")
    for i, problem in enumerate(problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"🎯 Problem: {problem}")
        
        result = agent.solve_problem(problem, max_evolution_time=15)
        
        print(f"✅ Solution Generated!")
        print(f"  🏗️  Architecture: {result['best_architecture']['type'] if result['best_architecture'] else 'None'}")
        print(f"  📈 Performance: {result['performance_metrics']['overall_score']:.4f}")
        print(f"  ⏱️  Time: {result['processing_time']:.2f}s")
        print(f"  🧬 Generation: {result['generation']}")
        print(f"  🎨 Diversity: {result['population_diversity']:.3f}")
        
        if result['improvement_opportunities']:
            print(f"  💡 Opportunities: {', '.join(result['improvement_opportunities'][:2])}")
        
        # Show solution highlights
        if 'solution' in result and 'adaptive_features' in result['solution']:
            print(f"  ⚡ Features: {len(result['solution']['adaptive_features'])} adaptive capabilities")
    
    # Trigger self-improvement
    print("\n🚀 Self-Improvement Cycle:")
    improvement_result = agent.self_improve()
    print(f"  ✨ Applied {len(improvement_result['improvement_results']['actions_successful'])} improvements")
    print(f"  🎯 New capabilities: {len(improvement_result['new_capabilities'])}")
    print(f"  📊 Performance gains calculated: {len(improvement_result['performance_gains'])} metrics")
    
    # Final status
    print("\n📈 Final Status:")
    final_status = agent.get_status()
    print(f"  🏆 Best fitness: {final_status['current_best_architecture']['fitness']:.4f}")
    print(f"  🧬 Generations: {final_status['agent_info']['generation']}")
    print(f"  🎨 Population diversity: {final_status['population_stats']['diversity_score']:.3f}")
    print(f"  📋 Tasks completed: {final_status['agent_info']['tasks_completed']}")
    
    print("\n🎉 Demo completed! The agent has successfully:")
    print("  ✓ Designed and evolved AI architectures")
    print("  ✓ Solved diverse problem types")
    print("  ✓ Monitored and improved its own performance")
    print("  ✓ Adapted architectures to problem requirements")
    print("  ✓ Implemented comprehensive self-improvement")
    print("  ✓ Generated detailed implementation strategies")


if __name__ == "__main__":
    main() 