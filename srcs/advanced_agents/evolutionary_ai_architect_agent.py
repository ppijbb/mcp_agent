"""
Evolutionary AI Architect & Self-Improving Agent

Main agent class that orchestrates architecture evolution and self-improvement.
"""

import time
import random
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .genome import ArchitectureGenome, PerformanceMetrics
from .architect import AIArchitectureDesigner  
from .improvement_engine import SelfImprovementEngine

logger = logging.getLogger(__name__)


class EvolutionaryAIArchitectAgent:
    """Main evolutionary AI agent that combines architecture design and self-improvement"""
    
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
        
        # Initialize population
        self.initialize_population()
        
        logger.info(f"Initialized {self.name} with population size {len(self.population)}")
    
    def initialize_population(self):
        """Initialize diverse population of AI architectures"""
        self.population = []
        architecture_types = ['transformer', 'cnn', 'hybrid']
        
        for i in range(self.population_size):
            arch_type = architecture_types[i % len(architecture_types)]
            genome = self.architect.generate_random_architecture(architecture_type=arch_type)
            self.population.append(genome)
        
        logger.info(f"Initialized population with {len(self.population)} architectures")
    
    def solve_problem(self, problem_description: str, 
                     constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main problem-solving interface"""
        start_time = time.time()
        
        logger.info(f"🧠 Solving: {problem_description}")
        
        # Analyze problem
        problem_analysis = self.analyze_problem(problem_description, constraints)
        
        # Evolve architectures for this problem
        self.evolve_population(generations=5, problem_context=problem_analysis)
        
        # Generate solution
        solution = self.generate_solution(problem_analysis)
        
        # Assess performance
        processing_time = time.time() - start_time
        task_results = {
            'problem': problem_description,
            'solution': solution,
            'processing_time': processing_time,
            'success': True,
            'accuracy': self.current_best.fitness_score if self.current_best else 0.5
        }
        
        # Self-improvement cycle
        performance = self.improvement_engine.assess_performance(task_results)
        opportunities = self.improvement_engine.identify_improvement_opportunities()
        improvement_strategy = self.improvement_engine.generate_improvement_strategy(opportunities)
        
        self.task_history.append(task_results)
        
        result = {
            'solution': solution,
            'problem_analysis': problem_analysis,
            'best_architecture': asdict(self.current_best) if self.current_best else None,
            'performance_metrics': asdict(performance),
            'improvement_opportunities': opportunities,
            'improvement_strategy': improvement_strategy,
            'generation': self.generation_count,
            'processing_time': processing_time
        }
        
        logger.info(f"✅ Problem solved in {processing_time:.2f}s")
        return result
    
    def analyze_problem(self, problem_description: str, 
                       constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze problem to determine optimal approach"""
        analysis = {
            'problem_type': 'general',
            'complexity': 'medium',
            'suggested_architecture_type': 'hybrid'
        }
        
        problem_lower = problem_description.lower()
        
        if any(word in problem_lower for word in ['image', 'vision', 'visual']):
            analysis.update({
                'problem_type': 'computer_vision',
                'suggested_architecture_type': 'cnn'
            })
        elif any(word in problem_lower for word in ['text', 'language', 'nlp']):
            analysis.update({
                'problem_type': 'natural_language',
                'suggested_architecture_type': 'transformer'
            })
        elif any(word in problem_lower for word in ['sequence', 'time', 'prediction']):
            analysis.update({
                'problem_type': 'sequence_modeling',
                'suggested_architecture_type': 'hybrid'
            })
        
        if any(word in problem_lower for word in ['complex', 'advanced']):
            analysis['complexity'] = 'high'
        elif any(word in problem_lower for word in ['simple', 'basic']):
            analysis['complexity'] = 'low'
        
        return analysis
    
    def evolve_population(self, generations: int = 5, problem_context: Dict[str, Any] = None):
        """Evolve population through selection, crossover, and mutation"""
        for gen in range(generations):
            # Evaluate all architectures
            for genome in self.population:
                fitness = self.architect.evaluate_architecture(genome, problem_context)
                genome.fitness_score = fitness
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            
            # Update current best
            if not self.current_best or self.population[0].fitness_score > self.current_best.fitness_score:
                self.current_best = self.population[0]
                logger.debug(f"New best: {self.current_best.fitness_score:.4f}")
            
            # Create next generation
            new_population = []
            
            # Keep best performers (elitism)
            elite_size = len(self.population) // 4
            new_population.extend(self.population[:elite_size])
            
            # Generate offspring
            while len(new_population) < len(self.population):
                if random.random() < 0.7:  # Crossover
                    parent1 = self.tournament_selection()
                    parent2 = self.tournament_selection()
                    offspring = self.architect.crossover_architectures(parent1, parent2)
                else:  # Mutation
                    parent = self.tournament_selection()
                    offspring = self.architect.mutate_architecture(parent)
                
                new_population.append(offspring)
            
            self.population = new_population
            self.generation_count += 1
        
        logger.info(f"Evolution completed: {generations} generations")
    
    def tournament_selection(self, tournament_size: int = 3) -> ArchitectureGenome:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def generate_solution(self, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate solution based on best architecture"""
        if not self.current_best:
            return {"error": "No viable architecture found"}
        
        solution = {
            'recommended_architecture': {
                'id': self.current_best.unique_id,
                'type': problem_analysis.get('suggested_architecture_type', 'hybrid'),
                'layers': self.current_best.layers,
                'hyperparameters': self.current_best.hyperparameters,
                'fitness_score': self.current_best.fitness_score
            },
            'implementation_steps': [
                "1. Initialize the recommended architecture",
                "2. Configure hyperparameters as specified",
                "3. Train using evolutionary optimization",
                "4. Monitor performance and adapt as needed",
                "5. Deploy with continuous improvement"
            ],
            'expected_performance': {
                'accuracy_estimate': f"{self.current_best.fitness_score * 100:.1f}%",
                'complexity_rating': len(self.current_best.layers),
                'training_time_estimate': f"{len(self.current_best.layers) * 10} minutes"
            },
            'adaptive_features': [
                "Architecture evolves based on performance feedback",
                "Hyperparameters auto-tune during training",
                "Model complexity adjusts to requirements",
                f"Optimized for {problem_analysis.get('problem_type', 'general')} tasks"
            ]
        }
        
        return solution
    
    def self_improve(self) -> Dict[str, Any]:
        """Trigger self-improvement cycle"""
        logger.info("🚀 Starting self-improvement...")
        
        opportunities = self.improvement_engine.identify_improvement_opportunities()
        strategy = self.improvement_engine.generate_improvement_strategy(opportunities)
        
        # Apply improvements
        improvements_applied = []
        for action in strategy['actions']:
            if action['type'] == 'architecture_evolution':
                self.evolve_population(generations=3)
                improvements_applied.append('evolved_population')
            elif action['type'] == 'optimization':
                self.optimize_population()
                improvements_applied.append('optimized_population')
        
        logger.info("✨ Self-improvement completed")
        return {
            'strategy': strategy,
            'improvements_applied': improvements_applied,
            'new_best_fitness': self.current_best.fitness_score if self.current_best else 0.0
        }
    
    def optimize_population(self):
        """Optimize current population by removing poor performers"""
        threshold = 0.3
        self.population = [g for g in self.population if g.fitness_score > threshold]
        
        # Refill population if too small
        while len(self.population) < self.population_size // 2:
            new_genome = self.architect.generate_random_architecture()
            self.population.append(new_genome)
        
        logger.debug(f"Population optimized to {len(self.population)} architectures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
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
            'performance_summary': self.improvement_engine.get_performance_summary(),
            'capabilities': [
                'AI Architecture Design',
                'Evolutionary Optimization', 
                'Self-Performance Monitoring',
                'Adaptive Problem Solving',
                'Meta-Learning'
            ],
            'recent_activity': self._get_recent_activity()
        }
    
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
    
    def _get_recent_activity(self) -> List[str]:
        """Get summary of recent agent activity"""
        activity = []
        
        if self.task_history:
            last_task = self.task_history[-1]
            activity.append(f"Solved: {last_task['problem'][:50]}...")
        
        if len(self.improvement_engine.performance_history) > 0:
            latest_performance = self.improvement_engine.performance_history[-1].overall_score()
            activity.append(f"Current performance: {latest_performance:.3f}")
        else:
            activity.append("Agent initialized - ready for tasks")
        
        return activity


def main():
    """Demo of the Evolutionary AI Architect Agent"""
    print("🧠 Evolutionary AI Architect Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = EvolutionaryAIArchitectAgent("EvoAI-Alpha", population_size=8)
    
    # Show initial status
    print("\n📊 Initial Status:")
    status = agent.get_status()
    print(f"  • Agent: {status['agent_info']['name']}")
    print(f"  • Population: {status['agent_info']['population_size']}")
    print(f"  • Diversity: {status['population_stats']['diversity_score']:.3f}")
    
    # Solve problems
    problems = [
        "Design an AI for medical image analysis",
        "Create a chatbot for customer service",
        "Build a system for stock price prediction"
    ]
    
    print("\n🔧 Solving Problems:")
    for i, problem in enumerate(problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"Problem: {problem}")
        
        result = agent.solve_problem(problem)
        
        print(f"✅ Solution generated!")
        print(f"  Architecture: {result['best_architecture']['type'] if result['best_architecture'] else 'None'}")
        print(f"  Performance: {result['performance_metrics']['overall_score']:.4f}")
        print(f"  Time: {result['processing_time']:.2f}s")
    
    # Self-improvement
    print("\n🚀 Self-Improvement:")
    improvement = agent.self_improve()
    print(f"  Applied improvements: {improvement['improvements_applied']}")
    print(f"  New best fitness: {improvement['new_best_fitness']:.4f}")
    
    # Final status
    print("\n📈 Final Status:")
    final_status = agent.get_status()
    print(f"  • Best fitness: {final_status['current_best_architecture']['fitness']:.4f}")
    print(f"  • Tasks completed: {final_status['agent_info']['tasks_completed']}")
    print(f"  • Generation: {final_status['agent_info']['generation']}")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()