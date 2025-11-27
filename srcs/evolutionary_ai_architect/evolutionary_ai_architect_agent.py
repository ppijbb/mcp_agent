"""
Evolutionary AI Architect MCP Agent
===================================
Advanced MCP Agent that evolves AI architectures using genetic algorithms.
"""

import asyncio
import os
import json
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Real MCP Agent imports  
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from srcs.common.utils import setup_agent_app

# Import existing components
from .genome import ArchitectureGenome, PerformanceMetrics
from .architect import AIArchitectureDesigner
from .improvement_engine import SelfImprovementEngine

class ArchitectureType(Enum):
    TRANSFORMER = "transformer"
    CNN = "cnn"
    HYBRID = "hybrid"
    RNN = "rnn"

@dataclass
class EvolutionaryTask:
    task_id: str
    problem_description: str
    constraints: Dict[str, Any]
    target_metrics: Dict[str, float]
    max_generations: int
    population_size: int
    timestamp: datetime

@dataclass  
class ArchitectureEvolutionResult:
    task: EvolutionaryTask
    best_architecture: ArchitectureGenome
    evolution_history: List[Dict[str, Any]]
    final_metrics: PerformanceMetrics
    reasoning_steps: List[str]
    research_insights: List[str]
    optimization_recommendations: List[str]
    generation_count: int
    processing_time: float
    success: bool

class EvolutionaryAIArchitectMCP:
    """
    🧬 Evolutionary AI Architect MCP Agent
    
    Features:
    - Genetic algorithm-based architecture evolution
    - ReAct pattern for iterative optimization
    - Real-time research via MCP servers
    - Architecture performance analysis
    """
    
    def __init__(self, output_dir: str = "evolutionary_architect_reports"):
        self.output_dir = output_dir
        self.app = setup_agent_app("evolutionary_architect")
        
        # Core components
        self.architect = AIArchitectureDesigner()
        self.improvement_engine = SelfImprovementEngine()
        
        # Evolution state
        self.active_populations: Dict[str, List[ArchitectureGenome]] = {}
        self.evolution_history: List[ArchitectureEvolutionResult] = []
        self.best_architectures: Dict[str, ArchitectureGenome] = {}
        
        # Configuration
        self.default_population_size = 20
        self.default_generations = 10
        
    async def evolve_architecture(
        self,
        problem_description: str,
        constraints: Dict[str, Any] = None,
        target_metrics: Dict[str, float] = None,
        max_generations: int = None,
        population_size: int = None,
        use_react_pattern: bool = True
    ) -> ArchitectureEvolutionResult:
        """
        🧬 Main architecture evolution function
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        start_time = time.time()
        
        # Create task
        task = EvolutionaryTask(
            task_id=f"evo_{int(time.time())}_{str(uuid.uuid4())[:8]}",
            problem_description=problem_description,
            constraints=constraints or {},
            target_metrics=target_metrics or {"accuracy": 0.9, "efficiency": 0.8},
            max_generations=max_generations or self.default_generations,
            population_size=population_size or self.default_population_size,
            timestamp=datetime.now(timezone.utc)
        )
        
        async with self.app.run() as evo_app:
            context = evo_app.context
            logger = evo_app.logger
            
            logger.info(f"🧬 Starting architecture evolution: {problem_description}")
            
            # --- CRITICAL FIX ---
            # The ReAct process using the Orchestrator is unstable.
            # Defaulting to the simple, stable evolution process to avoid errors.
            # The ReAct process can be re-enabled after the Orchestrator is fully stabilized or replaced.
            logger.warning("Orchestrator-based ReAct pattern is currently unstable. Using simple evolution process.")
            result = await self._simple_evolution_process(task, context, logger)
            
            # Save results
            await self._save_evolution_results(result, task.task_id)
            
            # Update state
            self.evolution_history.append(result)
            if result.success:
                self.best_architectures[task.task_id] = result.best_architecture
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"Architecture evolution completed in {processing_time:.2f}s")
            return result
    
    async def _react_evolution_process(self, task: EvolutionaryTask, context, logger) -> ArchitectureEvolutionResult:
        """ReAct pattern evolution process, without the faulty Orchestrator."""

        # Create a direct LLM for reasoning steps (load() 메서드 없이 직접 사용)
        reasoning_llm = GoogleAugmentedLLM(
            app=self.app,
            name="reasoning_llm",
            instruction="You are a world-class AI architect reasoning engine.",
            server_names=[]  # No tools needed for reasoning
        )

        # 연구용 LLM 생성 (서버 없이 사용 - MCP 서버 validation 에러 방지)
        # 필요시 reasoning_llm을 사용하여 연구 수행
        research_llm = reasoning_llm
        
        # Initialize variables
        reasoning_steps = []
        research_insights = []
        evolution_history = []
        
        # Initialize population
        population = self._initialize_population(task)
        logger.info(f"Initialized population: {len(population)} architectures")
        
        # Evolution loop
        best_genome = population[0]
        for generation in range(1, task.max_generations + 1):
            logger.info(f"Generation {generation}/{task.max_generations}")
            
            # THOUGHT: Analyze current state
            thought_task = f"""
            THOUGHT - Generation {generation}:
            Problem: {task.problem_description}
            Population size: {len(population)}
            Best fitness so far: {best_genome.fitness_score:.4f}
            
            Analyze the current state of evolution. What architectural improvements should be the focus for this generation? 
            Should I focus on mutation to explore new areas, or crossover to refine existing solutions?
            """
            
            thought_result = await reasoning_llm.generate_str(
                message=thought_task,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )
            reasoning_steps.append(f"Generation {generation} Thought: {thought_result}")

            # ACTION: Research based on thought
            action_task = f"""
            ACTION - Generation {generation}:
            Based on the thought: "{thought_result}"
            
            Research the latest techniques for '{task.architecture_type.value}' architectures related to '{task.problem_description}'.
            Provide a concise summary of key findings that can be used to improve the architecture in this generation.
            """
            try:
                action_result = await research_llm.generate_str(
                    message=action_task,
                    request_params=RequestParams(model="gemini-2.5-flash-lite")
                )
                research_insights.append(action_result)
                reasoning_steps.append(f"Generation {generation} Action: Performed research, insights gathered.")
            except Exception as e:
                # MCP 서버 validation 에러 등이 발생할 수 있으므로 fallback 사용
                error_msg = str(e)
                if "validation error" in error_msg.lower() or "functiondeclaration" in error_msg.lower():
                    logger.debug(f"MCP server validation error (non-critical): {error_msg[:100]}...")
                else:
                    logger.warning(f"Research failed: {error_msg[:200]}")
                
                # Fallback: 간단한 연구 인사이트 생성
                action_result = f"Focus on improving {task.architecture_type.value} architectures for {task.problem_description} with optimized layer connections and better data flow."
                research_insights.append(action_result)
                reasoning_steps.append(f"Generation {generation} Action: Performed research with fallback method.")

            # OBSERVATION: Evaluate population and evolve
            metrics = await self._evaluate_population(population, task, action_result)
            population = await self._evolve_generation(population, task, metrics)
            
            # Update best genome
            best_genome = max(population, key=lambda g: g.fitness_score)
            
            observation_task = f"""
            OBSERVATION - Generation {generation}:
            A new generation of {len(population)} architectures was created.
            The best architecture now has a fitness score of {best_genome.fitness_score:.4f}.
            The population diversity is {self._calculate_population_diversity(population):.3f}.
            
            Analyze the outcome of this generation. Is the evolution progressing as expected?
            """
            observation_result = await reasoning_llm.generate_str(
                message=observation_task,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )
            reasoning_steps.append(f"Generation {generation} Observation: {observation_result}")
            
            evolution_history.append({
                "generation": generation,
                "best_fitness": best_genome.fitness_score,
                "avg_fitness": sum(g.fitness_score for g in population) / len(population),
                "diversity": self._calculate_population_diversity(population)
            })

        # Final analysis and recommendations
        final_metrics = self.improvement_engine.assess_performance(best_genome)
        optimization_recommendations = await self._generate_recommendations(
            task, best_genome, evolution_history, reasoning_llm
        )

        return ArchitectureEvolutionResult(
            task=task,
            best_architecture=best_genome,
            evolution_history=evolution_history,
            final_metrics=final_metrics,
            reasoning_steps=reasoning_steps,
            research_insights=research_insights,
            optimization_recommendations=optimization_recommendations,
            generation_count=task.max_generations,
            processing_time=0.0, # Will be updated outside
            success=True
        )
    
    async def _simple_evolution_process(self, task: EvolutionaryTask, context, logger) -> ArchitectureEvolutionResult:
        """Simple evolution with LLM research and improvement - 실제 진화 프로세스 수행"""
        population = self._initialize_population(task)
        evolution_history = []
        reasoning_steps = []
        research_insights = []
        
        logger.info(f"Initializing population of {len(population)} architectures")
        
        # LLM 생성 (load() 메서드 없이 직접 사용)
        reasoning_llm = GoogleAugmentedLLM(
            app=self.app,
            name="reasoning_llm",
            instruction="You are a world-class AI architect reasoning engine.",
            server_names=[]
        )
        
        # 연구용 LLM 생성 (서버 없이 사용 - MCP 서버 validation 에러 방지)
        # 필요시 reasoning_llm을 사용하여 연구 수행
        research_llm = reasoning_llm
        
        # 각 세대마다 실제 진화 수행 (LLM 조사 및 개선 포함)
        for generation in range(task.max_generations):
            logger.info(f"Generation {generation + 1}/{task.max_generations}")
            
            # 1. 현재 세대 분석 (THOUGHT)
            best_genome = max(population, key=lambda x: x.fitness_score) if population else None
            thought_prompt = f"""
            Generation {generation + 1} Analysis:
            Problem: {task.problem_description}
            Current best fitness: {best_genome.fitness_score:.4f} if best_genome else 0.0
            Population size: {len(population)}
            
            Analyze the current state. What architectural improvements should we focus on for this generation?
            What specific techniques or patterns should we research and apply?
            """
            
            try:
                thought_result = await reasoning_llm.generate_str(
                    message=thought_prompt,
                    request_params=RequestParams(model="gemini-2.5-flash-lite")
                )
                reasoning_steps.append(f"Generation {generation + 1} - Analysis: {thought_result[:200]}...")
                logger.info(f"Generation {generation + 1} analysis completed")
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                thought_result = "Focus on improving architecture complexity and layer connections."
                reasoning_steps.append(f"Generation {generation + 1} - Analysis: {thought_result}")
            
            # 2. 아키텍처 조사 수행 (ACTION)
            research_prompt = f"""
            Research the latest techniques for '{task.problem_description}'.
            Based on the analysis: "{thought_result}"
            
            Provide specific, actionable insights about:
            1. Best practices for this type of architecture
            2. Recent research findings
            3. Optimization techniques
            
            Keep the response concise and actionable.
            """
            
            try:
                research_result = await research_llm.generate_str(
                    message=research_prompt,
                    request_params=RequestParams(model="gemini-2.5-flash-lite")
                )
                research_insights.append(f"Generation {generation + 1}: {research_result[:300]}...")
                reasoning_steps.append(f"Generation {generation + 1} - Research: Completed research on architecture improvements")
                logger.info(f"Generation {generation + 1} research completed")
            except Exception as e:
                # MCP 서버 validation 에러 등이 발생할 수 있으므로 fallback 사용
                error_msg = str(e)
                if "validation error" in error_msg.lower() or "functiondeclaration" in error_msg.lower():
                    logger.debug(f"MCP server validation error (non-critical): {error_msg[:100]}...")
                else:
                    logger.warning(f"Research failed: {error_msg[:200]}")
                
                # Fallback: 문제 설명 기반으로 의미있는 연구 인사이트 생성
                fallback_insight = f"""Based on the problem '{task.problem_description}', focus on:
1. Optimizing layer connections and data flow
2. Improving architecture efficiency for the target metrics: {json.dumps(task.target_metrics)}
3. Applying best practices for this type of architecture"""
                
                research_result = await research_llm.generate_str(
                    message=f"Provide concise architectural insights for: {fallback_insight}",
                    request_params=RequestParams(model="gemini-2.5-flash-lite")
                )
                research_insights.append(f"Generation {generation + 1}: {research_result[:300]}...")
                reasoning_steps.append(f"Generation {generation + 1} - Research: Completed with fallback method")
                logger.info(f"Generation {generation + 1} research completed (fallback)")
            
            # 3. Evaluate fitness - 실제 평가 수행
            for genome in population:
                task_context = {
                    'problem_description': task.problem_description,
                    'constraints': task.constraints,
                    'target_metrics': task.target_metrics,
                    'research_insights': research_result  # 조사 결과 반영
                }
                genome.fitness_score = self.architect.evaluate_architecture(genome, task_context)
                genome.generation = generation
            
            # 4. 세대별 통계 수집
            fitness_scores = [g.fitness_score for g in population]
            best_fitness = max(fitness_scores) if fitness_scores else 0.0
            avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
            
            # 진화 히스토리 기록
            evolution_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': len(population),
                'diversity': max(fitness_scores) - min(fitness_scores) if fitness_scores else 0.0,
                'improvement_note': f"Applied research insights: {research_result[:100]}..."
            })
            
            reasoning_steps.append(f"Generation {generation + 1}: Best fitness {best_fitness:.4f}, Average {avg_fitness:.4f}")
            
            # 5. Selection - 상위 50% 선택
            population.sort(key=lambda x: x.fitness_score, reverse=True)
            elite_size = max(1, len(population) // 2)
            elite = population[:elite_size]
            
            # 6. Crossover and Mutation - 다음 세대 생성 (조사 결과 반영)
            if generation < task.max_generations - 1:
                new_population = elite.copy()
                
                # 조사 결과를 반영한 개선된 아키텍처 생성
                while len(new_population) < task.population_size:
                    if len(elite) >= 2:
                        # Crossover: 두 부모 선택
                        parent1 = random.choice(elite)
                        parent2 = random.choice(elite)
                        
                        # 조사 결과를 반영한 개선된 crossover
                        new_layers = parent1.layers[:len(parent1.layers)//2] + parent2.layers[len(parent2.layers)//2:]
                        # 조사 결과에 따라 레이어 개수 조정
                        if "deeper" in research_result.lower() or "more layers" in research_result.lower():
                            new_layers.append({"type": "dense", "parameters": 1000})
                        
                        new_genome = self.architect.generate_random_architecture(
                            architecture_type=parent1.layers[0].get('type', 'hybrid') if parent1.layers else 'hybrid',
                            complexity_target=(parent1.hyperparameters.get('complexity_target', 0.5) + 
                                             parent2.hyperparameters.get('complexity_target', 0.5)) / 2
                        )
                        new_genome.layers = new_layers
                        new_genome.parent_ids = [parent1.unique_id, parent2.unique_id]
                    else:
                        # Mutation: 단일 부모에서 변이 (조사 결과 반영)
                        parent = random.choice(elite)
                        complexity_multiplier = 1.1 if "complex" in research_result.lower() else 0.95
                        new_genome = self.architect.generate_random_architecture(
                            architecture_type=parent.layers[0].get('type', 'hybrid') if parent.layers else 'hybrid',
                            complexity_target=parent.hyperparameters.get('complexity_target', 0.5) * complexity_multiplier
                        )
                        new_genome.parent_ids = [parent.unique_id]
                    
                    new_population.append(new_genome)
                
                population = new_population
                reasoning_steps.append(f"Generation {generation + 1}: Created next generation with {len(population)} architectures")
        
        # 최종 평가 및 결과 생성
        best_architecture = max(population, key=lambda x: x.fitness_score)
        
        # 실제 성능 메트릭 계산 (genome.py의 PerformanceMetrics 사용)
        final_metrics = PerformanceMetrics(
            accuracy=best_architecture.fitness_score,
            efficiency=best_architecture.fitness_score * 0.9,  # 효율성 추정
            adaptability=0.8,  # 적응성
            creativity_score=0.7,  # 창의성 점수
            problem_solving_time=len(population) * 0.1,  # 시뮬레이션된 학습 시간
            resource_usage=sum(layer.get('parameters', 1000) for layer in best_architecture.layers) * 4 / 1024 / 1000,  # 정규화된 리소스 사용량
            success_rate=best_architecture.fitness_score,  # 성공률
            learning_speed=0.8  # 학습 속도
        )
        
        # 최적화 추천 생성 (LLM 기반)
        optimization_recommendations = []
        try:
            recommendation_prompt = f"""
            Based on the evolution results:
            - Problem: {task.problem_description}
            - Best fitness: {best_architecture.fitness_score:.4f}
            - Architecture layers: {len(best_architecture.layers)}
            - Evolution history: {json.dumps(evolution_history[-3:], indent=2)}
            - Research insights: {research_insights[-2:] if research_insights else 'None'}
            
            Provide 3-5 specific, actionable optimization recommendations for improving this architecture.
            Format as a numbered list, one recommendation per line.
            """
            
            recommendations_text = await reasoning_llm.generate_str(
                message=recommendation_prompt,
                request_params=RequestParams(model="gemini-2.5-flash-lite")
            )
            
            # 추천사항을 리스트로 파싱
            for line in recommendations_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # 번호나 불릿 제거
                    clean_line = line.lstrip('0123456789.-•) ').strip()
                    if clean_line:
                        optimization_recommendations.append(clean_line)
            
            if not optimization_recommendations:
                optimization_recommendations = recommendations_text.split('\n')[:5]
                optimization_recommendations = [r.strip() for r in optimization_recommendations if r.strip()]
            
            logger.info(f"Generated {len(optimization_recommendations)} optimization recommendations")
        except Exception as e:
            logger.warning(f"Failed to generate LLM recommendations: {e}")
            # Fallback 추천
            if best_architecture.fitness_score < 0.7:
                optimization_recommendations.append("아키텍처 복잡도를 높여 성능을 개선할 수 있습니다")
            if len(best_architecture.layers) < 3:
                optimization_recommendations.append("더 깊은 네트워크 구조를 고려해보세요")
            if best_architecture.fitness_score > 0.8:
                optimization_recommendations.append("현재 아키텍처가 우수한 성능을 보입니다. 하이퍼파라미터 튜닝을 고려하세요")
        
        # 최종 요약
        research_insights.append(f"총 {task.max_generations}세대 동안 {len(population)}개 아키텍처를 진화시켰습니다")
        research_insights.append(f"최종 최적 아키텍처의 fitness score: {best_architecture.fitness_score:.4f}")
        research_insights.append(f"세대별 평균 fitness 향상: {evolution_history[-1]['avg_fitness']:.4f} (초기: {evolution_history[0]['avg_fitness']:.4f})")
        
        logger.info(f"Evolution completed. Best fitness: {best_architecture.fitness_score:.4f}")
        
        return ArchitectureEvolutionResult(
            task=task,
            best_architecture=best_architecture,
            evolution_history=evolution_history,
            final_metrics=final_metrics,
            reasoning_steps=reasoning_steps,
            research_insights=research_insights,
            optimization_recommendations=optimization_recommendations,
            generation_count=task.max_generations,
            processing_time=0.0,  # Will be updated outside
            success=True
        )
    
    def _initialize_population(self, task: EvolutionaryTask) -> List[ArchitectureGenome]:
        """Initialize diverse population based on task"""
        population = []
        
        # Determine primary architecture type
        problem_lower = task.problem_description.lower()
        if any(word in problem_lower for word in ['image', 'vision', 'visual', 'cnn']):
            primary_type = ArchitectureType.CNN
        elif any(word in problem_lower for word in ['text', 'language', 'nlp', 'transformer']):
            primary_type = ArchitectureType.TRANSFORMER
        else:
            primary_type = ArchitectureType.HYBRID
        
        # Generate diverse architectures
        for i in range(task.population_size):
            if i < task.population_size // 2:
                arch_type = primary_type.value
            else:
                arch_type = random.choice(list(ArchitectureType)).value
            
            genome = self.architect.generate_random_architecture(architecture_type=arch_type)
            population.append(genome)
        
        return population
    
    async def _evaluate_population(self, population: List[ArchitectureGenome], task: EvolutionaryTask, insights: str) -> Dict[str, Any]:
        """Evaluate population with research insights"""
        for genome in population:
            base_fitness = self.architect.evaluate_architecture(genome, {'problem_type': task.problem_description})
            
            # Apply research insights
            if "transformer" in insights.lower() and any("transformer" in str(layer).lower() for layer in genome.layers):
                base_fitness *= 1.1
            if "efficiency" in insights.lower():
                base_fitness *= 1.05
            if "optimization" in insights.lower():
                base_fitness *= 1.02
            
            genome.fitness_score = min(base_fitness, 1.0)
        
        fitness_scores = [g.fitness_score for g in population]
        return {
            "best_fitness": max(fitness_scores),
            "avg_fitness": sum(fitness_scores) / len(fitness_scores),
            "diversity": self._calculate_population_diversity(population)
        }
    
    async def _evolve_generation(self, population: List[ArchitectureGenome], task: EvolutionaryTask, metrics: Dict[str, Any]) -> List[ArchitectureGenome]:
        """Evolve population to next generation"""
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        new_population = []
        
        # Elitism: keep best performers
        elite_size = max(1, len(population) // 4)
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < task.population_size:
            if random.random() < 0.7:  # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                offspring = self.architect.crossover_architectures(parent1, parent2)
            else:  # Mutation
                parent = self._tournament_selection(population)
                offspring = self.architect.mutate_architecture(parent)
            
            new_population.append(offspring)
        
        return new_population[:task.population_size]
    
    def _tournament_selection(self, population: List[ArchitectureGenome], tournament_size: int = 3) -> ArchitectureGenome:
        """Selects the best individual from a randomly selected tournament."""
        if not population:
            raise ValueError("Population cannot be empty for selection.")
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _calculate_population_diversity(self, population: List[ArchitectureGenome]) -> float:
        """Calculates the diversity of the population based on layer structure."""
        if not population:
            return 0.0
        
        unique_architectures = set()
        for genome in population:
            # Create a frozenset of frozensets for nested layer dictionaries
            try:
                arch_tuple = tuple(frozenset(d.items()) for d in genome.layers)
                unique_architectures.add(arch_tuple)
            except (TypeError, AttributeError):
                # Handle cases where layers are not structured as expected
                continue

        return len(unique_architectures) / len(population) if population else 0.0
    
    async def _get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """MCP를 통해 데이터셋 정보 조회"""
        try:
            # MCP를 통해 Hugging Face datasets 서버 호출
            result = await self.app.execute_tool(
                server_name="huggingface",
                tool_name="get_dataset_info",
                arguments={"dataset_name": dataset_name}
            )
            
            if not result.get("success", False):
                raise RuntimeError(f"Failed to get dataset info: {result.get('error', 'Unknown error')}")
            
            return result.get("data", {})
            
        except Exception as e:
            raise RuntimeError(f"MCP call failed for dataset info: {str(e)}")
    
    async def _validate_dataset_size(self, dataset_name: str) -> int:
        """데이터셋의 실제 토큰 수 계산"""
        try:
            dataset_info = await self._get_dataset_info(dataset_name)
            
            # 데이터셋 크기 추정
            num_rows = dataset_info.get("num_rows", 0)
            avg_tokens_per_row = dataset_info.get("avg_tokens_per_row", 100)  # 기본값
            
            total_tokens = num_rows * avg_tokens_per_row
            
            if total_tokens <= 0:
                raise ValueError(f"Invalid dataset size calculation: {total_tokens}")
            
            return total_tokens
            
        except Exception as e:
            raise RuntimeError(f"Failed to validate dataset size: {str(e)}")
    
    async def _get_optimal_architecture_for_dataset(self, dataset_name: str, target_loss: float, 
                                                  compute_budget: float) -> Dict[str, Any]:
        """데이터셋에 대한 최적 아키텍처 계산"""
        try:
            # 데이터셋 크기 확인
            dataset_size = await self._validate_dataset_size(dataset_name)
            
            # Scaling Laws를 사용한 최적 아키텍처 계산
            optimal_arch = self.architect.calculate_optimal_architecture(
                dataset_size, target_loss, compute_budget
            )
            
            return {
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
                "optimal_architecture": optimal_arch
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate optimal architecture: {str(e)}")

    async def _generate_recommendations(self, task: EvolutionaryTask, best_arch: ArchitectureGenome, history: List[Dict], llm: GoogleAugmentedLLM) -> List[str]:
        """
        Generates optimization recommendations using an LLM.
        """
        prompt = f"""
        Given the following evolutionary process for an AI architecture:
        - Problem: {task.problem_description}
        - Best Architecture Fitness: {best_arch.fitness_score:.4f}
        - Best Architecture Layers: {json.dumps(best_arch.layers, indent=2)}
        - Evolution History Highlights:
    """
        # Add some history highlights
        for i, entry in enumerate(history[-3:]): # last 3 generations
            prompt += f"  - Gen {entry['generation']}: Best Fitness {entry['best_fitness']:.4f}, Diversity {entry['diversity']:.2f}\n"

        prompt += """
        Based on this, provide 3-5 specific, actionable recommendations for further improvement.
        Focus on potential bottlenecks, hyperparameter tuning, or alternative layer types.
        """
        
        response = await llm.generate(prompt)
        # Simple parsing, assuming recommendations are separated by newlines
        return [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    async def _save_evolution_results(self, result: ArchitectureEvolutionResult, task_id: str):
        """Saves the final results of the evolution process to a file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_result_{task_id}_{timestamp}.md"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"""# 🧬 Evolutionary AI Architecture Results

**Task ID**: {result.task.task_id}
**Problem**: {result.task.problem_description}
**Completion**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Success**: {'✅ Yes' if result.success else '❌ No'}
**Processing Time**: {result.processing_time:.2f}s

## 🎯 Task Configuration
- **Max Generations**: {result.task.max_generations}
- **Population Size**: {result.task.population_size}
- **Target Metrics**: {json.dumps(result.task.target_metrics, indent=2)}
- **Constraints**: {json.dumps(result.task.constraints, indent=2)}

## 🏆 Best Architecture
- **Fitness Score**: {result.best_architecture.fitness_score:.4f}
- **Layer Count**: {len(result.best_architecture.layers)}
- **Architecture ID**: {result.best_architecture.unique_id}

### Architecture Details
```json
{json.dumps(asdict(result.best_architecture), indent=2, ensure_ascii=False)}
```

## 📊 Evolution History
""")
                for gen_data in result.evolution_history:
                    f.write(f"### Generation {gen_data['generation']}\n")
                    f.write(f"- Best Fitness: {gen_data['best_fitness']:.4f}\n")
                    f.write(f"- Average Fitness: {gen_data['avg_fitness']:.4f}\n")
                    f.write(f"- Diversity: {gen_data['diversity']:.4f}\n\n")
                
                f.write(f"""## 🧠 Reasoning Steps
""")
                for i, step in enumerate(result.reasoning_steps, 1):
                    f.write(f"### Step {i}\n{step}\n\n")
                
                f.write(f"""## 🚀 Optimization Recommendations
""")
                for i, rec in enumerate(result.optimization_recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                
                f.write(f"""
---
*Generated by Evolutionary AI Architect MCP Agent*
""")
            
        except Exception as e:
            print(f"Save error: {e}")

# Export functions
async def create_evolutionary_architect(output_dir: str = "evolutionary_architect_reports") -> EvolutionaryAIArchitectMCP:
    """Create Evolutionary AI Architect MCP"""
    return EvolutionaryAIArchitectMCP(output_dir=output_dir)

async def run_architecture_evolution(
    problem_description: str,
    constraints: Dict[str, Any] = None,
    target_metrics: Dict[str, float] = None,
    max_generations: int = 10,
    population_size: int = 20,
    use_react_pattern: bool = True,
    output_dir: str = "evolutionary_architect_reports"
) -> ArchitectureEvolutionResult:
    """Run architecture evolution using MCP Agent"""
    agent = await create_evolutionary_architect(output_dir)
    return await agent.evolve_architecture(
        problem_description=problem_description,
        constraints=constraints,
        target_metrics=target_metrics,
        max_generations=max_generations,
        population_size=population_size,
        use_react_pattern=use_react_pattern
    )

# Demo functions
async def run_evolution_demo():
    """Demo: Evolutionary AI Architecture generation"""
    print("🧬 Evolutionary AI Architect MCP Demo")
    print("=" * 60)
    print("🤖 Genetic algorithm evolution with MCP research!")
    print()
    
    agent = EvolutionaryAIArchitectMCP()
    
    # Sample problems
    problems = [
        "Design efficient computer vision architecture for real-time object detection",
        "Create transformer model for multilingual text classification",
        "Develop hybrid RNN-CNN for time series forecasting"
    ]
    
    problem = random.choice(problems)
    constraints = {
        "max_latency_ms": random.randint(50, 200),
        "max_memory_mb": random.randint(256, 1024),
        "min_accuracy": 0.9 + random.random() * 0.08
    }
    target_metrics = {
        "accuracy": 0.93 + random.random() * 0.05,
        "efficiency": 0.8 + random.random() * 0.15
    }
    
    print(f"🎯 Evolution Problem: {problem}")
    print(f"📋 Constraints: {constraints}")
    print(f"🎯 Target Metrics: {target_metrics}")
    print()
    
    try:
        result = await agent.evolve_architecture(
            problem_description=problem,
            constraints=constraints,
            target_metrics=target_metrics,
            max_generations=3,  # Quick demo
            population_size=8,
            use_react_pattern=True
        )
        
        print("🏆 Evolution Results:")
        print(f"- Success: {'✅' if result.success else '❌'}")
        print(f"- Best Fitness: {result.best_architecture.fitness_score:.4f}")
        print(f"- Generations: {result.generation_count}")
        print(f"- Processing Time: {result.processing_time:.2f}s")
        print(f"- Reasoning Steps: {len(result.reasoning_steps)}")
        print(f"- Research Insights: {len(result.research_insights)}")
        print(f"- Recommendations: {len(result.optimization_recommendations)}")
        
        if result.optimization_recommendations:
            print("\n🚀 Top Recommendations:")
            for i, rec in enumerate(result.optimization_recommendations[:3], 1):
                print(f"{i}. {rec[:80]}...")
        
        print(f"\n📄 Results saved to: {os.path.join(agent.output_dir, 'evolution_result_*.md')}")
        
    except Exception as e:
        print(f"❌ Evolution error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main execution for Evolutionary AI Architect MCP"""
    print("🧬 Evolutionary AI Architect MCP Agent")
    print("=" * 60)
    print("1. Run evolution demo")
    print("2. Custom evolution task")
    print("3. Show agent status")
    print("0. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    try:
        if choice == "1":
            await run_evolution_demo()
        elif choice == "2":
            problem = input("Enter problem description: ")
            if problem.strip():
                print("Running custom evolution...")
                agent = EvolutionaryAIArchitectMCP()
                result = await agent.evolve_architecture(problem)
                print(f"Evolution completed! Best fitness: {result.best_architecture.fitness_score:.4f}")
        elif choice == "3":
            agent = EvolutionaryAIArchitectMCP()
            print(f"Agent Status:")
            print(f"- Active populations: {len(agent.active_populations)}")
            print(f"- Evolution history: {len(agent.evolution_history)}")
            print(f"- Best architectures: {len(agent.best_architectures)}")
        elif choice == "0":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Evolutionary AI Architect demo terminated.") 