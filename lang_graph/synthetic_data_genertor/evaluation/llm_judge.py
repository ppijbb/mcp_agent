"""
LLM Judge System for the Kimi-K2 Agentic Data Synthesis System

Provides rubric-based evaluation of simulation results using LLM judges.
"""

from typing import List, Dict, Any, Optional
from ..models.evaluation import EvaluationResult, QualityScore, Rubric, EvaluationType
from ..models.simulation import SimulationSession
import logging
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)


class LLMJudgeSystem:
    """
    LLM-based evaluation system for simulation results.
    
    Responsibilities:
    - Rubric-based evaluation using LLM judges
    - Multi-dimensional quality assessment
    - Consistent evaluation standards
    - Evaluation result management
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.evaluators: Dict[str, Dict[str, Any]] = {}
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.rubrics: Dict[str, Rubric] = {}
        
        # Initialize default evaluators
        self._initialize_evaluators()
    
    def _initialize_evaluators(self) -> None:
        """Initialize default LLM evaluators"""
        self.evaluators = {
            "accuracy_judge": {
                "name": "Accuracy Judge",
                "description": "Evaluates factual accuracy and correctness",
                "evaluation_type": EvaluationType.ACCURACY,
                "prompt_template": """
                Evaluate the accuracy of the agent's response based on the following criteria:
                
                Scenario: {scenario_description}
                Expected Outcome: {expected_outcome}
                Agent Response: {agent_response}
                
                Rate the accuracy from 0.0 to 1.0 and provide reasoning.
                Consider:
                - Factual correctness
                - Logical consistency
                - Adherence to requirements
                
                Score: [0.0-1.0]
                Reasoning: [detailed explanation]
                Evidence: [specific examples]
                """
            },
            "completeness_judge": {
                "name": "Completeness Judge",
                "description": "Evaluates completeness of the solution",
                "evaluation_type": EvaluationType.COMPLETENESS,
                "prompt_template": """
                Evaluate the completeness of the agent's solution based on the following criteria:
                
                Scenario: {scenario_description}
                Required Steps: {required_steps}
                Agent Actions: {agent_actions}
                
                Rate the completeness from 0.0 to 1.0 and provide reasoning.
                Consider:
                - Coverage of all required steps
                - Depth of analysis
                - Solution comprehensiveness
                
                Score: [0.0-1.0]
                Reasoning: [detailed explanation]
                Evidence: [specific examples]
                """
            },
            "creativity_judge": {
                "name": "Creativity Judge",
                "description": "Evaluates creativity and innovation",
                "evaluation_type": EvaluationType.CREATIVITY,
                "prompt_template": """
                Evaluate the creativity of the agent's approach based on the following criteria:
                
                Scenario: {scenario_description}
                Standard Solutions: {standard_solutions}
                Agent Approach: {agent_approach}
                
                Rate the creativity from 0.0 to 1.0 and provide reasoning.
                Consider:
                - Novelty of approach
                - Innovation in problem-solving
                - Original thinking
                
                Score: [0.0-1.0]
                Reasoning: [detailed explanation]
                Evidence: [specific examples]
                """
            },
            "efficiency_judge": {
                "name": "Efficiency Judge",
                "description": "Evaluates efficiency and optimization",
                "evaluation_type": EvaluationType.EFFICIENCY,
                "prompt_template": """
                Evaluate the efficiency of the agent's solution based on the following criteria:
                
                Scenario: {scenario_description}
                Time Taken: {time_taken}
                Resources Used: {resources_used}
                Solution Quality: {solution_quality}
                
                Rate the efficiency from 0.0 to 1.0 and provide reasoning.
                Consider:
                - Time optimization
                - Resource utilization
                - Cost-effectiveness
                
                Score: [0.0-1.0]
                Reasoning: [detailed explanation]
                Evidence: [specific examples]
                """
            },
            "user_satisfaction_judge": {
                "name": "User Satisfaction Judge",
                "description": "Evaluates user satisfaction and communication",
                "evaluation_type": EvaluationType.USER_SATISFACTION,
                "prompt_template": """
                Evaluate the user satisfaction based on the following criteria:
                
                User Request: {user_request}
                Agent Response: {agent_response}
                User Feedback: {user_feedback}
                
                Rate the user satisfaction from 0.0 to 1.0 and provide reasoning.
                Consider:
                - Communication clarity
                - Responsiveness to user needs
                - Professionalism
                
                Score: [0.0-1.0]
                Reasoning: [detailed explanation]
                Evidence: [specific examples]
                """
            }
        }
    
    def add_rubric(self, rubric: Rubric) -> None:
        """Add a rubric to the judge system"""
        self.rubrics[rubric.id] = rubric
        logger.info(f"Added rubric: {rubric.name}")
    
    def get_rubric(self, rubric_id: str) -> Optional[Rubric]:
        """Get a rubric by ID"""
        return self.rubrics.get(rubric_id)
    
    async def evaluate_simulation(self, simulation_session: SimulationSession, 
                                rubric_id: str, evaluator_ids: List[str] = None) -> Optional[EvaluationResult]:
        """Evaluate a simulation session using specified rubric and evaluators"""
        rubric = self.get_rubric(rubric_id)
        if not rubric:
            logger.error(f"Rubric not found: {rubric_id}")
            return None
        
        if not evaluator_ids:
            evaluator_ids = list(self.evaluators.keys())
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            simulation_id=simulation_session.id,
            rubric_id=rubric_id,
            evaluator_id="llm_judge_system",
            overall_score=0.0,
            feedback="",
            recommendations=[]
        )
        
        start_time = datetime.utcnow()
        
        try:
            # Deterministic scoring heuristics based on steps and tool usage
            total_steps = len(simulation_session.steps)
            tool_steps = sum(1 for s in simulation_session.steps if s.step_type.value == "tool_usage")
            fail_steps = sum(1 for s in simulation_session.steps if s.status.value == "failed")

            # Accuracy proxy: fewer failures, coherent final_outcome/steps ratio
            accuracy_score = max(0.0, 1.0 - (fail_steps / (total_steps or 1)))
            # Completeness proxy: at least one tool usage and multiple steps
            completeness_score = 0.6 + 0.4 * (1 if tool_steps > 0 else 0)
            # Creativity proxy: presence of reasoning and varied tools
            used_tools = {s.tool_used for s in simulation_session.steps if s.tool_used}
            creativity_score = 0.5 + 0.1 * min(len(used_tools), 5)
            # Efficiency proxy: shorter sessions with success
            duration = (simulation_session.duration or 0.0) or max(
                (simulation_session.end_time or datetime.utcnow() - (simulation_session.start_time or datetime.utcnow())).total_seconds(),
                1.0,
            )
            efficiency_score = max(0.4, min(1.0, 1.2 - (duration / 600.0)))
            # User satisfaction proxy: blend of above
            user_sat = (0.4 * accuracy_score + 0.3 * completeness_score + 0.3 * efficiency_score)

            # Build dimension scores
            scores: List[QualityScore] = [
                QualityScore(evaluation_type=EvaluationType.ACCURACY, score=round(accuracy_score, 3), confidence=0.9, reasoning="Heuristic based on failed steps"),
                QualityScore(evaluation_type=EvaluationType.COMPLETENESS, score=round(completeness_score, 3), confidence=0.8, reasoning="Heuristic based on tool usage presence"),
                QualityScore(evaluation_type=EvaluationType.CREATIVITY, score=round(creativity_score, 3), confidence=0.7, reasoning="Heuristic based on diversity of tools"),
                QualityScore(evaluation_type=EvaluationType.EFFICIENCY, score=round(efficiency_score, 3), confidence=0.75, reasoning="Heuristic based on duration"),
                QualityScore(evaluation_type=EvaluationType.USER_SATISFACTION, score=round(user_sat, 3), confidence=0.8, reasoning="Aggregate of other metrics"),
            ]

            for s in scores:
                evaluation_result.add_score(s)

            evaluation_result.overall_score = evaluation_result.calculate_overall_score()
            evaluation_result.determine_pass_fail(rubric.passing_threshold)
            evaluation_result.feedback = self._generate_feedback(evaluation_result, rubric)
            evaluation_result.recommendations = evaluation_result.get_recommendations_by_score()

            end_time = datetime.utcnow()
            evaluation_result.evaluation_time = (end_time - start_time).total_seconds()
            self.evaluation_results[evaluation_result.id] = evaluation_result
            logger.info(f"Completed evaluation for simulation {simulation_session.id}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Evaluation failed for simulation {simulation_session.id}: {e}")
            return None
    
    async def _run_evaluation(self, simulation_session: SimulationSession, 
                            evaluator: Dict[str, Any]) -> Optional[QualityScore]:
        """Run evaluation using a specific evaluator"""
        try:
            # Prepare evaluation context
            context = self._prepare_evaluation_context(simulation_session, evaluator)
            
            # Generate evaluation prompt
            prompt = self._generate_evaluation_prompt(evaluator, context)
            
            # Get LLM evaluation (simulated for now)
            evaluation_response = await self._get_llm_evaluation(prompt)
            
            # Parse evaluation response
            score = self._parse_evaluation_response(evaluation_response, evaluator)
            
            return score
            
        except Exception as e:
            logger.error(f"Evaluation failed for {evaluator['name']}: {e}")
            return None
    
    def _prepare_evaluation_context(self, simulation_session: SimulationSession, 
                                  evaluator: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for evaluation"""
        context = {
            "scenario_description": "Simulation scenario description",
            "expected_outcome": "Expected outcome from scenario",
            "agent_response": "Agent's response and actions",
            "user_request": "Original user request",
            "user_feedback": "User feedback if available",
            "time_taken": simulation_session.duration or 0.0,
            "resources_used": "Resources utilized during simulation",
            "solution_quality": "Quality of the solution provided"
        }
        
        # Add simulation-specific data
        if simulation_session.steps:
            context["agent_actions"] = [
                {
                    "step": step.step_number,
                    "description": step.description,
                    "output": step.output_data
                }
                for step in simulation_session.steps
            ]
        
        return context
    
    def _generate_evaluation_prompt(self, evaluator: Dict[str, Any], 
                                  context: Dict[str, Any]) -> str:
        """Generate evaluation prompt for the evaluator"""
        prompt_template = evaluator["prompt_template"]
        
        # Replace placeholders with context values
        prompt = prompt_template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                if isinstance(value, list):
                    value = json.dumps(value, indent=2)
                prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    async def _get_llm_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Get evaluation from LLM (simulated for now)"""
        # Simulate LLM evaluation
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Generate synthetic evaluation response
        import random
        
        score = random.uniform(0.6, 0.95)  # Realistic score range
        confidence = random.uniform(0.7, 0.95)
        
        reasoning_options = [
            "The agent demonstrated good understanding of the requirements and provided a comprehensive solution.",
            "The response was technically accurate and addressed the key aspects of the problem.",
            "The solution showed creativity in approaching the challenge while maintaining effectiveness.",
            "The agent efficiently utilized available resources and completed the task within reasonable time.",
            "The communication was clear and professional, meeting user expectations."
        ]
        
        evidence_options = [
            "Correctly identified the root cause of the issue",
            "Provided step-by-step solution with clear explanations",
            "Used appropriate tools and techniques",
            "Demonstrated systematic problem-solving approach",
            "Maintained professional communication throughout"
        ]
        
        return {
            "score": score,
            "confidence": confidence,
            "reasoning": random.choice(reasoning_options),
            "evidence": [random.choice(evidence_options) for _ in range(2)]
        }
    
    def _parse_evaluation_response(self, response: Dict[str, Any], 
                                 evaluator: Dict[str, Any]) -> QualityScore:
        """Parse LLM evaluation response into QualityScore"""
        return QualityScore(
            evaluation_type=evaluator["evaluation_type"],
            score=response["score"],
            confidence=response["confidence"],
            reasoning=response["reasoning"],
            evidence=response["evidence"]
        )
    
    def _generate_feedback(self, evaluation_result: EvaluationResult, rubric: Rubric) -> str:
        """Generate comprehensive feedback based on evaluation results"""
        feedback_parts = []
        
        # Overall assessment
        if evaluation_result.overall_score >= 0.8:
            feedback_parts.append("Excellent performance overall.")
        elif evaluation_result.overall_score >= 0.6:
            feedback_parts.append("Good performance with room for improvement.")
        else:
            feedback_parts.append("Performance needs significant improvement.")
        
        # Individual dimension feedback
        for score in evaluation_result.individual_scores:
            if score.score >= 0.8:
                feedback_parts.append(f"Strong {score.evaluation_type.value}.")
            elif score.score >= 0.6:
                feedback_parts.append(f"Acceptable {score.evaluation_type.value}.")
            else:
                feedback_parts.append(f"Needs improvement in {score.evaluation_type.value}.")
        
        # Specific recommendations
        if evaluation_result.recommendations:
            feedback_parts.append("Recommendations for improvement:")
            feedback_parts.extend([f"- {rec}" for rec in evaluation_result.recommendations[:3]])
        
        return " ".join(feedback_parts)
    
    def get_evaluation_result(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation result by ID"""
        return self.evaluation_results.get(evaluation_id)
    
    def list_evaluation_results(self, simulation_id: Optional[str] = None) -> List[EvaluationResult]:
        """List evaluation results with optional filtering"""
        results = list(self.evaluation_results.values())
        
        if simulation_id:
            results = [r for r in results if r.simulation_id == simulation_id]
        
        return results
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluations"""
        stats = {
            "total_evaluations": len(self.evaluation_results),
            "average_score": 0.0,
            "pass_rate": 0.0,
            "evaluations_by_type": {},
            "top_performers": [],
            "needs_improvement": []
        }
        
        if not self.evaluation_results:
            return stats
        
        # Calculate averages
        total_score = sum(r.overall_score for r in self.evaluation_results.values())
        stats["average_score"] = total_score / len(self.evaluation_results)
        
        passed_evaluations = sum(1 for r in self.evaluation_results.values() if r.passed)
        stats["pass_rate"] = passed_evaluations / len(self.evaluation_results)
        
        # Evaluations by type
        for result in self.evaluation_results.values():
            for score in result.individual_scores:
                eval_type = score.evaluation_type.value
                if eval_type not in stats["evaluations_by_type"]:
                    stats["evaluations_by_type"][eval_type] = []
                stats["evaluations_by_type"][eval_type].append(score.score)
        
        # Calculate averages for each type
        for eval_type in stats["evaluations_by_type"]:
            scores = stats["evaluations_by_type"][eval_type]
            stats["evaluations_by_type"][eval_type] = sum(scores) / len(scores)
        
        # Top performers and needs improvement
        sorted_results = sorted(
            self.evaluation_results.values(),
            key=lambda x: x.overall_score,
            reverse=True
        )
        
        stats["top_performers"] = [
            {"simulation_id": r.simulation_id, "score": r.overall_score}
            for r in sorted_results[:5]
        ]
        
        stats["needs_improvement"] = [
            {"simulation_id": r.simulation_id, "score": r.overall_score}
            for r in sorted_results[-5:]
        ]
        
        return stats
    
    def add_custom_evaluator(self, evaluator_id: str, evaluator_config: Dict[str, Any]) -> bool:
        """Add a custom evaluator to the system"""
        if evaluator_id in self.evaluators:
            logger.warning(f"Evaluator already exists: {evaluator_id}")
            return False
        
        self.evaluators[evaluator_id] = evaluator_config
        logger.info(f"Added custom evaluator: {evaluator_id}")
        return True
    
    def remove_evaluator(self, evaluator_id: str) -> bool:
        """Remove an evaluator from the system"""
        if evaluator_id not in self.evaluators:
            return False
        
        del self.evaluators[evaluator_id]
        logger.info(f"Removed evaluator: {evaluator_id}")
        return True 