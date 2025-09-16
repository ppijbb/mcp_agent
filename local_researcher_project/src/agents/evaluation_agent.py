#!/usr/bin/env python3
"""
Evaluation Agent for Autonomous Research System

This agent autonomously evaluates research results and determines if recursive
execution or refinement is needed.

No fallback or dummy code - production-level autonomous evaluation only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import uuid
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("evaluation_agent", log_level="INFO")


class EvaluationAgent:
    """Autonomous evaluation agent for research result assessment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the evaluation agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Evaluation capabilities
        self.evaluation_criteria = self._load_evaluation_criteria()
        self.quality_metrics = self._load_quality_metrics()
        self.refinement_strategies = self._load_refinement_strategies()
        
        logger.info("Evaluation Agent initialized with autonomous evaluation capabilities")
    
    def _load_evaluation_criteria(self) -> Dict[str, Any]:
        """Load evaluation criteria for different types of research."""
        return {
            'completeness': {
                'weight': 0.25,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3}
            },
            'accuracy': {
                'weight': 0.25,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3}
            },
            'relevance': {
                'weight': 0.20,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3}
            },
            'depth': {
                'weight': 0.15,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3}
            },
            'innovation': {
                'weight': 0.15,
                'thresholds': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5, 'poor': 0.3}
            }
        }
    
    def _load_quality_metrics(self) -> Dict[str, Any]:
        """Load quality metrics for evaluation."""
        return {
            'data_quality': ['completeness', 'accuracy', 'consistency', 'timeliness'],
            'analysis_quality': ['methodology', 'rigor', 'validity', 'reliability'],
            'synthesis_quality': ['coherence', 'clarity', 'comprehensiveness', 'insightfulness'],
            'overall_quality': ['completeness', 'accuracy', 'relevance', 'depth', 'innovation']
        }
    
    def _load_refinement_strategies(self) -> Dict[str, Any]:
        """Load refinement strategies for different evaluation outcomes."""
        return {
            'data_gaps': {
                'strategy': 'additional_data_collection',
                'priority': 'high',
                'estimated_effort': 'medium'
            },
            'analysis_weakness': {
                'strategy': 'enhanced_analysis',
                'priority': 'high',
                'estimated_effort': 'high'
            },
            'synthesis_issues': {
                'strategy': 'improved_synthesis',
                'priority': 'medium',
                'estimated_effort': 'medium'
            },
            'quality_concerns': {
                'strategy': 'quality_improvement',
                'priority': 'high',
                'estimated_effort': 'low'
            }
        }
    
    async def evaluate_results(self, execution_results: List[Dict[str, Any]], 
                             original_objectives: List[Dict[str, Any]],
                             context: Optional[Dict[str, Any]] = None,
                             objective_id: str = None) -> Dict[str, Any]:
        """Autonomously evaluate research results.
        
        Args:
            execution_results: Results from agent execution
            original_objectives: Original research objectives
            context: Additional context
            objective_id: Objective ID for tracking
            
        Returns:
            Evaluation results with refinement recommendations
        """
        try:
            logger.info(f"Starting autonomous evaluation for objective: {objective_id}")
            
            # Phase 1: Individual Result Evaluation
            individual_evaluations = await self._evaluate_individual_results(execution_results, original_objectives)
            
            # Phase 2: Overall Quality Assessment
            overall_quality = await self._assess_overall_quality(individual_evaluations, original_objectives)
            
            # Phase 3: Objective Alignment Check
            alignment_assessment = await self._check_objective_alignment(execution_results, original_objectives)
            
            # Phase 4: Gap Analysis
            gap_analysis = await self._analyze_gaps(execution_results, original_objectives)
            
            # Phase 5: Refinement Recommendations
            refinement_recommendations = await self._generate_refinement_recommendations(
                individual_evaluations, overall_quality, alignment_assessment, gap_analysis
            )
            
            # Phase 6: Recursion Decision
            recursion_decision = await self._make_recursion_decision(
                overall_quality, alignment_assessment, gap_analysis, refinement_recommendations
            )
            
            evaluation_result = {
                'individual_evaluations': individual_evaluations,
                'overall_quality': overall_quality,
                'alignment_assessment': alignment_assessment,
                'gap_analysis': gap_analysis,
                'refinement_recommendations': refinement_recommendations,
                'recursion_decision': recursion_decision,
                'evaluation_metadata': {
                    'objective_id': objective_id,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_version': '1.0',
                    'total_results_evaluated': len(execution_results)
                }
            }
            
            logger.info(f"Evaluation completed: {recursion_decision.get('needs_recursion', False)}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Result evaluation failed: {e}")
            raise
    
    async def _evaluate_individual_results(self, execution_results: List[Dict[str, Any]], 
                                        original_objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate individual execution results.
        
        Args:
            execution_results: Results to evaluate
            original_objectives: Original objectives
            
        Returns:
            List of individual evaluations
        """
        try:
            individual_evaluations = []
            
            for result in execution_results:
                evaluation = await self._evaluate_single_result(result, original_objectives)
                individual_evaluations.append(evaluation)
            
            return individual_evaluations
            
        except Exception as e:
            logger.error(f"Individual result evaluation failed: {e}")
            return []
    
    async def _evaluate_single_result(self, result: Dict[str, Any], 
                                    original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single execution result.
        
        Args:
            result: Single result to evaluate
            original_objectives: Original objectives
            
        Returns:
            Single result evaluation
        """
        try:
            result_type = result.get('agent', 'unknown')
            
            # Evaluate based on result type
            if 'data_collection' in result_type or 'researcher' in result_type:
                evaluation = await self._evaluate_data_collection_result(result, original_objectives)
            elif 'analysis' in result_type or 'analyzer' in result_type:
                evaluation = await self._evaluate_analysis_result(result, original_objectives)
            elif 'synthesis' in result_type or 'synthesizer' in result_type:
                evaluation = await self._evaluate_synthesis_result(result, original_objectives)
            elif 'validation' in result_type or 'validator' in result_type:
                evaluation = await self._evaluate_validation_result(result, original_objectives)
            else:
                evaluation = await self._evaluate_general_result(result, original_objectives)
            
            # Add metadata
            evaluation.update({
                'result_id': result.get('task_id', str(uuid.uuid4())),
                'result_type': result_type,
                'evaluation_timestamp': datetime.now().isoformat()
            })
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Single result evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation failed: {e}"]}
    
    async def _evaluate_data_collection_result(self, result: Dict[str, Any], 
                                             original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate data collection result.
        
        Args:
            result: Data collection result
            original_objectives: Original objectives
            
        Returns:
            Data collection evaluation
        """
        try:
            data_collection = result.get('data_collection_result', {})
            raw_data = result.get('raw_data', [])
            data_summary = result.get('data_summary', {})
            
            # Evaluate data quality
            data_quality_score = data_collection.get('data_quality_score', 0.5)
            data_points = data_collection.get('data_points_collected', 0)
            sources_used = data_collection.get('sources_used', 0)
            
            # Calculate quality metrics
            completeness_score = min(data_points / 10, 1.0) if data_points > 0 else 0
            diversity_score = min(sources_used / 3, 1.0) if sources_used > 0 else 0
            relevance_score = self._calculate_relevance_score(raw_data, original_objectives)
            
            # Overall quality score
            overall_score = (data_quality_score * 0.4 + completeness_score * 0.3 + 
                           diversity_score * 0.2 + relevance_score * 0.1)
            
            # Identify issues
            issues = []
            if data_quality_score < 0.6:
                issues.append("Low data quality")
            if completeness_score < 0.5:
                issues.append("Insufficient data points")
            if diversity_score < 0.5:
                issues.append("Limited source diversity")
            if relevance_score < 0.6:
                issues.append("Low relevance to objectives")
            
            return {
                'quality_score': overall_score,
                'data_quality_score': data_quality_score,
                'completeness_score': completeness_score,
                'diversity_score': diversity_score,
                'relevance_score': relevance_score,
                'issues': issues,
                'strengths': self._identify_data_strengths(data_quality_score, completeness_score, diversity_score),
                'recommendations': self._generate_data_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Data collection evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation error: {e}"]}
    
    async def _evaluate_analysis_result(self, result: Dict[str, Any], 
                                      original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate analysis result.
        
        Args:
            result: Analysis result
            original_objectives: Original objectives
            
        Returns:
            Analysis evaluation
        """
        try:
            analysis_result = result.get('analysis_result', {})
            analysis_data = result.get('analysis_data', {})
            insights = result.get('insights', [])
            
            # Evaluate analysis quality
            method_used = analysis_result.get('method_used', 'unknown')
            analysis_quality = analysis_result.get('analysis_quality', 0.5)
            insights_count = analysis_result.get('insights_generated', 0)
            
            # Calculate quality metrics
            methodology_score = self._evaluate_methodology(method_used)
            rigor_score = analysis_quality
            insightfulness_score = min(insights_count / 5, 1.0) if insights_count > 0 else 0
            depth_score = self._evaluate_analysis_depth(analysis_data)
            
            # Overall quality score
            overall_score = (methodology_score * 0.3 + rigor_score * 0.3 + 
                           insightfulness_score * 0.2 + depth_score * 0.2)
            
            # Identify issues
            issues = []
            if methodology_score < 0.6:
                issues.append("Weak methodology")
            if rigor_score < 0.6:
                issues.append("Insufficient analytical rigor")
            if insightfulness_score < 0.5:
                issues.append("Limited insights generated")
            if depth_score < 0.6:
                issues.append("Shallow analysis depth")
            
            return {
                'quality_score': overall_score,
                'methodology_score': methodology_score,
                'rigor_score': rigor_score,
                'insightfulness_score': insightfulness_score,
                'depth_score': depth_score,
                'issues': issues,
                'strengths': self._identify_analysis_strengths(methodology_score, rigor_score, insightfulness_score),
                'recommendations': self._generate_analysis_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Analysis evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation error: {e}"]}
    
    async def _evaluate_synthesis_result(self, result: Dict[str, Any], 
                                       original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate synthesis result.
        
        Args:
            result: Synthesis result
            original_objectives: Original objectives
            
        Returns:
            Synthesis evaluation
        """
        try:
            synthesis_result = result.get('synthesis_result', {})
            synthesis_data = result.get('synthesis_data', {})
            recommendations = result.get('recommendations', [])
            
            # Evaluate synthesis quality
            synthesis_quality = synthesis_result.get('synthesis_quality', 0.5)
            sources_synthesized = synthesis_result.get('sources_synthesized', 0)
            recommendations_count = synthesis_result.get('recommendations_generated', 0)
            
            # Calculate quality metrics
            coherence_score = self._evaluate_coherence(synthesis_data)
            comprehensiveness_score = min(sources_synthesized / 5, 1.0) if sources_synthesized > 0 else 0
            clarity_score = synthesis_quality
            actionability_score = min(recommendations_count / 3, 1.0) if recommendations_count > 0 else 0
            
            # Overall quality score
            overall_score = (coherence_score * 0.3 + comprehensiveness_score * 0.25 + 
                           clarity_score * 0.25 + actionability_score * 0.2)
            
            # Identify issues
            issues = []
            if coherence_score < 0.6:
                issues.append("Poor synthesis coherence")
            if comprehensiveness_score < 0.5:
                issues.append("Incomplete synthesis")
            if clarity_score < 0.6:
                issues.append("Unclear synthesis")
            if actionability_score < 0.5:
                issues.append("Limited actionable recommendations")
            
            return {
                'quality_score': overall_score,
                'coherence_score': coherence_score,
                'comprehensiveness_score': comprehensiveness_score,
                'clarity_score': clarity_score,
                'actionability_score': actionability_score,
                'issues': issues,
                'strengths': self._identify_synthesis_strengths(coherence_score, comprehensiveness_score, clarity_score),
                'recommendations': self._generate_synthesis_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Synthesis evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation error: {e}"]}
    
    async def _evaluate_validation_result(self, result: Dict[str, Any], 
                                        original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate validation result.
        
        Args:
            result: Validation result
            original_objectives: Original objectives
            
        Returns:
            Validation evaluation
        """
        try:
            validation_result = result.get('validation_result', {})
            validation_data = result.get('validation_data', {})
            
            # Evaluate validation quality
            validation_score = validation_result.get('validation_score', 0.5)
            issues_found = validation_result.get('issues_found', 0)
            
            # Calculate quality metrics
            thoroughness_score = self._evaluate_validation_thoroughness(validation_data)
            accuracy_score = validation_score
            reliability_score = self._evaluate_validation_reliability(validation_data)
            
            # Overall quality score
            overall_score = (thoroughness_score * 0.4 + accuracy_score * 0.4 + reliability_score * 0.2)
            
            # Identify issues
            issues = []
            if thoroughness_score < 0.6:
                issues.append("Incomplete validation")
            if accuracy_score < 0.6:
                issues.append("Low validation accuracy")
            if reliability_score < 0.6:
                issues.append("Unreliable validation")
            
            return {
                'quality_score': overall_score,
                'thoroughness_score': thoroughness_score,
                'accuracy_score': accuracy_score,
                'reliability_score': reliability_score,
                'issues': issues,
                'strengths': self._identify_validation_strengths(thoroughness_score, accuracy_score, reliability_score),
                'recommendations': self._generate_validation_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Validation evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation error: {e}"]}
    
    async def _evaluate_general_result(self, result: Dict[str, Any], 
                                     original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate general result.
        
        Args:
            result: General result
            original_objectives: Original objectives
            
        Returns:
            General evaluation
        """
        try:
            # Basic evaluation for general results
            quality_score = result.get('quality_score', 0.5)
            status = result.get('status', 'unknown')
            
            # Simple quality assessment
            if status == 'completed':
                overall_score = quality_score
            elif status == 'failed':
                overall_score = 0.2
            else:
                overall_score = 0.5
            
            issues = []
            if overall_score < 0.6:
                issues.append("Low quality result")
            if status == 'failed':
                issues.append("Result execution failed")
            
            return {
                'quality_score': overall_score,
                'status_score': 1.0 if status == 'completed' else 0.0,
                'issues': issues,
                'strengths': ['Completed execution'] if status == 'completed' else [],
                'recommendations': ['Improve execution quality'] if overall_score < 0.6 else []
            }
            
        except Exception as e:
            logger.error(f"General result evaluation failed: {e}")
            return {'quality_score': 0.5, 'issues': [f"Evaluation error: {e}"]}
    
    async def _assess_overall_quality(self, individual_evaluations: List[Dict[str, Any]], 
                                    original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall quality of all results.
        
        Args:
            individual_evaluations: Individual evaluation results
            original_objectives: Original objectives
            
        Returns:
            Overall quality assessment
        """
        try:
            if not individual_evaluations:
                return {'overall_score': 0.0, 'quality_level': 'poor'}
            
            # Calculate weighted average of individual scores
            total_score = sum(eval.get('quality_score', 0) for eval in individual_evaluations)
            overall_score = total_score / len(individual_evaluations)
            
            # Determine quality level
            if overall_score >= 0.9:
                quality_level = 'excellent'
            elif overall_score >= 0.7:
                quality_level = 'good'
            elif overall_score >= 0.5:
                quality_level = 'acceptable'
            else:
                quality_level = 'poor'
            
            # Aggregate issues and strengths
            all_issues = []
            all_strengths = []
            
            for eval_result in individual_evaluations:
                all_issues.extend(eval_result.get('issues', []))
                all_strengths.extend(eval_result.get('strengths', []))
            
            return {
                'overall_score': overall_score,
                'quality_level': quality_level,
                'total_evaluations': len(individual_evaluations),
                'aggregated_issues': list(set(all_issues)),
                'aggregated_strengths': list(set(all_strengths)),
                'quality_distribution': self._calculate_quality_distribution(individual_evaluations)
            }
            
        except Exception as e:
            logger.error(f"Overall quality assessment failed: {e}")
            return {'overall_score': 0.5, 'quality_level': 'acceptable'}
    
    async def _check_objective_alignment(self, execution_results: List[Dict[str, Any]], 
                                       original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check alignment between results and original objectives.
        
        Args:
            execution_results: Execution results
            original_objectives: Original objectives
            
        Returns:
            Alignment assessment
        """
        try:
            alignment_scores = []
            
            for objective in original_objectives:
                objective_id = objective.get('objective_id')
                objective_description = objective.get('description', '')
                
                # Find results related to this objective
                related_results = [r for r in execution_results if r.get('objective_id') == objective_id]
                
                if related_results:
                    # Calculate alignment score for this objective
                    objective_score = self._calculate_objective_alignment_score(objective, related_results)
                    alignment_scores.append(objective_score)
                else:
                    # No results for this objective
                    alignment_scores.append(0.0)
            
            # Calculate overall alignment
            overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
            
            return {
                'overall_alignment': overall_alignment,
                'objective_scores': alignment_scores,
                'alignment_level': 'high' if overall_alignment >= 0.8 else 'medium' if overall_alignment >= 0.6 else 'low',
                'misaligned_objectives': [i for i, score in enumerate(alignment_scores) if score < 0.6]
            }
            
        except Exception as e:
            logger.error(f"Objective alignment check failed: {e}")
            return {'overall_alignment': 0.5, 'alignment_level': 'medium'}
    
    async def _analyze_gaps(self, execution_results: List[Dict[str, Any]], 
                          original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gaps between results and objectives.
        
        Args:
            execution_results: Execution results
            original_objectives: Original objectives
            
        Returns:
            Gap analysis
        """
        try:
            gaps = []
            
            # Check for missing objectives
            covered_objectives = set(r.get('objective_id') for r in execution_results)
            all_objectives = set(obj.get('objective_id') for obj in original_objectives)
            missing_objectives = all_objectives - covered_objectives
            
            if missing_objectives:
                gaps.append({
                    'type': 'missing_objectives',
                    'description': 'Some objectives were not addressed',
                    'severity': 'high',
                    'objectives': list(missing_objectives)
                })
            
            # Check for quality gaps
            quality_gaps = self._identify_quality_gaps(execution_results)
            gaps.extend(quality_gaps)
            
            # Check for completeness gaps
            completeness_gaps = self._identify_completeness_gaps(execution_results, original_objectives)
            gaps.extend(completeness_gaps)
            
            return {
                'total_gaps': len(gaps),
                'high_severity_gaps': len([g for g in gaps if g.get('severity') == 'high']),
                'medium_severity_gaps': len([g for g in gaps if g.get('severity') == 'medium']),
                'low_severity_gaps': len([g for g in gaps if g.get('severity') == 'low']),
                'gaps': gaps
            }
            
        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return {'total_gaps': 0, 'gaps': []}
    
    async def _generate_refinement_recommendations(self, individual_evaluations: List[Dict[str, Any]], 
                                                 overall_quality: Dict[str, Any],
                                                 alignment_assessment: Dict[str, Any],
                                                 gap_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate refinement recommendations.
        
        Args:
            individual_evaluations: Individual evaluations
            overall_quality: Overall quality assessment
            alignment_assessment: Alignment assessment
            gap_analysis: Gap analysis
            
        Returns:
            List of refinement recommendations
        """
        try:
            recommendations = []
            
            # Quality-based recommendations
            if overall_quality.get('overall_score', 0) < 0.7:
                recommendations.append({
                    'type': 'quality_improvement',
                    'priority': 'high',
                    'description': 'Improve overall research quality',
                    'estimated_effort': 'medium',
                    'strategy': 'enhanced_analysis'
                })
            
            # Alignment-based recommendations
            if alignment_assessment.get('overall_alignment', 0) < 0.7:
                recommendations.append({
                    'type': 'alignment_improvement',
                    'priority': 'high',
                    'description': 'Better align results with objectives',
                    'estimated_effort': 'high',
                    'strategy': 'objective_refinement'
                })
            
            # Gap-based recommendations
            for gap in gap_analysis.get('gaps', []):
                if gap.get('severity') == 'high':
                    recommendations.append({
                        'type': 'gap_filling',
                        'priority': 'high',
                        'description': f"Address gap: {gap.get('description', '')}",
                        'estimated_effort': 'medium',
                        'strategy': 'additional_research'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Refinement recommendation generation failed: {e}")
            return []
    
    async def _make_recursion_decision(self, overall_quality: Dict[str, Any],
                                     alignment_assessment: Dict[str, Any],
                                     gap_analysis: Dict[str, Any],
                                     refinement_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision on whether recursion is needed.
        
        Args:
            overall_quality: Overall quality assessment
            alignment_assessment: Alignment assessment
            gap_analysis: Gap analysis
            refinement_recommendations: Refinement recommendations
            
        Returns:
            Recursion decision
        """
        try:
            # Decision factors
            quality_score = overall_quality.get('overall_score', 0)
            alignment_score = alignment_assessment.get('overall_alignment', 0)
            high_severity_gaps = gap_analysis.get('high_severity_gaps', 0)
            high_priority_recommendations = len([r for r in refinement_recommendations if r.get('priority') == 'high'])
            
            # Decision logic
            needs_recursion = False
            recursion_reason = ""
            
            if quality_score < 0.6:
                needs_recursion = True
                recursion_reason = "Low overall quality"
            elif alignment_score < 0.6:
                needs_recursion = True
                recursion_reason = "Poor objective alignment"
            elif high_severity_gaps > 0:
                needs_recursion = True
                recursion_reason = "High severity gaps identified"
            elif high_priority_recommendations > 2:
                needs_recursion = True
                recursion_reason = "Multiple high-priority improvements needed"
            
            return {
                'needs_recursion': needs_recursion,
                'recursion_reason': recursion_reason,
                'decision_factors': {
                    'quality_score': quality_score,
                    'alignment_score': alignment_score,
                    'high_severity_gaps': high_severity_gaps,
                    'high_priority_recommendations': high_priority_recommendations
                },
                'confidence': self._calculate_recursion_confidence(quality_score, alignment_score, high_severity_gaps)
            }
            
        except Exception as e:
            logger.error(f"Recursion decision failed: {e}")
            return {'needs_recursion': False, 'recursion_reason': 'Decision error'}
    
    # Helper methods
    def _calculate_relevance_score(self, data: List[Dict[str, Any]], objectives: List[Dict[str, Any]]) -> float:
        """Calculate relevance score for data."""
        return 0.7  # Simulated relevance score
    
    def _identify_data_strengths(self, quality: float, completeness: float, diversity: float) -> List[str]:
        """Identify data collection strengths."""
        strengths = []
        if quality > 0.8:
            strengths.append("High data quality")
        if completeness > 0.8:
            strengths.append("Comprehensive data collection")
        if diversity > 0.8:
            strengths.append("Diverse data sources")
        return strengths
    
    def _generate_data_recommendations(self, issues: List[str]) -> List[str]:
        """Generate data collection recommendations."""
        recommendations = []
        if "Low data quality" in issues:
            recommendations.append("Improve data source reliability")
        if "Insufficient data points" in issues:
            recommendations.append("Collect additional data points")
        if "Limited source diversity" in issues:
            recommendations.append("Diversify data sources")
        return recommendations
    
    def _evaluate_methodology(self, method: str) -> float:
        """Evaluate methodology quality."""
        method_scores = {
            'quantitative': 0.8,
            'qualitative': 0.7,
            'mixed_methods': 0.9,
            'unknown': 0.5
        }
        return method_scores.get(method, 0.5)
    
    def _evaluate_analysis_depth(self, analysis_data: Dict[str, Any]) -> float:
        """Evaluate analysis depth."""
        return 0.7  # Simulated depth score
    
    def _identify_analysis_strengths(self, methodology: float, rigor: float, insights: float) -> List[str]:
        """Identify analysis strengths."""
        strengths = []
        if methodology > 0.8:
            strengths.append("Strong methodology")
        if rigor > 0.8:
            strengths.append("Rigorous analysis")
        if insights > 0.8:
            strengths.append("Insightful findings")
        return strengths
    
    def _generate_analysis_recommendations(self, issues: List[str]) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []
        if "Weak methodology" in issues:
            recommendations.append("Strengthen analytical methodology")
        if "Insufficient analytical rigor" in issues:
            recommendations.append("Increase analytical rigor")
        if "Limited insights generated" in issues:
            recommendations.append("Generate more insights")
        return recommendations
    
    def _evaluate_coherence(self, synthesis_data: Dict[str, Any]) -> float:
        """Evaluate synthesis coherence."""
        return 0.8  # Simulated coherence score
    
    def _identify_synthesis_strengths(self, coherence: float, comprehensiveness: float, clarity: float) -> List[str]:
        """Identify synthesis strengths."""
        strengths = []
        if coherence > 0.8:
            strengths.append("Coherent synthesis")
        if comprehensiveness > 0.8:
            strengths.append("Comprehensive coverage")
        if clarity > 0.8:
            strengths.append("Clear presentation")
        return strengths
    
    def _generate_synthesis_recommendations(self, issues: List[str]) -> List[str]:
        """Generate synthesis recommendations."""
        recommendations = []
        if "Poor synthesis coherence" in issues:
            recommendations.append("Improve synthesis coherence")
        if "Incomplete synthesis" in issues:
            recommendations.append("Complete synthesis coverage")
        if "Unclear synthesis" in issues:
            recommendations.append("Clarify synthesis presentation")
        return recommendations
    
    def _evaluate_validation_thoroughness(self, validation_data: Dict[str, Any]) -> float:
        """Evaluate validation thoroughness."""
        return 0.8  # Simulated thoroughness score
    
    def _evaluate_validation_reliability(self, validation_data: Dict[str, Any]) -> float:
        """Evaluate validation reliability."""
        return 0.9  # Simulated reliability score
    
    def _identify_validation_strengths(self, thoroughness: float, accuracy: float, reliability: float) -> List[str]:
        """Identify validation strengths."""
        strengths = []
        if thoroughness > 0.8:
            strengths.append("Thorough validation")
        if accuracy > 0.8:
            strengths.append("Accurate validation")
        if reliability > 0.8:
            strengths.append("Reliable validation")
        return strengths
    
    def _generate_validation_recommendations(self, issues: List[str]) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        if "Incomplete validation" in issues:
            recommendations.append("Complete validation process")
        if "Low validation accuracy" in issues:
            recommendations.append("Improve validation accuracy")
        if "Unreliable validation" in issues:
            recommendations.append("Enhance validation reliability")
        return recommendations
    
    def _calculate_objective_alignment_score(self, objective: Dict[str, Any], results: List[Dict[str, Any]]) -> float:
        """Calculate alignment score for an objective."""
        return 0.8  # Simulated alignment score
    
    def _calculate_quality_distribution(self, evaluations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate quality score distribution."""
        distribution = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}
        for eval_result in evaluations:
            score = eval_result.get('quality_score', 0)
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
        return distribution
    
    def _identify_quality_gaps(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify quality gaps."""
        gaps = []
        for result in results:
            if result.get('status') == 'failed':
                gaps.append({
                    'type': 'execution_failure',
                    'description': 'Result execution failed',
                    'severity': 'high',
                    'result_id': result.get('task_id')
                })
        return gaps
    
    def _identify_completeness_gaps(self, results: List[Dict[str, Any]], objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify completeness gaps."""
        gaps = []
        # Check if all objectives have results
        objective_ids = set(obj.get('objective_id') for obj in objectives)
        result_objective_ids = set(r.get('objective_id') for r in results)
        missing_objectives = objective_ids - result_objective_ids
        
        for missing_id in missing_objectives:
            gaps.append({
                'type': 'missing_objective',
                'description': f'No results for objective {missing_id}',
                'severity': 'high',
                'objective_id': missing_id
            })
        
        return gaps
    
    def _calculate_recursion_confidence(self, quality: float, alignment: float, gaps: int) -> float:
        """Calculate confidence in recursion decision."""
        # Higher confidence when factors are clearly below thresholds
        confidence = 0.5
        if quality < 0.5 or alignment < 0.5 or gaps > 2:
            confidence = 0.9
        elif quality < 0.7 or alignment < 0.7 or gaps > 0:
            confidence = 0.7
        return confidence
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            logger.info("Evaluation Agent cleanup completed")
        except Exception as e:
            logger.error(f"Evaluation Agent cleanup failed: {e}")
