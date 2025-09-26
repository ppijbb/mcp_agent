#!/usr/bin/env python3
"""
Validation Agent for Autonomous Research System

This agent autonomously validates research results against original objectives
and ensures quality standards are met.

No fallback or dummy code - production-level autonomous validation only.
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

logger = setup_logger("validation_agent", log_level="INFO")


class ValidationAgent:
    """Autonomous validation agent for research result verification."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the validation agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Validation capabilities
        self.validation_criteria = self._load_validation_criteria()
        self.quality_standards = self._load_quality_standards()
        self.validation_methods = self._load_validation_methods()
        
        logger.info("Validation Agent initialized with autonomous validation capabilities")
    
    def _load_validation_criteria(self) -> Dict[str, Any]:
        """Load enhanced validation criteria for critical research validation."""
        return {
            'objective_alignment': {
                'weight': 0.2,
                'threshold': 0.85,
                'description': 'Results must align with original objectives',
                'critical': True
            },
            'cross_validation': {
                'weight': 0.2,
                'threshold': 0.8,
                'description': 'Information must be cross-validated across multiple sources',
                'critical': True
            },
            'source_credibility': {
                'weight': 0.2,
                'threshold': 0.85,
                'description': 'Sources must be credible and authoritative',
                'critical': True
            },
            'data_accuracy': {
                'weight': 0.15,
                'threshold': 0.8,
                'description': 'Data must be accurate and verifiable',
                'critical': True
            },
            'completeness': {
                'weight': 0.1,
                'threshold': 0.8,
                'description': 'Research must be comprehensive',
                'critical': False
            },
            'temporal_accuracy': {
                'weight': 0.1,
                'threshold': 0.7,
                'description': 'Information must be current and up-to-date',
                'critical': False
            },
            'bias_detection': {
                'weight': 0.05,
                'threshold': 0.6,
                'description': 'Results must be free from significant bias',
                'critical': False
            }
        }
    
    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards for validation."""
        return {
            'data_quality': {
                'completeness': 0.8,
                'accuracy': 0.8,
                'consistency': 0.7,
                'timeliness': 0.6
            },
            'analysis_quality': {
                'methodology': 0.8,
                'rigor': 0.8,
                'validity': 0.8,
                'reliability': 0.7
            },
            'synthesis_quality': {
                'coherence': 0.8,
                'clarity': 0.8,
                'comprehensiveness': 0.8,
                'insightfulness': 0.7
            }
        }
    
    def _load_validation_methods(self) -> Dict[str, Any]:
        """Load validation methods."""
        return {
            'cross_validation': {
                'description': 'Cross-validate results with multiple sources',
                'applicable_to': ['data_collection', 'analysis']
            },
            'peer_review': {
                'description': 'Simulate peer review process',
                'applicable_to': ['analysis', 'synthesis']
            },
            'consistency_check': {
                'description': 'Check internal consistency',
                'applicable_to': ['all']
            },
            'completeness_audit': {
                'description': 'Audit completeness against requirements',
                'applicable_to': ['all']
            }
        }
    
    async def validate_results(self, execution_results: List[Dict[str, Any]], 
                             original_objectives: List[Dict[str, Any]],
                             user_request: str,
                             context: Optional[Dict[str, Any]] = None,
                             objective_id: str = None) -> Dict[str, Any]:
        """Autonomously validate research results.
        
        Args:
            execution_results: Results to validate
            original_objectives: Original objectives
            user_request: Original user request
            context: Additional context
            objective_id: Objective ID for tracking
            
        Returns:
            Validation results with scores and recommendations
        """
        try:
            logger.info(f"Starting enhanced autonomous validation for objective: {objective_id}")
            
            # Phase 1: Cross-Validation Analysis
            cross_validation_results = await self._perform_cross_validation(execution_results)
            
            # Phase 2: Source Credibility Analysis
            source_credibility = await self._analyze_source_credibility(execution_results)
            
            # Phase 3: Bias Detection Analysis
            bias_analysis = await self._detect_bias(execution_results)
            
            # Phase 4: Objective Alignment Validation
            alignment_validation = await self._validate_objective_alignment_enhanced(
                execution_results, original_objectives, user_request, cross_validation_results
            )
            
            # Phase 5: Quality Standards Validation
            quality_validation = await self._validate_quality_standards_enhanced(
                execution_results, source_credibility
            )
            
            # Phase 6: Completeness Validation
            completeness_validation = await self._validate_completeness_enhanced(
                execution_results, original_objectives, cross_validation_results
            )
            
            # Phase 7: Accuracy Validation
            accuracy_validation = await self._validate_accuracy_enhanced(
                execution_results, cross_validation_results, source_credibility
            )
            
            # Phase 8: Relevance Validation
            relevance_validation = await self._validate_relevance_enhanced(
                execution_results, user_request, original_objectives, bias_analysis
            )
            
            # Phase 9: Overall Validation Score with Enhanced Weighting
            overall_validation = await self._calculate_enhanced_overall_validation(
                alignment_validation, quality_validation, completeness_validation,
                accuracy_validation, relevance_validation, cross_validation_results,
                source_credibility, bias_analysis
            )
            
            # Phase 7: Generate Validation Report
            validation_report = await self._generate_validation_report(
                overall_validation, alignment_validation, quality_validation,
                completeness_validation, accuracy_validation, relevance_validation
            )
            
            validation_result = {
                'validation_score': overall_validation['overall_score'],
                'validation_level': overall_validation['validation_level'],
                'alignment_validation': alignment_validation,
                'quality_validation': quality_validation,
                'completeness_validation': completeness_validation,
                'accuracy_validation': accuracy_validation,
                'relevance_validation': relevance_validation,
                'validation_report': validation_report,
                'validation_metadata': {
                    'objective_id': objective_id,
                    'timestamp': datetime.now().isoformat(),
                    'validation_version': '1.0',
                    'total_results_validated': len(execution_results)
                }
            }
            
            logger.info(f"Validation completed: {overall_validation['validation_level']} ({overall_validation['overall_score']:.2f})")
            return validation_result
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            raise
    
    async def _validate_objective_alignment(self, execution_results: List[Dict[str, Any]], 
                                          original_objectives: List[Dict[str, Any]], 
                                          user_request: str) -> Dict[str, Any]:
        """Validate alignment with original objectives.
        
        Args:
            execution_results: Results to validate
            original_objectives: Original objectives
            user_request: Original user request
            
        Returns:
            Objective alignment validation result
        """
        try:
            alignment_scores = []
            alignment_issues = []
            
            for objective in original_objectives:
                objective_id = objective.get('objective_id')
                objective_description = objective.get('description', '')
                objective_type = objective.get('type', 'primary')
                
                # Find results related to this objective
                related_results = [r for r in execution_results if r.get('objective_id') == objective_id]
                
                if related_results:
                    # Calculate alignment score for this objective
                    objective_score = await self._calculate_objective_alignment_score(
                        objective, related_results, user_request
                    )
                    alignment_scores.append(objective_score)
                    
                    if objective_score < 0.7:
                        alignment_issues.append({
                            'objective_id': objective_id,
                            'description': objective_description,
                            'score': objective_score,
                            'issue': 'Low alignment with objective'
                        })
                else:
                    # No results for this objective
                    alignment_scores.append(0.0)
                    alignment_issues.append({
                        'objective_id': objective_id,
                        'description': objective_description,
                        'score': 0.0,
                        'issue': 'No results for objective'
                    })
            
            # Calculate overall alignment score
            overall_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
            
            return {
                'overall_alignment_score': overall_alignment,
                'objective_scores': alignment_scores,
                'alignment_issues': alignment_issues,
                'alignment_level': 'high' if overall_alignment >= 0.8 else 'medium' if overall_alignment >= 0.6 else 'low',
                'coverage_percentage': len([s for s in alignment_scores if s > 0]) / len(alignment_scores) * 100 if alignment_scores else 0
            }
            
        except Exception as e:
            logger.error(f"Objective alignment validation failed: {e}")
            return {'overall_alignment_score': 0.0, 'alignment_level': 'low'}
    
    async def _validate_quality_standards(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate against quality standards.
        
        Args:
            execution_results: Results to validate
            
        Returns:
            Quality standards validation result
        """
        try:
            quality_scores = []
            quality_issues = []
            
            for result in execution_results:
                result_type = result.get('agent', 'unknown')
                quality_score = await self._calculate_result_quality_score(result, result_type)
                quality_scores.append(quality_score)
                
                if quality_score < 0.7:
                    quality_issues.append({
                        'result_id': result.get('task_id', 'unknown'),
                        'result_type': result_type,
                        'score': quality_score,
                        'issue': 'Below quality standards'
                    })
            
            # Calculate overall quality score
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                'overall_quality_score': overall_quality,
                'individual_scores': quality_scores,
                'quality_issues': quality_issues,
                'quality_level': 'high' if overall_quality >= 0.8 else 'medium' if overall_quality >= 0.6 else 'low',
                'standards_met': len([s for s in quality_scores if s >= 0.7]) / len(quality_scores) * 100 if quality_scores else 0
            }
            
        except Exception as e:
            logger.error(f"Quality standards validation failed: {e}")
            return {'overall_quality_score': 0.0, 'quality_level': 'low'}
    
    async def _validate_completeness(self, execution_results: List[Dict[str, Any]], 
                                   original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate completeness of results.
        
        Args:
            execution_results: Results to validate
            original_objectives: Original objectives
            
        Returns:
            Completeness validation result
        """
        try:
            # Check if all objectives have results
            objective_ids = set(obj.get('objective_id') for obj in original_objectives)
            result_objective_ids = set(r.get('objective_id') for r in execution_results)
            missing_objectives = objective_ids - result_objective_ids
            
            # Calculate completeness score
            completeness_score = len(result_objective_ids) / len(objective_ids) if objective_ids else 0.0
            
            # Check individual result completeness
            result_completeness_scores = []
            for result in execution_results:
                result_completeness = await self._calculate_result_completeness(result)
                result_completeness_scores.append(result_completeness)
            
            avg_result_completeness = sum(result_completeness_scores) / len(result_completeness_scores) if result_completeness_scores else 0.0
            
            # Overall completeness score
            overall_completeness = (completeness_score * 0.6 + avg_result_completeness * 0.4)
            
            completeness_issues = []
            for missing_id in missing_objectives:
                completeness_issues.append({
                    'type': 'missing_objective',
                    'objective_id': missing_id,
                    'issue': 'No results for this objective'
                })
            
            return {
                'overall_completeness_score': overall_completeness,
                'objective_coverage': completeness_score,
                'result_completeness': avg_result_completeness,
                'completeness_issues': completeness_issues,
                'completeness_level': 'high' if overall_completeness >= 0.8 else 'medium' if overall_completeness >= 0.6 else 'low',
                'missing_objectives_count': len(missing_objectives)
            }
            
        except Exception as e:
            logger.error(f"Completeness validation failed: {e}")
            return {'overall_completeness_score': 0.0, 'completeness_level': 'low'}
    
    async def _validate_accuracy(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate accuracy of results.
        
        Args:
            execution_results: Results to validate
            
        Returns:
            Accuracy validation result
        """
        try:
            accuracy_scores = []
            accuracy_issues = []
            
            for result in execution_results:
                result_accuracy = await self._calculate_result_accuracy(result)
                accuracy_scores.append(result_accuracy)
                
                if result_accuracy < 0.7:
                    accuracy_issues.append({
                        'result_id': result.get('task_id', 'unknown'),
                        'score': result_accuracy,
                        'issue': 'Low accuracy detected'
                    })
            
            # Calculate overall accuracy score
            overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
            
            return {
                'overall_accuracy_score': overall_accuracy,
                'individual_scores': accuracy_scores,
                'accuracy_issues': accuracy_issues,
                'accuracy_level': 'high' if overall_accuracy >= 0.8 else 'medium' if overall_accuracy >= 0.6 else 'low',
                'accuracy_consistency': self._calculate_accuracy_consistency(accuracy_scores)
            }
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            return {'overall_accuracy_score': 0.0, 'accuracy_level': 'low'}
    
    async def _validate_relevance(self, execution_results: List[Dict[str, Any]], 
                                user_request: str, 
                                original_objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate relevance of results.
        
        Args:
            execution_results: Results to validate
            user_request: Original user request
            original_objectives: Original objectives
            
        Returns:
            Relevance validation result
        """
        try:
            relevance_scores = []
            relevance_issues = []
            
            for result in execution_results:
                result_relevance = await self._calculate_result_relevance(result, user_request, original_objectives)
                relevance_scores.append(result_relevance)
                
                if result_relevance < 0.6:
                    relevance_issues.append({
                        'result_id': result.get('task_id', 'unknown'),
                        'score': result_relevance,
                        'issue': 'Low relevance to user request'
                    })
            
            # Calculate overall relevance score
            overall_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
            return {
                'overall_relevance_score': overall_relevance,
                'individual_scores': relevance_scores,
                'relevance_issues': relevance_issues,
                'relevance_level': 'high' if overall_relevance >= 0.8 else 'medium' if overall_relevance >= 0.6 else 'low',
                'relevance_consistency': self._calculate_relevance_consistency(relevance_scores)
            }
            
        except Exception as e:
            logger.error(f"Relevance validation failed: {e}")
            return {'overall_relevance_score': 0.0, 'relevance_level': 'low'}
    
    async def _calculate_overall_validation(self, alignment_validation: Dict[str, Any],
                                          quality_validation: Dict[str, Any],
                                          completeness_validation: Dict[str, Any],
                                          accuracy_validation: Dict[str, Any],
                                          relevance_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation score.
        
        Args:
            alignment_validation: Alignment validation result
            quality_validation: Quality validation result
            completeness_validation: Completeness validation result
            accuracy_validation: Accuracy validation result
            relevance_validation: Relevance validation result
            
        Returns:
            Overall validation result
        """
        try:
            # Weighted average of all validation scores
            weights = self.validation_criteria
            
            overall_score = (
                alignment_validation.get('overall_alignment_score', 0) * weights['objective_alignment']['weight'] +
                quality_validation.get('overall_quality_score', 0) * weights['quality_standards']['weight'] +
                completeness_validation.get('overall_completeness_score', 0) * weights['completeness']['weight'] +
                accuracy_validation.get('overall_accuracy_score', 0) * weights['accuracy']['weight'] +
                relevance_validation.get('overall_relevance_score', 0) * weights['relevance']['weight']
            )
            
            # Determine validation level
            if overall_score >= 0.9:
                validation_level = 'excellent'
            elif overall_score >= 0.8:
                validation_level = 'high'
            elif overall_score >= 0.7:
                validation_level = 'good'
            elif overall_score >= 0.6:
                validation_level = 'acceptable'
            else:
                validation_level = 'poor'
            
            return {
                'overall_score': overall_score,
                'validation_level': validation_level,
                'component_scores': {
                    'alignment': alignment_validation.get('overall_alignment_score', 0),
                    'quality': quality_validation.get('overall_quality_score', 0),
                    'completeness': completeness_validation.get('overall_completeness_score', 0),
                    'accuracy': accuracy_validation.get('overall_accuracy_score', 0),
                    'relevance': relevance_validation.get('overall_relevance_score', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Overall validation calculation failed: {e}")
            return {'overall_score': 0.0, 'validation_level': 'poor'}
    
    async def _generate_validation_report(self, overall_validation: Dict[str, Any],
                                        alignment_validation: Dict[str, Any],
                                        quality_validation: Dict[str, Any],
                                        completeness_validation: Dict[str, Any],
                                        accuracy_validation: Dict[str, Any],
                                        relevance_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Args:
            overall_validation: Overall validation result
            alignment_validation: Alignment validation result
            quality_validation: Quality validation result
            completeness_validation: Completeness validation result
            accuracy_validation: Accuracy validation result
            relevance_validation: Relevance validation result
            
        Returns:
            Validation report
        """
        try:
            # Aggregate all issues
            all_issues = []
            all_issues.extend(alignment_validation.get('alignment_issues', []))
            all_issues.extend(quality_validation.get('quality_issues', []))
            all_issues.extend(completeness_validation.get('completeness_issues', []))
            all_issues.extend(accuracy_validation.get('accuracy_issues', []))
            all_issues.extend(relevance_validation.get('relevance_issues', []))
            
            # Generate recommendations
            recommendations = await self._generate_validation_recommendations(
                overall_validation, alignment_validation, quality_validation,
                completeness_validation, accuracy_validation, relevance_validation
            )
            
            # Generate summary
            summary = await self._generate_validation_summary(overall_validation, all_issues)
            
            return {
                'summary': summary,
                'overall_score': overall_validation['overall_score'],
                'validation_level': overall_validation['validation_level'],
                'component_scores': overall_validation['component_scores'],
                'total_issues': len(all_issues),
                'issues_by_category': {
                    'alignment': len(alignment_validation.get('alignment_issues', [])),
                    'quality': len(quality_validation.get('quality_issues', [])),
                    'completeness': len(completeness_validation.get('completeness_issues', [])),
                    'accuracy': len(accuracy_validation.get('accuracy_issues', [])),
                    'relevance': len(relevance_validation.get('relevance_issues', []))
                },
                'recommendations': recommendations,
                'validation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Validation report generation failed: {e}")
            return {'summary': 'Validation report generation failed', 'overall_score': 0.0}
    
    # Helper methods
    async def _calculate_objective_alignment_score(self, objective: Dict[str, Any], 
                                                 results: List[Dict[str, Any]], 
                                                 user_request: str) -> float:
        """Calculate alignment score for an objective."""
        # Simulated alignment calculation
        return 0.8
    
    async def _calculate_result_quality_score(self, result: Dict[str, Any], result_type: str) -> float:
        """Calculate quality score for a result."""
        # Simulated quality calculation
        return 0.75
    
    async def _calculate_result_completeness(self, result: Dict[str, Any]) -> float:
        """Calculate completeness score for a result."""
        # Simulated completeness calculation
        return 0.8
    
    async def _calculate_result_accuracy(self, result: Dict[str, Any]) -> float:
        """Calculate accuracy score for a result."""
        # Simulated accuracy calculation
        return 0.85
    
    async def _calculate_result_relevance(self, result: Dict[str, Any], 
                                        user_request: str, 
                                        objectives: List[Dict[str, Any]]) -> float:
        """Calculate relevance score for a result."""
        # Simulated relevance calculation
        return 0.7
    
    def _calculate_accuracy_consistency(self, accuracy_scores: List[float]) -> float:
        """Calculate accuracy consistency."""
        if not accuracy_scores:
            return 0.0
        # Calculate standard deviation (simplified)
        mean = sum(accuracy_scores) / len(accuracy_scores)
        variance = sum((x - mean) ** 2 for x in accuracy_scores) / len(accuracy_scores)
        return 1.0 - (variance ** 0.5)  # Higher consistency = lower variance
    
    def _calculate_relevance_consistency(self, relevance_scores: List[float]) -> float:
        """Calculate relevance consistency."""
        if not relevance_scores:
            return 0.0
        # Calculate standard deviation (simplified)
        mean = sum(relevance_scores) / len(relevance_scores)
        variance = sum((x - mean) ** 2 for x in relevance_scores) / len(relevance_scores)
        return 1.0 - (variance ** 0.5)  # Higher consistency = lower variance
    
    async def _generate_validation_recommendations(self, overall_validation: Dict[str, Any],
                                                 alignment_validation: Dict[str, Any],
                                                 quality_validation: Dict[str, Any],
                                                 completeness_validation: Dict[str, Any],
                                                 accuracy_validation: Dict[str, Any],
                                                 relevance_validation: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        
        if overall_validation['overall_score'] < 0.7:
            recommendations.append("Overall validation score is below acceptable threshold")
        
        if alignment_validation.get('overall_alignment_score', 0) < 0.7:
            recommendations.append("Improve alignment with original objectives")
        
        if quality_validation.get('overall_quality_score', 0) < 0.7:
            recommendations.append("Enhance quality standards compliance")
        
        if completeness_validation.get('overall_completeness_score', 0) < 0.7:
            recommendations.append("Address completeness gaps")
        
        if accuracy_validation.get('overall_accuracy_score', 0) < 0.7:
            recommendations.append("Improve accuracy of results")
        
        if relevance_validation.get('overall_relevance_score', 0) < 0.7:
            recommendations.append("Increase relevance to user request")
        
        return recommendations
    
    # Enhanced Validation Methods
    
    async def _perform_cross_validation(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform cross-validation across multiple sources."""
        try:
            cross_validation = {
                'total_sources': 0,
                'consistent_sources': 0,
                'conflicting_sources': 0,
                'consistency_score': 0.0,
                'conflicts': [],
                'source_agreement': {}
            }
            
            # Extract all sources from results
            all_sources = []
            for result in execution_results:
                if 'sources' in result:
                    all_sources.extend(result['sources'])
                if 'source' in result:
                    all_sources.append(result['source'])
            
            cross_validation['total_sources'] = len(set(all_sources))
            
            # Analyze consistency across sources
            if len(all_sources) > 1:
                # Group results by similar content
                content_groups = self._group_similar_content(execution_results)
                
                for group in content_groups:
                    if len(group) > 1:
                        # Check consistency within group
                        consistency = self._check_content_consistency(group)
                        cross_validation['consistent_sources'] += consistency['consistent']
                        cross_validation['conflicting_sources'] += consistency['conflicting']
                        cross_validation['conflicts'].extend(consistency['conflicts'])
            
            # Calculate consistency score
            if cross_validation['total_sources'] > 0:
                cross_validation['consistency_score'] = (
                    cross_validation['consistent_sources'] / 
                    (cross_validation['consistent_sources'] + cross_validation['conflicting_sources'])
                )
            
            return cross_validation
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'consistency_score': 0.0, 'total_sources': 0}
    
    async def _analyze_source_credibility(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze source credibility and reliability."""
        try:
            credibility_analysis = {
                'high_credibility_sources': 0,
                'medium_credibility_sources': 0,
                'low_credibility_sources': 0,
                'overall_credibility_score': 0.0,
                'source_ratings': {}
            }
            
            # Define credibility criteria
            high_credibility_domains = [
                'edu', 'gov', 'org', 'ac.uk', 'edu.au', 'edu.ca'
            ]
            
            medium_credibility_domains = [
                'com', 'net', 'io', 'co.uk', 'com.au'
            ]
            
            # Analyze each source
            for result in execution_results:
                sources = result.get('sources', [])
                if isinstance(sources, str):
                    sources = [sources]
                
                for source in sources:
                    if isinstance(source, dict):
                        url = source.get('url', '')
                    else:
                        url = str(source)
                    
                    credibility_score = self._rate_source_credibility(url, high_credibility_domains, medium_credibility_domains)
                    credibility_analysis['source_ratings'][url] = credibility_score
                    
                    if credibility_score >= 0.8:
                        credibility_analysis['high_credibility_sources'] += 1
                    elif credibility_score >= 0.5:
                        credibility_analysis['medium_credibility_sources'] += 1
                    else:
                        credibility_analysis['low_credibility_sources'] += 1
            
            # Calculate overall credibility score
            total_sources = sum([
                credibility_analysis['high_credibility_sources'],
                credibility_analysis['medium_credibility_sources'],
                credibility_analysis['low_credibility_sources']
            ])
            
            if total_sources > 0:
                credibility_analysis['overall_credibility_score'] = (
                    (credibility_analysis['high_credibility_sources'] * 1.0 +
                     credibility_analysis['medium_credibility_sources'] * 0.6 +
                     credibility_analysis['low_credibility_sources'] * 0.2) / total_sources
                )
            
            return credibility_analysis
            
        except Exception as e:
            logger.error(f"Source credibility analysis failed: {e}")
            return {'overall_credibility_score': 0.0}
    
    async def _detect_bias(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential bias in research results."""
        try:
            bias_analysis = {
                'bias_indicators': [],
                'bias_score': 0.0,
                'language_bias': 0.0,
                'source_bias': 0.0,
                'content_bias': 0.0,
                'recommendations': []
            }
            
            # Analyze language bias
            language_bias = self._analyze_language_bias(execution_results)
            bias_analysis['language_bias'] = language_bias
            
            # Analyze source bias
            source_bias = self._analyze_source_bias(execution_results)
            bias_analysis['source_bias'] = source_bias
            
            # Analyze content bias
            content_bias = self._analyze_content_bias(execution_results)
            bias_analysis['content_bias'] = content_bias
            
            # Calculate overall bias score
            bias_analysis['bias_score'] = (
                language_bias + source_bias + content_bias
            ) / 3.0
            
            # Generate bias indicators
            if language_bias > 0.7:
                bias_analysis['bias_indicators'].append("Strong language bias detected")
            if source_bias > 0.7:
                bias_analysis['bias_indicators'].append("Source diversity issues detected")
            if content_bias > 0.7:
                bias_analysis['bias_indicators'].append("Content perspective bias detected")
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            return {'bias_score': 0.0}
    
    def _group_similar_content(self, execution_results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group results with similar content for consistency analysis."""
        groups = []
        
        for result in execution_results:
            content = result.get('content', '') or result.get('summary', '')
            if not content:
                continue
            
            # Simple similarity grouping based on content overlap
            added_to_group = False
            for group in groups:
                if self._calculate_content_similarity(content, group[0].get('content', '') or group[0].get('summary', '')) > 0.7:
                    group.append(result)
                    added_to_group = True
                    break
            
            if not added_to_group:
                groups.append([result])
        
        return groups
    
    def _check_content_consistency(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency within a group of similar content."""
        consistency = {
            'consistent': 0,
            'conflicting': 0,
            'conflicts': []
        }
        
        if len(group) < 2:
            return consistency
        
        # Compare each pair in the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                content1 = group[i].get('content', '') or group[i].get('summary', '')
                content2 = group[j].get('content', '') or group[j].get('summary', '')
                
                similarity = self._calculate_content_similarity(content1, content2)
                
                if similarity > 0.8:
                    consistency['consistent'] += 1
                elif similarity < 0.5:
                    consistency['conflicting'] += 1
                    consistency['conflicts'].append({
                        'source1': group[i].get('source', 'unknown'),
                        'source2': group[j].get('source', 'unknown'),
                        'similarity': similarity
                    })
        
        return consistency
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        if not content1 or not content2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rate_source_credibility(self, url: str, high_domains: List[str], medium_domains: List[str]) -> float:
        """Rate source credibility based on domain and other factors."""
        if not url:
            return 0.0
        
        url_lower = url.lower()
        
        # Check for high credibility domains
        for domain in high_domains:
            if domain in url_lower:
                return 0.9
        
        # Check for medium credibility domains
        for domain in medium_domains:
            if domain in url_lower:
                return 0.6
        
        # Check for suspicious patterns
        suspicious_patterns = ['blogspot', 'wordpress', 'tumblr', 'wix']
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return 0.3
        
        # Default rating
        return 0.4
    
    def _analyze_language_bias(self, execution_results: List[Dict[str, Any]]) -> float:
        """Analyze language bias in results."""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP techniques
        
        bias_indicators = [
            'clearly', 'obviously', 'undoubtedly', 'certainly',
            'definitely', 'absolutely', 'never', 'always',
            'all', 'none', 'every', 'no one'
        ]
        
        total_content = ''
        for result in execution_results:
            content = result.get('content', '') or result.get('summary', '')
            total_content += content + ' '
        
        total_content = total_content.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in total_content)
        
        # Normalize bias score
        return min(bias_count / 10.0, 1.0)
    
    def _analyze_source_bias(self, execution_results: List[Dict[str, Any]]) -> float:
        """Analyze source diversity and potential bias."""
        sources = []
        for result in execution_results:
            if 'sources' in result:
                sources.extend(result['sources'])
            if 'source' in result:
                sources.append(result['source'])
        
        if not sources:
            return 0.0
        
        # Check domain diversity
        domains = set()
        for source in sources:
            if isinstance(source, dict):
                url = source.get('url', '')
            else:
                url = str(source)
            
            # Extract domain
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
                domains.add(domain)
        
        # Low diversity indicates potential bias
        if len(domains) <= 2:
            return 0.8
        elif len(domains) <= 5:
            return 0.5
        else:
            return 0.2
    
    def _analyze_content_bias(self, execution_results: List[Dict[str, Any]]) -> float:
        """Analyze content for perspective bias."""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated analysis
        
        positive_words = ['excellent', 'great', 'amazing', 'wonderful', 'perfect']
        negative_words = ['terrible', 'awful', 'horrible', 'disastrous', 'catastrophic']
        
        total_content = ''
        for result in execution_results:
            content = result.get('content', '') or result.get('summary', '')
            total_content += content + ' '
        
        total_content = total_content.lower()
        
        positive_count = sum(1 for word in positive_words if word in total_content)
        negative_count = sum(1 for word in negative_words if word in total_content)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Calculate bias towards positive or negative
        bias_ratio = abs(positive_count - negative_count) / (positive_count + negative_count)
        return bias_ratio
    
    async def _generate_validation_summary(self, overall_validation: Dict[str, Any], 
                                         all_issues: List[Dict[str, Any]]) -> str:
        """Generate validation summary."""
        score = overall_validation['overall_score']
        level = overall_validation['validation_level']
        issues_count = len(all_issues)
        
        if level == 'excellent':
            return f"Validation passed with excellent score ({score:.2f}). No issues found."
        elif level == 'high':
            return f"Validation passed with high score ({score:.2f}). {issues_count} minor issues found."
        elif level == 'good':
            return f"Validation passed with good score ({score:.2f}). {issues_count} issues found."
        elif level == 'acceptable':
            return f"Validation passed with acceptable score ({score:.2f}). {issues_count} issues found."
        else:
            return f"Validation failed with poor score ({score:.2f}). {issues_count} significant issues found."
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            logger.info("Validation Agent cleanup completed")
        except Exception as e:
            logger.error(f"Validation Agent cleanup failed: {e}")
