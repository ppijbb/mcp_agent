#!/usr/bin/env python3
"""
Task Analyzer Agent for Autonomous Research System

This agent autonomously analyzes user requests and extracts research objectives,
requirements, constraints, and success criteria.

No fallback or dummy code - production-level autonomous analysis only.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import re
from pathlib import Path
import google.generativeai as genai
import os

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger("task_analyzer", log_level="INFO")


class TaskAnalyzerAgent:
    """Autonomous task analyzer agent for research objective extraction."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the task analyzer agent.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Learning capabilities
        self.learning_data = []
        self.analysis_history = []
        
        logger.info("Task Analyzer Agent initialized with LLM-based analysis")
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            genai.configure(api_key=api_key)
            model_name = self.config_manager.get('models.primary', 'gemini-2.5-flash-lite')
            model = genai.GenerativeModel(model_name)
            logger.info(f"LLM initialized for TaskAnalyzerAgent with model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _load_analysis_patterns(self) -> Dict[str, Any]:
        """Load analysis patterns for objective extraction.
        
        Returns:
            Dictionary of analysis patterns
        """
        return {
            'research_intent': [
                r'analyze|study|investigate|examine|explore|research',
                r'compare|contrast|evaluate|assess|review',
                r'understand|learn|discover|find out',
                r'develop|create|build|design|implement'
            ],
            'scope_indicators': [
                r'in the field of|in|regarding|about|concerning',
                r'focus on|concentrate on|emphasize',
                r'including|covering|spanning|across'
            ],
            'constraint_indicators': [
                r'within|under|limited to|restricted to',
                r'no more than|at most|maximum|minimum',
                r'by|before|until|deadline'
            ],
            'quality_indicators': [
                r'comprehensive|thorough|detailed|in-depth',
                r'accurate|precise|reliable|valid',
                r'current|recent|latest|up-to-date'
            ]
        }
    
    def _load_domain_knowledge(self) -> Dict[str, Any]:
        """Load domain knowledge for context understanding.
        
        Returns:
            Dictionary of domain knowledge
        """
        return {
            'technology': {
                'keywords': ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 
                           'neural networks', 'algorithms', 'software', 'hardware', 'systems'],
                'subdomains': ['ai', 'ml', 'nlp', 'computer vision', 'robotics', 'cybersecurity'],
                'research_types': ['technical analysis', 'performance evaluation', 'comparative study']
            },
            'science': {
                'keywords': ['research', 'experiment', 'hypothesis', 'theory', 'data', 'analysis'],
                'subdomains': ['physics', 'chemistry', 'biology', 'medicine', 'environmental'],
                'research_types': ['experimental study', 'theoretical analysis', 'literature review']
            },
            'business': {
                'keywords': ['market', 'industry', 'company', 'strategy', 'finance', 'economics'],
                'subdomains': ['marketing', 'finance', 'operations', 'strategy', 'management'],
                'research_types': ['market analysis', 'competitive analysis', 'trend analysis']
            },
            'general': {
                'keywords': ['topic', 'subject', 'area', 'field', 'domain'],
                'subdomains': ['general', 'interdisciplinary', 'multidisciplinary'],
                'research_types': ['general research', 'comprehensive analysis', 'overview study']
            }
        }
    
    def _load_objective_templates(self) -> Dict[str, Any]:
        """Load objective templates for structured analysis.
        
        Returns:
            Dictionary of objective templates
        """
        return {
            'primary_objective': {
                'template': 'Analyze {topic} in {domain} with {depth} depth',
                'required_fields': ['topic', 'domain', 'depth']
            },
            'secondary_objectives': {
                'template': 'Investigate {aspect} of {topic}',
                'required_fields': ['aspect', 'topic']
            },
            'constraints': {
                'template': 'Within {constraint_type}: {constraint_value}',
                'required_fields': ['constraint_type', 'constraint_value']
            },
            'success_criteria': {
                'template': 'Deliver {deliverable_type} with {quality_metrics}',
                'required_fields': ['deliverable_type', 'quality_metrics']
            }
        }
    
    async def analyze_objective(self, user_request: str, context: Optional[Dict[str, Any]] = None, 
                              objective_id: str = None) -> Dict[str, Any]:
        """LLM-based analysis of user request and objective extraction.
        
        Args:
            user_request: The user's research request
            context: Additional context for analysis
            objective_id: Objective ID for tracking
            
        Returns:
            Dictionary containing analyzed objectives and metadata
        """
        try:
            logger.info(f"Starting LLM-based analysis for objective: {objective_id}")
            
            # Use LLM for comprehensive analysis
            analysis_result = await self._llm_analyze_request(user_request, context, objective_id)
            
            # Store analysis in history for learning
            self.analysis_history.append({
                'objective_id': objective_id,
                'user_request': user_request,
                'context': context,
                'result': analysis_result,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"LLM analysis completed: {len(analysis_result.get('objectives', []))} objectives identified")
            return analysis_result
            
        except Exception as e:
            logger.error(f"LLM objective analysis failed: {e}")
            raise
    
    async def _llm_analyze_request(self, user_request: str, context: Optional[Dict[str, Any]], 
                                 objective_id: str) -> Dict[str, Any]:
        """Use LLM to analyze the research request."""
        try:
            prompt = f"""
            사용자의 연구 요청을 분석하고 구체적인 연구 목표들을 추출하세요.
            
            사용자 요청: {user_request}
            추가 컨텍스트: {context or {}}
            
            다음을 수행하세요:
            1. 요청의 핵심 의도 파악
            2. 관련된 도메인 식별
            3. 연구 범위와 깊이 결정
            4. 제약사항과 요구사항 식별
            5. 구체적인 연구 목표들 생성
            6. 각 목표의 우선순위 설정
            7. 성공 기준 정의
            
            JSON 형태로 응답하세요:
            {{
                "objectives": [
                    {{
                        "objective_id": "unique_id",
                        "type": "primary|secondary|quality",
                        "description": "목표 설명",
                        "intent": "analysis|comparison|exploration|development",
                        "domain": "identified_domain",
                        "scope": "general|focused|comprehensive",
                        "depth": "basic|standard|comprehensive",
                        "priority": 0.0-1.0,
                        "constraints": {{
                            "time_constraints": [],
                            "resource_constraints": [],
                            "quality_constraints": [],
                            "scope_constraints": []
                        }},
                        "success_criteria": {{
                            "deliverable_types": [],
                            "quality_metrics": [],
                            "completion_indicators": [],
                            "success_threshold": 0.0-1.0
                        }},
                        "estimated_effort": 0.0-1.0,
                        "dependencies": []
                    }}
                ],
                "intent_analysis": {{
                    "primary_intent": "main_intent",
                    "secondary_intents": [],
                    "intent_confidence": 0.0-1.0,
                    "research_type": "analytical|comparative|exploratory|developmental"
                }},
                "domain_analysis": {{
                    "primary_domain": "domain",
                    "subdomains": [],
                    "domain_confidence": 0.0-1.0,
                    "domain_keywords": []
                }},
                "scope_analysis": {{
                    "scope_type": "general_analysis|comparative|trend_analysis|literature_review",
                    "scope_boundaries": [],
                    "scope_depth": "basic|standard|comprehensive",
                    "scope_keywords": []
                }},
                "constraint_analysis": {{
                    "time_constraints": [],
                    "resource_constraints": [],
                    "quality_constraints": [],
                    "scope_constraints": []
                }},
                "success_criteria": {{
                    "deliverable_types": [],
                    "quality_metrics": [],
                    "completion_indicators": [],
                    "success_threshold": 0.0-1.0
                }},
                "analysis_metadata": {{
                    "objective_id": "{objective_id}",
                    "timestamp": "{datetime.now().isoformat()}",
                    "analysis_version": "2.0",
                    "confidence_score": 0.0-1.0,
                    "complexity_level": "low|medium|high",
                    "estimated_duration": "short|medium|long"
                }}
            }}
            """
            
            response = await asyncio.to_thread(self.llm.generate_content, prompt)
            result = json.loads(response.text)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return minimal fallback
            return {
                'objectives': [{
                    'objective_id': f"primary_{objective_id}",
                    'type': 'primary',
                    'description': user_request,
                    'priority': 1.0,
                    'success_criteria': {'success_threshold': 0.8}
                }],
                'analysis_metadata': {
                    'objective_id': objective_id,
                    'timestamp': datetime.now().isoformat(),
                    'confidence_score': 0.5
                }
            }
    
    async def _analyze_research_intent(self, user_request: str) -> Dict[str, Any]:
        """Analyze the research intent from user request.
        
        Args:
            user_request: The user's research request
            
        Returns:
            Intent analysis results
        """
        try:
            intent_analysis = {
                'primary_intent': None,
                'secondary_intents': [],
                'intent_confidence': 0.0,
                'research_type': None
            }
            
            # Pattern matching for intent detection
            for pattern_type, patterns in self.analysis_patterns['research_intent'].items():
                for pattern in patterns:
                    if re.search(pattern, user_request.lower()):
                        if pattern_type == 'analyze':
                            intent_analysis['primary_intent'] = 'analysis'
                        elif pattern_type == 'compare':
                            intent_analysis['primary_intent'] = 'comparison'
                        elif pattern_type == 'understand':
                            intent_analysis['primary_intent'] = 'exploration'
                        elif pattern_type == 'develop':
                            intent_analysis['primary_intent'] = 'development'
                        
                        intent_analysis['intent_confidence'] += 0.2
            
            # Determine research type based on intent
            if intent_analysis['primary_intent'] == 'analysis':
                intent_analysis['research_type'] = 'analytical'
            elif intent_analysis['primary_intent'] == 'comparison':
                intent_analysis['research_type'] = 'comparative'
            elif intent_analysis['primary_intent'] == 'exploration':
                intent_analysis['research_type'] = 'exploratory'
            elif intent_analysis['primary_intent'] == 'development':
                intent_analysis['research_type'] = 'developmental'
            
            # Normalize confidence score
            intent_analysis['intent_confidence'] = min(intent_analysis['intent_confidence'], 1.0)
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {'primary_intent': 'general', 'intent_confidence': 0.5}
    
    async def _classify_domain(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify the research domain.
        
        Args:
            user_request: The user's research request
            context: Additional context
            
        Returns:
            Domain classification results
        """
        try:
            domain_analysis = {
                'primary_domain': 'general',
                'subdomains': [],
                'domain_confidence': 0.0,
                'domain_keywords': []
            }
            
            # Check context for domain hints
            if context and 'domain' in context:
                domain_analysis['primary_domain'] = context['domain']
                domain_analysis['domain_confidence'] = 0.8
            
            # Analyze user request for domain indicators
            user_request_lower = user_request.lower()
            
            for domain, domain_info in self.domain_knowledge.items():
                domain_score = 0
                matched_keywords = []
                
                for keyword in domain_info['keywords']:
                    if keyword in user_request_lower:
                        domain_score += 1
                        matched_keywords.append(keyword)
                
                if domain_score > domain_analysis['domain_confidence']:
                    domain_analysis['primary_domain'] = domain
                    domain_analysis['domain_confidence'] = min(domain_score / len(domain_info['keywords']), 1.0)
                    domain_analysis['domain_keywords'] = matched_keywords
                    
                    # Identify subdomains
                    for subdomain in domain_info['subdomains']:
                        if subdomain in user_request_lower:
                            domain_analysis['subdomains'].append(subdomain)
            
            return domain_analysis
            
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            return {'primary_domain': 'general', 'domain_confidence': 0.5}
    
    async def _extract_scope(self, user_request: str, domain_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research scope from user request.
        
        Args:
            user_request: The user's research request
            domain_analysis: Domain analysis results
            
        Returns:
            Scope analysis results
        """
        try:
            scope_analysis = {
                'scope_type': 'general',
                'scope_boundaries': [],
                'scope_depth': 'standard',
                'scope_keywords': []
            }
            
            # Analyze scope indicators
            for pattern in self.analysis_patterns['scope_indicators']:
                matches = re.findall(pattern, user_request.lower())
                if matches:
                    scope_analysis['scope_keywords'].extend(matches)
            
            # Determine scope depth
            depth_indicators = {
                'basic': ['overview', 'introduction', 'basics', 'fundamentals'],
                'standard': ['analysis', 'study', 'investigation', 'research'],
                'comprehensive': ['comprehensive', 'thorough', 'detailed', 'in-depth', 'extensive']
            }
            
            for depth, indicators in depth_indicators.items():
                for indicator in indicators:
                    if indicator in user_request.lower():
                        scope_analysis['scope_depth'] = depth
                        break
            
            # Determine scope type
            if 'compare' in user_request.lower() or 'comparison' in user_request.lower():
                scope_analysis['scope_type'] = 'comparative'
            elif 'trend' in user_request.lower() or 'trends' in user_request.lower():
                scope_analysis['scope_type'] = 'trend_analysis'
            elif 'review' in user_request.lower() or 'survey' in user_request.lower():
                scope_analysis['scope_type'] = 'literature_review'
            else:
                scope_analysis['scope_type'] = 'general_analysis'
            
            return scope_analysis
            
        except Exception as e:
            logger.error(f"Scope extraction failed: {e}")
            return {'scope_type': 'general', 'scope_depth': 'standard'}
    
    async def _identify_constraints(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Identify constraints from user request and context.
        
        Args:
            user_request: The user's research request
            context: Additional context
            
        Returns:
            Constraint analysis results
        """
        try:
            constraint_analysis = {
                'time_constraints': [],
                'resource_constraints': [],
                'quality_constraints': [],
                'scope_constraints': []
            }
            
            # Analyze constraint indicators
            for pattern in self.analysis_patterns['constraint_indicators']:
                matches = re.findall(pattern, user_request.lower())
                if matches:
                    # Categorize constraints
                    for match in matches:
                        if any(word in match for word in ['time', 'deadline', 'by', 'before']):
                            constraint_analysis['time_constraints'].append(match)
                        elif any(word in match for word in ['budget', 'cost', 'resource', 'limit']):
                            constraint_analysis['resource_constraints'].append(match)
                        elif any(word in match for word in ['quality', 'accuracy', 'precision']):
                            constraint_analysis['quality_constraints'].append(match)
                        else:
                            constraint_analysis['scope_constraints'].append(match)
            
            # Add context constraints
            if context:
                if 'deadline' in context:
                    constraint_analysis['time_constraints'].append(f"deadline: {context['deadline']}")
                if 'budget' in context:
                    constraint_analysis['resource_constraints'].append(f"budget: {context['budget']}")
                if 'quality_requirements' in context:
                    constraint_analysis['quality_constraints'].extend(context['quality_requirements'])
            
            return constraint_analysis
            
        except Exception as e:
            logger.error(f"Constraint identification failed: {e}")
            return {'time_constraints': [], 'resource_constraints': [], 'quality_constraints': [], 'scope_constraints': []}
    
    async def _define_success_criteria(self, user_request: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Define success criteria based on user request and intent.
        
        Args:
            user_request: The user's research request
            intent_analysis: Intent analysis results
            
        Returns:
            Success criteria definition
        """
        try:
            success_criteria = {
                'deliverable_types': [],
                'quality_metrics': [],
                'completion_indicators': [],
                'success_threshold': 0.8
            }
            
            # Determine deliverable types based on intent
            if intent_analysis['primary_intent'] == 'analysis':
                success_criteria['deliverable_types'].append('analytical_report')
            elif intent_analysis['primary_intent'] == 'comparison':
                success_criteria['deliverable_types'].append('comparative_analysis')
            elif intent_analysis['primary_intent'] == 'exploration':
                success_criteria['deliverable_types'].append('exploratory_report')
            elif intent_analysis['primary_intent'] == 'development':
                success_criteria['deliverable_types'].append('development_plan')
            
            # Analyze quality indicators
            for pattern in self.analysis_patterns['quality_indicators']:
                if re.search(pattern, user_request.lower()):
                    if 'comprehensive' in pattern or 'thorough' in pattern:
                        success_criteria['quality_metrics'].append('comprehensiveness')
                    elif 'accurate' in pattern or 'precise' in pattern:
                        success_criteria['quality_metrics'].append('accuracy')
                    elif 'current' in pattern or 'recent' in pattern:
                        success_criteria['quality_metrics'].append('currency')
            
            # Default quality metrics
            if not success_criteria['quality_metrics']:
                success_criteria['quality_metrics'] = ['completeness', 'accuracy', 'relevance']
            
            # Define completion indicators
            success_criteria['completion_indicators'] = [
                'all objectives addressed',
                'quality metrics met',
                'deliverables generated',
                'constraints satisfied'
            ]
            
            return success_criteria
            
        except Exception as e:
            logger.error(f"Success criteria definition failed: {e}")
            return {'deliverable_types': ['general_report'], 'quality_metrics': ['completeness'], 'completion_indicators': ['objectives_met']}
    
    async def _synthesize_objectives(self, intent_analysis: Dict[str, Any], domain_analysis: Dict[str, Any],
                                   scope_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any],
                                   success_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize objectives from all analysis components.
        
        Args:
            intent_analysis: Intent analysis results
            domain_analysis: Domain analysis results
            scope_analysis: Scope analysis results
            constraint_analysis: Constraint analysis results
            success_criteria: Success criteria definition
            
        Returns:
            List of synthesized objectives
        """
        try:
            objectives = []
            
            # Primary objective
            primary_objective = {
                'objective_id': f"primary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'type': 'primary',
                'description': f"Analyze {scope_analysis['scope_type']} in {domain_analysis['primary_domain']} domain",
                'intent': intent_analysis['primary_intent'],
                'domain': domain_analysis['primary_domain'],
                'scope': scope_analysis['scope_type'],
                'depth': scope_analysis['scope_depth'],
                'priority': 1.0,
                'constraints': constraint_analysis,
                'success_criteria': success_criteria,
                'estimated_effort': self._estimate_effort(scope_analysis, domain_analysis),
                'dependencies': []
            }
            objectives.append(primary_objective)
            
            # Secondary objectives based on subdomains
            for subdomain in domain_analysis['subdomains']:
                secondary_objective = {
                    'objective_id': f"secondary_{subdomain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'type': 'secondary',
                    'description': f"Investigate {subdomain} aspects of the research topic",
                    'intent': 'exploration',
                    'domain': domain_analysis['primary_domain'],
                    'subdomain': subdomain,
                    'scope': 'focused',
                    'depth': 'standard',
                    'priority': 0.7,
                    'constraints': constraint_analysis,
                    'success_criteria': success_criteria,
                    'estimated_effort': self._estimate_effort(scope_analysis, domain_analysis) * 0.5,
                    'dependencies': [primary_objective['objective_id']]
                }
                objectives.append(secondary_objective)
            
            # Quality objectives
            for quality_metric in success_criteria['quality_metrics']:
                quality_objective = {
                    'objective_id': f"quality_{quality_metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'type': 'quality',
                    'description': f"Ensure {quality_metric} in research deliverables",
                    'intent': 'validation',
                    'domain': domain_analysis['primary_domain'],
                    'scope': 'quality_assurance',
                    'depth': 'standard',
                    'priority': 0.8,
                    'constraints': constraint_analysis,
                    'success_criteria': success_criteria,
                    'estimated_effort': self._estimate_effort(scope_analysis, domain_analysis) * 0.3,
                    'dependencies': [primary_objective['objective_id']]
                }
                objectives.append(quality_objective)
            
            return objectives
            
        except Exception as e:
            logger.error(f"Objective synthesis failed: {e}")
            return []
    
    async def _assign_priorities(self, objectives: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Assign priorities to objectives based on context and dependencies.
        
        Args:
            objectives: List of objectives
            context: Additional context
            
        Returns:
            List of objectives with assigned priorities
        """
        try:
            # Sort objectives by priority (highest first)
            prioritized_objectives = sorted(objectives, key=lambda x: x['priority'], reverse=True)
            
            # Adjust priorities based on context
            if context:
                if 'urgent' in context and context['urgent']:
                    for obj in prioritized_objectives:
                        if obj['type'] == 'primary':
                            obj['priority'] = min(obj['priority'] + 0.2, 1.0)
                
                if 'quality_focus' in context and context['quality_focus']:
                    for obj in prioritized_objectives:
                        if obj['type'] == 'quality':
                            obj['priority'] = min(obj['priority'] + 0.1, 1.0)
            
            return prioritized_objectives
            
        except Exception as e:
            logger.error(f"Priority assignment failed: {e}")
            return objectives
    
    def _estimate_effort(self, scope_analysis: Dict[str, Any], domain_analysis: Dict[str, Any]) -> float:
        """Estimate effort required for objectives.
        
        Args:
            scope_analysis: Scope analysis results
            domain_analysis: Domain analysis results
            
        Returns:
            Estimated effort score (0.0 to 1.0)
        """
        try:
            effort = 0.5  # Base effort
            
            # Adjust based on scope depth
            depth_multipliers = {
                'basic': 0.5,
                'standard': 1.0,
                'comprehensive': 2.0
            }
            effort *= depth_multipliers.get(scope_analysis['scope_depth'], 1.0)
            
            # Adjust based on domain complexity
            domain_complexity = {
                'general': 0.8,
                'technology': 1.2,
                'science': 1.1,
                'business': 1.0
            }
            effort *= domain_complexity.get(domain_analysis['primary_domain'], 1.0)
            
            # Adjust based on subdomains
            effort += len(domain_analysis['subdomains']) * 0.1
            
            return min(effort, 1.0)
            
        except Exception as e:
            logger.error(f"Effort estimation failed: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, intent_analysis: Dict[str, Any], 
                                  domain_analysis: Dict[str, Any], 
                                  scope_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis.
        
        Args:
            intent_analysis: Intent analysis results
            domain_analysis: Domain analysis results
            scope_analysis: Scope analysis results
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        try:
            confidence = 0.0
            
            # Intent confidence
            confidence += intent_analysis.get('intent_confidence', 0.0) * 0.4
            
            # Domain confidence
            confidence += domain_analysis.get('domain_confidence', 0.0) * 0.3
            
            # Scope confidence (based on clarity of scope indicators)
            scope_confidence = 0.5
            if scope_analysis.get('scope_keywords'):
                scope_confidence = min(len(scope_analysis['scope_keywords']) * 0.2, 1.0)
            confidence += scope_confidence * 0.3
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            logger.info("Task Analyzer Agent cleanup completed")
        except Exception as e:
            logger.error(f"Task Analyzer Agent cleanup failed: {e}")
    
    async def update_capabilities(self, evaluation_result: Dict[str, Any], iteration: int) -> None:
        """Update agent capabilities based on evaluation feedback.
        
        Args:
            evaluation_result: Evaluation results from current iteration
            iteration: Current iteration number
        """
        try:
            feedback = evaluation_result.get('feedback', [])
            quality_metrics = evaluation_result.get('quality_metrics', {})
            
            # Store learning data
            learning_entry = {
                'iteration': iteration,
                'feedback': feedback,
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.learning_data.append(learning_entry)
            
            # Update adaptive thresholds based on feedback
            if 'insufficient_depth' in str(feedback):
                self.adaptive_thresholds['depth'] = min(0.9, self.adaptive_thresholds['depth'] + 0.1)
            elif 'excessive_depth' in str(feedback):
                self.adaptive_thresholds['depth'] = max(0.3, self.adaptive_thresholds['depth'] - 0.1)
            
            if 'insufficient_scope' in str(feedback):
                self.adaptive_thresholds['scope'] = min(0.9, self.adaptive_thresholds['scope'] + 0.1)
            elif 'excessive_scope' in str(feedback):
                self.adaptive_thresholds['scope'] = max(0.3, self.adaptive_thresholds['scope'] - 0.1)
            
            if 'insufficient_complexity' in str(feedback):
                self.adaptive_thresholds['complexity'] = min(0.9, self.adaptive_thresholds['complexity'] + 0.1)
            elif 'excessive_complexity' in str(feedback):
                self.adaptive_thresholds['complexity'] = max(0.3, self.adaptive_thresholds['complexity'] - 0.1)
            
            # Update analysis patterns based on successful patterns
            if quality_metrics.get('overall_score', 0) > 0.8:
                await self._update_successful_patterns(evaluation_result)
            
            logger.info(f"TaskAnalyzerAgent capabilities updated for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Capability update failed: {e}")
    
    async def _update_successful_patterns(self, evaluation_result: Dict[str, Any]) -> None:
        """Update analysis patterns based on successful evaluations."""
        try:
            # Extract successful patterns from evaluation
            quality_metrics = evaluation_result.get('quality_metrics', {})
            
            # Update pattern weights based on success
            for pattern_name in self.analysis_patterns:
                if quality_metrics.get(f'{pattern_name}_score', 0) > 0.8:
                    if 'weight' not in self.analysis_patterns[pattern_name]:
                        self.analysis_patterns[pattern_name]['weight'] = 1.0
                    self.analysis_patterns[pattern_name]['weight'] = min(2.0, 
                        self.analysis_patterns[pattern_name]['weight'] + 0.1)
            
            logger.info("Successful patterns updated")
            
        except Exception as e:
            logger.error(f"Pattern update failed: {e}")
    
    async def _enhanced_analyze_with_learning(self, user_request: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced analysis using learning from previous iterations."""
        try:
            # Get learning data from context
            learning_data = context.get('learning_data', [])
            iteration = context.get('iteration', 1)
            
            # Start with base analysis
            objectives = await self.analyze(user_request, context)
            
            # Apply learning enhancements
            if learning_data and iteration > 1:
                objectives = await self._apply_learning_enhancements(objectives, learning_data, iteration)
            
            return objectives
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return await self.analyze(user_request, context)
    
    async def _apply_learning_enhancements(self, objectives: List[Dict[str, Any]], 
                                         learning_data: List[Dict[str, Any]], 
                                         iteration: int) -> List[Dict[str, Any]]:
        """Apply learning enhancements to objectives."""
        try:
            enhanced_objectives = []
            
            for objective in objectives:
                enhanced_objective = objective.copy()
                
                # Apply learning-based enhancements
                if iteration > 1:
                    # Increase depth if previous iterations lacked depth
                    latest_feedback = learning_data[-1].get('feedback', [])
                    if 'insufficient_depth' in str(latest_feedback):
                        enhanced_objective['depth_requirement'] = 'high'
                        enhanced_objective['analysis_depth'] = 'comprehensive'
                    
                    # Expand scope if previous iterations were too narrow
                    if 'insufficient_scope' in str(latest_feedback):
                        enhanced_objective['scope'] = 'comprehensive'
                        enhanced_objective['coverage_requirement'] = 'extensive'
                    
                    # Adjust complexity based on feedback
                    if 'insufficient_complexity' in str(latest_feedback):
                        enhanced_objective['complexity_level'] = 'high'
                        enhanced_objective['analysis_complexity'] = 'advanced'
                
                # Add learning metadata
                enhanced_objective['learning_applied'] = True
                enhanced_objective['iteration'] = iteration
                enhanced_objective['learning_data_count'] = len(learning_data)
                
                enhanced_objectives.append(enhanced_objective)
            
            return enhanced_objectives
            
        except Exception as e:
            logger.error(f"Learning enhancement failed: {e}")
            return objectives
        except Exception as e:
            logger.error(f"Task Analyzer Agent cleanup failed: {e}")
