"""
Benchmark Evaluation System for Local Researcher
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    RACE_SCORE = "race_score"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    SOURCE_CREDIBILITY = "source_credibility"
    BIAS_DETECTION = "bias_detection"
    CITATION_QUALITY = "citation_quality"


@dataclass
class EvaluationResult:
    """Evaluation result data structure."""
    metric: EvaluationMetric
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: str


class BenchmarkEvaluator:
    """Advanced benchmark evaluation system for research results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the benchmark evaluator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Evaluation weights
        self.weights = {
            EvaluationMetric.RACE_SCORE: 0.25,
            EvaluationMetric.COMPLETENESS: 0.20,
            EvaluationMetric.ACCURACY: 0.20,
            EvaluationMetric.RELEVANCE: 0.15,
            EvaluationMetric.COHERENCE: 0.10,
            EvaluationMetric.SOURCE_CREDIBILITY: 0.05,
            EvaluationMetric.BIAS_DETECTION: 0.03,
            EvaluationMetric.CITATION_QUALITY: 0.02
        }
    
    async def evaluate_research(self, research_result: Dict[str, Any], 
                              original_query: str) -> Dict[str, Any]:
        """Comprehensive evaluation of research results."""
        try:
            self.logger.info(f"Starting comprehensive evaluation for query: {original_query}")
            
            evaluation_results = {}
            overall_score = 0.0
            
            # Evaluate each metric
            for metric in EvaluationMetric:
                try:
                    result = await self._evaluate_metric(metric, research_result, original_query)
                    evaluation_results[metric.value] = result
                    overall_score += result.score * self.weights[metric]
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {metric.value}: {e}")
                    evaluation_results[metric.value] = EvaluationResult(
                        metric=metric,
                        score=0.0,
                        max_score=1.0,
                        details={'error': str(e)},
                        timestamp=datetime.now().isoformat()
                    )
            
            # Calculate overall score
            overall_score = min(overall_score, 1.0)  # Cap at 1.0
            
            # Generate evaluation summary
            summary = await self._generate_evaluation_summary(evaluation_results, overall_score)
            
            return {
                'success': True,
                'overall_score': overall_score,
                'evaluation_results': evaluation_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'query': original_query
            }
            
        except Exception as e:
            self.logger.error(f"Research evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _evaluate_metric(self, metric: EvaluationMetric, 
                             research_result: Dict[str, Any], 
                             original_query: str) -> EvaluationResult:
        """Evaluate a specific metric."""
        if metric == EvaluationMetric.RACE_SCORE:
            return await self._calculate_race_score(research_result, original_query)
        elif metric == EvaluationMetric.COMPLETENESS:
            return await self._evaluate_completeness(research_result, original_query)
        elif metric == EvaluationMetric.ACCURACY:
            return await self._evaluate_accuracy(research_result, original_query)
        elif metric == EvaluationMetric.RELEVANCE:
            return await self._evaluate_relevance(research_result, original_query)
        elif metric == EvaluationMetric.COHERENCE:
            return await self._evaluate_coherence(research_result, original_query)
        elif metric == EvaluationMetric.SOURCE_CREDIBILITY:
            return await self._evaluate_source_credibility(research_result, original_query)
        elif metric == EvaluationMetric.BIAS_DETECTION:
            return await self._detect_bias(research_result, original_query)
        elif metric == EvaluationMetric.CITATION_QUALITY:
            return await self._evaluate_citation_quality(research_result, original_query)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    async def _calculate_race_score(self, research_result: Dict[str, Any], 
                                  original_query: str) -> EvaluationResult:
        """Calculate RACE (Research Accuracy, Completeness, and Effectiveness) score."""
        try:
            # Extract key components
            content = research_result.get('content', '')
            sources = research_result.get('sources', [])
            methodology = research_result.get('methodology', '')
            
            # RACE components
            accuracy_score = await self._calculate_accuracy_component(content, sources)
            completeness_score = await self._calculate_completeness_component(content, original_query)
            effectiveness_score = await self._calculate_effectiveness_component(content, methodology)
            
            # Weighted RACE score
            race_score = (accuracy_score * 0.4 + completeness_score * 0.3 + effectiveness_score * 0.3)
            
            return EvaluationResult(
                metric=EvaluationMetric.RACE_SCORE,
                score=race_score,
                max_score=1.0,
                details={
                    'accuracy_component': accuracy_score,
                    'completeness_component': completeness_score,
                    'effectiveness_component': effectiveness_score,
                    'race_score': race_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"RACE score calculation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.RACE_SCORE,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _calculate_accuracy_component(self, content: str, sources: List[Dict]) -> float:
        """Calculate accuracy component of RACE score."""
        try:
            # Check for factual claims and source backing
            factual_claims = self._extract_factual_claims(content)
            source_backed_claims = 0
            
            for claim in factual_claims:
                if self._is_claim_source_backed(claim, sources):
                    source_backed_claims += 1
            
            if not factual_claims:
                return 0.5  # Neutral score if no factual claims
            
            accuracy = source_backed_claims / len(factual_claims)
            return min(accuracy, 1.0)
            
        except Exception as e:
            self.logger.error(f"Accuracy component calculation failed: {e}")
            return 0.0
    
    async def _calculate_completeness_component(self, content: str, query: str) -> float:
        """Calculate completeness component of RACE score."""
        try:
            # Extract query keywords
            query_keywords = self._extract_keywords(query)
            content_keywords = self._extract_keywords(content)
            
            # Calculate keyword coverage
            covered_keywords = set(query_keywords) & set(content_keywords)
            completeness = len(covered_keywords) / len(query_keywords) if query_keywords else 0.5
            
            # Check for comprehensive coverage (length and structure)
            content_length_score = min(len(content) / 1000, 1.0)  # Normalize to 1000 chars
            structure_score = self._evaluate_content_structure(content)
            
            # Combined completeness score
            final_score = (completeness * 0.6 + content_length_score * 0.2 + structure_score * 0.2)
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Completeness component calculation failed: {e}")
            return 0.0
    
    async def _calculate_effectiveness_component(self, content: str, methodology: str) -> float:
        """Calculate effectiveness component of RACE score."""
        try:
            # Check for clear methodology
            methodology_score = 0.5
            if methodology and len(methodology) > 50:
                methodology_score = 0.8
            
            # Check for clear conclusions
            conclusion_indicators = ['conclusion', 'summary', 'findings', 'results']
            has_conclusion = any(indicator in content.lower() for indicator in conclusion_indicators)
            conclusion_score = 0.8 if has_conclusion else 0.3
            
            # Check for actionable insights
            action_indicators = ['recommend', 'suggest', 'imply', 'indicate', 'show']
            has_actions = any(indicator in content.lower() for indicator in action_indicators)
            action_score = 0.7 if has_actions else 0.4
            
            effectiveness = (methodology_score * 0.4 + conclusion_score * 0.3 + action_score * 0.3)
            return min(effectiveness, 1.0)
            
        except Exception as e:
            self.logger.error(f"Effectiveness component calculation failed: {e}")
            return 0.0
    
    async def _evaluate_completeness(self, research_result: Dict[str, Any], 
                                   original_query: str) -> EvaluationResult:
        """Evaluate completeness of research results."""
        try:
            content = research_result.get('content', '')
            sources = research_result.get('sources', [])
            
            # Check query coverage
            query_coverage = self._calculate_query_coverage(content, original_query)
            
            # Check source diversity
            source_diversity = self._calculate_source_diversity(sources)
            
            # Check content depth
            content_depth = self._calculate_content_depth(content)
            
            # Combined completeness score
            completeness_score = (query_coverage * 0.4 + source_diversity * 0.3 + content_depth * 0.3)
            
            return EvaluationResult(
                metric=EvaluationMetric.COMPLETENESS,
                score=completeness_score,
                max_score=1.0,
                details={
                    'query_coverage': query_coverage,
                    'source_diversity': source_diversity,
                    'content_depth': content_depth,
                    'completeness_score': completeness_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Completeness evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.COMPLETENESS,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _evaluate_accuracy(self, research_result: Dict[str, Any], 
                               original_query: str) -> EvaluationResult:
        """Evaluate accuracy of research results."""
        try:
            content = research_result.get('content', '')
            sources = research_result.get('sources', [])
            
            # Check for factual consistency
            factual_consistency = self._check_factual_consistency(content)
            
            # Check source reliability
            source_reliability = self._check_source_reliability(sources)
            
            # Check for contradictions
            contradiction_score = self._check_contradictions(content)
            
            # Combined accuracy score
            accuracy_score = (factual_consistency * 0.4 + source_reliability * 0.4 + contradiction_score * 0.2)
            
            return EvaluationResult(
                metric=EvaluationMetric.ACCURACY,
                score=accuracy_score,
                max_score=1.0,
                details={
                    'factual_consistency': factual_consistency,
                    'source_reliability': source_reliability,
                    'contradiction_score': contradiction_score,
                    'accuracy_score': accuracy_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Accuracy evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.ACCURACY,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _evaluate_relevance(self, research_result: Dict[str, Any], 
                                original_query: str) -> EvaluationResult:
        """Evaluate relevance of research results."""
        try:
            content = research_result.get('content', '')
            
            # Calculate semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(content, original_query)
            
            # Check topic alignment
            topic_alignment = self._check_topic_alignment(content, original_query)
            
            # Check query-specific content
            query_specificity = self._check_query_specificity(content, original_query)
            
            # Combined relevance score
            relevance_score = (semantic_similarity * 0.5 + topic_alignment * 0.3 + query_specificity * 0.2)
            
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE,
                score=relevance_score,
                max_score=1.0,
                details={
                    'semantic_similarity': semantic_similarity,
                    'topic_alignment': topic_alignment,
                    'query_specificity': query_specificity,
                    'relevance_score': relevance_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Relevance evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _evaluate_coherence(self, research_result: Dict[str, Any], 
                                original_query: str) -> EvaluationResult:
        """Evaluate coherence of research results."""
        try:
            content = research_result.get('content', '')
            
            # Check logical flow
            logical_flow = self._check_logical_flow(content)
            
            # Check paragraph structure
            paragraph_structure = self._check_paragraph_structure(content)
            
            # Check transition quality
            transition_quality = self._check_transition_quality(content)
            
            # Combined coherence score
            coherence_score = (logical_flow * 0.4 + paragraph_structure * 0.3 + transition_quality * 0.3)
            
            return EvaluationResult(
                metric=EvaluationMetric.COHERENCE,
                score=coherence_score,
                max_score=1.0,
                details={
                    'logical_flow': logical_flow,
                    'paragraph_structure': paragraph_structure,
                    'transition_quality': transition_quality,
                    'coherence_score': coherence_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Coherence evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.COHERENCE,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _evaluate_source_credibility(self, research_result: Dict[str, Any], 
                                         original_query: str) -> EvaluationResult:
        """Evaluate source credibility."""
        try:
            sources = research_result.get('sources', [])
            
            if not sources:
                return EvaluationResult(
                    metric=EvaluationMetric.SOURCE_CREDIBILITY,
                    score=0.0,
                    max_score=1.0,
                    details={'error': 'No sources found'},
                    timestamp=datetime.now().isoformat()
                )
            
            credibility_scores = []
            for source in sources:
                score = self._calculate_source_credibility_score(source)
                credibility_scores.append(score)
            
            avg_credibility = sum(credibility_scores) / len(credibility_scores)
            
            return EvaluationResult(
                metric=EvaluationMetric.SOURCE_CREDIBILITY,
                score=avg_credibility,
                max_score=1.0,
                details={
                    'individual_scores': credibility_scores,
                    'average_credibility': avg_credibility,
                    'total_sources': len(sources)
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Source credibility evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.SOURCE_CREDIBILITY,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _detect_bias(self, research_result: Dict[str, Any], 
                         original_query: str) -> EvaluationResult:
        """Detect bias in research results."""
        try:
            content = research_result.get('content', '')
            sources = research_result.get('sources', [])
            
            # Check for language bias
            language_bias = self._detect_language_bias(content)
            
            # Check for source bias
            source_bias = self._detect_source_bias(sources)
            
            # Check for confirmation bias
            confirmation_bias = self._detect_confirmation_bias(content, original_query)
            
            # Combined bias score (lower is better)
            bias_score = 1.0 - (language_bias * 0.4 + source_bias * 0.3 + confirmation_bias * 0.3)
            
            return EvaluationResult(
                metric=EvaluationMetric.BIAS_DETECTION,
                score=bias_score,
                max_score=1.0,
                details={
                    'language_bias': language_bias,
                    'source_bias': source_bias,
                    'confirmation_bias': confirmation_bias,
                    'bias_score': bias_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.BIAS_DETECTION,
                score=0.5,  # Neutral score on error
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    async def _evaluate_citation_quality(self, research_result: Dict[str, Any], 
                                       original_query: str) -> EvaluationResult:
        """Evaluate citation quality."""
        try:
            sources = research_result.get('sources', [])
            content = research_result.get('content', '')
            
            if not sources:
                return EvaluationResult(
                    metric=EvaluationMetric.CITATION_QUALITY,
                    score=0.0,
                    max_score=1.0,
                    details={'error': 'No sources found'},
                    timestamp=datetime.now().isoformat()
                )
            
            # Check citation format
            citation_format_score = self._check_citation_format(sources)
            
            # Check citation relevance
            citation_relevance_score = self._check_citation_relevance(sources, content)
            
            # Check citation recency
            citation_recency_score = self._check_citation_recency(sources)
            
            # Combined citation quality score
            citation_score = (citation_format_score * 0.4 + citation_relevance_score * 0.4 + citation_recency_score * 0.2)
            
            return EvaluationResult(
                metric=EvaluationMetric.CITATION_QUALITY,
                score=citation_score,
                max_score=1.0,
                details={
                    'citation_format': citation_format_score,
                    'citation_relevance': citation_relevance_score,
                    'citation_recency': citation_recency_score,
                    'citation_score': citation_score
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Citation quality evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.CITATION_QUALITY,
                score=0.0,
                max_score=1.0,
                details={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    # ==================== HELPER METHODS ====================
    
    def _extract_factual_claims(self, content: str) -> List[str]:
        """Extract factual claims from content."""
        # Simple pattern matching for factual claims
        patterns = [
            r'[A-Z][^.]*is [^.]*\.',
            r'[A-Z][^.]*are [^.]*\.',
            r'[A-Z][^.]*was [^.]*\.',
            r'[A-Z][^.]*were [^.]*\.',
            r'[A-Z][^.]*has [^.]*\.',
            r'[A-Z][^.]*have [^.]*\.',
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            claims.extend(matches)
        
        return claims
    
    def _is_claim_source_backed(self, claim: str, sources: List[Dict]) -> bool:
        """Check if a claim is backed by sources."""
        # Simple keyword matching
        claim_keywords = set(claim.lower().split())
        
        for source in sources:
            source_text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
            source_keywords = set(source_text.split())
            
            # Check for keyword overlap
            overlap = len(claim_keywords & source_keywords)
            if overlap > len(claim_keywords) * 0.3:  # 30% overlap threshold
                return True
        
        return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an'}
        
        keywords = [word for word in words if word not in stop_words]
        return keywords
    
    def _calculate_query_coverage(self, content: str, query: str) -> float:
        """Calculate how well the content covers the query."""
        query_keywords = self._extract_keywords(query)
        content_keywords = self._extract_keywords(content)
        
        if not query_keywords:
            return 0.5
        
        covered = set(query_keywords) & set(content_keywords)
        return len(covered) / len(query_keywords)
    
    def _calculate_source_diversity(self, sources: List[Dict]) -> float:
        """Calculate diversity of sources."""
        if not sources:
            return 0.0
        
        # Check domain diversity
        domains = set()
        for source in sources:
            url = source.get('url', '')
            if url:
                try:
                    domain = url.split('/')[2] if '://' in url else url.split('/')[0]
                    domains.add(domain)
                except:
                    pass
        
        # Normalize diversity score
        diversity = len(domains) / min(len(sources), 10)  # Cap at 10 sources
        return min(diversity, 1.0)
    
    def _calculate_content_depth(self, content: str) -> float:
        """Calculate depth of content."""
        # Check for multiple sections/paragraphs
        paragraphs = content.split('\n\n')
        depth_score = min(len(paragraphs) / 5, 1.0)  # Normalize to 5 paragraphs
        
        # Check for detailed explanations
        word_count = len(content.split())
        length_score = min(word_count / 500, 1.0)  # Normalize to 500 words
        
        return (depth_score * 0.5 + length_score * 0.5)
    
    def _check_factual_consistency(self, content: str) -> float:
        """Check for factual consistency in content."""
        # Simple consistency check based on sentence structure
        sentences = content.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for contradictory statements (simplified)
        contradiction_indicators = ['however', 'but', 'although', 'despite', 'contrary']
        contradictions = sum(1 for indicator in contradiction_indicators if indicator in content.lower())
        
        # Lower contradiction count is better
        consistency = max(0, 1.0 - (contradictions / len(sentences)))
        return consistency
    
    def _check_source_reliability(self, sources: List[Dict]) -> float:
        """Check reliability of sources."""
        if not sources:
            return 0.0
        
        reliability_scores = []
        for source in sources:
            url = source.get('url', '').lower()
            title = source.get('title', '').lower()
            
            # Check for reliable domains
            reliable_domains = ['edu', 'gov', 'org', 'ac.uk', 'edu.au']
            domain_score = 0.5  # Default score
            
            for domain in reliable_domains:
                if domain in url:
                    domain_score = 0.8
                    break
            
            # Check for academic indicators
            academic_indicators = ['research', 'study', 'journal', 'paper', 'academic']
            academic_score = 0.3
            for indicator in academic_indicators:
                if indicator in title or indicator in url:
                    academic_score = 0.7
                    break
            
            reliability_scores.append((domain_score + academic_score) / 2)
        
        return sum(reliability_scores) / len(reliability_scores)
    
    def _check_contradictions(self, content: str) -> float:
        """Check for internal contradictions."""
        # Simple contradiction detection
        contradiction_pairs = [
            ('always', 'never'),
            ('all', 'none'),
            ('every', 'no'),
            ('increase', 'decrease'),
            ('rise', 'fall'),
            ('positive', 'negative')
        ]
        
        contradictions = 0
        for pos, neg in contradiction_pairs:
            if pos in content.lower() and neg in content.lower():
                contradictions += 1
        
        # Return score (lower contradictions = higher score)
        return max(0, 1.0 - (contradictions / len(contradiction_pairs)))
    
    def _calculate_semantic_similarity(self, content: str, query: str) -> float:
        """Calculate semantic similarity between content and query."""
        # Simple keyword-based similarity
        content_keywords = set(self._extract_keywords(content))
        query_keywords = set(self._extract_keywords(query))
        
        if not query_keywords:
            return 0.5
        
        intersection = content_keywords & query_keywords
        union = content_keywords | query_keywords
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        return jaccard_similarity
    
    def _check_topic_alignment(self, content: str, query: str) -> float:
        """Check topic alignment between content and query."""
        # Simple topic keyword matching
        query_keywords = self._extract_keywords(query)
        content_keywords = self._extract_keywords(content)
        
        if not query_keywords:
            return 0.5
        
        matches = sum(1 for keyword in query_keywords if keyword in content_keywords)
        return matches / len(query_keywords)
    
    def _check_query_specificity(self, content: str, query: str) -> float:
        """Check if content addresses specific aspects of the query."""
        # Check for question words and specific terms
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        query_has_questions = any(word in query.lower() for word in question_words)
        
        if not query_has_questions:
            return 0.5
        
        # Check if content addresses the questions
        content_lower = content.lower()
        addressed_questions = sum(1 for word in question_words if word in query.lower() and word in content_lower)
        
        return addressed_questions / len([w for w in question_words if w in query.lower()])
    
    def _check_logical_flow(self, content: str) -> float:
        """Check logical flow of content."""
        # Check for transition words
        transition_words = ['first', 'second', 'third', 'next', 'then', 'finally', 'therefore', 'however', 'moreover', 'furthermore']
        
        content_lower = content.lower()
        transitions = sum(1 for word in transition_words if word in content_lower)
        
        # Normalize based on content length
        sentences = content.split('.')
        transition_score = transitions / len(sentences) if sentences else 0
        
        return min(transition_score * 10, 1.0)  # Scale up and cap at 1.0
    
    def _check_paragraph_structure(self, content: str) -> float:
        """Check paragraph structure."""
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            return 0.3
        
        # Check for topic sentences (first sentence of each paragraph)
        topic_sentences = 0
        for paragraph in paragraphs:
            if paragraph.strip():
                first_sentence = paragraph.split('.')[0].strip()
                if len(first_sentence) > 10:  # Reasonable topic sentence length
                    topic_sentences += 1
        
        return topic_sentences / len(paragraphs)
    
    def _check_transition_quality(self, content: str) -> float:
        """Check quality of transitions."""
        # Check for smooth transitions between sentences
        sentences = content.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Look for transition phrases
        transition_phrases = ['in addition', 'furthermore', 'moreover', 'on the other hand', 'in contrast', 'as a result', 'consequently']
        
        content_lower = content.lower()
        transitions = sum(1 for phrase in transition_phrases if phrase in content_lower)
        
        return min(transitions / len(sentences), 1.0)
    
    def _calculate_source_credibility_score(self, source: Dict) -> float:
        """Calculate credibility score for a single source."""
        url = source.get('url', '').lower()
        title = source.get('title', '').lower()
        
        score = 0.5  # Base score
        
        # Check domain credibility
        credible_domains = ['edu', 'gov', 'org', 'ac.uk', 'edu.au', 'harvard', 'mit', 'stanford', 'berkeley']
        for domain in credible_domains:
            if domain in url:
                score += 0.3
                break
        
        # Check for academic indicators
        academic_indicators = ['research', 'study', 'journal', 'paper', 'academic', 'university', 'institute']
        for indicator in academic_indicators:
            if indicator in title or indicator in url:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _detect_language_bias(self, content: str) -> float:
        """Detect language bias in content."""
        # Check for subjective language
        subjective_words = ['obviously', 'clearly', 'undoubtedly', 'certainly', 'definitely', 'absolutely']
        subjective_count = sum(1 for word in subjective_words if word in content.lower())
        
        # Check for emotional language
        emotional_words = ['amazing', 'terrible', 'incredible', 'awful', 'fantastic', 'horrible']
        emotional_count = sum(1 for word in emotional_words if word in content.lower())
        
        # Calculate bias score
        word_count = len(content.split())
        bias_score = (subjective_count + emotional_count) / max(word_count, 1)
        
        return min(bias_score * 100, 1.0)  # Scale and cap
    
    def _detect_source_bias(self, sources: List[Dict]) -> float:
        """Detect bias in source selection."""
        if not sources:
            return 0.0
        
        # Check for political bias indicators
        political_domains = ['foxnews', 'cnn', 'msnbc', 'breitbart', 'huffpost', 'dailywire']
        political_count = 0
        
        for source in sources:
            url = source.get('url', '').lower()
            for domain in political_domains:
                if domain in url:
                    political_count += 1
                    break
        
        # Calculate bias score
        bias_score = political_count / len(sources)
        return bias_score
    
    def _detect_confirmation_bias(self, content: str, query: str) -> float:
        """Detect confirmation bias."""
        # Check if content only supports one viewpoint
        # This is a simplified implementation
        query_keywords = self._extract_keywords(query)
        
        # Look for balanced perspectives
        balance_indicators = ['however', 'on the other hand', 'alternatively', 'contrary', 'different perspective']
        balance_count = sum(1 for indicator in balance_indicators if indicator in content.lower())
        
        # More balance indicators = less bias
        bias_score = max(0, 1.0 - (balance_count / 5))  # Normalize to 5 indicators
        return bias_score
    
    def _check_citation_format(self, sources: List[Dict]) -> float:
        """Check citation format quality."""
        if not sources:
            return 0.0
        
        format_scores = []
        for source in sources:
            score = 0.0
            
            # Check for required fields
            if source.get('title'):
                score += 0.3
            if source.get('url'):
                score += 0.3
            if source.get('authors') or source.get('author'):
                score += 0.2
            if source.get('published') or source.get('date'):
                score += 0.2
            
            format_scores.append(score)
        
        return sum(format_scores) / len(format_scores)
    
    def _check_citation_relevance(self, sources: List[Dict], content: str) -> float:
        """Check relevance of citations to content."""
        if not sources:
            return 0.0
        
        content_keywords = self._extract_keywords(content)
        relevance_scores = []
        
        for source in sources:
            source_text = f"{source.get('title', '')} {source.get('snippet', '')}"
            source_keywords = self._extract_keywords(source_text)
            
            # Calculate keyword overlap
            overlap = len(set(content_keywords) & set(source_keywords))
            relevance = overlap / max(len(content_keywords), 1)
            
            relevance_scores.append(min(relevance, 1.0))
        
        return sum(relevance_scores) / len(relevance_scores)
    
    def _check_citation_recency(self, sources: List[Dict]) -> float:
        """Check recency of citations."""
        if not sources:
            return 0.0
        
        current_year = datetime.now().year
        recency_scores = []
        
        for source in sources:
            published = source.get('published', '')
            if published:
                try:
                    # Extract year from published date
                    year_match = re.search(r'(\d{4})', published)
                    if year_match:
                        year = int(year_match.group(1))
                        age = current_year - year
                        # Score decreases with age
                        recency_score = max(0, 1.0 - (age / 10))  # 10 years = 0 score
                    else:
                        recency_score = 0.5  # Unknown date
                except:
                    recency_score = 0.5  # Error parsing date
            else:
                recency_score = 0.3  # No date provided
            
            recency_scores.append(recency_score)
        
        return sum(recency_scores) / len(recency_scores)
    
    def _evaluate_content_structure(self, content: str) -> float:
        """Evaluate content structure quality."""
        # Check for headings
        heading_patterns = [r'^#+\s', r'^\d+\.\s', r'^[A-Z][^.]*:$']
        headings = sum(1 for pattern in heading_patterns for _ in re.finditer(pattern, content, re.MULTILINE))
        
        # Check for lists
        list_patterns = [r'^\s*[-*]\s', r'^\s*\d+\.\s']
        lists = sum(1 for pattern in list_patterns for _ in re.finditer(pattern, content, re.MULTILINE))
        
        # Check for paragraphs
        paragraphs = len([p for p in content.split('\n\n') if p.strip()])
        
        # Calculate structure score
        structure_score = min((headings + lists) / max(paragraphs, 1), 1.0)
        return structure_score
    
    async def _generate_evaluation_summary(self, evaluation_results: Dict[str, EvaluationResult], 
                                         overall_score: float) -> Dict[str, Any]:
        """Generate evaluation summary."""
        try:
            # Calculate grade
            if overall_score >= 0.9:
                grade = "A+"
            elif overall_score >= 0.8:
                grade = "A"
            elif overall_score >= 0.7:
                grade = "B+"
            elif overall_score >= 0.6:
                grade = "B"
            elif overall_score >= 0.5:
                grade = "C+"
            elif overall_score >= 0.4:
                grade = "C"
            else:
                grade = "D"
            
            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for metric_name, result in evaluation_results.items():
                if result.score >= 0.8:
                    strengths.append(metric_name)
                elif result.score < 0.5:
                    weaknesses.append(metric_name)
            
            # Generate recommendations
            recommendations = []
            if EvaluationMetric.ACCURACY.value in weaknesses:
                recommendations.append("Improve source verification and fact-checking")
            if EvaluationMetric.COMPLETENESS.value in weaknesses:
                recommendations.append("Provide more comprehensive coverage of the topic")
            if EvaluationMetric.RELEVANCE.value in weaknesses:
                recommendations.append("Better align content with the research query")
            if EvaluationMetric.COHERENCE.value in weaknesses:
                recommendations.append("Improve logical flow and structure")
            if EvaluationMetric.SOURCE_CREDIBILITY.value in weaknesses:
                recommendations.append("Use more credible and authoritative sources")
            if EvaluationMetric.BIAS_DETECTION.value in weaknesses:
                recommendations.append("Reduce bias and provide balanced perspectives")
            if EvaluationMetric.CITATION_QUALITY.value in weaknesses:
                recommendations.append("Improve citation format and quality")
            
            return {
                'overall_score': overall_score,
                'grade': grade,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': recommendations,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation summary generation failed: {e}")
            return {
                'overall_score': overall_score,
                'grade': 'N/A',
                'strengths': [],
                'weaknesses': [],
                'recommendations': ['Evaluation summary generation failed'],
                'evaluation_timestamp': datetime.now().isoformat()
            }
