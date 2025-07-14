"""
Quality Filter for the Kimi-K2 Agentic Data Synthesis System

Filters and selects high-quality training data based on evaluation results.
"""

from typing import List, Dict, Any, Optional
from ..models.evaluation import EvaluationResult
from ..models.data import TrainingData, DataBatch, DataQuality
from ..models.simulation import SimulationSession
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Filters training data based on quality criteria.
    
    Responsibilities:
    - Quality threshold application
    - Data filtering and selection
    - Duplicate removal
    - Data normalization
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.filtered_data: List[TrainingData] = []
        self.rejected_data: List[TrainingData] = []
        self.duplicate_detection_enabled = True
        self.normalization_enabled = True
        
        # Quality criteria weights
        self.quality_weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "creativity": 0.15,
            "efficiency": 0.15,
            "user_satisfaction": 0.15
        }
    
    def filter_simulation_data(self, simulation_session: SimulationSession,
                             evaluation_result: EvaluationResult) -> Optional[TrainingData]:
        """Filter simulation data based on evaluation results"""
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(evaluation_result)
        
        # Create training data entry
        training_data = self._create_training_data(simulation_session, evaluation_result, quality_score)
        
        # Apply quality filter
        if quality_score >= self.quality_threshold:
            # Check for duplicates
            if not self._is_duplicate(training_data):
                self.filtered_data.append(training_data)
                logger.info(f"Accepted training data with quality score: {quality_score:.3f}")
                return training_data
            else:
                logger.info(f"Rejected duplicate training data")
                self.rejected_data.append(training_data)
                return None
        else:
            logger.info(f"Rejected training data with quality score: {quality_score:.3f} (threshold: {self.quality_threshold})")
            self.rejected_data.append(training_data)
            return None
    
    def _calculate_quality_score(self, evaluation_result: EvaluationResult) -> float:
        """Calculate weighted quality score from evaluation results"""
        if not evaluation_result.individual_scores:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for score in evaluation_result.individual_scores:
            weight = self.quality_weights.get(score.evaluation_type.value, 0.1)
            weighted_score += score.score * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return evaluation_result.overall_score
    
    def _create_training_data(self, simulation_session: SimulationSession,
                            evaluation_result: EvaluationResult,
                            quality_score: float) -> TrainingData:
        """Create training data from simulation session"""
        from ..models.data import Metadata
        
        # Create metadata
        metadata = Metadata(
            domain_id=simulation_session.domain_id,
            scenario_id=simulation_session.scenario_id,
            agent_ids=simulation_session.agent_ids,
            simulation_id=simulation_session.id,
            evaluation_id=evaluation_result.id,
            quality_score=quality_score,
            tags=["filtered", "high_quality"] if quality_score >= 0.8 else ["filtered"]
        )
        
        # Extract conversation history
        conversation_history = []
        for step in simulation_session.steps:
            if step.output_data and "response" in step.output_data:
                conversation_history.append({
                    "role": "assistant",
                    "content": step.output_data["response"],
                    "step": step.step_number
                })
        
        # Extract tool usage log
        tool_usage_log = []
        for step in simulation_session.steps:
            if step.tool_used:
                tool_usage_log.append({
                    "tool": step.tool_used,
                    "parameters": step.input_data.get("parameters", {}),
                    "result": step.output_data.get("result", "")
                })
        
        # Create final outcome
        final_outcome = {
            "status": "completed" if simulation_session.status.value == "completed" else "failed",
            "quality_score": quality_score,
            "evaluation_feedback": evaluation_result.feedback
        }
        
        # Create quality metrics
        quality_metrics = {
            "overall": quality_score,
            "accuracy": 0.0,
            "completeness": 0.0,
            "creativity": 0.0,
            "efficiency": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Add individual scores
        for score in evaluation_result.individual_scores:
            quality_metrics[score.evaluation_type.value] = score.score
        
        # Create training data
        training_data = TrainingData(
            metadata=metadata,
            conversation_history=conversation_history,
            tool_usage_log=tool_usage_log,
            final_outcome=final_outcome,
            quality_metrics=quality_metrics
        )
        
        return training_data
    
    def _is_duplicate(self, training_data: TrainingData) -> bool:
        """Check if training data is a duplicate"""
        if not self.duplicate_detection_enabled:
            return False
        
        # Simple duplicate detection based on conversation content
        new_conversation = training_data.get_conversation_text()
        
        for existing_data in self.filtered_data:
            existing_conversation = existing_data.get_conversation_text()
            
            # Calculate similarity (simple approach - can be enhanced with semantic similarity)
            similarity = self._calculate_text_similarity(new_conversation, existing_conversation)
            
            if similarity > 0.8:  # 80% similarity threshold
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def filter_batch(self, data_batch: DataBatch) -> DataBatch:
        """Filter a batch of training data"""
        filtered_batch = DataBatch(
            name=f"{data_batch.name} (Filtered)",
            description=f"Quality-filtered version of {data_batch.name}",
            quality_threshold=self.quality_threshold
        )
        
        for training_data in data_batch.training_data:
            # Validate data
            if not training_data.validate_data():
                continue
            
            # Check quality threshold
            quality_score = training_data.quality_metrics.get("overall", 0.0)
            
            if quality_score >= self.quality_threshold:
                # Check for duplicates
                if not self._is_duplicate_in_batch(training_data, filtered_batch.training_data):
                    filtered_batch.add_training_data(training_data)
        
        # Calculate batch statistics
        filtered_batch.calculate_quality_metrics()
        
        logger.info(f"Filtered batch: {len(data_batch.training_data)} -> {len(filtered_batch.training_data)} items")
        return filtered_batch
    
    def _is_duplicate_in_batch(self, training_data: TrainingData, existing_data: List[TrainingData]) -> bool:
        """Check for duplicates within a batch"""
        if not self.duplicate_detection_enabled:
            return False
        
        new_conversation = training_data.get_conversation_text()
        
        for existing in existing_data:
            existing_conversation = existing.get_conversation_text()
            similarity = self._calculate_text_similarity(new_conversation, existing_conversation)
            
            if similarity > 0.8:
                return True
        
        return False
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set quality threshold for filtering"""
        if 0.0 <= threshold <= 1.0:
            self.quality_threshold = threshold
            logger.info(f"Quality threshold set to: {threshold}")
        else:
            logger.warning(f"Invalid quality threshold: {threshold}. Must be between 0.0 and 1.0")
    
    def set_quality_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for quality criteria"""
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Quality weights do not sum to 1.0: {total_weight}")
        
        self.quality_weights = weights.copy()
        logger.info(f"Quality weights updated: {weights}")
    
    def enable_duplicate_detection(self, enabled: bool = True) -> None:
        """Enable or disable duplicate detection"""
        self.duplicate_detection_enabled = enabled
        logger.info(f"Duplicate detection {'enabled' if enabled else 'disabled'}")
    
    def enable_normalization(self, enabled: bool = True) -> None:
        """Enable or disable data normalization"""
        self.normalization_enabled = enabled
        logger.info(f"Data normalization {'enabled' if enabled else 'disabled'}")
    
    def get_filtered_data(self, min_quality: Optional[float] = None) -> List[TrainingData]:
        """Get filtered data with optional minimum quality threshold"""
        if min_quality is None:
            return self.filtered_data.copy()
        
        return [
            data for data in self.filtered_data
            if data.quality_metrics.get("overall", 0.0) >= min_quality
        ]
    
    def get_rejected_data(self) -> List[TrainingData]:
        """Get rejected data"""
        return self.rejected_data.copy()
    
    def clear_data(self) -> None:
        """Clear all filtered and rejected data"""
        self.filtered_data.clear()
        self.rejected_data.clear()
        logger.info("Cleared all filtered and rejected data")
    
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get statistics about filtering process"""
        stats = {
            "total_filtered": len(self.filtered_data),
            "total_rejected": len(self.rejected_data),
            "acceptance_rate": 0.0,
            "quality_threshold": self.quality_threshold,
            "average_quality_filtered": 0.0,
            "average_quality_rejected": 0.0,
            "quality_distribution": {
                "excellent": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        total_processed = len(self.filtered_data) + len(self.rejected_data)
        if total_processed > 0:
            stats["acceptance_rate"] = len(self.filtered_data) / total_processed
        
        # Calculate average quality scores
        if self.filtered_data:
            avg_quality_filtered = sum(
                data.quality_metrics.get("overall", 0.0) for data in self.filtered_data
            ) / len(self.filtered_data)
            stats["average_quality_filtered"] = avg_quality_filtered
        
        if self.rejected_data:
            avg_quality_rejected = sum(
                data.quality_metrics.get("overall", 0.0) for data in self.rejected_data
            ) / len(self.rejected_data)
            stats["average_quality_rejected"] = avg_quality_rejected
        
        # Quality distribution
        for data in self.filtered_data:
            quality = data.quality_metrics.get("overall", 0.0)
            if quality >= 0.9:
                stats["quality_distribution"]["excellent"] += 1
            elif quality >= 0.8:
                stats["quality_distribution"]["high"] += 1
            elif quality >= 0.7:
                stats["quality_distribution"]["medium"] += 1
            else:
                stats["quality_distribution"]["low"] += 1
        
        return stats
    
    def export_filtered_data(self, format: str = "json") -> str:
        """Export filtered data in specified format"""
        if format.lower() == "json":
            import json
            data_list = [data.model_dump() for data in self.filtered_data]
            return json.dumps(data_list, indent=2, default=str)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return ""
    
    def import_filtered_data(self, data_list: List[Dict[str, Any]]) -> int:
        """Import filtered data from list"""
        imported_count = 0
        
        for data_dict in data_list:
            try:
                training_data = TrainingData(**data_dict)
                if training_data.validate_data():
                    self.filtered_data.append(training_data)
                    imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import data item: {e}")
        
        logger.info(f"Imported {imported_count} training data items")
        return imported_count 