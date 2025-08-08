"""
Data Generator for the Kimi-K2 Agentic Data Synthesis System

Generates and exports training data in various formats for model training.
"""

from typing import List, Dict, Any, Optional
from ..models.data import TrainingData, DataBatch, DataFormat, Metadata
from ..models.evaluation import EvaluationResult
from ..models.simulation import SimulationSession
import logging
from datetime import datetime
import json
import csv
import os

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Generates and exports training data for model training.
    
    Responsibilities:
    - Data format conversion
    - Batch processing
    - Data validation
    - Metadata management
    """
    
    def __init__(self, output_directory: str = "output"):
        self.output_directory = output_directory
        self.generated_batches: Dict[str, DataBatch] = {}
        self.export_formats = {
            DataFormat.JSON: self._export_json,
            DataFormat.JSONL: self._export_jsonl,
            DataFormat.CSV: self._export_csv
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
    
    def create_training_batch(self, name: str, description: str,
                            training_data: List[TrainingData],
                            quality_threshold: float = 0.7) -> DataBatch:
        """Create a new training data batch"""
        batch = DataBatch(
            name=name,
            description=description,
            quality_threshold=quality_threshold
        )
        
        # Add training data to batch
        for data in training_data:
            batch.add_training_data(data)
        
        # Validate and calculate metrics
        batch.validate_batch()
        batch.calculate_quality_metrics()
        
        # Store batch
        self.generated_batches[batch.id] = batch
        
        logger.info(f"Created training batch: {name} with {len(training_data)} items")
        return batch
    
    def generate_batch_from_simulations(self, name: str, description: str,
                                      simulation_sessions: List[SimulationSession],
                                      evaluation_results: List[EvaluationResult],
                                      quality_threshold: float = 0.7) -> Optional[DataBatch]:
        """Generate training batch from simulation sessions and evaluations"""
        if len(simulation_sessions) != len(evaluation_results):
            logger.error("Number of simulation sessions and evaluation results must match")
            return None
        
        training_data_list = []
        
        for session, evaluation in zip(simulation_sessions, evaluation_results):
            # Create training data from simulation
            training_data = self._create_training_data_from_simulation(session, evaluation)
            if training_data and training_data.validate_data():
                training_data_list.append(training_data)
        
        if not training_data_list:
            logger.warning("No valid training data generated from simulations")
            return None
        
        return self.create_training_batch(name, description, training_data_list, quality_threshold)
    
    def _create_training_data_from_simulation(self, session: SimulationSession,
                                            evaluation: EvaluationResult) -> Optional[TrainingData]:
        """Create training data from a simulation session and evaluation"""
        try:
            # Create metadata
            metadata = Metadata(
                domain_id=session.domain_id,
                scenario_id=session.scenario_id,
                agent_ids=session.agent_ids,
                simulation_id=session.id,
                evaluation_id=evaluation.id,
                quality_score=evaluation.overall_score,
                tags=["generated", "simulation"]
            )
            
            # Extract conversation history
            conversation_history = []
            for step in session.steps:
                # steps may be dicts when passed from LangGraph state; normalize access
                out = step.get("output_data") if isinstance(step, dict) else step.output_data
                thought = step.get("agent_thought") if isinstance(step, dict) else step.agent_thought
                step_no = step.get("step_number") if isinstance(step, dict) else step.step_number
                start_time = step.get("start_time") if isinstance(step, dict) else (step.start_time.isoformat() if getattr(step, "start_time", None) else None)
                if out and "response" in out:
                    turn = {
                        "role": "assistant",
                        "content": out["response"],
                        "step": step_no,
                        "timestamp": start_time,
                    }
                    if thought:
                        turn["thought"] = thought
                    conversation_history.append(turn)
            
            # Extract tool usage log
            tool_usage_log = []
            for step in session.steps:
                tool_used = step.get("tool_used") if isinstance(step, dict) else step.tool_used
                if tool_used:
                    input_data = step.get("input_data") if isinstance(step, dict) else step.input_data
                    output_data = step.get("output_data") if isinstance(step, dict) else step.output_data
                    start_time = step.get("start_time") if isinstance(step, dict) else (step.start_time.isoformat() if getattr(step, "start_time", None) else None)
                    duration = step.get("duration") if isinstance(step, dict) else step.duration
                    tool_usage_log.append({
                        "tool": tool_used,
                        "parameters": (input_data or {}).get("parameters", {}),
                        "result": (output_data or {}).get("result", ""),
                        "timestamp": start_time,
                        "duration": duration,
                    })
            
            # Create final outcome
            final_outcome = {
                "status": session.status.value,
                "quality_score": evaluation.overall_score,
                "evaluation_feedback": evaluation.feedback,
                "total_steps": len(session.steps),
                "completed_steps": len(session.get_completed_steps()),
                "duration": session.duration
            }
            
            # Create quality metrics
            quality_metrics = {
                "overall": evaluation.overall_score,
                "passed": evaluation.passed
            }
            
            # Add individual scores
            for score in evaluation.individual_scores:
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
            
        except Exception as e:
            logger.error(f"Failed to create training data from simulation: {e}")
            return None
    
    def export_batch(self, batch_id: str, format: DataFormat = DataFormat.JSON,
                    output_path: Optional[str] = None) -> Optional[str]:
        """Export a batch to specified format"""
        batch = self.generated_batches.get(batch_id)
        if not batch:
            logger.error(f"Batch not found: {batch_id}")
            return None
        
        if format not in self.export_formats:
            logger.error(f"Unsupported export format: {format}")
            return None
        
        # Generate output path
        if not output_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{batch.name}_{timestamp}.{format.value}"
            output_path = os.path.join(self.output_directory, filename)
        
        try:
            # Export using appropriate formatter
            export_func = self.export_formats[format]
            result = export_func(batch, output_path)
            
            if result:
                logger.info(f"Exported batch {batch.name} to {output_path}")
                return output_path
            else:
                logger.error(f"Failed to export batch {batch.name}")
                return None
                
        except Exception as e:
            logger.error(f"Export failed for batch {batch.name}: {e}")
            return None
    
    def _export_json(self, batch: DataBatch, output_path: str) -> bool:
        """Export batch to JSON format"""
        try:
            data = {
                "batch_info": {
                    "id": batch.id,
                    "name": batch.name,
                    "description": batch.description,
                    "created_at": batch.created_at.isoformat(),
                    "batch_size": batch.batch_size,
                    "quality_threshold": batch.quality_threshold,
                    "average_quality_score": batch.average_quality_score
                },
                "training_data": [data.model_dump() for data in batch.training_data]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    def _export_jsonl(self, batch: DataBatch, output_path: str) -> bool:
        """Export batch to JSONL format"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for data in batch.training_data:
                    json.dump(data.model_dump(), f, default=str)
                    f.write('\n')
            
            return True
            
        except Exception as e:
            logger.error(f"JSONL export failed: {e}")
            return False
    
    def _export_csv(self, batch: DataBatch, output_path: str) -> bool:
        """Export batch to CSV format"""
        try:
            if not batch.training_data:
                return False
            
            # Define CSV fields
            fields = [
                'id', 'domain_id', 'scenario_id', 'simulation_id',
                'quality_score', 'conversation_length', 'tool_usage_count',
                'status', 'created_at'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for data in batch.training_data:
                    row = {
                        'id': data.id,
                        'domain_id': data.metadata.domain_id,
                        'scenario_id': data.metadata.scenario_id,
                        'simulation_id': data.metadata.simulation_id,
                        'quality_score': data.quality_metrics.get('overall', 0.0),
                        'conversation_length': len(data.conversation_history),
                        'tool_usage_count': len(data.tool_usage_log),
                        'status': data.final_outcome.get('status', 'unknown'),
                        'created_at': data.created_at.isoformat()
                    }
                    writer.writerow(row)
            
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def get_batch(self, batch_id: str) -> Optional[DataBatch]:
        """Get a batch by ID"""
        return self.generated_batches.get(batch_id)
    
    def list_batches(self) -> List[DataBatch]:
        """List all generated batches"""
        return list(self.generated_batches.values())
    
    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch"""
        if batch_id not in self.generated_batches:
            return False
        
        batch_name = self.generated_batches[batch_id].name
        del self.generated_batches[batch_id]
        logger.info(f"Deleted batch: {batch_name}")
        return True
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about all batches"""
        stats = {
            "total_batches": len(self.generated_batches),
            "total_training_data": 0,
            "average_batch_size": 0.0,
            "average_quality_score": 0.0,
            "batches_by_quality": {
                "excellent": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        if not self.generated_batches:
            return stats
        
        total_data = 0
        total_quality = 0.0
        
        for batch in self.generated_batches.values():
            total_data += batch.batch_size
            total_quality += batch.average_quality_score
            
            # Quality distribution
            quality = batch.average_quality_score
            if quality >= 0.9:
                stats["batches_by_quality"]["excellent"] += 1
            elif quality >= 0.8:
                stats["batches_by_quality"]["high"] += 1
            elif quality >= 0.7:
                stats["batches_by_quality"]["medium"] += 1
            else:
                stats["batches_by_quality"]["low"] += 1
        
        stats["total_training_data"] = total_data
        stats["average_batch_size"] = total_data / len(self.generated_batches)
        stats["average_quality_score"] = total_quality / len(self.generated_batches)
        
        return stats
    
    def validate_batch(self, batch_id: str) -> Dict[str, Any]:
        """Validate a batch and return validation results"""
        batch = self.get_batch(batch_id)
        if not batch:
            return {"valid": False, "errors": ["Batch not found"]}
        
        validation_results = batch.validate_batch()
        
        # Additional validation
        additional_errors = []
        
        # Check for empty batch
        if batch.batch_size == 0:
            additional_errors.append("Batch is empty")
        
        # Check quality distribution
        low_quality_count = sum(
            1 for data in batch.training_data
            if data.quality_metrics.get("overall", 0.0) < batch.quality_threshold
        )
        
        if low_quality_count > batch.batch_size * 0.1:  # More than 10% low quality
            additional_errors.append(f"Too many low-quality items: {low_quality_count}")
        
        validation_results["errors"].extend(additional_errors)
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def merge_batches(self, batch_ids: List[str], name: str, description: str) -> Optional[DataBatch]:
        """Merge multiple batches into a single batch"""
        if len(batch_ids) < 2:
            logger.error("Need at least 2 batches to merge")
            return None
        
        # Collect all training data
        all_training_data = []
        
        for batch_id in batch_ids:
            batch = self.get_batch(batch_id)
            if not batch:
                logger.error(f"Batch not found: {batch_id}")
                return None
            
            all_training_data.extend(batch.training_data)
        
        if not all_training_data:
            logger.error("No training data to merge")
            return None
        
        # Create merged batch
        merged_batch = self.create_training_batch(
            name=name,
            description=description,
            training_data=all_training_data
        )
        
        logger.info(f"Merged {len(batch_ids)} batches into {name}")
        return merged_batch
    
    def split_batch(self, batch_id: str, split_ratio: float = 0.8,
                   train_name: str = None, test_name: str = None) -> Optional[Dict[str, DataBatch]]:
        """Split a batch into training and test sets"""
        batch = self.get_batch(batch_id)
        if not batch:
            logger.error(f"Batch not found: {batch_id}")
            return None
        
        if not 0.0 < split_ratio < 1.0:
            logger.error(f"Invalid split ratio: {split_ratio}")
            return None
        
        # Sort by quality score
        sorted_data = sorted(
            batch.training_data,
            key=lambda x: x.quality_metrics.get("overall", 0.0),
            reverse=True
        )
        
        # Split data
        split_index = int(len(sorted_data) * split_ratio)
        train_data = sorted_data[:split_index]
        test_data = sorted_data[split_index:]
        
        # Generate names if not provided
        if not train_name:
            train_name = f"{batch.name}_train"
        if not test_name:
            test_name = f"{batch.name}_test"
        
        # Create batches
        train_batch = self.create_training_batch(
            name=train_name,
            description=f"Training split from {batch.name}",
            training_data=train_data
        )
        
        test_batch = self.create_training_batch(
            name=test_name,
            description=f"Test split from {batch.name}",
            training_data=test_data
        )
        
        logger.info(f"Split batch {batch.name}: {len(train_data)} train, {len(test_data)} test")
        
        return {
            "train": train_batch,
            "test": test_batch
        } 