"""
Model Manager for Agentic Agent Trainer System

Manages model checkpoints, versioning, and model loading/saving.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import shutil

from ..models.training import ModelCheckpoint, ModelStatus, TrainingAlgorithm

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model checkpoints and versions.
    
    Responsibilities:
    - Save and load model checkpoints
    - Track model versions
    - Manage best model
    - Model evaluation and comparison
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10
    ):
        """
        Initialize model manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # Track checkpoints
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        self.best_checkpoint: Optional[ModelCheckpoint] = None
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
        logger.info(f"ModelManager initialized with checkpoint_dir: {checkpoint_dir}")
    
    def _load_existing_checkpoints(self) -> None:
        """Load existing checkpoints from disk"""
        metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    
                for checkpoint_data in data.get("checkpoints", []):
                    checkpoint = ModelCheckpoint(**checkpoint_data)
                    self.checkpoints[checkpoint.checkpoint_id] = checkpoint
                    
                    if checkpoint.is_best:
                        self.best_checkpoint = checkpoint
                
                logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
            except Exception as e:
                logger.error(f"Error loading checkpoints metadata: {e}")
    
    def _save_checkpoints_metadata(self) -> None:
        """Save checkpoints metadata to disk"""
        metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        try:
            data = {
                "checkpoints": [
                    checkpoint.model_dump() for checkpoint in self.checkpoints.values()
                ],
                "best_checkpoint_id": self.best_checkpoint.checkpoint_id if self.best_checkpoint else None,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving checkpoints metadata: {e}")
    
    def save_checkpoint(
        self,
        model_path: str,
        model_name: str,
        algorithm: TrainingAlgorithm,
        training_step: int = 0,
        epoch: int = 0,
        training_loss: float = 0.0,
        validation_loss: Optional[float] = None,
        average_reward: float = 0.0,
        evaluation_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelCheckpoint:
        """
        Save a model checkpoint.
        
        Args:
            model_path: Path to the model files
            model_name: Name of the model
            algorithm: Training algorithm used
            training_step: Current training step
            epoch: Current epoch
            training_loss: Training loss
            validation_loss: Validation loss (optional)
            average_reward: Average reward
            evaluation_metrics: Evaluation metrics (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Created checkpoint
        """
        # Generate checkpoint ID and version
        checkpoint_id = f"{model_name}_step_{training_step}_{int(datetime.utcnow().timestamp())}"
        model_version = f"{algorithm.value}_v{len(self.checkpoints) + 1}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to checkpoint directory
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                # Copy entire directory
                dest_path = checkpoint_path / "model"
                shutil.copytree(model_path, dest_path, dirs_exist_ok=True)
            else:
                # Copy single file
                shutil.copy2(model_path, checkpoint_path)
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(checkpoint_path)
        
        # Create checkpoint metadata
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_name=model_name,
            model_path=str(checkpoint_path),
            model_version=model_version,
            algorithm=algorithm,
            training_step=training_step,
            epoch=epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
            average_reward=average_reward,
            evaluation_metrics=evaluation_metrics or {},
            status=ModelStatus.READY,
            model_size_mb=model_size_mb,
            metadata=metadata or {}
        )
        
        # Store checkpoint
        self.checkpoints[checkpoint_id] = checkpoint
        
        # Save metadata
        self._save_checkpoints_metadata()
        
        logger.info(f"Saved checkpoint: {checkpoint_id} at step {training_step}")
        
        return checkpoint
    
    def mark_as_best(
        self,
        checkpoint_id: str,
        evaluation_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Mark a checkpoint as the best model.
        
        Args:
            checkpoint_id: ID of the checkpoint
            evaluation_metrics: Evaluation metrics (optional)
        """
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return
        
        # Unmark previous best
        if self.best_checkpoint:
            self.best_checkpoint.is_best = False
            self.best_checkpoint.status = ModelStatus.READY
        
        # Mark new best
        checkpoint.mark_as_best()
        if evaluation_metrics:
            checkpoint.update_metrics(evaluation_metrics)
        
        self.best_checkpoint = checkpoint
        
        # Save metadata
        self._save_checkpoints_metadata()
        
        logger.info(f"Marked checkpoint {checkpoint_id} as best model")
    
    def get_best_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Get the best checkpoint"""
        return self.best_checkpoint
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        """Get a checkpoint by ID"""
        return self.checkpoints.get(checkpoint_id)
    
    def list_checkpoints(
        self,
        algorithm: Optional[TrainingAlgorithm] = None,
        status: Optional[ModelStatus] = None,
        limit: Optional[int] = None
    ) -> List[ModelCheckpoint]:
        """
        List checkpoints with optional filters.
        
        Args:
            algorithm: Filter by algorithm
            status: Filter by status
            limit: Maximum number to return
            
        Returns:
            List of checkpoints
        """
        checkpoints = list(self.checkpoints.values())
        
        # Filter by algorithm
        if algorithm:
            checkpoints = [c for c in checkpoints if c.algorithm == algorithm]
        
        # Filter by status
        if status:
            checkpoints = [c for c in checkpoints if c.status == status]
        
        # Sort by training step (descending)
        checkpoints.sort(key=lambda x: x.training_step, reverse=True)
        
        # Limit results
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
            
        Returns:
            True if deleted, False otherwise
        """
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return False
        
        # Don't delete best checkpoint
        if checkpoint.is_best:
            logger.warning(f"Cannot delete best checkpoint {checkpoint_id}")
            return False
        
        # Delete checkpoint directory
        checkpoint_path = Path(checkpoint.model_path)
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        # Remove from tracking
        del self.checkpoints[checkpoint_id]
        
        # Save metadata
        self._save_checkpoints_metadata()
        
        logger.info(f"Deleted checkpoint {checkpoint_id}")
        return True
    
    def cleanup_old_checkpoints(self) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Returns:
            Number of checkpoints deleted
        """
        # Get all checkpoints sorted by training step
        all_checkpoints = sorted(
            self.checkpoints.values(),
            key=lambda x: x.training_step,
            reverse=True
        )
        
        # Keep best checkpoint and most recent ones
        to_keep = set()
        if self.best_checkpoint:
            to_keep.add(self.best_checkpoint.checkpoint_id)
        
        # Keep most recent checkpoints
        for checkpoint in all_checkpoints[:self.max_checkpoints]:
            to_keep.add(checkpoint.checkpoint_id)
        
        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in all_checkpoints:
            if checkpoint.checkpoint_id not in to_keep:
                if self.delete_checkpoint(checkpoint.checkpoint_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old checkpoints")
        return deleted_count
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in MB"""
        total_size = 0
        
        if model_path.is_dir():
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        else:
            total_size = model_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_checkpoint_id": self.best_checkpoint.checkpoint_id if self.best_checkpoint else None,
            "checkpoint_dir": str(self.checkpoint_dir),
            "max_checkpoints": self.max_checkpoints,
            "checkpoints_by_algorithm": {
                algo.value: len([c for c in self.checkpoints.values() if c.algorithm == algo])
                for algo in TrainingAlgorithm
            }
        }


