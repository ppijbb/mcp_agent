"""
GRPO (Group Relative Policy Optimization) Trainer

Implements GRPO algorithm for training agents using group-based preferences.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import numpy as np

from ...models.training import TrainingConfig, TrainingAlgorithm

logger = logging.getLogger(__name__)


class GroupDataset(Dataset):
    """Dataset for GRPO groups"""
    
    def __init__(
        self,
        groups: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.groups = groups
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Flatten groups into samples
        self.samples = []
        for group in groups:
            for sample in group["samples"]:
                self.samples.append({
                    **sample,
                    "group_id": group["group_id"],
                    "group_size": group["group_size"],
                    "average_reward": group["average_reward"]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize prompt
        prompt_encoded = self.tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize response
        response_encoded = self.tokenizer(
            sample["response"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_encoded["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoded["attention_mask"].squeeze(),
            "response_input_ids": response_encoded["input_ids"].squeeze(),
            "response_attention_mask": response_encoded["attention_mask"].squeeze(),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "group_id": sample["group_id"],
            "group_size": sample["group_size"],
            "average_reward": torch.tensor(sample["average_reward"], dtype=torch.float32),
            "metadata": sample.get("metadata", {})
        }


class GRPOTrainer:
    """
    GRPO Trainer for agent learning.
    
    Implements Group Relative Policy Optimization algorithm for training
    agents using group-based preferences.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            self.model = model
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        self.model.to(self.device)
        
        logger.info(f"GRPOTrainer initialized with model: {config.base_model}")
    
    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        group_rewards: torch.Tensor,
        beta: float = 0.1
    ) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        Args:
            log_probs: Log probabilities from model
            rewards: Individual sample rewards
            group_rewards: Average rewards for groups
            beta: Temperature parameter
            
        Returns:
            GRPO loss
        """
        # Relative reward: reward - group_average_reward
        relative_rewards = rewards - group_rewards
        
        # GRPO loss: -log(sigma(beta * (log_prob + relative_reward)))
        # This encourages higher log prob for samples with positive relative reward
        loss = -torch.nn.functional.logsigmoid(beta * (log_probs + relative_rewards)).mean()
        return loss
    
    def train(
        self,
        groups: List[Dict[str, Any]],
        validation_groups: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train the model using GRPO.
        
        Args:
            groups: List of groups for training
            validation_groups: Optional validation groups
            
        Returns:
            Training results dictionary
        """
        if not groups:
            raise ValueError("No groups provided for training")
        
        logger.info(f"Starting GRPO training with {len(groups)} groups")
        
        # Create dataset
        train_dataset = GroupDataset(
            groups,
            self.tokenizer,
            max_length=self.config.algorithm_params.get("max_length", 512)
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        beta = self.config.algorithm_params.get("beta", 0.1)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                prompt_ids = batch["prompt_input_ids"].to(self.device)
                response_ids = batch["response_input_ids"].to(self.device)
                rewards = batch["reward"].to(self.device)
                group_rewards = batch["average_reward"].to(self.device)
                
                # Get log probabilities
                log_probs = self._get_log_probs(
                    self.model,
                    prompt_ids,
                    response_ids
                )
                
                # Compute GRPO loss
                loss = self.compute_grpo_loss(
                    log_probs,
                    rewards,
                    group_rewards,
                    beta
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Batch {batch_idx+1}, Loss: {loss.item():.4f}"
                    )
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        results = {
            "algorithm": TrainingAlgorithm.GRPO.value,
            "total_epochs": self.config.num_epochs,
            "total_batches": num_batches,
            "average_loss": avg_loss,
            "final_loss": avg_loss,
            "num_groups": len(groups),
            "total_samples": len(train_dataset)
        }
        
        logger.info(f"GRPO training completed. Average loss: {avg_loss:.4f}")
        return results
    
    def _get_log_probs(
        self,
        model: Any,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get log probabilities for responses given prompts.
        
        Args:
            model: Model to use
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs
            
        Returns:
            Log probabilities
        """
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Extract log probs for response tokens
        response_log_probs = log_probs[:, -response_ids.size(1):, :]
        
        # Get log prob for each token in response
        response_token_log_probs = torch.gather(
            response_log_probs,
            dim=2,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Average log prob per sequence
        avg_log_probs = response_token_log_probs.mean(dim=1)
        
        return avg_log_probs
    
    def save_model(self, output_path: str) -> None:
        """Save the trained model"""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_path}")

