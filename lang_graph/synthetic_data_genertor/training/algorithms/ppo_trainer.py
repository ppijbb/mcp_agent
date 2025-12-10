"""
PPO (Proximal Policy Optimization) Trainer

Implements PPO algorithm for training agents using rewards from agent self-evaluation.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import numpy as np

from ...models.training import TrainingConfig, TrainingAlgorithm
from ..data_processor import ProcessedTrainingSample

logger = logging.getLogger(__name__)


class PPODataset(Dataset):
    """Dataset for PPO training"""
    
    def __init__(
        self,
        samples: List[ProcessedTrainingSample],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize prompt
        prompt_encoded = self.tokenizer(
            sample.prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize response
        response_encoded = self.tokenizer(
            sample.response,
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
            "reward": torch.tensor(sample.reward, dtype=torch.float32),
            "metadata": sample.metadata
        }


class PPOTrainer:
    """
    PPO Trainer for agent learning.
    
    Implements Proximal Policy Optimization algorithm for training
    agents using rewards from agent self-evaluation.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize PPO trainer.
        
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
        
        # Value function (optional, for advantage estimation)
        self.use_value_function = config.algorithm_params.get("use_value_function", False)
        if self.use_value_function:
            # Simple value head (can be enhanced)
            self.value_head = torch.nn.Linear(
                self.model.config.hidden_size,
                1
            ).to(self.device)
        
        # Store old policy for PPO
        self.old_policy = None
        
        logger.info(f"PPOTrainer initialized with model: {config.base_model}")
    
    def compute_ppo_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float = 0.2
    ) -> torch.Tensor:
        """
        Compute PPO clipped loss.
        
        Args:
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            clip_epsilon: Clipping parameter
            
        Returns:
            PPO loss
        """
        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        # PPO loss: min(ratio * advantage, clipped_ratio * advantage)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        return loss
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: Reward signals
            values: Value estimates (optional)
            gamma: Discount factor
            lambda_: GAE parameter
            
        Returns:
            Advantage estimates
        """
        if values is None:
            # Simple advantage: reward - baseline
            baseline = rewards.mean()
            advantages = rewards - baseline
        else:
            # GAE computation
            advantages = torch.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                advantages[t] = delta + gamma * lambda_ * last_gae
                last_gae = advantages[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train(
        self,
        samples: List[ProcessedTrainingSample],
        validation_samples: Optional[List[ProcessedTrainingSample]] = None
    ) -> Dict[str, Any]:
        """
        Train the model using PPO.
        
        Args:
            samples: List of training samples with rewards
            validation_samples: Optional validation samples
            
        Returns:
            Training results dictionary
        """
        if not samples:
            raise ValueError("No samples provided for training")
        
        logger.info(f"Starting PPO training with {len(samples)} samples")
        
        # Create dataset
        train_dataset = PPODataset(
            samples,
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
        
        if self.use_value_function:
            value_optimizer = torch.optim.AdamW(
                self.value_head.parameters(),
                lr=self.config.learning_rate
            )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        clip_epsilon = self.config.algorithm_params.get("clip_epsilon", 0.2)
        gamma = self.config.algorithm_params.get("gamma", 0.99)
        lambda_ = self.config.algorithm_params.get("lambda", 0.95)
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # Store old policy log probs for first pass
            old_log_probs_list = []
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                prompt_ids = batch["prompt_input_ids"].to(self.device)
                response_ids = batch["response_input_ids"].to(self.device)
                rewards = batch["reward"].to(self.device)
                
                # Get log probabilities from current policy
                new_log_probs = self._get_log_probs(
                    self.model,
                    prompt_ids,
                    response_ids
                )
                
                # Store old log probs on first pass
                if epoch == 0 and batch_idx == 0:
                    old_log_probs_list = new_log_probs.detach().clone()
                else:
                    # Use stored old log probs
                    if len(old_log_probs_list) < len(new_log_probs):
                        # Extend if needed
                        old_log_probs = old_log_probs_list[:len(new_log_probs)]
                    else:
                        old_log_probs = old_log_probs_list[:len(new_log_probs)]
                
                # Compute advantages
                if self.use_value_function:
                    # Get value estimates
                    # This is simplified - in practice, would compute from model hidden states
                    values = torch.zeros_like(rewards)  # Placeholder
                    advantages = self.compute_advantages(rewards, values, gamma, lambda_)
                else:
                    advantages = self.compute_advantages(rewards)
                
                # Compute PPO loss
                loss = self.compute_ppo_loss(
                    new_log_probs,
                    old_log_probs,
                    advantages,
                    clip_epsilon
                )
                
                # Value function loss (if used)
                if self.use_value_function:
                    value_loss = torch.nn.functional.mse_loss(values, rewards)
                    loss = loss + 0.5 * value_loss
                
                # Backward pass
                optimizer.zero_grad()
                if self.use_value_function:
                    value_optimizer.zero_grad()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                optimizer.step()
                if self.use_value_function:
                    value_optimizer.step()
                
                # Update old log probs for next iteration
                old_log_probs_list = new_log_probs.detach().clone()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.num_epochs}, "
                        f"Batch {batch_idx+1}, Loss: {loss.item():.4f}, "
                        f"Avg Reward: {rewards.mean().item():.4f}"
                    )
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        results = {
            "algorithm": TrainingAlgorithm.PPO.value,
            "total_epochs": self.config.num_epochs,
            "total_batches": num_batches,
            "average_loss": avg_loss,
            "final_loss": avg_loss,
            "num_samples": len(samples),
            "average_reward": sum(s.reward for s in samples) / len(samples) if samples else 0.0
        }
        
        logger.info(f"PPO training completed. Average loss: {avg_loss:.4f}")
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
        if self.use_value_function:
            torch.save(self.value_head.state_dict(), f"{output_path}/value_head.pt")
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        if self.use_value_function:
            value_head_path = f"{model_path}/value_head.pt"
            if os.path.exists(value_head_path):
                self.value_head.load_state_dict(torch.load(value_head_path))
        logger.info(f"Model loaded from {model_path}")

