"""
DPO (Direct Preference Optimization) Trainer

Implements DPO algorithm for training agents using preference pairs.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np

from ...models.training import TrainingConfig, TrainingAlgorithm
from ..data_processor import PreferencePair

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """Dataset for DPO preference pairs"""
    
    def __init__(
        self,
        preference_pairs: List[PreferencePair],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        
        # Tokenize prompt
        prompt_encoded = self.tokenizer(
            pair.prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize chosen response
        chosen_encoded = self.tokenizer(
            pair.chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_encoded = self.tokenizer(
            pair.rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt_input_ids": prompt_encoded["input_ids"].squeeze(),
            "prompt_attention_mask": prompt_encoded["attention_mask"].squeeze(),
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(),
            "metadata": pair.metadata
        }


class DPOTrainer:
    """
    DPO Trainer for agent learning.
    
    Implements Direct Preference Optimization algorithm for training
    agents using preference pairs (chosen vs rejected responses).
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize DPO trainer.
        
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
        
        # Reference model for DPO (frozen copy)
        self.reference_model = None
        
        logger.info(f"DPOTrainer initialized with model: {config.base_model}")
    
    def _create_reference_model(self):
        """Create a frozen reference model for DPO"""
        if self.reference_model is None:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.reference_model.to(self.device)
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False
            logger.info("Created frozen reference model for DPO")
    
    def compute_dpo_loss(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        beta: float = 0.1
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Args:
            policy_logps: Log probabilities from policy model
            reference_logps: Log probabilities from reference model
            beta: Temperature parameter
            
        Returns:
            DPO loss
        """
        # DPO loss: -log(sigma(beta * (log_p_policy - log_p_ref)))
        log_ratios = policy_logps - reference_logps
        loss = -torch.nn.functional.logsigmoid(beta * log_ratios).mean()
        return loss
    
    def train(
        self,
        preference_pairs: List[PreferencePair],
        validation_pairs: Optional[List[PreferencePair]] = None
    ) -> Dict[str, Any]:
        """
        Train the model using DPO.
        
        Args:
            preference_pairs: List of preference pairs for training
            validation_pairs: Optional validation pairs
            
        Returns:
            Training results dictionary
        """
        if not preference_pairs:
            raise ValueError("No preference pairs provided for training")
        
        logger.info(f"Starting DPO training with {len(preference_pairs)} preference pairs")
        
        # Create reference model
        self._create_reference_model()
        
        # Create dataset
        train_dataset = PreferenceDataset(
            preference_pairs,
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
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                
                # Get log probabilities from policy model
                policy_chosen_logps = self._get_log_probs(
                    self.model,
                    prompt_ids,
                    chosen_ids
                )
                policy_rejected_logps = self._get_log_probs(
                    self.model,
                    prompt_ids,
                    rejected_ids
                )
                
                # Get log probabilities from reference model
                ref_chosen_logps = self._get_log_probs(
                    self.reference_model,
                    prompt_ids,
                    chosen_ids
                )
                ref_rejected_logps = self._get_log_probs(
                    self.reference_model,
                    prompt_ids,
                    rejected_ids
                )
                
                # Compute DPO loss
                chosen_loss = self.compute_dpo_loss(
                    policy_chosen_logps,
                    ref_chosen_logps,
                    beta
                )
                rejected_loss = self.compute_dpo_loss(
                    policy_rejected_logps,
                    ref_rejected_logps,
                    beta
                )
                
                # Combined loss (prefer chosen, avoid rejected)
                loss = chosen_loss - rejected_loss
                
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
            "algorithm": TrainingAlgorithm.DPO.value,
            "total_epochs": self.config.num_epochs,
            "total_batches": num_batches,
            "average_loss": avg_loss,
            "final_loss": avg_loss,
            "num_preference_pairs": len(preference_pairs)
        }
        
        logger.info(f"DPO training completed. Average loss: {avg_loss:.4f}")
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
        with torch.no_grad():
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

