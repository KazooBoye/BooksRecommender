"""
Training script for the book recommendation model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import yaml
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model import BookRecommenderModel, create_model
from src.utils.tokenizer import BookTokenizer
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BookDataset(Dataset):
    """Dataset for book recommendation training."""
    
    def __init__(self, data_path: str, tokenizer: BookTokenizer, config: Dict):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config['max_sequence_length']
        
        # Load data
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} books from {data_path}")
        
        # Prepare tags
        self.tag_vocab = self._load_tag_vocab()
        
    def _load_tag_vocab(self) -> Dict[str, int]:
        """Load tag vocabulary."""
        tag_vocab_path = os.path.join(self.config['processed_path'], 'tag_vocab.json')
        if os.path.exists(tag_vocab_path):
            with open(tag_vocab_path, 'r') as f:
                return json.load(f)
        else:
            # Create tag vocabulary from data
            return self._create_tag_vocab()
    
    def _create_tag_vocab(self) -> Dict[str, int]:
        """Create tag vocabulary from data."""
        all_tags = set()
        
        for _, row in self.data.iterrows():
            if pd.notna(row.get('genre')):
                tags = str(row['genre']).split(',')
                all_tags.update([tag.strip().lower() for tag in tags])
        
        tag_vocab = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        
        # Save vocabulary
        tag_vocab_path = os.path.join(self.config['processed_path'], 'tag_vocab.json')
        os.makedirs(os.path.dirname(tag_vocab_path), exist_ok=True)
        with open(tag_vocab_path, 'w') as f:
            json.dump(tag_vocab, f, indent=2)
        
        return tag_vocab
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Get text
        text = row.get('combined_text', '')
        if pd.isna(text):
            text = f"{row.get('title', '')} {row.get('author', '')} {row.get('description', '')}"
        
        # Tokenize
        tokens = self.tokenizer.encode(str(text), max_length=self.max_length)
        
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
        
        # Get tags
        tag_vector = self._get_tag_vector(row)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'tags': torch.tensor(tag_vector, dtype=torch.float),
            'book_id': idx
        }
    
    def _get_tag_vector(self, row: pd.Series) -> List[float]:
        """Convert book tags to binary vector."""
        tag_vector = [0.0] * len(self.tag_vocab)
        
        if pd.notna(row.get('genre')):
            tags = str(row['genre']).split(',')
            for tag in tags:
                tag = tag.strip().lower()
                if tag in self.tag_vocab:
                    tag_vector[self.tag_vocab[tag]] = 1.0
        
        return tag_vector


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similar books."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Book embeddings [batch_size, embedding_dim]
            labels: Book similarity labels [batch_size, batch_size]
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks for positive and negative pairs
        batch_size = embeddings.size(0)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        
        # Remove diagonal (self-similarity)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute contrastive loss
        pos_pairs = similarity_matrix[labels == 1]
        neg_pairs = similarity_matrix[labels == 0]
        
        if len(pos_pairs) > 0 and len(neg_pairs) > 0:
            pos_loss = -torch.log(torch.sigmoid(pos_pairs)).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_pairs)).mean()
            loss = pos_loss + neg_loss
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


class BookTrainer:
    """Trainer for book recommendation model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize tokenizer
        self.tokenizer = BookTokenizer(config)
        
        # Create datasets
        self.train_dataset = BookDataset(
            os.path.join(config['data']['processed_path'], 'train.csv'),
            self.tokenizer,
            config['data']
        )
        
        self.val_dataset = BookDataset(
            os.path.join(config['data']['processed_path'], 'val.csv'),
            self.tokenizer,
            config['data']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )
        
        # Create model
        model_config = {**config['model'], 'vocab_size': len(self.tokenizer.vocab)}
        self.model = create_model(model_config).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Create scheduler
        total_steps = len(self.train_loader) * config['training']['num_epochs']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Loss functions
        self.tag_loss_fn = nn.BCEWithLogitsLoss()
        self.contrastive_loss_fn = ContrastiveLoss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_tag_loss = 0.0
        total_contrastive_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch['input_ids'], batch['attention_mask'])
            
            # Tag prediction loss
            tag_loss = self.tag_loss_fn(outputs['tag_logits'], batch['tags'])
            
            # Contrastive loss (simplified - using tag similarity)
            tag_similarity = self._compute_tag_similarity(batch['tags'])
            contrastive_loss = self.contrastive_loss_fn(outputs['book_embeddings'], tag_similarity)
            
            # Combined loss
            loss = (self.config['training']['tag_prediction_loss_weight'] * tag_loss + 
                   self.config['training']['contrastive_loss_weight'] * contrastive_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clipping']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_tag_loss += tag_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tag_loss': f"{tag_loss.item():.4f}",
                'cont_loss': f"{contrastive_loss.item():.4f}"
            })
            
            self.global_step += 1
        
        # Return epoch metrics
        num_batches = len(self.train_loader)
        return {
            'train_loss': total_loss / num_batches,
            'train_tag_loss': total_tag_loss / num_batches,
            'train_contrastive_loss': total_contrastive_loss / num_batches,
        }
    
    def _compute_tag_similarity(self, tags: torch.Tensor) -> torch.Tensor:
        """Compute tag-based similarity matrix."""
        # Compute cosine similarity between tag vectors
        tags_norm = F.normalize(tags, p=2, dim=1)
        similarity = torch.matmul(tags_norm, tags_norm.T)
        
        # Convert to binary similarity (threshold at 0.5)
        return (similarity > 0.5).float()
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_tag_loss = 0.0
        all_tag_preds = []
        all_tag_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['input_ids'], batch['attention_mask'])
                
                # Tag prediction loss
                tag_loss = self.tag_loss_fn(outputs['tag_logits'], batch['tags'])
                
                # Update metrics
                total_loss += tag_loss.item()
                total_tag_loss += tag_loss.item()
                
                # Collect predictions for metrics
                all_tag_preds.append(outputs['tag_probs'].cpu())
                all_tag_targets.append(batch['tags'].cpu())
        
        # Compute validation metrics
        all_tag_preds = torch.cat(all_tag_preds, dim=0)
        all_tag_targets = torch.cat(all_tag_targets, dim=0)
        
        tag_metrics = compute_metrics(all_tag_preds, all_tag_targets)
        
        num_batches = len(self.val_loader)
        return {
            'val_loss': total_loss / num_batches,
            'val_tag_loss': total_tag_loss / num_batches,
            **tag_metrics
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join('models', f'checkpoint_epoch_{self.current_epoch}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join('models', 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved with validation loss: {metrics['val_loss']:.4f}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            logger.info(f"Epoch {epoch}: {all_metrics}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(all_metrics, is_best)
        
        logger.info("Training completed!")


def main():
    """Main training script."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train book recommendation model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create trainer and start training
    trainer = BookTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()