"""
Custom transformer-based model for book semantic understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = Q.size(0)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Prepare mask for multi-head attention if provided
        if mask is not None:
            # If mask is 2D (batch_size, seq_len), convert to proper attention mask
            if mask.dim() == 2:
                # Create causal mask for attention (batch_size, seq_len, seq_len)
                attention_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
                # For each position, mask out padding positions in keys
                key_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
                attention_mask = attention_mask * key_mask
                # Expand for all heads (batch_size, num_heads, seq_len, seq_len)
                mask = attention_mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(context)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class BookSemanticEncoder(nn.Module):
    """Custom transformer encoder for book semantic understanding."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.vocab_size = config['vocab_size']
        self.d_model = config['embedding_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.d_ff = config['hidden_dim']
        self.max_seq_len = config['max_sequence_length']
        self.dropout = config['dropout']
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def create_padding_mask(self, x: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """Create padding mask for attention."""
        # Create mask where non-padding tokens are 1, padding tokens are 0
        mask = (x != pad_token).float()  # (batch_size, seq_len)
        return mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            # Convert to float if it's not already
            attention_mask = attention_mask.float()
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout_layer(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attention_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x
    
    def get_book_embedding(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get book-level embedding by pooling token embeddings."""
        # Get token embeddings
        token_embeddings = self.forward(input_ids, attention_mask)
        
        # Mean pooling over valid tokens
        if attention_mask is not None:
            mask = attention_mask.squeeze(1).squeeze(1).float()  # [batch_size, seq_len]
            masked_embeddings = token_embeddings * mask.unsqueeze(-1)
            book_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            book_embedding = token_embeddings.mean(dim=1)
        
        return book_embedding


class TagPredictionHead(nn.Module):
    """Multi-label tag prediction head."""
    
    def __init__(self, d_model: int, num_tags: int, dropout: float = 0.1):
        super().__init__()
        self.num_tags = num_tags
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_tags)
        )
        
    def forward(self, book_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict tags for books."""
        logits = self.classifier(book_embeddings)
        return logits


class BookRecommenderModel(nn.Module):
    """Complete book recommendation model."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.encoder = BookSemanticEncoder(config)
        self.tag_predictor = TagPredictionHead(
            config['embedding_dim'], 
            config['num_tags'], 
            config['dropout']
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Get book embeddings
        book_embeddings = self.encoder.get_book_embedding(input_ids, attention_mask)
        
        # Predict tags
        tag_logits = self.tag_predictor(book_embeddings)
        tag_probs = torch.sigmoid(tag_logits)
        
        return {
            'book_embeddings': book_embeddings,
            'tag_logits': tag_logits,
            'tag_probs': tag_probs
        }
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to embedding vector."""
        return self.encoder.get_book_embedding(input_ids, attention_mask)
    
    def predict_tags(self, book_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict tags from book embeddings."""
        tag_logits = self.tag_predictor(book_embeddings)
        return torch.sigmoid(tag_logits)


def create_model(config: Dict) -> BookRecommenderModel:
    """Factory function to create model."""
    return BookRecommenderModel(config)