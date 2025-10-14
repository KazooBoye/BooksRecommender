"""
Custom tokenizer for book text data.
"""

import os
import json
import re
from typing import List, Dict, Optional, Union
from collections import Counter
import pickle


class BookTokenizer:
    """Custom tokenizer for book recommendation system."""
    
    def __init__(self, config: Dict, vocab_path: Optional[str] = None):
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        self.min_frequency = config.get('min_word_frequency', 5)
        self.max_length = config.get('max_sequence_length', 512)
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        
        # Token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        
        # Vocabulary
        self.vocab = {}
        self.id_to_token = {}
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens."""
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """Build vocabulary from text corpus."""
        print("Building vocabulary...")
        
        # Tokenize all texts and count words
        word_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize_text(text)
            word_counts.update(tokens)
        
        # Filter by frequency and select top words
        filtered_words = [
            word for word, count in word_counts.items() 
            if count >= self.min_frequency
        ]
        
        # Sort by frequency (most common first)
        filtered_words = sorted(filtered_words, key=word_counts.get, reverse=True)
        
        # Keep only top vocab_size - 4 words (excluding special tokens)
        max_vocab = self.vocab_size - len(self.vocab)
        filtered_words = filtered_words[:max_vocab]
        
        # Add words to vocabulary
        for word in filtered_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        
        # Save vocabulary if path provided
        if save_path:
            self.save_vocab(save_path)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Simple word tokenization (split on whitespace and punctuation)
        # Keep important punctuation as separate tokens
        text = re.sub(r'([.!?;:,])', r' \1 ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out empty tokens and very short tokens
        tokens = [token.strip() for token in tokens if len(token.strip()) > 0]
        
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               add_special_tokens: bool = True, 
               padding: bool = True, 
               truncation: bool = True) -> List[int]:
        """Encode text to token IDs."""
        
        if max_length is None:
            max_length = self.max_length
        
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            if add_special_tokens:
                # Keep CLS and SEP tokens
                tokens = tokens[:max_length-1] + [self.sep_token]
            else:
                tokens = tokens[:max_length]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        
        # Pad if necessary
        if padding and len(token_ids) < max_length:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in [self.pad_token, self.cls_token, self.sep_token]:
                    continue
                
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        # Join tokens back to text
        text = ' '.join(tokens)
        
        # Clean up punctuation spacing
        text = re.sub(r' ([.!?;:,])', r'\1', text)
        
        return text.strip()
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    add_special_tokens: bool = True,
                    padding: bool = True,
                    truncation: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""
        return [
            self.encode(text, max_length, add_special_tokens, padding, truncation)
            for text in texts
        ]
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'config': {
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'max_length': self.max_length,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'cls_token': self.cls_token,
                'sep_token': self.sep_token,
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as pickle for faster loading
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        # Also save as JSON for human readability
        json_path = path.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        try:
            # Try to load pickle first (faster)
            with open(path, 'rb') as f:
                vocab_data = pickle.load(f)
        except:
            # Fall back to JSON
            json_path = path.replace('.pkl', '.json')
            with open(json_path, 'r') as f:
                vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.id_to_token = vocab_data['id_to_token']
        
        # Update config
        config = vocab_data.get('config', {})
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        print(f"Vocabulary loaded from {path} ({len(self.vocab)} tokens)")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.unk_token_id)
    
    def id_to_token_fn(self, token_id: int) -> str:
        """Convert ID to token."""
        return self.id_to_token.get(token_id, self.unk_token)


def create_tokenizer_from_data(data_path: str, config: Dict, save_path: str) -> BookTokenizer:
    """Create tokenizer from training data."""
    import pandas as pd
    
    # Load training data
    df = pd.read_csv(data_path)
    
    # Extract all text
    texts = []
    text_columns = ['title', 'author', 'description', 'combined_text']
    
    for col in text_columns:
        if col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())
    
    # Create tokenizer
    tokenizer = BookTokenizer(config)
    
    # Build vocabulary
    tokenizer.build_vocab(texts, save_path)
    
    return tokenizer


if __name__ == "__main__":
    # Example usage
    config = {
        'vocab_size': 50000,
        'min_word_frequency': 5,
        'max_sequence_length': 512
    }
    
    # Create tokenizer from training data
    tokenizer = create_tokenizer_from_data(
        'data/processed/train.csv',
        config,
        'data/processed/tokenizer.pkl'
    )