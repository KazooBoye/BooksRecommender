"""
Inference module for book recommendations.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from dataclasses import dataclass
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.model import BookRecommenderModel
from src.utils.tokenizer import BookTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Result of a book recommendation."""
    book_id: str
    title: str
    author: str
    description: str
    similarity_score: float
    tag_scores: Dict[str, float]
    combined_score: float


class BookRecommendationEngine:
    """Engine for generating book recommendations."""
    
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            import yaml
            self.config = yaml.safe_load(f)
        
        # Load tokenizer
        tokenizer_path = os.path.join(
            self.config['data']['processed_path'], 
            'tokenizer.pkl'
        )
        self.tokenizer = BookTokenizer(self.config['data'], tokenizer_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load book database
        self.book_database = self._load_book_database()
        self.book_embeddings = self._load_book_embeddings()
        
        # Load tag vocabulary
        self.tag_vocab = self._load_tag_vocab()
        self.id_to_tag = {v: k for k, v in self.tag_vocab.items()}
        
        logger.info(f"Recommendation engine initialized with {len(self.book_database)} books")
    
    def _load_model(self, model_path: str) -> BookRecommenderModel:
        """Load trained model."""
        # Simple approach: just load with weights_only=False since we trust our own model files
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with proper configuration
        model_config = checkpoint['config']['model']
        model_config['vocab_size'] = len(self.tokenizer.vocab)
        
        from src.models.model import create_model
        model = create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _load_book_database(self) -> pd.DataFrame:
        """Load book database."""
        # Try to load from processed data first
        processed_path = os.path.join(self.config['data']['processed_path'], 'all_books.csv')
        
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
        else:
            # Fall back to train data (in practice, you'd have a separate book database)
            train_path = os.path.join(self.config['data']['processed_path'], 'train.csv')
            val_path = os.path.join(self.config['data']['processed_path'], 'val.csv')
            test_path = os.path.join(self.config['data']['processed_path'], 'test.csv')
            
            dfs = []
            for path in [train_path, val_path, test_path]:
                if os.path.exists(path):
                    dfs.append(pd.read_csv(path))
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                # Save combined database
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                df.to_csv(processed_path, index=False)
            else:
                raise FileNotFoundError("No book data found")
        
        # Add book IDs if not present
        if 'book_id' not in df.columns:
            df['book_id'] = df.index.astype(str)
        
        return df
    
    def _load_book_embeddings(self) -> Optional[torch.Tensor]:
        """Load pre-computed book embeddings."""
        embeddings_path = os.path.join('models', 'book_embeddings.pt')
        
        if os.path.exists(embeddings_path):
            embeddings = torch.load(embeddings_path, map_location=self.device)
            logger.info(f"Loaded pre-computed embeddings for {embeddings.shape[0]} books")
            return embeddings
        else:
            logger.info("No pre-computed embeddings found. Will compute on-the-fly.")
            return None
    
    def _load_tag_vocab(self) -> Dict[str, int]:
        """Load tag vocabulary."""
        tag_vocab_path = os.path.join(self.config['data']['processed_path'], 'tag_vocab.json')
        
        if os.path.exists(tag_vocab_path):
            with open(tag_vocab_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning("Tag vocabulary not found. Using empty vocabulary.")
            return {}
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode user query to embedding."""
        # Tokenize query
        tokens = self.tokenizer.encode(query)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        
        # Encode with model
        with torch.no_grad():
            query_embedding = self.model.encode_text(input_ids, attention_mask)
        
        return query_embedding.squeeze(0)  # Remove batch dimension
    
    def get_book_embedding(self, book_idx: int) -> torch.Tensor:
        """Get embedding for a book."""
        if self.book_embeddings is not None and book_idx < len(self.book_embeddings):
            return self.book_embeddings[book_idx]
        else:
            # Compute embedding on-the-fly
            book_data = self.book_database.iloc[book_idx]
            text = book_data.get('combined_text', '')
            if pd.isna(text):
                text = f"{book_data.get('title', '')} {book_data.get('author', '')} {book_data.get('description', '')}"
            
            tokens = self.tokenizer.encode(str(text))
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
            
            with torch.no_grad():
                book_embedding = self.model.encode_text(input_ids, attention_mask)
            
            return book_embedding.squeeze(0)
    
    def compute_similarity_scores(self, query_embedding: torch.Tensor) -> np.ndarray:
        """Compute similarity scores between query and all books."""
        num_books = len(self.book_database)
        similarities = np.zeros(num_books)
        
        query_embedding = query_embedding.cpu().numpy()
        query_norm = np.linalg.norm(query_embedding)
        
        for i in range(num_books):
            book_embedding = self.get_book_embedding(i).cpu().numpy()
            book_norm = np.linalg.norm(book_embedding)
            
            if query_norm > 0 and book_norm > 0:
                # Cosine similarity
                similarity = np.dot(query_embedding, book_embedding) / (query_norm * book_norm)
                similarities[i] = similarity
        
        return similarities
    
    def compute_tag_scores(self, query: str) -> Dict[str, float]:
        """Compute tag relevance scores for query."""
        # Encode query
        tokens = self.tokenizer.encode(query)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        
        # Predict tags
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            tag_probs = outputs['tag_probs'].squeeze(0).cpu().numpy()
        
        # Convert to tag scores
        tag_scores = {}
        for tag_id, prob in enumerate(tag_probs):
            if tag_id in self.id_to_tag:
                tag_name = self.id_to_tag[tag_id]
                tag_scores[tag_name] = float(prob)
        
        return tag_scores
    
    def match_books_by_tags(self, query_tag_scores: Dict[str, float], top_k: int = 100) -> List[int]:
        """Find books that match query tags."""
        book_scores = []
        
        for idx, book_data in self.book_database.iterrows():
            # Get book tags
            book_tags = []
            if pd.notna(book_data.get('genre')):
                book_tags = [tag.strip().lower() for tag in str(book_data['genre']).split(',')]
            
            # Compute tag overlap score
            tag_score = 0.0
            for tag in book_tags:
                if tag in query_tag_scores:
                    tag_score += query_tag_scores[tag]
            
            if tag_score > 0:
                book_scores.append((idx, tag_score))
        
        # Sort by tag score and return top-k
        book_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in book_scores[:top_k]]
    
    def recommend_books(self, query: str, num_recommendations: int = 10, 
                       use_tags: bool = True, use_semantic: bool = True) -> List[RecommendationResult]:
        """Generate book recommendations for a query."""
        logger.info(f"Generating recommendations for query: '{query}'")
        
        recommendations = []
        
        # Get semantic similarity scores
        semantic_scores = None
        if use_semantic:
            query_embedding = self.encode_query(query)
            semantic_scores = self.compute_similarity_scores(query_embedding)
        
        # Get tag-based scores
        tag_scores = None
        tag_candidates = None
        if use_tags and self.tag_vocab:
            tag_scores = self.compute_tag_scores(query)
            tag_candidates = self.match_books_by_tags(tag_scores, top_k=min(500, len(self.book_database)))
        
        # Determine candidate books
        if tag_candidates is not None:
            candidate_indices = tag_candidates
        else:
            candidate_indices = list(range(len(self.book_database)))
        
        # Score all candidates
        scored_books = []
        
        for book_idx in candidate_indices:
            book_data = self.book_database.iloc[book_idx]
            
            # Get scores
            semantic_score = semantic_scores[book_idx] if semantic_scores is not None else 0.0
            
            # Get book tag scores
            book_tag_scores = {}
            book_tags = []
            if pd.notna(book_data.get('genre')):
                book_tags = [tag.strip().lower() for tag in str(book_data['genre']).split(',')]
            
            tag_match_score = 0.0
            if tag_scores:
                for tag in book_tags:
                    if tag in tag_scores:
                        book_tag_scores[tag] = tag_scores[tag]
                        tag_match_score += tag_scores[tag]
                
                # Normalize by number of tags
                if len(book_tags) > 0:
                    tag_match_score /= len(book_tags)
            
            # Combine scores
            config = self.config.get('inference', {})
            semantic_weight = config.get('semantic_weight', 0.7)
            tag_weight = config.get('tag_weight', 0.3)
            
            combined_score = (semantic_weight * semantic_score + 
                            tag_weight * tag_match_score)
            
            # Create recommendation result
            result = RecommendationResult(
                book_id=str(book_data.get('book_id', book_idx)),
                title=str(book_data.get('title', 'Unknown Title')),
                author=str(book_data.get('author', 'Unknown Author')),
                description=str(book_data.get('description', ''))[:200] + '...',
                similarity_score=float(semantic_score),
                tag_scores=book_tag_scores,
                combined_score=float(combined_score)
            )
            
            scored_books.append(result)
        
        # Sort by combined score and return top recommendations
        scored_books.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply diversity filtering
        final_recommendations = self._apply_diversity_filter(
            scored_books, 
            num_recommendations,
            diversity_weight=config.get('diversity_weight', 0.2)
        )
        
        logger.info(f"Generated {len(final_recommendations)} recommendations")
        return final_recommendations
    
    def _apply_diversity_filter(self, recommendations: List[RecommendationResult], 
                               num_recommendations: int, diversity_weight: float = 0.2) -> List[RecommendationResult]:
        """Apply diversity filtering to recommendations."""
        if len(recommendations) <= num_recommendations:
            return recommendations
        
        # Simple diversity: avoid recommending multiple books by same author
        selected = []
        seen_authors = set()
        
        # First pass: select diverse authors
        for rec in recommendations:
            author = rec.author.lower()
            if author not in seen_authors:
                selected.append(rec)
                seen_authors.add(author)
                
                if len(selected) >= num_recommendations:
                    break
        
        # Second pass: fill remaining slots with highest scores
        if len(selected) < num_recommendations:
            remaining_recs = [rec for rec in recommendations if rec not in selected]
            selected.extend(remaining_recs[:num_recommendations - len(selected)])
        
        return selected[:num_recommendations]
    
    def compute_book_embeddings(self, save_path: str = None) -> torch.Tensor:
        """Pre-compute embeddings for all books in database."""
        logger.info("Computing embeddings for all books...")
        
        num_books = len(self.book_database)
        embedding_dim = self.config['model']['embedding_dim']
        embeddings = torch.zeros(num_books, embedding_dim)
        
        batch_size = 32
        for i in range(0, num_books, batch_size):
            batch_end = min(i + batch_size, num_books)
            batch_texts = []
            
            for j in range(i, batch_end):
                book_data = self.book_database.iloc[j]
                text = book_data.get('combined_text', '')
                if pd.isna(text):
                    text = f"{book_data.get('title', '')} {book_data.get('author', '')} {book_data.get('description', '')}"
                batch_texts.append(str(text))
            
            # Tokenize batch
            batch_tokens = [self.tokenizer.encode(text) for text in batch_texts]
            max_len = max(len(tokens) for tokens in batch_tokens)
            
            # Pad sequences
            padded_tokens = []
            attention_masks = []
            for tokens in batch_tokens:
                padded = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
                mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
                padded_tokens.append(padded)
                attention_masks.append(mask)
            
            # Convert to tensors
            input_ids = torch.tensor(padded_tokens, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
            
            # Compute embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_text(input_ids, attention_mask)
                embeddings[i:batch_end] = batch_embeddings.cpu()
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {batch_end}/{num_books} books")
        
        # Save embeddings
        if save_path is None:
            save_path = os.path.join('models', 'book_embeddings.pt')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
        logger.info(f"Saved embeddings to {save_path}")
        
        return embeddings


def main():
    """Main inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate book recommendations')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='Model path')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Config path')
    parser.add_argument('--num_recs', type=int, default=10, help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create recommendation engine
    engine = BookRecommendationEngine(args.model, args.config)
    
    # Generate recommendations
    recommendations = engine.recommend_books(args.query, args.num_recs)
    
    # Print results
    print(f"\nRecommendations for: '{args.query}'\n")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.title}")
        print(f"   Author: {rec.author}")
        print(f"   Score: {rec.combined_score:.3f} (semantic: {rec.similarity_score:.3f})")
        print(f"   Description: {rec.description}")
        if rec.tag_scores:
            top_tags = sorted(rec.tag_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top Tags: {', '.join([f'{tag}({score:.2f})' for tag, score in top_tags])}")
        print()


if __name__ == "__main__":
    main()