#!/usr/bin/env python3
"""
Test script to measure inference performance improvements.
"""

import time
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.inference.recommend import BookRecommendationEngine

def test_inference_speed():
    """Test inference speed with different queries."""
    
    print("=== Book Recommendation Inference Speed Test ===\n")
    
    # Initialize recommendation engine
    print("Initializing recommendation engine...")
    start_time = time.time()
    
    try:
        engine = BookRecommendationEngine(
            model_path='models/best_model.pt',
            config_path='configs/model_config.yaml'
        )
        init_time = time.time() - start_time
        print(f"✓ Initialization completed in {init_time:.2f} seconds\n")
        
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Test queries
    test_queries = [
        "fantasy adventure magic",
        "science fiction space",
        "romance love story",
        "mystery detective crime",
        "historical fiction war"
    ]
    
    print("Testing inference speed with different queries...")
    print("-" * 60)
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: '{query}'")
        
        start_time = time.time()
        try:
            recommendations = engine.recommend_books(query, num_recommendations=10)
            query_time = time.time() - start_time
            total_time += query_time
            
            print(f"  ✓ Generated {len(recommendations)} recommendations in {query_time:.3f} seconds")
            if recommendations:
                print(f"  Top result: {recommendations[0].title} (score: {recommendations[0].combined_score:.3f})")
            
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
        
        print()
    
    # Summary
    avg_time = total_time / len(test_queries)
    print("-" * 60)
    print(f"Performance Summary:")
    print(f"  • Total query time: {total_time:.3f} seconds")
    print(f"  • Average per query: {avg_time:.3f} seconds")
    print(f"  • Queries per second: {1/avg_time:.1f}")
    
    if avg_time < 1.0:
        print("  🚀 Excellent performance!")
    elif avg_time < 5.0:
        print("  ✓ Good performance")
    else:
        print("  ⚠️  Performance could be improved")

if __name__ == "__main__":
    test_inference_speed()