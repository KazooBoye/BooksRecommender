"""
Data collection module for gathering book data from multiple sources.
"""

import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Optional
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects book data from multiple sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BookRecommender/1.0 (Educational Project)'
        })
    
    def collect_goodreads_data(self, output_path: str) -> None:
        """
        Collect data from Goodreads dataset.
        Downloads and processes the Kaggle Goodreads 100k books dataset.
        """
        logger.info("Setting up Goodreads data collection...")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Instructions for manual download
        dataset_url = "https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k"
        download_instructions = f"""
        
        📚 GOODREADS DATASET COLLECTION INSTRUCTIONS
        ==========================================
        
        This script will help you set up the Goodreads 100k books dataset.
        
        STEP 1: Download the dataset
        ---------------------------
        1. Go to: {dataset_url}
        2. Click "Download" button (requires Kaggle account)
        3. Save the downloaded ZIP file to: {output_path}
        
        STEP 2: Extract the dataset
        ---------------------------
        The script will automatically extract the CSV file when you run it again.
        
        Expected file: GoodReads_100k_books.csv
        Dataset size: ~120MB
        Number of books: ~100,000
        
        """
        
        print(download_instructions)
        logger.info(f"Goodreads data collection setup completed. Please download manually to: {output_path}")
        
        # Check if dataset already exists
        csv_path = os.path.join(output_path, "GoodReads_100k_books.csv")
        zip_path = os.path.join(output_path, "goodreads-books-100k.zip")
        
        # Try to extract if ZIP file exists
        if os.path.exists(zip_path) and not os.path.exists(csv_path):
            try:
                logger.info("Found ZIP file, extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path)
                logger.info("✅ Dataset extracted successfully!")
            except Exception as e:
                logger.error(f"❌ Error extracting ZIP file: {e}")
        
        # Verify dataset
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, nrows=5)  # Read first 5 rows to check
                logger.info(f"✅ Dataset found and verified!")
                logger.info(f"   📊 Columns: {list(df.columns)}")
                logger.info(f"   📈 Sample shape: {df.shape}")
                
                # Save column information for preprocessing
                metadata = {
                    "source": "Kaggle Goodreads 100k",
                    "url": dataset_url,
                    "filename": "GoodReads_100k_books.csv",
                    "columns": list(df.columns),
                    "sample_data": df.head(2).to_dict('records')
                }
                
                with open(os.path.join(output_path, "dataset_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                logger.error(f"❌ Error reading dataset: {e}")
        else:
            logger.warning("⚠️  Dataset not found. Please download manually.")
    
    def collect_openlibrary_data(self, book_ids: List[str], output_path: str) -> None:
        """
        Collect book metadata from OpenLibrary API.
        """
        logger.info(f"Collecting OpenLibrary data for {len(book_ids)} books...")
        
        books_data = []
        base_url = "https://openlibrary.org/api/books"
        
        for i, book_id in enumerate(book_ids):
            try:
                # Rate limiting
                time.sleep(0.1)
                
                params = {
                    'bibkeys': f'ISBN:{book_id}',
                    'format': 'json',
                    'jscmd': 'data'
                }
                
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                if data:
                    books_data.append(data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(book_ids)} books")
                    
            except Exception as e:
                logger.error(f"Error collecting data for book {book_id}: {e}")
                continue
        
        # Save collected data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(books_data, f, indent=2)
        
        logger.info(f"Saved {len(books_data)} books to {output_path}")
    
    def collect_book_content(self, book_info: Dict, output_path: str) -> Optional[Dict]:
        """
        Collect book content from various sources.
        """
        try:
            # Try Project Gutenberg for public domain books
            if self._is_public_domain(book_info):
                content = self._get_gutenberg_content(book_info)
                if content:
                    return content
            
            # Try other sources for book summaries/descriptions
            content = self._get_book_summary(book_info)
            return content
            
        except Exception as e:
            logger.error(f"Error collecting content for {book_info.get('title', 'Unknown')}: {e}")
            return None
    
    def _is_public_domain(self, book_info: Dict) -> bool:
        """Check if book is likely in public domain."""
        pub_year = book_info.get('publication_year', 0)
        return pub_year < 1923  # Simplified public domain check
    
    def _get_gutenberg_content(self, book_info: Dict) -> Optional[Dict]:
        """Get content from Project Gutenberg."""
        # Implementation for Project Gutenberg API
        # This would search and download full text for public domain books
        logger.info(f"Attempting to get Gutenberg content for: {book_info.get('title')}")
        return None  # Placeholder
    
    def _get_book_summary(self, book_info: Dict) -> Optional[Dict]:
        """Get book summary/description from various sources."""
        # Implementation for scraping book summaries
        # Could use Google Books API, Amazon, etc.
        logger.info(f"Attempting to get summary for: {book_info.get('title')}")
        return None  # Placeholder
    
    def create_tag_vocabulary(self, books_data: List[Dict]) -> Dict[str, int]:
        """
        Create tag vocabulary from collected book data.
        """
        logger.info("Creating tag vocabulary...")
        
        tag_counts = {}
        
        for book in books_data:
            # Extract tags from genres, themes, subjects
            tags = []
            
            # From genres
            if 'genres' in book:
                if isinstance(book['genres'], str):
                    # Handle string format like "Fiction, Romance, Historical"
                    genre_list = [g.strip() for g in book['genres'].split(',')]
                    tags.extend(genre_list)
                elif isinstance(book['genres'], list):
                    tags.extend(book['genres'])
            
            # From subjects/themes
            if 'subjects' in book:
                if isinstance(book['subjects'], str):
                    subject_list = [s.strip() for s in book['subjects'].split(',')]
                    tags.extend(subject_list)
                elif isinstance(book['subjects'], list):
                    tags.extend(book['subjects'])
            
            # From review keywords (if available)
            if 'keywords' in book:
                if isinstance(book['keywords'], str):
                    keyword_list = [k.strip() for k in book['keywords'].split(',')]
                    tags.extend(keyword_list)
                elif isinstance(book['keywords'], list):
                    tags.extend(book['keywords'])
            
            # Count tag occurrences
            for tag in tags:
                if tag:
                    tag = tag.lower().strip()
                    if tag and len(tag) > 1:  # Filter out single characters
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Filter tags by minimum frequency
        min_freq = self.config.get('min_tag_frequency', 10)
        filtered_tags = {
            tag: count for tag, count in tag_counts.items() 
            if count >= min_freq
        }
        
        # Create tag-to-index mapping
        tag_vocab = {tag: idx for idx, tag in enumerate(sorted(filtered_tags.keys()))}
        
        logger.info(f"Created vocabulary with {len(tag_vocab)} tags")
        return tag_vocab
    
    def verify_goodreads_dataset(self, file_path: str) -> Dict:
        """
        Verify and analyze the downloaded Goodreads dataset.
        """
        logger.info(f"Verifying Goodreads dataset: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            # Read dataset info
            df = pd.read_csv(file_path, nrows=1000)  # Sample for analysis
            
            info = {
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                "columns": list(df.columns),
                "shape": df.shape,
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_values": {}
            }
            
            # Get sample values for each column
            for col in df.columns:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    info["sample_values"][col] = non_null_values.iloc[0]
            
            logger.info(f"✅ Dataset verified successfully!")
            logger.info(f"   📊 Shape: {info['shape']}")
            logger.info(f"   📁 Size: {info['file_size_mb']:.1f} MB")
            logger.info(f"   📋 Columns: {info['columns']}")
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Error verifying dataset: {e}")
            raise


def main():
    """Main data collection script."""
    import yaml
    
    # Load configuration
    config_path = 'configs/model_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration if config file doesn't exist
        config = {
            'data': {
                'goodreads_path': 'data/raw/goodreads/',
                'openlibrary_path': 'data/raw/openlibrary/',
                'min_tag_frequency': 10
            }
        }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Initialize collector
    collector = DataCollector(config['data'])
    
    # Collect Goodreads data
    goodreads_path = config['data']['goodreads_path']
    collector.collect_goodreads_data(goodreads_path)
    
    # Verify dataset if it exists
    csv_path = os.path.join(goodreads_path, "GoodReads_100k_books.csv")
    if os.path.exists(csv_path):
        try:
            dataset_info = collector.verify_goodreads_dataset(csv_path)
            
            # Save dataset info
            with open(os.path.join(goodreads_path, "dataset_info.json"), 'w') as f:
                json.dump(dataset_info, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Dataset verification failed: {e}")
    
    # Example OpenLibrary collection (optional)
    # sample_book_ids = ['9780261102385', '9780547928227', '9780544003415']
    # collector.collect_openlibrary_data(sample_book_ids, 'data/raw/openlibrary/books.json')
    
    logger.info("Data collection setup completed!")


if __name__ == "__main__":
    main()