"""
Data preprocessing module for cleaning and preparing book data.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses book data for model training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vocab = {}
        self.tag_vocab = {}
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from multiple sources."""
        logger.info("Loading raw data...")
        
        data = {}
        
        # Load Goodreads data
        goodreads_path = self.config['goodreads_path']
        if os.path.exists(goodreads_path):
            data['goodreads'] = self._load_goodreads_data(goodreads_path)
        
        # Load OpenLibrary data
        openlibrary_path = self.config['openlibrary_path']
        if os.path.exists(openlibrary_path):
            data['openlibrary'] = self._load_openlibrary_data(openlibrary_path)
        
        # Load content data
        content_path = self.config['content_path']
        if os.path.exists(content_path):
            data['content'] = self._load_content_data(content_path)
        
        logger.info(f"Loaded data from {len(data)} sources")
        return data
    
    def _load_goodreads_data(self, path: str) -> pd.DataFrame:
        """Load and parse Goodreads data from Kaggle dataset."""
        # Look for the specific Kaggle Goodreads dataset file
        kaggle_file = os.path.join(path, "GoodReads_100k_books.csv")
        
        if os.path.exists(kaggle_file):
            logger.info(f"Loading Kaggle Goodreads dataset: {kaggle_file}")
            try:
                df = pd.read_csv(kaggle_file)
                logger.info(f"Loaded {len(df)} books from Kaggle Goodreads dataset")
                
                # Log dataset structure for debugging
                logger.info(f"Dataset columns: {list(df.columns)}")
                logger.info(f"Dataset shape: {df.shape}")
                
                return df
            except Exception as e:
                logger.error(f"Error loading Kaggle dataset: {e}")
                return pd.DataFrame()
        
        # Fallback: Look for any CSV files in the directory
        if os.path.isdir(path):
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                logger.warning(f"No CSV files found in {path}")
                return pd.DataFrame()
            
            # Load the largest CSV file (likely the main dataset)
            csv_files_with_size = [(f, os.path.getsize(os.path.join(path, f))) for f in csv_files]
            largest_file = max(csv_files_with_size, key=lambda x: x[1])[0]
            
            try:
                df = pd.read_csv(os.path.join(path, largest_file))
                logger.info(f"Loaded {len(df)} books from {largest_file}")
                return df
            except Exception as e:
                logger.error(f"Error loading CSV file {largest_file}: {e}")
                return pd.DataFrame()
        
        logger.warning(f"Goodreads data path not found: {path}")
        return pd.DataFrame()
    
    def _load_openlibrary_data(self, path: str) -> pd.DataFrame:
        """Load and parse OpenLibrary data."""
        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        
        all_data = []
        for file in json_files:
            with open(os.path.join(path, file), 'r') as f:
                data = json.load(f)
                all_data.extend(data if isinstance(data, list) else [data])
        
        df = pd.DataFrame(all_data)
        logger.info(f"Loaded {len(df)} books from OpenLibrary")
        return df
    
    def _load_content_data(self, path: str) -> pd.DataFrame:
        """Load book content data."""
        # Load text content files
        content_files = [f for f in os.listdir(path) if f.endswith(('.txt', '.json'))]
        
        content_data = []
        for file in content_files:
            file_path = os.path.join(path, file)
            if file.endswith('.json'):
                with open(file_path, 'r') as f:
                    content_data.append(json.load(f))
            else:
                with open(file_path, 'r') as f:
                    content_data.append({'content': f.read(), 'source': file})
        
        df = pd.DataFrame(content_data)
        logger.info(f"Loaded content for {len(df)} books")
        return df
    
    def merge_datasets(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data from multiple sources."""
        logger.info("Merging datasets...")
        
        if 'goodreads' not in data or data['goodreads'].empty:
            raise ValueError("Goodreads data is required as primary dataset")
        
        merged_df = data['goodreads'].copy()
        
        # Standardize column names
        merged_df = self._standardize_columns(merged_df)
        
        # Merge with OpenLibrary data if available
        if 'openlibrary' in data and not data['openlibrary'].empty:
            merged_df = self._merge_openlibrary(merged_df, data['openlibrary'])
        
        # Merge with content data if available
        if 'content' in data and not data['content'].empty:
            merged_df = self._merge_content(merged_df, data['content'])
        
        logger.info(f"Merged dataset has {len(merged_df)} books")
        return merged_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across datasets for Kaggle Goodreads format."""
        # Common column mappings for Kaggle Goodreads dataset
        # Based on typical Goodreads dataset structure
        column_mapping = {
            # Book identification
            'book_id': 'book_id',
            'bookID': 'book_id',
            'id': 'book_id',
            
            # Title variations
            'title': 'title',
            'book_title': 'title',
            'name': 'title',
            
            # Author variations
            'authors': 'author',
            'author': 'author',
            'author_name': 'author',
            'writer': 'author',
            
            # Description/summary variations
            'description': 'description',
            'summary': 'description',
            'plot': 'description',
            'book_description': 'description',
            
            # Genre/category variations
            'genres': 'genre',
            'genre': 'genre',
            'categories': 'genre',
            'category': 'genre',
            'shelves': 'genre',
            
            # Rating variations
            'average_rating': 'rating',
            'avg_rating': 'rating',
            'rating': 'rating',
            'averageRating': 'rating',
            
            # Publication year variations
            'publication_year': 'publication_year',
            'published_year': 'publication_year',
            'year_published': 'publication_year',
            'publication_date': 'publication_year',
            'published_date': 'publication_year',
            'year': 'publication_year',
            
            # Pages/length
            'num_pages': 'pages',
            'pages': 'pages',
            'page_count': 'pages',
            
            # ISBN variations
            'isbn': 'isbn',
            'isbn10': 'isbn',
            'isbn13': 'isbn',
            
            # Rating count
            'ratings_count': 'ratings_count',
            'rating_count': 'ratings_count',
            'num_ratings': 'ratings_count',
            
            # Language
            'language_code': 'language',
            'language': 'language',
            'lang': 'language',
            
            # Publisher
            'publisher': 'publisher',
            'publishers': 'publisher',
        }
        
        # Create a copy to avoid modifying original
        df_standardized = df.copy()
        
        # Apply column mappings
        columns_to_rename = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df_standardized.columns and old_name != new_name:
                columns_to_rename[old_name] = new_name
        
        if columns_to_rename:
            df_standardized = df_standardized.rename(columns=columns_to_rename)
            logger.info(f"Renamed columns: {columns_to_rename}")
        
        # Log final column structure
        logger.info(f"Standardized columns: {list(df_standardized.columns)}")
        
        return df_standardized
    
    def _merge_openlibrary(self, main_df: pd.DataFrame, ol_df: pd.DataFrame) -> pd.DataFrame:
        """Merge OpenLibrary data with main dataset."""
        # Implementation would depend on the specific structure of OpenLibrary data
        # For now, just return the main dataframe
        return main_df
    
    def _merge_content(self, main_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
        """Merge content data with main dataset."""
        # Implementation would depend on how content is linked to books
        # For now, just return the main dataframe
        return main_df
    
    def clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text data."""
        logger.info("Cleaning text data...")
        
        text_columns = ['title', 'author', 'description']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text field."""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from raw data."""
        logger.info("Creating features...")
        
        # Create combined text feature
        df['combined_text'] = df.apply(self._combine_text_features, axis=1)
        
        # Create numeric features
        df = self._create_numeric_features(df)
        
        # Create categorical features
        df = self._create_categorical_features(df)
        
        return df
    
    def _combine_text_features(self, row: pd.Series) -> str:
        """Combine multiple text fields into single feature."""
        parts = []
        
        # Add title
        if pd.notna(row.get('title')):
            parts.append(str(row['title']))
        
        # Add author
        if pd.notna(row.get('author')):
            parts.append(str(row['author']))
        
        # Add description
        if pd.notna(row.get('description')):
            parts.append(str(row['description']))
        
        # Add genre
        if pd.notna(row.get('genre')):
            parts.append(str(row['genre']))
        
        return ' '.join(parts)
    
    def _create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric features."""
        # Publication year processing
        if 'publication_year' in df.columns:
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
            df['publication_decade'] = (df['publication_year'] // 10) * 10
        
        # Rating processing
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating_category'] = pd.cut(df['rating'], bins=[0, 2, 3, 4, 5], labels=['low', 'medium', 'good', 'excellent'])
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical features."""
        # Process genres
        if 'genre' in df.columns:
            df['primary_genre'] = df['genre'].apply(self._extract_primary_genre)
        
        return df
    
    def _extract_primary_genre(self, genre_str) -> str:
        """Extract primary genre from genre string with better handling."""
        if pd.isna(genre_str) or genre_str == '':
            return 'unknown'
        
        try:
            # Convert to string
            genre_text = str(genre_str).lower().strip()
            
            # Handle empty strings
            if not genre_text or genre_text == 'nan':
                return 'unknown'
            
            # Handle different separators
            separators = [',', ';', '|', '/', '\\', ' > ']
            genres = [genre_text]
            
            for sep in separators:
                if sep in genre_text:
                    genres = [g.strip() for g in genre_text.split(sep)]
                    break
            
            # Get the first non-empty genre
            for genre in genres:
                genre = genre.strip()
                if genre and len(genre) > 1:
                    # Clean up common prefixes/suffixes
                    genre = re.sub(r'^(genre|category|shelf):\s*', '', genre)
                    genre = re.sub(r'\s*\(.*\)$', '', genre)  # Remove parenthetical notes
                    
                    # Return cleaned genre
                    return genre.strip() if genre.strip() else 'unknown'
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Error processing genre '{genre_str}': {e}")
            return 'unknown'
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on quality criteria."""
        logger.info("Filtering data...")
        
        initial_count = len(df)
        df_filtered = df.copy()
        
        # Remove books without titles
        if 'title' in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=['title'])
            df_filtered = df_filtered[df_filtered['title'].str.strip() != '']
            logger.info(f"After title filter: {len(df_filtered)} books")
        
        # Handle description filtering more carefully
        if 'description' in df_filtered.columns:
            # Remove books without descriptions
            df_filtered = df_filtered.dropna(subset=['description'])
            
            # Filter by minimum description length
            min_desc_length = 50
            desc_lengths = df_filtered['description'].astype(str).str.len()
            df_filtered = df_filtered[desc_lengths >= min_desc_length]
            logger.info(f"After description filter: {len(df_filtered)} books")
        else:
            logger.warning("No 'description' column found - skipping description filter")
        
        # Filter by publication year more carefully
        if 'publication_year' in df_filtered.columns:
            # Convert to numeric, handling various formats
            pub_year_numeric = pd.to_numeric(df_filtered['publication_year'], errors='coerce')
            df_filtered = df_filtered.copy()
            df_filtered['publication_year'] = pub_year_numeric
            
            # Filter by reasonable year range
            year_mask = (df_filtered['publication_year'] >= 1800) & (df_filtered['publication_year'] <= 2025)
            df_filtered = df_filtered[year_mask]
            logger.info(f"After publication year filter: {len(df_filtered)} books")
        else:
            logger.warning("No 'publication_year' column found - skipping year filter")
        
        # Filter by rating if available
        if 'rating' in df_filtered.columns:
            # Remove books with invalid ratings
            rating_numeric = pd.to_numeric(df_filtered['rating'], errors='coerce')
            df_filtered = df_filtered.copy()
            df_filtered['rating'] = rating_numeric
            
            # Keep books with ratings between 0 and 5
            rating_mask = (df_filtered['rating'] >= 0) & (df_filtered['rating'] <= 5)
            df_filtered = df_filtered[rating_mask]
            logger.info(f"After rating filter: {len(df_filtered)} books")
        
        # Remove completely empty rows
        df_filtered = df_filtered.dropna(how='all')
        
        final_count = len(df_filtered)
        removed_count = initial_count - final_count
        logger.info(f"Filtered from {initial_count} to {final_count} books ({removed_count} removed)")
        
        return df_filtered
    
    def create_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits."""
        logger.info("Creating train/test splits...")
        
        train_size = self.config['train_split']
        val_size = self.config['val_split']
        test_size = self.config['test_split']
        
        # Prepare stratification column
        stratify_col = None
        if 'primary_genre' in df.columns:
            # Check genre distribution and group rare genres
            genre_counts = df['primary_genre'].value_counts()
            min_samples_per_class = 2  # Minimum samples needed for stratification
            
            # Group genres with fewer than min_samples_per_class into 'other'
            rare_genres = genre_counts[genre_counts < min_samples_per_class].index
            if len(rare_genres) > 0:
                logger.info(f"Grouping {len(rare_genres)} rare genres into 'other' category")
                df_copy = df.copy()
                df_copy.loc[df_copy['primary_genre'].isin(rare_genres), 'primary_genre'] = 'other'
                stratify_col = df_copy['primary_genre']
            else:
                stratify_col = df['primary_genre']
                
            logger.info(f"Genre distribution for stratification: {stratify_col.value_counts().head(10).to_dict()}")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_size + test_size),
            random_state=42,
            stratify=stratify_col
        )
        
        # Second split: val vs test
        # Use the same stratification logic for the temp_df
        temp_stratify_col = None
        if 'primary_genre' in temp_df.columns:
            temp_genre_counts = temp_df['primary_genre'].value_counts()
            temp_rare_genres = temp_genre_counts[temp_genre_counts < 2].index
            if len(temp_rare_genres) > 0:
                temp_df_copy = temp_df.copy()
                temp_df_copy.loc[temp_df_copy['primary_genre'].isin(temp_rare_genres), 'primary_genre'] = 'other'
                temp_stratify_col = temp_df_copy['primary_genre']
            else:
                temp_stratify_col = temp_df['primary_genre']
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(test_size / (val_size + test_size)),
            random_state=42,
            stratify=temp_stratify_col
        )
        
        logger.info(f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save processed data to disk."""
        logger.info("Saving processed data...")
        
        output_dir = self.config['processed_path']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        # Save metadata
        metadata = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'features': list(train_df.columns),
            'config': self.config
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")


def main():
    """Main preprocessing script."""
    import yaml
    
    # Load configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config['data'])
    
    # Load and process data
    raw_data = preprocessor.load_raw_data()
    merged_df = preprocessor.merge_datasets(raw_data)
    cleaned_df = preprocessor.clean_text_data(merged_df)
    featured_df = preprocessor.create_features(cleaned_df)
    filtered_df = preprocessor.filter_data(featured_df)
    
    # Create splits and save
    train_df, val_df, test_df = preprocessor.create_train_test_split(filtered_df)
    preprocessor.save_processed_data(train_df, val_df, test_df)
    
    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()