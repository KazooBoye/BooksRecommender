# Development Log

## Project Initialization - October 14, 2025

### Project Setup
- Created initial project structure following ML project guidelines
- Established folder hierarchy: data/, notebooks/, src/, models/, results/, configs/
- Defined system architecture for semantic book recommendation system

### System Architecture Design
- **Approach**: Two-stage semantic model with tag prediction
- **Stage 1**: Custom transformer-based book content encoder
- **Stage 2**: Multi-label tag prediction network  
- **Stage 3**: Query-book matching with similarity scoring

### Technical Decisions
- Decision to train custom model from scratch instead of using pretrained models
- Multi-source data strategy: Goodreads + OpenLibrary + content scraping
- Semantic embedding approach combined with tag-based matching
- Support for various input types: titles, authors, descriptions

### Implementation Progress
- **Core Architecture**: Implemented custom transformer model with multi-head attention
- **Data Pipeline**: Created data collection and preprocessing modules
- **Training Framework**: Developed training loop with contrastive and tag prediction losses
- **Tokenization**: Custom tokenizer for book text data with vocabulary management
- **Inference Engine**: Recommendation system with similarity scoring and diversity filtering
- **Evaluation**: Comprehensive metrics for tag prediction and embedding quality

### Dataset Integration - October 14, 2025

#### Updated Data Collection
- **New Dataset**: Integrated Kaggle Goodreads 100k books dataset
- **Source**: https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k
- **Features**: ~100,000 books with comprehensive metadata
- **Format**: Single CSV file (~120MB) with 13 columns
- **Automated Setup**: Script provides download instructions and auto-extraction

#### Enhanced Data Preprocessing
- **Flexible Column Mapping**: Handles various Goodreads column name formats
- **Robust Data Filtering**: Better handling of missing/invalid data
- **Genre Processing**: Enhanced genre extraction with multiple separator support
- **Data Validation**: Comprehensive error handling and logging
- **Sample Data Generation**: Fallback sample data for testing

#### Key Improvements
1. **Dataset Compatibility**: 
   - Supports standard Goodreads CSV format
   - Handles common column variations (authors/author, average_rating/rating, etc.)
   - Flexible file detection and loading

2. **Data Quality Assurance**:
   - Validates data types and ranges
   - Filters invalid entries (missing titles, too short descriptions)
   - Handles publication year and rating validation

3. **Enhanced Genre Processing**:
   - Supports multiple separators (comma, semicolon, pipe, etc.)
   - Cleans genre names and removes common prefixes
   - Extracts primary genre for stratified sampling

4. **Robust Error Handling**:
   - Graceful handling of missing columns
   - Detailed logging for debugging
   - Fallback mechanisms for corrupted data

### Files Created
- `src/models/model.py`: Custom transformer architecture for book encoding
- `src/models/train.py`: Training pipeline with multi-task learning
- `src/data/collect_data.py`: **UPDATED** - Kaggle dataset integration with auto-setup
- `src/data/preprocess.py`: **UPDATED** - Enhanced preprocessing for Goodreads format
- `src/utils/tokenizer.py`: Custom tokenizer with vocabulary building
- `src/utils/metrics.py`: Evaluation metrics for recommendations
- `src/utils/config.py`: Configuration management utilities
- `src/inference/recommend.py`: Recommendation engine implementation
- `notebooks/01_data_exploration.ipynb`: Data analysis and visualization
- `configs/model_config.yaml`: Model and training configuration
- `requirements.txt`: **CLEANED** - Removed unused dependencies
- `environment.yml`: **CLEANED** - Streamlined conda environment
- `test_data_pipeline.py`: **NEW** - Pipeline testing script

### Dataset Details
- **Name**: Goodreads 100k Books Dataset
- **Size**: ~100,000 books, 120MB CSV file
- **Columns**: 13 features including title, authors, rating, description, genres
- **Quality**: Clean, well-structured data suitable for ML training
- **License**: CC0 Public Domain

### Data Pipeline Status
- ✅ **Collection**: Automated setup with clear download instructions
- ✅ **Preprocessing**: Flexible handling of Goodreads format
- ✅ **Validation**: Comprehensive data quality checks
- ✅ **Testing**: Test script for pipeline verification
- ⏳ **Ready for Training**: Awaiting dataset download

### Next Steps
1. **Download Dataset**: Get Kaggle Goodreads 100k dataset manually
2. **Run Preprocessing**: Process the dataset through the pipeline
3. **Train Tokenizer**: Build vocabulary from processed text
4. **Model Training**: Train the semantic recommendation model
5. **Evaluation**: Test recommendation quality and performance

### Technical Notes
- Model supports both semantic similarity and tag-based recommendations
- Configuration-driven design for easy experimentation
- Comprehensive logging and checkpointing
- GPU acceleration support with fallback to CPU
- Robust data pipeline handles real-world data quality issues