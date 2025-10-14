# Book Recommendation System

A machine learning system that recommends books based on user text input (book titles, author names, or descriptions) using semantic understanding and tag-based matching.

## Project Overview

This system uses a custom-trained semantic model to understand book content and generate recommendations without requiring user interaction history. The approach combines:

1. **Semantic Encoding**: Custom transformer-based encoder for book content
2. **Tag Prediction**: Multi-label classification for book characteristics
3. **Similarity Matching**: Cosine similarity and tag-based scoring for recommendations

## System Architecture

```
User Input → Text Preprocessing → Semantic Encoding → Tag Generation → 
Book Matching → Similarity Scoring → Ranking → Top-K Recommendations
```

## Project Structure

```
BooksRecommender/
├── data/
│   ├── raw/              # Raw datasets (Goodreads, OpenLibrary, etc.)
│   └── processed/        # Processed and cleaned datasets
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code and scripts
│   ├── data/            # Data collection and preprocessing
│   ├── models/          # Model architectures and training
│   ├── inference/       # Inference and recommendation logic
│   └── utils/           # Utility functions
├── models/              # Saved models and checkpoints
├── results/             # Results, plots, and visualizations
├── configs/             # Configuration files
└── requirements.txt     # Python dependencies
```

## Features

- **Custom Model Training**: Train your own semantic model from scratch
- **Multi-source Data**: Combine Goodreads, OpenLibrary, and content data
- **Semantic Understanding**: Handle various input types (titles, authors, descriptions)
- **Tag-based Matching**: Use predicted tags for enhanced recommendations
- **Scalable Architecture**: Designed for efficient inference

## Getting Started

1. **Environment Setup**:
   ```bash
   conda env create -f environment.yml
   conda activate mlenv
   ```

2. **Data Collection**:
   ```bash
   python src/data/collect_data.py
   ```

3. **Data Preprocessing**:
   ```bash
   python src/data/preprocess.py
   ```

4. **Model Training**:
   ```bash
   python src/models/train.py --config configs/model_config.yaml
   ```

5. **Inference**:
   ```bash
   python src/inference/recommend.py --query "fantasy adventure magic"
   ```

## Model Architecture

### Stage 1: Book Content Encoder
- Custom transformer-based architecture
- Input: Book metadata + content features
- Output: 768-dimensional semantic embeddings

### Stage 2: Tag Prediction Network
- Multi-label classification head
- Predicts book characteristics and themes
- Supports dynamic tag vocabulary

### Stage 3: Recommendation Engine
- Combines semantic similarity and tag matching
- Implements ranking algorithms for diverse recommendations
- Supports configurable recommendation strategies

## Datasets

- **Goodreads Dataset**: Book metadata, ratings, reviews
- **OpenLibrary**: Additional metadata and book information
- **Project Gutenberg**: Full text content for public domain books
- **Custom Scraped Data**: Book summaries and descriptions

## Evaluation Metrics

- **Semantic Similarity**: Cosine similarity between embeddings
- **Tag Accuracy**: Multi-label classification metrics
- **Recommendation Quality**: Relevance and diversity measures
- **User Study**: Manual evaluation of recommendation quality

## License

This project is part of a Machine Learning and Data Mining course.