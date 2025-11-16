#!/bin/bash
set -e

echo "=== Music Recommender Training Pipeline ==="

# Check if models already exist
if [ -f "/app/models/extraction_pipeline.joblib" ] && [ -f "/app/models/hybrid_model.joblib" ]; then
    echo "✓ Models already exist. Skipping training."
    exit 0
fi

echo "⚠ Models not found. Starting training pipeline..."

# Download processed data from GitHub release
echo "📦 Downloading processed_data.zip from GitHub..."
if [ ! -f "/app/data/processed/matched_metadata.csv" ]; then
    cd /app/data/processed
    curl -L -o precessed_data.zip "${GITHUB_RELEASE_URL}"
    
    echo "📦 Extracting processed_data.zip..."
    unzip -q precessed_data.zip
    rm precessed_data.zip
    echo "✓ Processed data extracted"
else
    echo "✓ Processed data already exists"
fi

# Download Spotify dataset
echo "📦 Downloading Spotify dataset from Kaggle..."
if [ ! -f "/app/data/raw/spotify-12m-songs/tracks_features.csv" ]; then
    mkdir -p /app/data/raw/spotify-12m-songs
    cd /app/data/raw
    
    curl -L -o spotify-12m-songs.zip \
        "https://www.kaggle.com/api/v1/datasets/download/rodolfofigueroa/spotify-12m-songs"
    
    echo "📦 Extracting Spotify dataset..."
    unzip -q spotify-12m-songs.zip -d spotify-12m-songs
    rm spotify-12m-songs.zip
    echo "✓ Spotify dataset extracted"
else
    echo "✓ Spotify dataset already exists"
fi

# Run training script
echo "🚀 Starting model training..."
cd /app
uv run python src/music_recommender/scripts/training_stats.py

# Verify models were created
if [ -f "/app/models/extraction_pipeline.joblib" ] && [ -f "/app/models/hybrid_model.joblib" ]; then
    echo "✓ Training completed successfully!"
    echo "✓ Models saved to /app/models/"
else
    echo "❌ Training failed - models not found"
    exit 1
fi

echo "=== Training Pipeline Complete ==="