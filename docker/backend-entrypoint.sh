#!/bin/bash
set -e

echo "=== Music Recommender Backend Starting ==="

# Function to download data if missing
download_data() {
    # Download processed data if missing
    if [ ! -f "/app/data/processed/matched_metadata.csv" ]; then
        echo "📦 Downloading processed_data.zip from GitHub..."
        cd /app/data/processed
        curl -L -o precessed_data.zip "${GITHUB_RELEASE_URL}"
        unzip -q precessed_data.zip
        rm precessed_data.zip
        echo "✓ Processed data extracted"
    fi

    # Download Spotify dataset if missing
    if [ ! -f "/app/data/raw/spotify-12m-songs/tracks_features.csv" ]; then
        echo "📦 Downloading Spotify dataset from Kaggle..."
        mkdir -p /app/data/raw/spotify-12m-songs
        cd /app/data/raw
        curl -L -o spotify-12m-songs.zip \
            "https://www.kaggle.com/api/v1/datasets/download/rodolfofigueroa/spotify-12m-songs"
        unzip -q spotify-12m-songs.zip -d spotify-12m-songs
        rm spotify-12m-songs.zip
        echo "✓ Spotify dataset extracted"
    fi
}

# Ensure data exists before starting server
echo "📋 Checking data availability..."
download_data

echo "✓ Data ready. Starting FastAPI server..."
echo "   (Model training will happen automatically if needed)"

# Start FastAPI server
cd /app
exec uv run uvicorn src.music_recommender.backend.backend:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info