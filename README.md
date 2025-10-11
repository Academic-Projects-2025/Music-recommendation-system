this is the project structure for now

music-recommender/
├── data/
│ ├── raw/ <- Original FMA audio + Spotify CSV
│ ├── interim/ <- Extracted librosa features
│ ├── processed/ <- Matched FMA-Spotify features
│ └── external/ <- Downloaded datasets
├── docs/ <- Documentation
├── models/ <- Saved .pkl/.joblib models
├── notebooks/ <- Jupyter notebooks
├── references/ <- Data dictionaries, FMA/Spotify docs
├── reports/
│ └── figures/ <- Feature distribution plots, etc.
├── tests/
│ └── **init**.py
├── api/
│ └── main.py <- FastAPI app
├── scripts/
│ ├── train.py <- CLI training script
│ └── download_audio.py <- yt-dlp wrapper
├── src/
│ └── music_recommender/
│ ├── **init**.py
│ ├── config.py <- Paths, model configs
│ ├── dataset.py <- FMA/Spotify downloaders
│ ├── features.py <- Librosa extraction
│ ├── plots.py <- Visualization utilities
│ └── modeling/
│ ├── **init**.py
│ ├── train.py <- Feature predictor training
│ └── predict.py <- Similarity search logic
├── requirements.txt
├── setup.cfg
├── pyproject.toml
├── README.md
└── .gitignore

```bash
uv init
uv sync
```
