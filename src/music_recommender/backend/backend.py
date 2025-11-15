import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from music_recommender.config import Config
from music_recommender.models.MFCC_recommender import MusicRecommender
from music_recommender.utils.logger import get_logger

logger = get_logger(context="api")
cfg = Config()

recommender: Optional[MusicRecommender] = None
extraction_pipeline = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    models_loaded: bool
    message: str


def startup_event():
    """Load models on startup"""
    global recommender, extraction_pipeline

    try:
        logger.info("Loading models...")

        extraction_pipeline_path = cfg.paths.models / "extraction_pipeline.joblib"
        if extraction_pipeline_path.exists():
            extraction_pipeline = joblib.load(extraction_pipeline_path)
            logger.info("Extraction pipeline loaded successfully")
        else:
            logger.warning(
                f"Extraction pipeline not found at {extraction_pipeline_path}"
            )

        hybrid_model_path = cfg.paths.models / "hybrid_model.joblib"
        spotify_dataset_path = (
            cfg.paths.data / "raw/spotify-12m-songs/tracks_features.csv"
        )

        logger.info(f"Looking for hybrid model at: {hybrid_model_path}")
        logger.info(f"Looking for dataset at: {spotify_dataset_path}")
        logger.info(f"Hybrid model exists: {hybrid_model_path.exists()}")
        logger.info(f"Dataset exists: {spotify_dataset_path.exists()}")

        if hybrid_model_path.exists() and spotify_dataset_path.exists():
            recommender = MusicRecommender(
                hybrid_model_path=hybrid_model_path,
                spotify_dataset_path=spotify_dataset_path,
                top_n=10,
            )
            logger.info("Music recommender initialized successfully")
        else:
            logger.error("Required model files not found")
            raise FileNotFoundError("Model files missing")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Don't fail startup, but log the error
        recommender = None
        extraction_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting up...")
    startup_event()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Music Recommender API",
    description="AI-powered music recommendation system based on audio analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="online",
        models_loaded=recommender is not None and extraction_pipeline is not None,
        message="Music Recommender API is running",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    models_loaded = recommender is not None and extraction_pipeline is not None

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        message="All systems operational" if models_loaded else "Models not loaded",
    )


@app.post("/recommend/from-audio")
async def recommend_from_audio(
    file: UploadFile = File(...),
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    """
    Get music recommendations by uploading an audio file

    Upload an MP3, WAV, or M4A file to get similar track recommendations
    based on audio feature analysis.
    """
    if recommender is None or extraction_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please ensure both extraction pipeline and recommender are loaded.",
        )

    if not file.filename.endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload MP3, WAV, or M4A files.",
        )

    tmp_file_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.info(f"Extracting features from {file.filename}")
        file_path_series = pd.Series([tmp_file_path])
        audio_features = extraction_pipeline.transform(file_path_series)
        audio_features_df = pd.DataFrame(audio_features)

        recommendations_df = recommender.get_recommendations_from_audio(
            audio_features=audio_features_df, top_n=top_n, return_scores=True
        )

        Path(tmp_file_path).unlink(missing_ok=True)

        recommendations_list = recommendations_df[
            ["name", "album", "artists", "similarity_score"]
        ].to_dict("records")

        return JSONResponse(
            content={
                "recommendations": recommendations_list,
                "count": len(recommendations_list),
                "filename": file.filename,
            }
        )

    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")

        # Clean up on error (safely)
        if tmp_file_path is not None:
            try:
                Path(tmp_file_path).unlink(missing_ok=True)
            except Exception:
                pass

        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
