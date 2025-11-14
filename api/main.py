from fastapi import FastAPI, File, UploadFile, HTTPException, Background
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np
import tempfile
import librosa
from datetime import datetime


from src.music_recommender.data.mfcc_extractor import MFCCExtractor  
from src.music_recommender.models.MFCC_recommender import MFCC_MusicRecommender 


app = FastAPI(
    title = "MFCC features based music recommender API",
    description = "API for music feature extraction and recommendations",
    version = "1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelState:
    """Global state for loaded models"""
    def __init__(self):
        self.mfcc_extractor = None
        self.recommender = None
        self.models = {}  # Store trained models
        self.track_database = []  # Store processed tracks

state = ModelState()

@app.on_event("startup")
async def load_models():
    try:
        state.mfcc_extractor = MFCCExtractor()
        hybrid_model_dir = Path()



        # Initialize recommender
        state.recommender = MFCC_MusicRecommender(state.models)

    except Exception as e:
        print(f"Warning: Could not load models - {e}")



class MFCCFeaturesResponse(BaseModel):
    """Response model for MFCC features"""
    track_id: str
    features: Dict[str, Any]
    duration: float
    sample_rate: int
    processing_time: float

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, List[float]]
    models: Optional[List[str]] = Field(
        default=None,
        description="List of model names to use. If None, uses all available models"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: Dict[str, float]
    confidence: Optional[Dict[str, float]] = None

class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    target_features: Dict[str, float]
    n_recommendations: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[Dict[str, Any]]
    query_features: Dict[str, float]

class TrackMetadata(BaseModel):
    """Track metadata model"""
    track_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    duration: float
    features: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    database_size: int
    timestamp: str



@app.get("/health", response_model = HealthResponse)
async def health_check():
    "Health check endpoint"
    return HealthResponse(
        status = "healthy",
        models_loaded=list(state.models.keys()),
        database_size=len(state.track_database),
        timestamp=datetime.now().isoformat()
    )

@app.post("/extract-features", response_model=MFCCFeaturesResponse)
async def extract_features(
    file: UploadFile = File(...),
    track_id: Optional[str]=None
):
    "Extract MFCC features from audio file"
    if state.mfcc_extractor is None:
        raise HTTPException(status_code = 503, detail = "MFCC extractor not initialized")
    
    start_time = datetime.now()

    with tempfile.NamedTemporaryFile(delete = False, suffix = Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract features
        features = state.mfcc_extractor.extractor(tmp_path)






"""
FastAPI Backend for Music Recommender System
main.py
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import joblib
import numpy as np
import tempfile
import librosa
from datetime import datetime

from src.music_recommender.mfcc_extractor import MFCCExtractor  # Your extractor
from src.music_recommender.recommender import MusicRecommender  # Your recommender

app = FastAPI(
    title="Music Recommender API",
    description="API for music feature extraction and recommendations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global State / Model Loading
# ============================================

class ModelState:
    """Global state for loaded models"""
    def __init__(self):
        self.mfcc_extractor = None
        self.hybrid_model = None  # Single hybrid model instead of multiple models
        self.recommender = None
        self.track_database = []  # Store processed tracks
        
state = ModelState()

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    try:
        # Initialize MFCC extractor
        state.mfcc_extractor = MFCCExtractor()
        
        # Load hybrid model
        model_path = Path("models/mfcc_hybrid_model.pkl")
        if model_path.exists():
            state.hybrid_model = joblib.load(model_path)
            print(f"✅ Hybrid model loaded from {model_path}")
        else:
            # Try to import and initialize the hybrid model class
            try:
                from src.music_recommender.models.mfcc_hybrid import MFCCHybridModel
                state.hybrid_model = MFCCHybridModel()
                
                # Try to load weights if they exist
                weights_path = Path("models/mfcc_hybrid_weights.pkl")
                if weights_path.exists():
                    state.hybrid_model.load(weights_path)
                    print(f"✅ Hybrid model loaded from {weights_path}")
                else:
                    print("⚠️ Warning: No saved hybrid model found. Using uninitialized model.")
            except ImportError as e:
                print(f"⚠️ Warning: Could not import MFCCHybridModel - {e}")
                state.hybrid_model = None
        
        # Initialize recommender with hybrid model
        if state.hybrid_model:
            state.recommender = MusicRecommender(state.hybrid_model)
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models - {e}")

# ============================================
# Pydantic Models (Request/Response Schemas)
# ============================================

class MFCCFeaturesResponse(BaseModel):
    """Response model for MFCC features"""
    track_id: str
    features: Dict[str, Any]
    duration: float
    sample_rate: int
    processing_time: float

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, List[float]]
    models: Optional[List[str]] = Field(
        default=None,
        description="List of model names to use. If None, uses all available models"
    )

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: Dict[str, float]
    confidence: Optional[Dict[str, float]] = None

class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    target_features: Dict[str, float]
    n_recommendations: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[Dict[str, Any]]
    query_features: Dict[str, float]

class TrackMetadata(BaseModel):
    """Track metadata model"""
    track_id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    duration: float
    features: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    database_size: int
    timestamp: str

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Music Recommender API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_status = []
    if state.hybrid_model:
        model_status.append("hybrid_model")
    
    return HealthResponse(
        status="healthy",
        models_loaded=model_status,
        database_size=len(state.track_database),
        timestamp=datetime.now().isoformat()
    )

@app.post("/extract-features", response_model=MFCCFeaturesResponse)
async def extract_features(
    file: UploadFile = File(...),
    track_id: Optional[str] = None
):
    """
    Extract MFCC features from an audio file
    
    Args:
        file: Audio file (mp3, wav, etc.)
        track_id: Optional track identifier
    
    Returns:
        Extracted MFCC features
    """
    if state.mfcc_extractor is None:
        raise HTTPException(status_code=503, detail="MFCC extractor not initialized")
    
    start_time = datetime.now()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract features
        features = state.mfcc_extractor.extract(tmp_path)
        
        # Load audio for metadata
        y, sr = librosa.load(tmp_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MFCCFeaturesResponse(
            track_id=track_id or file.filename,
            features=features,
            duration=duration,
            sample_rate=sr,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {str(e)}")
    
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using the hybrid model
    
    Args:
        request: Features and model selection
    
    Returns:
        Predictions for each requested attribute
    """
    if state.hybrid_model is None:
        raise HTTPException(status_code=503, detail="Hybrid model not loaded")
    
    try:
        # Prepare features
        feature_array = np.array([list(request.features.values())])
        
        # Make predictions using hybrid model
        predictions = state.hybrid_model.predict(feature_array)
        
        # If predictions is not a dict, convert it
        if not isinstance(predictions, dict):
            # Assuming the model predicts multiple targets
            # Adjust based on your hybrid model's output format
            target_names = request.models or ["valence", "energy", "danceability"]
            if isinstance(predictions, np.ndarray):
                predictions = {
                    name: float(pred) 
                    for name, pred in zip(target_names, predictions[0])
                }
            else:
                predictions = {"prediction": float(predictions)}
        
        return PredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-from-audio", response_model=PredictionResponse)
async def predict_from_audio(
    file: UploadFile = File(...),
    models: Optional[List[str]] = None
):
    # Extract features first
    features_response = await extract_features(file)
    
    # Make predictions
    prediction_request = PredictionRequest(
        features=features_response.features,
        models=models
    )
    
    return await predict(prediction_request)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if state.recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    if not state.track_database:
        raise HTTPException(
            status_code=404,
            detail="No tracks in database. Please add tracks first."
        )
    
    try:
        # Get recommendations using your recommender
        recommendations = state.recommender.recommend(
            target_features=request.target_features,
            n_recommendations=request.n_recommendations,
            track_database=state.track_database,
            filters=request.filters
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            query_features=request.target_features
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Recommendation failed: {str(e)}")

@app.post("/add-track", response_model=Dict[str, str])
async def add_track_to_database(
    file: UploadFile = File(...),
    metadata: Optional[str] = None  # JSON string of metadata
):
    
    import json
    
    # Extract features
    features_response = await extract_features(file)
    
    # Parse metadata
    meta = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    # Create track entry
    track = TrackMetadata(
        track_id=features_response.track_id,
        title=meta.get("title"),
        artist=meta.get("artist"),
        duration=features_response.duration,
        features=features_response.features
    )
    
    # Add to database
    state.track_database.append(track.dict())
    
    return {
        "message": "Track added successfully",
        "track_id": track.track_id,
        "database_size": len(state.track_database)
    }

@app.get("/tracks", response_model=List[TrackMetadata])
async def list_tracks(
    limit: int = 100,
    offset: int = 0
):
    
    return state.track_database[offset:offset + limit]

@app.delete("/tracks/{track_id}", response_model=Dict[str, str])
async def delete_track(track_id: str):
    
    initial_size = len(state.track_database)
    state.track_database = [
        t for t in state.track_database 
        if t["track_id"] != track_id
    ]
    
    if len(state.track_database) == initial_size:
        raise HTTPException(status_code=404, detail="Track not found")
    
    return {
        "message": "Track deleted successfully",
        "track_id": track_id
    }

@app.post("/batch-extract", response_model=List[MFCCFeaturesResponse])
async def batch_extract_features(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    
    results = []
    
    for file in files:
        try:
            result = await extract_features(file)
            results.append(result)
        except Exception as e:
            # Log error but continue with other files
            print(f"Error processing {file.filename}: {e}")
    
    return results

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )