import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from src.music_recommender.config import Config

cfg = Config()



class MFCCExtractor(BaseEstimator, TransformerMixin):
    """Extract MFCC features with statistics (mean, std, quartiles)"""

    def __init__(self,
                 sr: int = 22050,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 cache_dir: Path = None, 
                 enable_cache: bool = True
                 ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache

        if enable_cache and cache_dir:
            cache_dir.mkdir(parents=True, exist_ok = True)

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X) -> pd.DataFrame:
        """
        Extract mfcc features from audio data
        X: array of dicts from AudioLoader
        """
        feature_vectors = []

        for audio_dict in tqdm(X, desc='Extracting MFCC features'):
            audio = audio_dict["audio"]
            audio_path = audio_dict["path"]
            source_type = audio_dict["source_type"]

            # Try cache
            if self.enable_cache and source_type == "path":
                cache_key = self._get_cache_key(audio_path)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    feature_vectors.append(cached)
                    continue

            # Extract features 
            features = self._extract_mfcc_stats(audio)

            # Save to cache
            if self.enable_cache and source_type=="path":
                cache_key = self._get_cache_key(audio_path)
                self._save_to_cache(cache_key, features)

            feature_vectors.append(features)

        return pd.DataFrame(feature_vectors)
    

    def _extract_mfcc_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract MFCC and compute statistics"""
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Compute deltas
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Compute statistics
        stats = {}
        
        # MFCC statistics
        for i in range(self.n_mfcc):
            stats.update(self._make_stats(mfccs[i, :], f"mfcc_{i}"))
        
        # Delta statistics
        for i in range(self.n_mfcc):
            stats.update(self._make_stats(delta_mfccs[i, :], f"delta_{i}"))
        
        # Delta2 statistics
        for i in range(self.n_mfcc):
            stats.update(self._make_stats(delta2_mfccs[i, :], f"delta2_{i}"))
        
        return stats
    
    @staticmethod
    def _make_stats(feature_array: np.ndarray, name: str) -> Dict[str, float]:
        """Compute statistics for a feature array"""
        return {
            f"{name}_mean": float(np.mean(feature_array)),
            f"{name}_std": float(np.std(feature_array)),
            f"{name}_min": float(np.min(feature_array)),
            f"{name}_max": float(np.max(feature_array)),
            f"{name}_median": float(np.median(feature_array)),
            f"{name}_q25": float(np.percentile(feature_array, 25)),
            f"{name}_q75": float(np.percentile(feature_array, 75))
        }
    
    def _get_cache_key(self, audio_path: Path) -> str:
        """Generate cache key"""
        params = f"{audio_path}_{self.sr}_{self.n_mfcc}_{self.n_fft}_{self.hop_length}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Dict[str, float]:
        """Load from cache"""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, features: Dict[str, float]):
        """Save to cache"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass