import hashlib
import io
from pathlib import Path

import librosa
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


class AudioLoader(BaseEstimator, TransformerMixin):
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """
        X can be:
        - pd.Series of file paths (training)
        - List of file paths (training)
        - Single file path (inference)
        - List of bytes objects (inference - uploaded files)
        - Single bytes object (inference)
        """
        if isinstance(X, (str, Path)):
            X = [X]
        elif isinstance(X, bytes):
            X = [X]
        elif hasattr(X, "tolist"):
            X = X.tolist()

        results = []
        for item in tqdm(X, desc="Loading audio"):
            loaded = self._load_single(item)
            results.append(loaded)

        return np.array(results, dtype=object)

    def _load_single(self, item):
        """
        Returns dict with: audio, sr, path (or pseudo-path for caching)
        """
        if isinstance(item, (str, Path)):
            # Training mode: load from path
            audio, sr = librosa.load(item, sr=self.sr)
            return {"audio": audio, "sr": sr, "path": Path(item), "source_type": "path"}

        elif isinstance(item, bytes):
            audio, sr = librosa.load(io.BytesIO(item), sr=self.sr)

            audio_hash = hashlib.md5(item[:10000]).hexdigest()
            pseudo_path = Path(f"uploaded_{audio_hash}")

            return {
                "audio": audio,
                "sr": sr,
                "path": pseudo_path,
                "source_type": "bytes",
            }

        else:
            raise ValueError(f"Unsupported input type: {type(item)}")