from pathlib import Path
from typing import List, Optional, Dict
from music_recommender.models.mfcc_hybrid import MFCCHybridModel
import joblib
from music_recommender.config import Config
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from music_recommender.utils.logger import get_logger
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Objective did not converge")
logger = get_logger(context=__name__)
cfg = Config()


class MusicRecommender:
    def __init__(
        self,
        hybrid_model_path: Path,
        spotify_dataset_path: Path,
        feature_cols: Optional[Dict[str, List[str]]] = None,
        top_n: int = 10,
    ):
        logger.info("Initializing MusicRecommender...")

        self.top_n = top_n
        self.hybrid_model = joblib.load(hybrid_model_path)
        logger.info(f"Loaded hybrid model from {hybrid_model_path}")

        self.spotify_dataset = pd.read_csv(spotify_dataset_path)
        logger.info(f"Loaded {len(self.spotify_dataset)} tracks from Spotify dataset")

        self._add_tempo_bins()

        if feature_cols is None:
            self.feature_cols = {
                "continuous": [
                    "danceability",
                    "energy",
                    "loudness",
                    "speechiness",
                    "acousticness",
                    "instrumentalness",
                    "liveness",
                    "valence",
                ],
                "categorical": ["key", "mode", "tempo_bins"],
            }
        else:
            self.feature_cols = feature_cols
        self.modify_artist_col()
        self._prepare_spotify_features()
        logger.info("MusicRecommender initialized successfully")

    def modify_artist_col(self):
        """Remove brackets from artists column."""
        self.spotify_dataset["artists"] = (
            self.spotify_dataset["artists"].str.strip().str.strip("[]")
        )

    def _add_tempo_bins(self):
        bins = [0, 80, 100, 120, 140, 170, float("inf")]
        numeric_labels = [0, 1, 2, 3, 4, 5]
        self.spotify_dataset["tempo_bins"] = pd.cut(
            self.spotify_dataset["tempo"], bins=bins, labels=numeric_labels, right=False
        )

    def _prepare_spotify_features(self):
        self.scaler = StandardScaler()
        spotify_cont_scaled = self.scaler.fit_transform(
            self.spotify_dataset[self.feature_cols["continuous"]]
        )

        categ_dfs = []
        for col in self.feature_cols["categorical"]:
            encoded = pd.get_dummies(
                self.spotify_dataset[col], prefix=col, drop_first=False
            )
            categ_dfs.append(encoded)

        spotify_categ_encoded = pd.concat(categ_dfs, axis=1)

        self.categorical_columns = spotify_categ_encoded.columns.tolist()

        self.spotify_features_scaled = np.concatenate(
            [spotify_cont_scaled, spotify_categ_encoded.values], axis=1
        )

        logger.info(
            f"Prepared {self.spotify_features_scaled.shape[1]} features "
            f"for {self.spotify_features_scaled.shape[0]} tracks"
        )

    def _prepare_prediction_vector(
        self, predicted_features: pd.DataFrame
    ) -> np.ndarray:
        cont_features = predicted_features[self.feature_cols["continuous"]].fillna(
            predicted_features[self.feature_cols["continuous"]].median()
        )
        cont_scaled = self.scaler.transform(cont_features)

        categ_dfs = []
        for col in self.feature_cols["categorical"]:
            encoded = pd.get_dummies(
                predicted_features[col], prefix=col, drop_first=False
            )
            categ_dfs.append(encoded)

        categ_encoded = pd.concat(categ_dfs, axis=1)

        for col in self.categorical_columns:
            if col not in categ_encoded.columns:
                categ_encoded[col] = 0

        categ_encoded = categ_encoded[self.categorical_columns]

        return np.concatenate([cont_scaled, categ_encoded.values], axis=1)

    def predict_features(self, audio_features: pd.DataFrame) -> pd.DataFrame:
        """Predict music features from extracted audio features."""
        return self.hybrid_model.predict(audio_features)

    def get_recommendations_from_audio(
        self,
        audio_features: pd.DataFrame,
        top_n: Optional[int] = None,
        return_scores: bool = True,
    ) -> pd.DataFrame:
        predicted_features = self.predict_features(audio_features)

        return self.get_recommendations_from_predictions(
            predicted_features, top_n=top_n, return_scores=return_scores
        )

    def get_recommendations_from_predictions(
        self,
        predicted_features: pd.DataFrame,
        top_n: Optional[int] = None,
        return_scores: bool = True,
    ) -> pd.DataFrame:
        if top_n is None:
            top_n = self.top_n

        if len(predicted_features) > 1:
            logger.warning("Multiple predictions provided, using only the first one")
            predicted_features = predicted_features.iloc[[0]]

        prediction_vector = self._prepare_prediction_vector(predicted_features)

        similarities = cosine_similarity(
            prediction_vector, self.spotify_features_scaled
        )[0]

        top_indices = np.argsort(similarities)[-top_n:][::-1]
        recommendations = self.spotify_dataset.iloc[top_indices].copy()

        if return_scores:
            recommendations["similarity_score"] = similarities[top_indices]

        return recommendations

    def get_recommendations(
        self,
        features: pd.DataFrame,
        top_n: Optional[int] = None,
        return_scores: bool = True,
        features_type: str = "predicted",
    ) -> pd.DataFrame:
        if features_type == "audio":
            return self.get_recommendations_from_audio(features, top_n, return_scores)
        else:
            return self.get_recommendations_from_predictions(
                features, top_n, return_scores
            )

    def get_recommendations_simple(
        self,
        predicted_features: pd.DataFrame,
    ) -> List[dict]:
        recommendations = self.get_recommendations_from_predictions(
            predicted_features, top_n=self.top_n, return_scores=True
        )

        return recommendations[
            ["name", "album", "artists", "similarity_score"]
        ].to_dict("records")

    def batch_recommendations(
        self,
        features_list: List[pd.DataFrame],
        top_n: Optional[int] = None,
        features_type: str = "predicted",
    ) -> List[pd.DataFrame]:
        if top_n is None:
            top_n = self.top_n

        if features_type == "audio":
            return [
                self.get_recommendations_from_audio(features, top_n=top_n)
                for features in features_list
            ]
        else:
            return [
                self.get_recommendations_from_predictions(features, top_n=top_n)
                for features in features_list
            ]