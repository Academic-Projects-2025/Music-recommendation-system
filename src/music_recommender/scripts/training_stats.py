import os
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from src.music_recommender.config import Config
from src.music_recommender.data.pipeline import create_extraction_pipeline
from src.music_recommender.evaluation.evaluator import get_best_models, get_top_3_models
from src.music_recommender.models.hybrid import HybridModel
from src.music_recommender.models.model_registry import (
    MODEL_CLASS_LOOKUP,
    TARGET_GROUPS,
)
from src.music_recommender.training.trainer import train_models
from src.music_recommender.utils.helpers import tree
from src.music_recommender.utils.logger import get_logger

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Objective did not converge")


def training_script():
    logger = get_logger(context="training")
    cfg = Config()

    logger.info("Loading dataset...")
    audio_data = pd.read_csv(cfg.paths.processed / "matched_metadata.csv")

    bins = [0, 80, 100, 120, 140, 170, float("inf")]
    numeric_labels = [0, 1, 2, 3, 4, 5]
    audio_data["tempo_bins"] = pd.cut(
        audio_data["tempo"], bins=bins, labels=numeric_labels, right=False
    )

    X = audio_data["track_id"].map(
        lambda id: cfg.paths.processed / "audio" / f"{str(id).zfill(6)}.mp3"
    )
    y = audio_data[
        [
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo_bins",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    logger.info("Extracting audio features...")
    extraction_pipeline = create_extraction_pipeline(cfg)
    X_train_extracted = extraction_pipeline.fit_transform(X_train)
    X_test_extracted = extraction_pipeline.transform(X_test)
    joblib.dump(extraction_pipeline, cfg.paths.models / "extraction_pipeline.joblib")
    logger.success(
        f"The extraction pipline is saved in {cfg.paths.models / 'extraction_pipeline.joblib'}"
    )
    logger.info(f"Feature extraction complete. Shape: {X_train_extracted.shape}")

    if os.path.exists(cfg.paths.models / "model_comparison_results.csv"):
        logger.info(
            f"Found models training results in {cfg.paths.models / 'model_comparison_results.csv'} skipping the retrainig for now"
        )
        results_df = pd.read_csv(cfg.paths.models / "model_comparison_results.csv")
    else:
        logger.info("Training individual models...")
        results_df = train_models(
            X_train_extracted, X_test_extracted, y_train, y_test, TARGET_GROUPS, cfg
        )
        logger.info("Model training complete. Results saved.")

    best_models_df = get_best_models(results_df)
    top_3_df = get_top_3_models(results_df)

    top_3_dict = tree()
    for i in range(len(top_3_df)):
        task_type = top_3_df.iloc[i]["task_type"]
        group = top_3_df.iloc[i]["group"]
        rank = int(top_3_df.iloc[i]["rank"])

        top_3_dict[task_type][group][rank] = {
            "model_name": top_3_df.iloc[i]["model"],
            "primary_metric": top_3_df.iloc[i]["primary_metric"],
            "primary_score": top_3_df.iloc[i]["primary_score"],
            "best_params": top_3_df.iloc[i]["best_params"],
        }

    best_dict = tree()
    for i in range(len(best_models_df)):
        task_type = best_models_df.iloc[i]["task_type"]
        group = best_models_df.iloc[i]["group"]

        best_dict[task_type][group] = {
            "model_name": best_models_df.iloc[i]["best_model"],
            "primary_metric": best_models_df.iloc[i]["primary_metric"],
            "primary_score": best_models_df.iloc[i]["primary_score"],
            "best_params": best_models_df.iloc[i]["best_params"],
        }

    hybrid_model = HybridModel(
        top_models=top_3_dict,
        best_models=best_dict,
        skip_stacking={
            "regression": ["structure"],
        },
        target_groups=TARGET_GROUPS,
        lookup_table=MODEL_CLASS_LOOKUP,
        random_state=42,
        cv=4,
    )

    hybrid_model.fit(X_train_extracted, y_train)
    predictions = hybrid_model.predict(X_test_extracted)
    scores = hybrid_model.score(X_test_extracted, y_test)

    logger.info("\n=== Hybrid Model Stats Scores ===")
    for group_name, score_info in scores.items():
        logger.info(
            f"{group_name:15} | {score_info['metric']:8} = {score_info['score']:.4f} | Targets: {score_info['targets']}"
        )

    joblib.dump(hybrid_model, cfg.paths.models / "hybrid_model.joblib")
    logger.success(f"Model saved to {cfg.paths.models / 'hybrid_model.joblib'}")


if __name__ == "__main__":
    training_script()
