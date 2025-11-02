from sklearn.pipeline import Pipeline

from .extractors import SpectrogramExtractor, StatsFeatureExtractor
from .loaders import AudioLoader


def create_extraction_pipeline(cfg, save_cache=True, enable_cache=True):
    """Create the audio feature extraction pipeline"""
    return Pipeline(
        [
            ("audio_loader", AudioLoader(sr=22050)),
            (
                "spectrogram",
                SpectrogramExtractor(
                    new_sr=22050,
                    save_cache=save_cache,
                    cache_dir=cfg.paths.interim / "spectrogram_cache",
                ),
            ),
            (
                "features",
                StatsFeatureExtractor(
                    sr=22050,
                    enable_cache=enable_cache,
                    cache_path=cfg.paths.interim / "stats_feat",
                ),
            ),
        ]
    )
