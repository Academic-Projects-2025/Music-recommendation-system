from sklearn.pipeline import Pipeline

from .extractors import SpectrogramExtractor, StatsFeatureExtractor
from .loaders import AudioLoader
from .mfcc_extractor import MFCCExtractor


def create_MFCC_extraction_pipeline(cfg, save_cache=True, enable_cache=True):
    """Create the audio feature extraction pipeline"""
    return Pipeline(
        [
            ("audio_loader", AudioLoader(sr=22050)),
            (
                "MFCCs_extractor",
                MFCCExtractor(
                    sr=22050,
                    n_mfcc=13,
                    n_fft=2048,
                    hop_length=512,
                    cache_dir=cfg.paths.interim / "MFCC_cache",
                    enable_cache=True,
                ),
            ),
        ]
    )


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
