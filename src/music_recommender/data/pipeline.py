from sklearn.pipeline import Pipeline

from .extractors import MFCCExtractor
from .loaders import AudioLoader


def create_MFCC_extraction_pipeline(cfg, save_cache=True, enable_cache=True):
    """Create the audio feature extraction pipeline"""
    return Pipeline(
    [
        ("audio_loader", AudioLoader(sr=22050)),
        (
            "MFCCs_extractor",
            MFCCExtractor(
                 sr= 22050,
                 n_mfcc= 13,
                 n_fft= 2048,
                 hop_length= 512,
                 cache_dir=cfg.paths.interim / "MFCC_cache", 
                 enable_cache= True
            ),
        ),
    ]
)