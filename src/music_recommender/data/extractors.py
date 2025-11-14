import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict

import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from src.music_recommender.config import Config

cfg = Config()


class SpectrogramExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        new_sr: int = 22050,
        new_ch: int = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
        target_duration: float = 30.0,
        cache_dir: Path = cfg.paths.interim / "spectrogram_cache",
        save_cache: bool = False,
    ) -> None:
        super().__init__()
        self.new_ch = new_ch
        self.new_sr = new_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_duration = target_duration
        self.cache_dir = cache_dir
        self.save_cache = save_cache
        if save_cache and cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resample(
        aud_sr: tuple[np.ndarray, float], new_sr: int
    ) -> tuple[np.ndarray, float]:
        aud, sr = aud_sr
        if sr == new_sr:
            return aud, sr
        if aud.ndim == 1:
            res_aud = librosa.resample(aud, orig_sr=sr, target_sr=new_sr)
        else:
            res_aud = np.stack(
                [
                    librosa.resample(channel, orig_sr=sr, target_sr=new_sr)
                    for channel in aud
                ]
            )
        return res_aud, new_sr

    @staticmethod
    def _rechannel(
        aud_sr: tuple[np.ndarray, float], new_ch: int
    ) -> tuple[np.ndarray, float]:
        aud, sr = aud_sr
        n_ch = 1 if aud.ndim == 1 else aud.shape[0]
        if n_ch == new_ch:
            return aud_sr
        if new_ch == 1:
            res_aud = np.mean(aud, axis=0, keepdims=True)
            return res_aud, sr
        if new_ch == 2:
            res_aud = np.stack([aud, aud])
            return res_aud, sr
        else:
            raise ValueError(f"Unsupported number of channels: {new_ch}")

    @staticmethod
    def _pad_or_truncate(
        aud_sr: tuple[np.ndarray, float], target_duration: float
    ) -> tuple[np.ndarray, float]:
        aud, sr = aud_sr

        target_samples = int(sr * target_duration)
        current_samples = aud.shape[-1] if aud.ndim > 1 else len(aud)

        if current_samples > target_samples:
            if aud.ndim > 1:
                return aud[:, :target_samples], sr
            else:
                return aud[:target_samples], sr
        elif current_samples < target_samples:
            pad_samples = target_samples - current_samples
            if aud.ndim > 1:
                pad_width = ((0, 0), (0, pad_samples))
            else:
                pad_width = (0, pad_samples)
            return np.pad(aud, pad_width, mode="constant", constant_values=0), sr
        else:
            return aud, sr

    def _get_cache_path(self, audio_path: Path) -> Path:
        """Generate cache path from file path"""
        params_hash = f"{self.n_fft}_{self.hop_length}_{self.n_mels}_{self.new_sr}_{self.new_ch}_{self.target_duration}"
        return self.cache_dir / f"{audio_path.stem}_{params_hash}.npz"

    def _load_spectr(
        self,
        audio_dict: dict,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
    ) -> dict:
        """
        Takes dict with keys: audio, sr, path, source_type
        Returns dict with: mel_spectrogram, stft_spectrogram, audio, sr, path
        """

        audio = audio_dict["audio"]
        sr = audio_dict["sr"]
        audio_path = audio_dict["path"]
        source_type = audio_dict["source_type"]

        # Try to load from cache (only for real file paths)
        if self.save_cache and source_type == "path":
            cache_path = self._get_cache_path(audio_path)
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    return {
                        "mel_spectrogram": cached["mel_spectrogram"],
                        "stft_spectrogram": cached["stft_spectrogram"],
                        "audio": cached["audio"],
                        "sr": float(cached["sr"]),
                        "path": audio_path,
                    }
                except Exception:
                    # If cache is corrupted, recompute
                    pass

        try:
            aud_sr = (audio, sr)
            aud_sr = self._pad_or_truncate(
                aud_sr=aud_sr, target_duration=self.target_duration
            )
            aud_sr = self._rechannel(aud_sr=aud_sr, new_ch=self.new_ch)
            aud_sr = self._resample(aud_sr=aud_sr, new_sr=self.new_sr)
            aud, sr = aud_sr

            # Compute STFT for features that need it
            stft = librosa.stft(aud, n_fft=n_fft, hop_length=hop_length)
            stft_mag = np.abs(stft)

            # Compute mel spectrogram
            mel_spect = librosa.feature.melspectrogram(
                y=aud, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

            # Save to cache only for real file paths
            if self.save_cache and source_type == "path":
                cache_path = self._get_cache_path(audio_path)
                np.savez_compressed(
                    cache_path,
                    mel_spectrogram=mel_spect_db,
                    stft_spectrogram=stft_mag,
                    audio=aud,
                    sr=sr,
                )

            return {
                "mel_spectrogram": mel_spect_db,
                "stft_spectrogram": stft_mag,
                "audio": aud,
                "sr": sr,
                "path": audio_path,
            }
        except Exception as e:
            raise ValueError(f"Error processing audio from {audio_path}: {e}")

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """X is now array of dicts from AudioLoader"""
        results = []
        for audio_dict in tqdm(X, desc="Computing spectrograms"):
            result = self._load_spectr(
                audio_dict,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            results.append(result)
        return np.array(results, dtype=object)


class StatsFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sr: float = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        n_mfcc: int = 13,
        n_chroma: int = 12,
        f0_min: float = 65.41,
        f0_max: float = 2093.0,
        enable_cache: bool = True,
        cache_path: Path = cfg.paths.interim / "stats_feat",
    ) -> None:
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.n_mfcc = n_mfcc
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.enable_cache = enable_cache
        self.cache_path = cache_path
        if self.enable_cache and cache_path:
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, audio_path: Path, source_type: str) -> str | None:
        """Generate cache key from path"""
        if source_type != "path":
            return None  # Don't cache uploaded files

        params_str = f"{audio_path}_{self.sr}_{self.hop_length}_{self.n_fft}_{self.n_chroma}_{self.n_mfcc}_{self.f0_min}_{self.f0_max}"
        return hashlib.md5(params_str.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        return self.cache_path / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str):
        if cache_key is None:
            return None

        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def _save_to_cache(self, cache_key: str, stats: dict):
        """Save features to cache"""
        if cache_key is None:
            return

        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(stats, f)
        except:
            pass

    @staticmethod
    def make_stats(feature_array, features_name):
        return {
            f"{features_name}_mean": np.mean(feature_array),
            f"{features_name}_std": np.std(feature_array),
            f"{features_name}_min": np.min(feature_array),
            f"{features_name}_max": np.max(feature_array),
            f"{features_name}_median": np.median(feature_array),
            f"{features_name}_q25": np.percentile(feature_array, 25),
            f"{features_name}_q75": np.percentile(feature_array, 75),
        }

    @staticmethod
    def temporal_extract(
        audio: np.ndarray, stft_spectrogram: np.ndarray, sr: float, hop_length: int
    ):
        # Use STFT spectrogram for RMS
        rms = librosa.feature.rms(S=stft_spectrogram, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)

        envelope = rms[0]
        temporal_features = {}

        if len(envelope) > 0 and np.max(envelope) > 0:
            envelope_norm = envelope / np.max(envelope)

            # Attack time (time to reach 90% of max amplitude)
            attack_threshold = 0.9
            attack_frame = np.argmax(envelope_norm >= attack_threshold)
            temporal_features["attack_time"] = (attack_frame * hop_length) / sr

            # Temporal centroid (center of mass in time)
            times = np.arange(len(envelope_norm)) * hop_length / sr
            temporal_features["temporal_centroid"] = np.sum(times * envelope_norm) / (
                np.sum(envelope_norm) + 1e-8
            )
        else:
            temporal_features["attack_time"] = 0.0
            temporal_features["temporal_centroid"] = 0.0

        frame_features = {
            "zcr": zcr.flatten(),
            "rms": rms.flatten(),
        }

        frame_features.update(temporal_features)
        return frame_features

    @staticmethod
    def spectral_extract(
        audio: np.ndarray,
        stft_spectrogram: np.ndarray,
        sr: float,
        hop_length: int,
        n_mfcc: int,
    ):
        spectral_centroid = librosa.feature.spectral_centroid(
            S=stft_spectrogram, sr=sr, hop_length=hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=stft_spectrogram, sr=sr, hop_length=hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=stft_spectrogram, sr=sr, hop_length=hop_length
        )
        spectral_flatness = librosa.feature.spectral_flatness(
            S=stft_spectrogram, hop_length=hop_length
        )
        spectral_flux = librosa.onset.onset_strength(S=stft_spectrogram, sr=sr)

        mfccs = librosa.feature.mfcc(
            S=librosa.power_to_db(stft_spectrogram**2),
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
        )

        return {
            "spectral_centroid": spectral_centroid.flatten(),
            "spectral_bandwidth": spectral_bandwidth.flatten(),
            "spectral_rolloff": spectral_rolloff.flatten(),
            "spectral_flatness": spectral_flatness.flatten(),
            "spectral_flux": spectral_flux,
            "mfccs": mfccs,  # Shape: (13, n_frames)
        }

    @staticmethod
    def extract_rhythm_features(audio: np.ndarray, sr: float):
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

        # Compute beat strength only if beats detected
        beat_strength = np.mean(onset_env[beats]) if len(beats) > 0 else 0.0
        onset_rate = len(beats) / (len(audio) / sr) if len(audio) > 0 else 0.0
        if isinstance(tempo, np.ndarray):
            tempo = float(np.median(tempo))
        else:
            tempo = float(tempo)
        return {
            "tempo": tempo,  # Scalar
            "beat_strength": beat_strength,  # Scalar
            "onset_rate": onset_rate,  # Scalar
            "onset_strength": onset_env,  # Time series
        }

    @staticmethod
    def extract_chroma_features(
        audio: np.ndarray, sr: float, hop_length: int, n_chroma: int
    ):
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma
        )

        return {
            "chroma": chroma,
        }

    @staticmethod
    def extract_hpss_features(audio: np.ndarray):
        y_harmonic, y_percussive = librosa.effects.hpss(audio)

        h_energy = np.mean(y_harmonic**2)
        p_energy = np.mean(y_percussive**2)

        return {
            "harmonic_percussive_ratio": h_energy / (p_energy + 1e-8),  # Scalar
            "harmonic_energy": h_energy,  # Scalar
            "percussive_energy": p_energy,  # Scalar
        }, y_harmonic  # Return harmonic component for further analysis

    @staticmethod
    def extract_harmonic_features(
        y_harmonic, sr, hop_length=512, n_fft=2048, f0_min=65.41, f0_max=2093.0
    ):
        features = {}

        # Get fundamental frequency estimates
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_harmonic,
            fmin=f0_min,
            fmax=f0_max,
            sr=sr,
            hop_length=hop_length,
        )

        # Compute STFT
        stft = librosa.stft(y_harmonic, n_fft=n_fft, hop_length=hop_length)
        mag_spec = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Use only voiced frames
        valid_indices = np.where(voiced_flag)[0]
        valid_f0 = f0[voiced_flag]

        if len(valid_f0) > 0:
            inharmonicity_values = []
            t1_values = []
            t2_values = []
            t3_values = []

            for idx, f0_val in zip(valid_indices, valid_f0):
                if np.isnan(f0_val) or f0_val <= 0:
                    continue

                frame_spec = mag_spec[:, idx]

                # --- INHARMONICITY CALCULATION ---
                peaks, properties = find_peaks(
                    frame_spec, height=np.max(frame_spec) * 0.1
                )
                peak_freqs = freqs[peaks]
                peak_mags = properties["peak_heights"]

                if len(peak_freqs) > 0:
                    sorted_idx = np.argsort(peak_mags)[::-1]
                    peak_freqs = peak_freqs[sorted_idx[:10]]
                    peak_mags = peak_mags[sorted_idx[:10]]

                    deviations = []
                    for n in range(1, min(8, len(peak_freqs) + 1)):
                        expected_freq = f0_val * n
                        if expected_freq < freqs[-1]:
                            closest_idx = np.argmin(np.abs(peak_freqs - expected_freq))
                            deviation = (
                                np.abs(peak_freqs[closest_idx] - expected_freq)
                                / expected_freq
                            )
                            deviations.append(deviation)

                    if deviations:
                        inharmonicity_values.append(np.mean(deviations))

                # --- TRISTIMULUS CALCULATION ---
                harmonic_energies = []
                for n in range(1, 11):
                    harmonic_freq = f0_val * n
                    if harmonic_freq >= freqs[-1]:
                        break

                    bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
                    start_bin = max(0, bin_idx - 3)
                    end_bin = min(len(frame_spec), bin_idx + 4)
                    energy = np.sum(frame_spec[start_bin:end_bin])
                    harmonic_energies.append(energy)

                if len(harmonic_energies) >= 5:
                    total_energy = np.sum(harmonic_energies) + 1e-8

                    t1 = harmonic_energies[0] / total_energy
                    t2 = np.sum(harmonic_energies[1:4]) / total_energy
                    t3 = np.sum(harmonic_energies[4:]) / total_energy

                    t1_values.append(t1)
                    t2_values.append(t2)
                    t3_values.append(t3)

            # Aggregate features
            features["inharmonicity"] = (
                np.mean(inharmonicity_values) if inharmonicity_values else 0.0
            )
            features["tristimulus_1"] = np.mean(t1_values) if t1_values else 0.0
            features["tristimulus_2"] = np.mean(t2_values) if t2_values else 0.0
            features["tristimulus_3"] = np.mean(t3_values) if t3_values else 0.0
            features["f0_mean"] = np.mean(valid_f0)
            features["f0_std"] = np.std(valid_f0)
            features["voiced_ratio"] = np.sum(voiced_flag) / len(voiced_flag)
        else:
            # No voiced frames
            features["inharmonicity"] = 0.0
            features["tristimulus_1"] = 0.0
            features["tristimulus_2"] = 0.0
            features["tristimulus_3"] = 0.0
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0
            features["voiced_ratio"] = 0.0

        return features

    def extract_all_features(self, audio: np.ndarray, stft_spectrogram: np.ndarray):
        all_features = {}

        all_features.update(
            self.temporal_extract(audio, stft_spectrogram, self.sr, self.hop_length)
        )

        all_features.update(
            self.spectral_extract(
                audio, stft_spectrogram, self.sr, self.hop_length, self.n_mfcc
            )  # ← Pass n_mfcc
        )

        all_features.update(self.extract_rhythm_features(audio, self.sr))

        all_features.update(
            self.extract_chroma_features(audio, self.sr, self.hop_length, self.n_chroma)
        )

        hpss_features, y_harmonic = self.extract_hpss_features(audio)
        all_features.update(hpss_features)

        all_features.update(
            self.extract_harmonic_features(
                y_harmonic,
                self.sr,
                self.hop_length,
                self.n_fft,
                self.f0_min,
                self.f0_max,
            )  # ← Pass all params
        )

        return all_features

    def compute_statistics(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Convert all features to statistics (scalars)"""
        stats = {}

        for key, val in features.items():
            # Handle different feature types
            if isinstance(val, (int, float, np.integer, np.floating)):
                # Already a scalar (tempo, attack_time, etc.)
                stats[key] = float(val)

            elif isinstance(val, np.ndarray):
                if val.ndim == 1:
                    # 1D time series (RMS, spectral_centroid, etc.)
                    stats.update(self.make_stats(val, key))

                elif val.ndim == 2:
                    # 2D features (MFCCs, chroma)
                    if key == "mfccs":
                        # MFCCs: shape (13, n_frames)
                        for i in range(val.shape[0]):  # Iterate over 13 coefficients
                            mfcc_i = val[i, :]
                            stats.update(self.make_stats(mfcc_i, f"mfcc_{i}"))

                    elif key == "chroma":
                        # Chroma: shape (12, n_frames)
                        for i in range(val.shape[0]):  # Iterate over 12 pitch classes
                            chroma_i = val[i, :]
                            stats.update(self.make_stats(chroma_i, f"chroma_{i}"))

                    else:
                        # Generic 2D handling
                        for i in range(val.shape[0]):
                            stats.update(self.make_stats(val[i, :], f"{key}_{i}"))

        return stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is array of dicts from SpectrogramExtractor"""
        feature_vectors = []

        for item in tqdm(X, desc="Extracting features"):
            audio = item["audio"]
            stft_spectrogram = item["stft_spectrogram"]
            audio_path = item.get("path")

            # Determine source type from path
            source_type = (
                "path"
                if (audio_path and not str(audio_path).startswith("uploaded_"))
                else "bytes"
            )

            stats = None

            # Try cache only for real files
            if self.enable_cache and self.cache_path:
                cache_key = self._get_cache_key(audio_path, source_type)
                stats = self._load_from_cache(cache_key)

            if stats is None:
                features = self.extract_all_features(audio, stft_spectrogram)
                stats = self.compute_statistics(features)

                # Save to cache only for real files
                if self.enable_cache and self.cache_path:
                    cache_key = self._get_cache_key(audio_path, source_type)
                    self._save_to_cache(cache_key, stats)

            feature_vectors.append(stats)

        return pd.DataFrame(feature_vectors)
