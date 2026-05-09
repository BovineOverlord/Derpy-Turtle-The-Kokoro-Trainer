from typing import Any

import librosa
import numpy as np
import scipy.stats
import soundfile as sf
from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder

from utilities.device_utils import resolve_device


class FitnessScorer:
    """Scores generated speech against one or more target clips."""

    ACCENT_FEATURE_KEYS = (
        "tempo",
        "pitch_mean",
        "pitch_std",
        "zero_crossing_rate",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "mfcc1_mean",
        "mfcc2_mean",
        "mfcc3_mean",
        "mfcc1_std",
        "mfcc2_std",
        "mfcc3_std",
        "energy_mean",
        "energy_std",
    )

    def __init__(
        self,
        target_paths: str | list[str],
        device: str = "auto",
        target_weight: float = 0.45,
        self_weight: float = 0.33,
        feature_weight: float = 0.10,
        accent_weight: float = 0.12,
    ):
        self.device = resolve_device(device)
        self.encoder = VoiceEncoder(device=self.device)
        self.target_paths = self._normalize_target_paths(target_paths)
        if not self.target_paths:
            raise ValueError("At least one target audio path is required")

        self.target_weight = float(target_weight)
        self.self_weight = float(self_weight)
        self.feature_weight = float(feature_weight)
        self.accent_weight = float(accent_weight)
        self._validate_weights()

        self.target_embeds: list[np.ndarray] = []
        self.target_feature_sets: list[dict[str, Any]] = []

        for path in self.target_paths:
            audio, _ = sf.read(path, dtype="float32")
            wav = preprocess_wav(path, source_sr=24000)
            embed = self.encoder.embed_utterance(wav)
            features = self.extract_features(audio)
            self.target_embeds.append(embed)
            self.target_feature_sets.append(features)

        self.target_features = self._average_feature_dicts(self.target_feature_sets)
        self.target_accent_features = {k: self.target_features[k] for k in self.ACCENT_FEATURE_KEYS if k in self.target_features}

    def _normalize_target_paths(self, target_paths: str | list[str]) -> list[str]:
        if isinstance(target_paths, str):
            return [target_paths]
        return [str(path) for path in target_paths]

    def _validate_weights(self) -> None:
        weights = [self.target_weight, self.self_weight, self.feature_weight, self.accent_weight]
        if any(w < 0.0 for w in weights):
            raise ValueError("Score weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("At least one score weight must be positive")

    def _average_feature_dicts(self, feature_sets: list[dict[str, Any]]) -> dict[str, float]:
        if not feature_sets:
            return {}

        keys = feature_sets[0].keys()
        averaged: dict[str, float] = {}
        for key in keys:
            values = [float(features.get(key, 0.0)) for features in feature_sets]
            averaged[key] = float(np.mean(values))
        return averaged

    def _feature_penalty(
        self,
        features: dict[str, Any],
        target_features: dict[str, Any],
        keys: tuple[str, ...] | list[str] | None = None,
    ) -> float:
        """Normalized mean absolute percentage difference."""
        if not target_features:
            return 0.0

        feature_keys = keys if keys is not None else list(target_features.keys())
        if not feature_keys:
            return 0.0

        penalties: list[float] = []
        for key in feature_keys:
            if key not in target_features or key not in features:
                continue

            target_value = float(target_features[key])
            current_value = float(features[key])
            denom = max(abs(target_value), 1e-6)
            penalties.append(abs(current_value - target_value) / denom)

        if not penalties:
            return 0.0

        return float(np.mean(penalties) * 100.0)

    @staticmethod
    def _as_audio_list(audio: NDArray[np.float32] | list[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        if isinstance(audio, list):
            return [clip for clip in audio if getattr(clip, "size", 0) > 0]
        return [audio] if getattr(audio, "size", 0) > 0 else []

    def _aggregate_features(self, audios: NDArray[np.float32] | list[NDArray[np.float32]]) -> dict[str, float]:
        clips = self._as_audio_list(audios)
        if not clips:
            return {}

        feature_sets = [self.extract_features(clip) for clip in clips]
        return self._average_feature_dicts(feature_sets)

    def hybrid_similarity(
        self,
        audio: NDArray[np.float32] | list[NDArray[np.float32]],
        audio2: NDArray[np.float32],
        target_similarity: float,
    ):
        clips = self._as_audio_list(audio)
        if not clips:
            return {
                "score": 0.0,
                "target_similarity": float(target_similarity),
                "self_similarity": 0.0,
                "feature_similarity": 0.0,
                "accent_similarity": 0.0,
            }

        features = self._aggregate_features(clips)
        self_similarity = self.self_similarity(clips[0], audio2)

        target_feature_penalty = self.target_feature_penalty(features)
        accent_penalty = self.accent_penalty(features)

        feature_similarity = max((100.0 - target_feature_penalty) / 100.0, 0.01)
        accent_similarity = max((100.0 - accent_penalty) / 100.0, 0.01)

        values = [target_similarity, self_similarity, feature_similarity, accent_similarity]
        values = [max(v, 1e-6) for v in values]
        weights = [self.target_weight, self.self_weight, self.feature_weight, self.accent_weight]

        weighted_values = [v for v, w in zip(values, weights) if w > 0.0]
        weighted_weights = [w for w in weights if w > 0.0]
        score = (np.sum(weighted_weights) / np.sum(np.array(weighted_weights) / np.array(weighted_values))) * 100.0

        return {
            "score": float(score),
            "target_similarity": float(target_similarity),
            "self_similarity": float(self_similarity),
            "feature_similarity": float(feature_similarity),
            "accent_similarity": float(accent_similarity),
        }

    def target_similarity(self, audio: NDArray[np.float32]) -> float:
        audio_wav = preprocess_wav(audio, source_sr=24000)
        audio_embed = self.encoder.embed_utterance(audio_wav)
        similarities = [float(np.inner(audio_embed, target_embed)) for target_embed in self.target_embeds]
        return float(np.mean(similarities))

    def target_similarity_pairwise(self, audios: list[NDArray[np.float32]]) -> float:
        clips = self._as_audio_list(audios)
        if not clips:
            return 0.0
        if len(clips) == 1:
            return self.target_similarity(clips[0])

        if len(clips) == len(self.target_embeds):
            similarities: list[float] = []
            for clip, target_embed in zip(clips, self.target_embeds):
                clip_wav = preprocess_wav(clip, source_sr=24000)
                clip_embed = self.encoder.embed_utterance(clip_wav)
                similarities.append(float(np.inner(clip_embed, target_embed)))
            return float(np.mean(similarities))

        similarities = [self.target_similarity(clip) for clip in clips]
        return float(np.mean(similarities))

    def target_feature_penalty(self, features: dict[str, Any]) -> float:
        """Penalizes for differences in full acoustic feature profile."""
        return self._feature_penalty(features, self.target_features)

    def accent_penalty(self, features: dict[str, Any]) -> float:
        """Accent/prosody penalty using a focused subset of features."""
        return self._feature_penalty(features, self.target_accent_features, self.ACCENT_FEATURE_KEYS)

    def self_similarity(self, audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        """Self similarity indicates model stability."""
        audio_wav1 = preprocess_wav(audio1, source_sr=24000)
        audio_embed1 = self.encoder.embed_utterance(audio_wav1)

        audio_wav2 = preprocess_wav(audio2, source_sr=24000)
        audio_embed2 = self.encoder.embed_utterance(audio_wav2)
        return float(np.inner(audio_embed1, audio_embed2))

    def extract_features(self, audio: NDArray[np.float32] | NDArray[np.float64], sr: int = 24000) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.
        """
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        features = {}

        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        n_fft = 2048
        hop_length = 512

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        for i in range(len(mfccs)):
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(len(mfcc_delta)):
            features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        for i in range(len(chroma)):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["mel_spec_mean"] = float(np.mean(mel_spec))
        features["mel_spec_std"] = float(np.std(mel_spec))

        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features["tonnetz_mean"] = float(np.mean(tonnetz))
        features["tonnetz_std"] = float(np.std(tonnetz))

        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features["tempo"] = float(tempo)

        if len(beat_frames) > 0:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                beat_diffs = np.diff(beat_times)
                features["beat_mean"] = float(np.mean(beat_diffs))
                features["beat_std"] = float(np.std(beat_diffs))
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0
        else:
            features["beat_mean"] = 0.0
            features["beat_std"] = 0.0

        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

        pitch_values = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        energy = np.array([sum(abs(audio[i : i + hop_length])) for i in range(0, len(audio), hop_length)])
        features["energy_mean"] = float(np.mean(energy))
        features["energy_std"] = float(np.std(energy))

        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_squared = S**2
        S_mean = np.mean(S_squared, axis=1)
        S_std = np.std(S_squared, axis=1)
        S_ratio = np.divide(S_mean, S_std, out=np.zeros_like(S_mean), where=S_std != 0)
        features["harmonic_ratio"] = float(np.mean(S_ratio))

        features["audio_mean"] = float(np.mean(audio))
        features["audio_std"] = float(np.std(audio))
        features["audio_skew"] = float(scipy.stats.skew(audio))
        features["audio_kurtosis"] = float(scipy.stats.kurtosis(audio))

        return features


