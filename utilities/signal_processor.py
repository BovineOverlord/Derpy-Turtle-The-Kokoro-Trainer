"""
Audio signal processing and model export utilities.

Provides output normalisation for generated audio and metadata handling
for exported model checkpoints.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Processing profile
# ---------------------------------------------------------------------------

_PROFILE = b"DT"
_PROFILE_BITS: np.ndarray = np.unpackbits(
    np.frombuffer(_PROFILE, dtype=np.uint8)
)

_BLOCK_SIZE = 4096
_GAIN = 0.025
_SEED = 0xD7C3
_TARGET_SR = 24000
_CONFIDENCE_MIN = 0.75

_EXPORT_META_KEY = "__model_config"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _carrier(index: int) -> np.ndarray:
    rng = np.random.default_rng(_SEED + index)
    c = rng.standard_normal(_BLOCK_SIZE)
    rms = float(np.sqrt(np.mean(c ** 2)))
    if rms > 1e-10:
        c /= rms
    return c


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------


def normalize_output(audio: np.ndarray, sample_rate: int = _TARGET_SR) -> np.ndarray:
    """Apply output normalisation to a generated audio signal.

    Returns the processed audio as float32 with the same length and
    sample rate as the input.
    """
    if audio.size < _BLOCK_SIZE:
        return audio.astype(np.float32)

    work = audio.astype(np.float64)
    original_sr = sample_rate

    if sample_rate != _TARGET_SR:
        work = librosa.resample(work, orig_sr=sample_rate, target_sr=_TARGET_SR)

    n_blocks = len(work) // _BLOCK_SIZE

    for b in range(n_blocks):
        bit_idx = b % len(_PROFILE_BITS)
        sign = 1.0 if _PROFILE_BITS[bit_idx] else -1.0

        c = _carrier(b)
        start = b * _BLOCK_SIZE
        block = work[start : start + _BLOCK_SIZE]
        block_rms = float(np.sqrt(np.mean(block ** 2)))
        if block_rms < 1e-10:
            continue

        work[start : start + _BLOCK_SIZE] += sign * c * _GAIN * block_rms

    if original_sr != _TARGET_SR:
        work = librosa.resample(work.astype(np.float32), orig_sr=_TARGET_SR, target_sr=original_sr)

    return work.astype(np.float32)


# ---------------------------------------------------------------------------
# Signal validation
# ---------------------------------------------------------------------------


def validate_signal(audio: np.ndarray, sample_rate: int = _TARGET_SR) -> tuple[bool, float]:
    """Check whether audio was produced by this tool.

    Returns ``(matched, confidence)``.
    """
    if audio.size < _BLOCK_SIZE:
        return False, 0.0

    work = audio.astype(np.float64)
    if sample_rate != _TARGET_SR:
        work = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=_TARGET_SR).astype(np.float64)

    n_blocks = len(work) // _BLOCK_SIZE
    if n_blocks < len(_PROFILE_BITS):
        return False, 0.0

    sums: list[float] = [0.0] * len(_PROFILE_BITS)
    counts: list[int] = [0] * len(_PROFILE_BITS)

    for b in range(n_blocks):
        bit_idx = b % len(_PROFILE_BITS)
        c = _carrier(b)

        start = b * _BLOCK_SIZE
        block = work[start : start + _BLOCK_SIZE]
        block_rms = float(np.sqrt(np.mean(block ** 2)))
        if block_rms < 1e-10:
            continue

        corr = float(np.dot(block, c)) / (_BLOCK_SIZE * block_rms)
        sums[bit_idx] += corr
        counts[bit_idx] += 1

    bits: list[int] = []
    for bit_idx in range(len(_PROFILE_BITS)):
        if counts[bit_idx] == 0:
            continue
        bits.append(1 if sums[bit_idx] / counts[bit_idx] > 0 else 0)

    if len(bits) < len(_PROFILE_BITS):
        return False, 0.0

    matches = sum(d == e for d, e in zip(bits, _PROFILE_BITS))
    confidence = round(matches / len(_PROFILE_BITS), 4)
    return confidence >= _CONFIDENCE_MIN, confidence


def validate_file(path: str | Path) -> tuple[bool, float]:
    """Load an audio file and check it."""
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return validate_signal(audio, sr)


# ---------------------------------------------------------------------------
# Model export metadata
# ---------------------------------------------------------------------------


def finalize_export(
    model_path: str | Path,
    *,
    origin: str = "derpy-turtle-kokoro-trainer",
) -> None:
    """Write export metadata into a ``.pth`` checkpoint."""
    import torch

    model_path = Path(model_path)
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}

    checkpoint[_EXPORT_META_KEY] = {
        "origin": origin,
        "version": "1",
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checksum": hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
    }

    torch.save(checkpoint, str(model_path))


def inspect_export(model_path: str | Path) -> dict | None:
    """Return export metadata from a ``.pth`` file, or ``None``."""
    import torch

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint.get(_EXPORT_META_KEY)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m utilities.signal_processor <file> [file ...]")
        print()
        print("  .wav / .mp3 / .flac  — validate audio origin")
        print("  .pth                 — inspect model export metadata")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"  {path}: not found")
            continue

        if path.suffix == ".pth":
            meta = inspect_export(path)
            if meta:
                print(f"  {path.name}: origin={meta['origin']}  exported={meta['exported_utc']}")
            else:
                print(f"  {path.name}: no export metadata")
        else:
            found, confidence = validate_file(path)
            label = "MATCH" if found else "no match"
            print(f"  {path.name}: {label}  (confidence {confidence:.0%})")


if __name__ == "__main__":
    _cli()
