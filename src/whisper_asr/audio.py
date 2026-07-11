"""Audio loading, format detection, resampling."""

from __future__ import annotations

import io
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf


def detect_format(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    return "unknown"


def load_from_bytes(audio_bytes: bytes, target_sr: int = 16_000) -> np.ndarray:
    """Load audio bytes → mono float32 at target_sr."""
    fmt = detect_format(audio_bytes)

    if fmt == "wav":
        waveform, sr = sf.read(io.BytesIO(audio_bytes))
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            waveform, sr = librosa.load(tmp_path, sr=None, mono=False)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    waveform = np.asarray(waveform, dtype=np.float32)

    # stereo → mono
    if waveform.ndim == 2:
        axis = 0 if waveform.shape[0] <= 8 else 1
        waveform = waveform.mean(axis=axis).astype(np.float32)
    else:
        waveform = waveform.astype(np.float32)

    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    return waveform


def load_from_file(path: str, sr: int = 16_000) -> np.ndarray:
    """Load a WAV file → mono float32 at given sr."""
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    return waveform.astype(np.float32)
