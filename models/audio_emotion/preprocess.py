"""Audio preprocessing utilities for emotion recognition."""

from __future__ import annotations


def extract_audio_features(audio_path: str) -> dict:
    """Extract acoustic features from an audio file.

    Next steps:
    - Load waveform (librosa/torchaudio).
    - Compute MFCC, mel spectrogram, pitch/energy.
    - Return tensor-ready arrays for training/inference.
    """
    # TODO: Implement actual feature extraction.
    return {"audio_path": audio_path, "features": []}
