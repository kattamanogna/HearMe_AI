"""Audio emotion analysis using MFCC features and a pretrained SER model."""

from __future__ import annotations

from functools import lru_cache
import io
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_MFCC = 40
MAX_FRAMES = 128
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful"]
WEIGHTS_PATH = Path("models/audio_emotion/audio_emotion_cnn.pt")


class SERNet(nn.Module):
    """Compact speech-emotion classifier over MFCC inputs."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats.flatten(start_dim=1))


def _neutral_audio_response() -> dict[str, Any]:
    return {
        "emotion": "neutral",
        "confidence": 0.0,
        "probabilities": {"neutral": 0.0},
    }


@lru_cache(maxsize=1)
def _load_ser_model() -> nn.Module | None:
    """Load and cache pretrained SER model weights if available."""

    if not WEIGHTS_PATH.exists():
        logger.warning("SER weight file missing at %s; using neutral fallback.", WEIGHTS_PATH)
        return None

    model = SERNet(n_classes=len(EMOTION_LABELS))
    try:
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        logger.info("Loaded SER weights from %s", WEIGHTS_PATH)
        return model
    except Exception as exc:
        logger.warning("Failed to load SER weights from %s: %s", WEIGHTS_PATH, exc)
        return None


def warmup_audio_model() -> None:
    """Warm model cache at startup."""

    _load_ser_model()


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Apply simple noise normalization to stabilize MFCC extraction."""

    centered = waveform - np.mean(waveform)
    peak = np.max(np.abs(centered)) + 1e-8
    return centered / peak


def _extract_mfcc(audio_bytes: bytes) -> np.ndarray | None:
    if not audio_bytes:
        return None

    try:
        waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    if waveform.size == 0:
        return None

    waveform = _normalize_waveform(waveform)
    mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    mean = float(np.mean(mfcc))
    std = float(np.std(mfcc) + 1e-8)
    mfcc = (mfcc - mean) / std
    return mfcc.astype(np.float32)


def _predict_from_features(features: np.ndarray) -> dict[str, Any]:
    model = _load_ser_model()
    if model is None:
        return _neutral_audio_response()

    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
    try:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    except Exception as exc:
        logger.warning("Audio emotion inference failed: %s", exc)
        return _neutral_audio_response()

    probabilities = {
        label: float(score) for label, score in zip(EMOTION_LABELS, probs, strict=False)
    }
    emotion = max(probabilities, key=probabilities.get)
    confidence = float(probabilities[emotion])
    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def analyze_audio_emotion(audio_path: str) -> dict[str, Any]:
    """Analyze emotion from an audio file path and return probabilities/confidence."""

    try:
        audio_bytes = Path(audio_path).read_bytes()
    except Exception:
        return {
            "modality": "audio",
            "input": audio_path,
            **_neutral_audio_response(),
        }

    features = _extract_mfcc(audio_bytes)
    if features is None:
        return {
            "modality": "audio",
            "input": audio_path,
            **_neutral_audio_response(),
        }

    return {
        "modality": "audio",
        "input": audio_path,
        **_predict_from_features(features),
    }


def analyze_audio_emotion_bytes(audio_bytes: bytes) -> dict[str, Any] | None:
    """Variant for API usage from uploaded bytes payloads."""

    if not audio_bytes:
        return None

    features = _extract_mfcc(audio_bytes)
    if features is None:
        return _neutral_audio_response()

    return _predict_from_features(features)
