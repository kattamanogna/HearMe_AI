"""Audio emotion recognition service based on MFCC features and a tiny CNN.

This module intentionally uses a lightweight architecture so it can run in a
backend API process without a dedicated GPU.
"""

from __future__ import annotations

from functools import lru_cache
import io

import librosa
import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16_000
N_MFCC = 40
MAX_FRAMES = 128
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful"]
WEIGHTS_PATH = "models/audio_emotion/audio_emotion_cnn.pt"


class TinyAudioEmotionCNN(nn.Module):
    """Small convolutional network for MFCC-based emotion classification.

    Limitation:
    - This model is intentionally compact and may underfit nuanced emotional
      expression, especially across accents, recording conditions, and languages.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


@lru_cache(maxsize=1)
def load_audio_model() -> nn.Module:
    """Load the audio emotion model and cache it for reuse.

    If trained weights are available at ``WEIGHTS_PATH``, they are loaded.
    Otherwise, randomly initialized weights are used as a fallback.

    Limitation:
    - Fallback random weights are not meaningful for production predictions; they
      only preserve API behavior in development environments.
    """

    model = TinyAudioEmotionCNN(n_classes=len(EMOTION_LABELS))
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    except (FileNotFoundError, RuntimeError, OSError):
        # Keep randomly initialized weights when no compatible checkpoint exists.
        pass

    model.eval()
    return model


def extract_audio_features(audio_bytes: bytes) -> np.ndarray | None:
    """Extract normalized MFCC features from raw audio bytes.

    Returns:
        np.ndarray: ``(N_MFCC, MAX_FRAMES)`` feature matrix.
        None: when audio input is missing/empty.

    Limitation:
    - MFCC features drop phase and many paralinguistic details, so subtle
      emotions can be misclassified compared to larger end-to-end models.
    """

    if not audio_bytes:
        return None

    try:
        waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    if waveform.size == 0:
        return None

    mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    mean = np.mean(mfcc)
    std = np.std(mfcc) + 1e-8
    mfcc = (mfcc - mean) / std

    return mfcc.astype(np.float32)


def predict_audio_emotion(audio_bytes: bytes) -> dict[str, str | float] | None:
    """Predict emotion and confidence score from audio bytes.

    Returns None only when no audio payload is provided.
    """

    if not audio_bytes:
        return None

    features = extract_audio_features(audio_bytes)
    if features is None:
        return {"emotion": "neutral", "confidence": 0.0}

    model = load_audio_model()
    inputs = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=-1)[0]

    top_confidence, top_index = torch.max(probabilities, dim=0)
    return {
        "emotion": EMOTION_LABELS[int(top_index)],
        "confidence": float(top_confidence.item()),
    }
