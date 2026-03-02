"""Inference template for face emotion recognition model."""

from __future__ import annotations


class FaceEmotionInferencer:
    """Run inference on image frames for emotion recognition."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    def predict(self, frame_path: str) -> dict:
        """Predict emotion from face image/frame."""
        # TODO: Add model loading + forward pass.
        return {"emotion": "neutral", "confidence": 0.0, "embedding": []}
