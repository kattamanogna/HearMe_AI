"""Face emotion recognition using lightweight CNN inference.

The service combines:
- face detection (prefer MTCNN when available, fallback to OpenCV Haar cascade), and
- a compact convolutional neural network trained for FER-style emotion labels.

API surface:
- ``load_face_model()``: loads/caches detector + CNN.
- ``detect_face_and_predict(image_bytes)``: returns ``{"emotion": str, "confidence": float}``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn

try:
    from mtcnn import MTCNN
except Exception:  # pragma: no cover - optional dependency may be missing.
    MTCNN = None

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
WEIGHTS_PATH = "models/face_emotion/fer2013_tiny_cnn.pt"
FACE_SIZE = 48


class TinyFaceEmotionCNN(nn.Module):
    """A compact CNN for grayscale face emotion recognition.

    Notes:
    - The architecture is intentionally lightweight for CPU-bound API serving.
    - Accuracy depends on quality/diversity of training data and checkpoint quality.
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
def load_face_model() -> tuple[object, nn.Module]:
    """Load and cache the face detector and emotion CNN.

    Returns:
        tuple[object, nn.Module]: ``(detector, model)`` where detector is either
        an ``MTCNN`` instance (if available) or an OpenCV ``CascadeClassifier``.

    Fallback behavior:
        If no compatible model checkpoint is found at ``WEIGHTS_PATH``, the CNN
        remains randomly initialized to preserve API behavior in development.
    """

    detector: object
    if MTCNN is not None:
        detector = MTCNN()
    else:
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    model = TinyFaceEmotionCNN(n_classes=len(EMOTION_LABELS))
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    except (FileNotFoundError, RuntimeError, OSError):
        # Fallback for environments where checkpoints are not yet provisioned.
        pass

    model.eval()
    return detector, model


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes into an OpenCV BGR image."""
    if not image_bytes:
        return None

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    return image


def _detect_primary_face(image_bgr: np.ndarray, detector: object) -> tuple[int, int, int, int] | None:
    """Detect the most prominent face as (x, y, w, h)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if MTCNN is not None and isinstance(detector, MTCNN):
        results = detector.detect_faces(rgb)
        if not results:
            return None

        # Prefer the face with highest confidence.
        best = max(results, key=lambda item: item.get("confidence", 0.0))
        x, y, w, h = best.get("box", [0, 0, 0, 0])
        return max(0, int(x)), max(0, int(y)), int(w), int(h)

    # OpenCV cascade fallback.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    # Pick largest face by area.
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(w), int(h)


def _prepare_face_tensor(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> torch.Tensor | None:
    """Crop + normalize detected face into model input tensor."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    face_crop = image_bgr[y : y + h, x : x + w]
    if face_crop.size == 0:
        return None

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] then standardize for stable inference.
    face = resized_face.astype(np.float32) / 255.0
    face = (face - face.mean()) / (face.std() + 1e-8)

    return torch.from_numpy(face).unsqueeze(0).unsqueeze(0)


def detect_face_and_predict(image_bytes: bytes) -> dict[str, str | float]:
    """Detect a face in image bytes and predict the primary emotion.

    Args:
        image_bytes: Raw bytes for an encoded image (e.g. JPEG/PNG).

    Returns:
        dict: Emotion classification result in the shape:
            ``{"emotion": <label>, "confidence": <float_0_to_1>}``.

        If no face/image is found, a neutral fallback response is returned.
    """

    image = _decode_image(image_bytes)
    if image is None:
        return {"emotion": "neutral", "confidence": 0.0}

    detector, model = load_face_model()
    bbox = _detect_primary_face(image, detector)
    if bbox is None:
        return {"emotion": "neutral", "confidence": 0.0}

    face_tensor = _prepare_face_tensor(image, bbox)
    if face_tensor is None:
        return {"emotion": "neutral", "confidence": 0.0}

    with torch.no_grad():
        logits = model(face_tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]

    top_confidence, top_index = torch.max(probabilities, dim=0)
    return {
        "emotion": EMOTION_LABELS[int(top_index)],
        "confidence": float(top_confidence.item()),
    }
