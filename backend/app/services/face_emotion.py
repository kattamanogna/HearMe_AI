"""Face emotion analysis using DeepFace with detection validation."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _deepface() -> Any:
    from deepface import DeepFace  # type: ignore

    return DeepFace


def warmup_face_model() -> None:
    """Warm face-analysis stack at startup."""

    try:
        _deepface()
    except Exception as exc:  # pragma: no cover
        logger.warning("Face emotion model warmup failed: %s", exc)


def _decode_image(frame_source: str) -> np.ndarray | None:
    image = cv2.imread(frame_source)
    if image is None:
        return None
    return image


def analyze_face_emotion(frame_source: str) -> dict[str, Any]:
    """Analyze emotion from image path with face detection validation first."""

    image = _decode_image(frame_source)
    if image is None:
        return {
            "modality": "face",
            "input": frame_source,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": False,
        }

    deepface = _deepface()
    try:
        faces = deepface.extract_faces(img_path=image, detector_backend="opencv", enforce_detection=False)
    except Exception:
        faces = []

    if not faces:
        return {
            "modality": "face",
            "input": frame_source,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": False,
        }

    try:
        result = deepface.analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
    except Exception as exc:
        logger.warning("DeepFace inference failed: %s", exc)
        return {
            "modality": "face",
            "input": frame_source,
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": True,
        }

    payload = result[0] if isinstance(result, list) else result
    probs_raw = payload.get("emotion", {})
    probabilities = {str(k).lower(): float(v) / 100.0 for k, v in probs_raw.items()}
    dominant = str(payload.get("dominant_emotion", "neutral")).lower()
    confidence = probabilities.get(dominant, 0.0)

    return {
        "modality": "face",
        "input": frame_source,
        "emotion": dominant,
        "confidence": confidence,
        "probabilities": probabilities,
        "face_detected": True,
    }


def analyze_face_emotion_bytes(image_bytes: bytes) -> dict[str, Any]:
    """Variant for API usage from uploaded image bytes."""

    if not image_bytes:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": False,
        }

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": False,
        }

    deepface = _deepface()
    try:
        faces = deepface.extract_faces(img_path=image, detector_backend="opencv", enforce_detection=False)
    except Exception:
        faces = []
    if not faces:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": False,
        }

    try:
        result = deepface.analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
        payload = result[0] if isinstance(result, list) else result
        probs_raw = payload.get("emotion", {})
        probabilities = {str(k).lower(): float(v) / 100.0 for k, v in probs_raw.items()}
        dominant = str(payload.get("dominant_emotion", "neutral")).lower()
        return {
            "emotion": dominant,
            "confidence": probabilities.get(dominant, 0.0),
            "probabilities": probabilities,
            "face_detected": True,
        }
    except Exception:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probabilities": {"neutral": 1.0},
            "face_detected": True,
        }
