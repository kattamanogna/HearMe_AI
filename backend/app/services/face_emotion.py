"""Face emotion analysis with safe DeepFace handling and explicit fallbacks."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - dependency/runtime dependent.
    cv2 = None  # type: ignore[assignment]
    logger = logging.getLogger(__name__)
    logger.warning("OpenCV dependency unavailable, face model decode disabled: %s", exc)


logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace  # type: ignore
except Exception as exc:  # pragma: no cover - dependency/runtime dependent.
    DeepFace = None  # type: ignore[assignment]
    logger.warning("DeepFace dependency unavailable, face model disabled: %s", exc)


def warmup_face_model() -> None:
    """Warm face-analysis stack at startup."""

    if DeepFace is None:
        logger.warning("Skipping face model warmup because DeepFace is not installed.")


def _neutral_face_response(*, face_detected: bool = False) -> dict[str, Any]:
    return {
        "emotion": "neutral",
        "confidence": 0.0,
        "probabilities": {"neutral": 0.0},
        "face_detected": face_detected,
    }


def _decode_image(frame_source: str) -> np.ndarray | None:
    if cv2 is None:
        logger.warning("Face inference skipped: OpenCV is unavailable.")
        return None
    image = cv2.imread(frame_source)
    if image is None:
        logger.warning("Face inference skipped: image path could not be decoded: %s", frame_source)
    return image


def _analyze_face_image(image: np.ndarray | None) -> dict[str, Any]:
    if image is None:
        logger.warning("Face inference skipped: image is None.")
        return _neutral_face_response(face_detected=False)

    if DeepFace is None:
        logger.warning("Face inference fallback: DeepFace dependency is missing.")
        return _neutral_face_response(face_detected=False)

    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend="opencv",
            enforce_detection=False,
        )
    except Exception as exc:
        logger.warning("DeepFace face-detection error: %s", exc)
        return _neutral_face_response(face_detected=False)

    if not faces:
        logger.warning("No face detected in provided image.")
        return _neutral_face_response(face_detected=False)

    try:
        result = DeepFace.analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
    except Exception as exc:
        logger.warning("DeepFace emotion inference failed: %s", exc)
        return _neutral_face_response(face_detected=True)

    payload = result[0] if isinstance(result, list) else result
    probs_raw = payload.get("emotion", {})
    probabilities = {
        str(label).lower(): float(score) / 100.0
        for label, score in probs_raw.items()
    }

    if not probabilities:
        logger.warning("DeepFace returned empty emotion scores.")
        return _neutral_face_response(face_detected=True)

    dominant = str(payload.get("dominant_emotion", "neutral")).lower()
    confidence = float(probabilities.get(dominant, 0.0))

    return {
        "emotion": dominant,
        "confidence": confidence,
        "probabilities": probabilities,
        "face_detected": True,
    }


def analyze_face_emotion(frame_source: str) -> dict[str, Any]:
    """Analyze emotion from image path with robust safety handling."""

    image = _decode_image(frame_source)
    result = _analyze_face_image(image)
    return {
        "modality": "face",
        "input": frame_source,
        **result,
    }


def analyze_face_emotion_bytes(image_bytes: bytes) -> dict[str, Any]:
    """Analyze emotion from uploaded image bytes for API usage."""

    if not image_bytes:
        logger.warning("Face inference skipped: empty image payload.")
        return _neutral_face_response(face_detected=False)

    if cv2 is None:
        logger.warning("Face inference skipped: OpenCV is unavailable.")
        return _neutral_face_response(face_detected=False)

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    return _analyze_face_image(image)
