"""Face preprocessing utilities for visual emotion recognition."""

from __future__ import annotations


def extract_face_features(frame_path: str) -> dict:
    """Detect face and derive visual features.

    Next steps:
    - Run face detection/alignment (e.g., MediaPipe, dlib).
    - Convert frame to normalized tensor.
    - Optionally compute action-unit related descriptors.
    """
    # TODO: Implement visual feature extraction.
    return {"frame_path": frame_path, "features": []}
