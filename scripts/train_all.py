"""Orchestration script to train all HearMe_AI models sequentially."""

from models.audio_emotion.train import train_audio_emotion_model
from models.face_emotion.train import train_face_emotion_model
from models.text_emotion_intent.train import train_text_model


def train_all() -> None:
    """Run all training pipelines.

    Next steps:
    - Replace hardcoded paths with configuration.
    - Add experiment tracking and failure recovery.
    """
    train_text_model("data/text/train.jsonl", "artifacts/text_model", epochs=3)
    train_audio_emotion_model("data/audio", "artifacts/audio_model")
    train_face_emotion_model("data/face", "artifacts/face_model")


if __name__ == "__main__":
    train_all()
