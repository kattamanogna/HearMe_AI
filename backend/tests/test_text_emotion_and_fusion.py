import pytest

from app.services import text_emotion
from app.services.chat_response import generate_response
from app.services.emotional_intelligence import analyze_emotional_context
from app.services.fusion_engine import combine_predictions
from app.services.session_manager import store_interaction


def test_analyze_text_emotion_returns_structured_understanding(monkeypatch):
    def fake_classifier(_text):
        return [[
            {"label": "sadness", "score": 0.98},
            {"label": "anger", "score": 0.21},
        ]]

    monkeypatch.setattr(text_emotion, "_get_text_classifier", lambda: fake_classifier)

    result = text_emotion.analyze_text_emotion(
        "I feel overwhelmed because nobody understands me and I want to cry."
    )

    assert result["emotion"] == "sadness"
    assert result["primary_emotion"] == "sadness"
    assert result["secondary_emotion"] == "frustration"
    assert result["stress"] == "high"
    assert result["urgency"] == "low"
    assert result["intent"] == "seek emotional support"
    assert result["needs_advice"] is False
    assert result["wants_listening_only"] is False
    assert "loneliness" in result["topics"]
    assert "communication" in result["topics"]
    assert result["negative_statements"] == [
        "I feel overwhelmed because nobody understands me and I want to cry."
    ]


def test_message_understanding_detects_advice_questions_and_listening_only(monkeypatch):
    def fake_classifier(_text):
        return [[{"label": "fear", "score": 0.75}, {"label": "sadness", "score": 0.5}]]

    monkeypatch.setattr(text_emotion, "_get_text_classifier", lambda: fake_classifier)

    advice = text_emotion.analyze_text_emotion("How can I talk to my boss today?")
    listening = text_emotion.analyze_text_emotion("I need to vent; no advice, just listen.")

    assert advice["needs_advice"] is True
    assert advice["urgency"] == "medium"
    assert advice["questions"] == ["How can I talk to my boss today?"]
    assert "boss" in advice["people"]
    assert listening["needs_advice"] is False
    assert listening["wants_listening_only"] is True
    assert listening["intent"] == "vent or be heard"


def test_invalid_model_output_has_only_numeric_fallback_scores(monkeypatch):
    monkeypatch.setattr(text_emotion, "_get_text_classifier", lambda: lambda _text: [])

    result = text_emotion.analyze_text_emotion("I am feeling sad and overwhelmed")

    assert result["emotion"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["probabilities"] == {"neutral": 0.0}
    assert all(isinstance(score, float) for score in result["probabilities"].values())


def test_rank_emotions_ignores_invalid_values():
    primary, secondary, confidence = text_emotion._rank_emotions(
        {"sadness": "not a score", "fear": 0.2, "anger": float("nan")},
        "I am feeling sad and overwhelmed",
    )

    assert (primary, secondary, confidence) == ("fear", "sadness", 0.2)


def test_predict_text_endpoint_returns_json_for_sad_and_overwhelmed_input(monkeypatch):
    pytest.importorskip("fastapi")

    from app.main import create_app
    from fastapi.testclient import TestClient

    def fake_classifier(_text):
        return [[
            {"label": "anger", "score": 0.013245883397758007},
            {"label": "fear", "score": 0.1705426126718521},
            {"label": "sadness", "score": 0.01749451644718647},
            {"label": "surprise", "score": 0.7880874872207642},
        ]]

    monkeypatch.setattr(text_emotion, "_get_text_classifier", lambda: fake_classifier)

    response = TestClient(create_app()).post(
        "/api/v1/predict-text",
        json={"text": "i am feeling sad and overwhelmed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"] == "surprise"
    assert body["primary_emotion"] == "surprise"
    assert body["secondary_emotion"] == "sadness"
    assert body["confidence"] == 0.7880874872207642
    assert all(isinstance(score, float) for score in body["probabilities"].values())


def test_fusion_uses_text_when_other_models_fail():
    text_result = {"emotion": "sadness", "confidence": 0.98}
    audio_result = {"emotion": "neutral", "confidence": 0.0}
    face_result = None

    fused = combine_predictions(text_result, audio_result, face_result)

    assert fused["emotion"] == "sadness"
    assert fused["confidence"] == 0.98


def test_fusion_defaults_to_neutral_only_if_all_fail():
    fused = combine_predictions(
        {"emotion": "neutral", "confidence": 0.0},
        {"emotion": "neutral", "confidence": 0.0},
        None,
    )

    assert fused == {
        "emotion": "neutral",
        "confidence": 0.0,
        "probabilities": {"neutral": 0.0},
    }

from datetime import datetime, timezone

from app.services.chat_response import generate_response
from app.services.session_manager import get_chat_history, store_interaction


def test_chat_response_uses_recent_conversation_memory():
    session_id = "memory-test-session"
    first_response = "We focused on one grounding step."
    store_interaction(
        session_id,
        user_text="My name is Maya and I am anxious about work.",
        emotion="anxious",
        confidence=0.91,
        route="test",
        timestamp=datetime.now(timezone.utc).isoformat(),
        response_text=first_response,
    )

    response = generate_response(session_id, "sad", "It still feels hard today.")

    assert "Maya" in str(response["response_text"])
    assert "work stress" in str(response["response_text"])
    assert "Earlier emotions included anxious" in str(response["response_text"])


def test_session_memory_keeps_only_last_10_exchanges():
    session_id = "memory-limit-session"
    for index in range(12):
        store_interaction(
            session_id,
            user_text=f"Message {index}",
            emotion="neutral",
            confidence=0.5,
            route="test",
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_text=f"Advice {index}",
        )

    history = get_chat_history(session_id)

    assert len(history) == 10
    assert history[0]["text"] == "Message 2"
    assert history[-1]["response_text"] == "Advice 11"
