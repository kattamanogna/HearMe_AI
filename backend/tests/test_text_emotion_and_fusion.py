from app.services import text_emotion
from app.services.chat_response import generate_response
from app.services.emotional_intelligence import analyze_emotional_context
from app.services.fusion_engine import combine_predictions
from app.services.session_manager import store_interaction


def test_analyze_text_emotion_standardized_output(monkeypatch):
    def fake_classifier(_text):
        return [[{"label": "sadness", "score": 0.98}]]

    monkeypatch.setattr(text_emotion, "_get_text_classifier", lambda: fake_classifier)

    result = text_emotion.analyze_text_emotion("I feel awful")

    assert result == {"emotion": "sadness", "confidence": 0.98}


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


def test_emotional_context_detects_mixed_confused_intense_sarcasm():
    context = analyze_emotional_context(
        "Just great!! I am SO happy but also really scared and confused??",
        primary_emotion="fear",
    )

    assert context.confused is True
    assert context.intensity == "high"
    assert context.sarcastic is True
    assert "happy" in context.mixed_emotions
    assert "anxious" in context.mixed_emotions


def test_emotional_context_detects_hopelessness_gratitude_and_repetition():
    context = analyze_emotional_context(
        "Thanks for listening, but there is no point and nothing ever changes.",
        recent_user_messages=[
            "Nothing ever works for me.",
            "I never get better.",
        ],
    )

    assert context.grateful is True
    assert context.hopeless is True
    assert context.repeated_negative_thoughts is True


def test_generate_response_adjusts_tone_for_context(monkeypatch):
    monkeypatch.setattr("app.services.chat_response._load_hf_generator", lambda: None)
    session_id = "ei-test-session"
    store_interaction(
        session_id,
        user_text="Nothing ever works for me.",
        emotion="sad",
        confidence=0.9,
        route="test",
        timestamp="2026-08-06T00:00:00+00:00",
    )

    response = generate_response(
        session_id,
        "sad",
        "I am grateful you are here, but nothing ever changes and I feel confused??",
    )

    assert response["crisis_detected"] is False
    assert response["emotional_context"]["confused"] is True
    assert response["emotional_context"]["grateful"] is True
    assert response["emotional_context"]["repeated_negative_thoughts"] is True
    assert "simpler" in response["response_text"]
    assert "appreciate" in response["response_text"]
