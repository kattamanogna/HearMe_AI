from app.services.chat_response import generate_mental_health_response, generate_response


def test_generate_response_follows_compassionate_structure(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_CHAT_RESPONSE", "0")

    result = generate_response("session-1", "sadness", "I failed my exam and feel like I let everyone down")

    assert result == {
        "response_text": result["response_text"],
        "crisis_detected": False,
        "severity": "low",
    }
    response = str(result["response_text"])
    assert response.startswith("I'm really sorry this is weighing on you.")
    assert "Feeling low can be exhausting" in response
    assert "from what you shared, i failed my exam" in response.lower()
    assert "try one gentle next step" in response
    assert "You don't have to solve everything at once" in response
    assert response.endswith("?")
    assert "I understand you're feeling sadness" not in response


def test_generate_mental_health_response_reflects_current_message():
    response = generate_mental_health_response(
        "angry",
        conversation_history="My friend ignored my messages all week",
    )

    assert "I can hear how fired up" in response
    assert "from what you shared, my friend ignored my messages all week" in response.lower()
    assert response.endswith("?")


def test_generate_response_preserves_crisis_safety(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_CHAT_RESPONSE", "0")

    result = generate_response("session-1", "sad", "I want to kill myself")

    assert result["crisis_detected"] is True
    assert result["severity"] == "high"
    assert "988" in str(result["response_text"])
