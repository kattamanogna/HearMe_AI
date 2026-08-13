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
    assert "Because this sounds sad" in response
    assert "write the next school task on paper" in response
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
    assert "safety first" in str(result["response_text"])


def test_generate_response_routes_requested_safety_categories_to_crisis_support(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_CHAT_RESPONSE", "0")

    messages = [
        "I keep thinking about self harm",
        "My partner is abusive and I am not safe at home",
        "I am having a panic attack and can't breathe",
        "This is a crisis and I need help",
    ]

    for index, message in enumerate(messages):
        result = generate_response(f"session-crisis-{index}", "anxiety", message)

        assert result["crisis_detected"] is True
        assert result["severity"] == "high"
        response = str(result["response_text"])
        assert "safety first" in response
        assert "Because this sounds anxious" not in response


def test_generate_response_personalizes_anxiety_coping_to_work_context(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_CHAT_RESPONSE", "0")

    result = generate_response("session-anxiety", "anxiety", "My boss moved up the deadline and I feel anxious")

    response = str(result["response_text"])
    assert "Because this sounds anxious" in response
    assert "one work item" in response
    assert "two-minute breathing reset" in response


def test_generate_response_uses_emotion_fallback_when_context_is_unclear(monkeypatch):
    monkeypatch.setenv("ENABLE_HF_CHAT_RESPONSE", "0")

    result = generate_response("session-grounding", "fear", "Everything feels too much right now")

    response = str(result["response_text"])
    assert "5-4-3-2-1 grounding exercise" in response
    assert "break the next task" in response
