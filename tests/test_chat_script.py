from __future__ import annotations

from scripts import chat


def test_chat_resolves_first_served_model(monkeypatch) -> None:
    seen: list[str] = []

    def request(url, **kwargs):
        seen.append(url)
        assert kwargs["payload"] is None
        return {"data": [{"id": "policy"}]}

    monkeypatch.setattr(chat, "_request_json", request)

    assert chat._resolve_model("http://localhost:8000/v1", api_key=None, timeout=1) == (
        "policy"
    )
    assert seen == ["http://localhost:8000/v1/models"]


def test_chat_once_sends_openai_compatible_payload(monkeypatch) -> None:
    captured = {}

    def request(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(chat, "_request_json", request)
    messages = [{"role": "user", "content": "hi"}]

    result = chat._chat_once(
        "http://localhost:8000/v1",
        model="policy",
        messages=messages,
        api_key="secret",
        temperature=0.2,
        max_completion_tokens=32,
        reasoning_effort="low",
        timeout=4,
    )

    assert result == "hello"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["api_key"] == "secret"
    assert captured["payload"] == {
        "model": "policy",
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": 32,
        "reasoning_effort": "low",
    }


def test_chat_main_keeps_conversation_history(monkeypatch, capsys) -> None:
    prompts = iter(["first", "second", "/exit"])
    histories = []
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    monkeypatch.setattr(chat, "_resolve_model", lambda *args, **kwargs: "policy")

    def complete(base_url, **kwargs):
        histories.append(list(kwargs["messages"]))
        return f"answer-{len(histories)}"

    monkeypatch.setattr(chat, "_chat_once", complete)

    assert chat.main([]) == 0
    assert histories[0] == [{"role": "user", "content": "first"}]
    assert histories[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "second"},
    ]
    assert "answer-2" in capsys.readouterr().out
