from types import SimpleNamespace

import pytest

from app.chat import ChatClient


class _Resp:
    def __init__(self, payload=None, text="raw-text"):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def test_chatclient_requires_api_key():
    with pytest.raises(ValueError):
        ChatClient(api_key="", url="https://example.com")


def test_chatclient_requires_url_when_parameter_missing():
    with pytest.raises(ValueError):
        ChatClient(api_key="k", url=None)


def test_chatclient_chat_returns_choice_message_content(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        return _Resp(payload={"choices": [{"message": {"content": " hi "}}]})

    monkeypatch.setattr("app.chat.requests.post", fake_post)

    client = ChatClient(api_key="k", url="https://example.com")
    text = client.chat(messages=[{"role": "user", "content": "hello"}], top_p=0.8)

    assert text == "hi"
    assert captured["payload"]["top_p"] == 0.8


def test_chatclient_chat_returns_raw_text_on_non_json(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return _Resp(payload=None, text="fallback")

    monkeypatch.setattr("app.chat.requests.post", fake_post)

    client = ChatClient(api_key="k", url="https://example.com")
    assert client.chat(messages=[{"role": "user", "content": "hello"}]) == "fallback"
