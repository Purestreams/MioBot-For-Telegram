import asyncio

import app.ai_model as ai_model
from app.ai_model import (
    LLMProvider,
    LLMSettings,
    _chat_completion_ark,
    _chat_completion_zan,
    _clean_dict,
    _coerce_provider,
    _default_model_for_provider,
)


def test_coerce_provider_alias_and_values():
    assert _coerce_provider("azure_openai") == LLMProvider.AZURE
    assert _coerce_provider("azure") == LLMProvider.AZURE
    assert _coerce_provider("ollama") == LLMProvider.OLLAMA
    assert _coerce_provider("ark") == LLMProvider.ARK
    assert _coerce_provider("zan") == LLMProvider.ZAN
    assert _coerce_provider("openai-compatible") == LLMProvider.ZAN
    # Unknown values currently fall back to ARK.
    assert _coerce_provider("unknown") == LLMProvider.ARK


def test_clean_dict_removes_none_values_only():
    payload = {"a": 1, "b": None, "c": "", "d": False}
    assert _clean_dict(payload) == {"a": 1, "c": "", "d": False}


def test_default_model_for_each_provider():
    settings = LLMSettings(
        provider=LLMProvider.ARK,
        ark_model="ark-model",
        zan_model="zan-model",
        azure_deployment="azure-model",
        ollama_model="ollama-model",
    )

    assert _default_model_for_provider(settings, LLMProvider.ARK) == "ark-model"
    assert _default_model_for_provider(settings, LLMProvider.ZAN) == "zan-model"
    assert _default_model_for_provider(settings, LLMProvider.AZURE) == "azure-model"
    assert _default_model_for_provider(settings, LLMProvider.OLLAMA) == "ollama-model"


def test_load_settings_from_env_disables_thinking_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)

    settings = ai_model._load_settings_from_env()

    assert settings.enable_thinking is False


def test_chat_completion_ark_uses_env_driven_thinking_toggle(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(ai_model.httpx, "AsyncClient", _FakeClient)

    settings = LLMSettings(
        provider=LLMProvider.ARK,
        enable_thinking=True,
        ark_endpoint="https://example.invalid/chat",
        ark_api_key="secret",
        ark_model="seed-thinking",
        request_timeout=12.0,
    )

    result = asyncio.run(
        _chat_completion_ark(
            settings=settings,
            messages=[{"role": "user", "content": "hello"}],
            model=None,
            response_format=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
        )
    )

    assert result.content == "ok"
    assert captured["json"]["thinking"] == {"type": "enabled"}


def test_chat_completion_zan_uses_openai_compatible_payload(monkeypatch):
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(ai_model.httpx, "AsyncClient", _FakeClient)
    settings = LLMSettings(
        provider=LLMProvider.ZAN,
        enable_thinking=True,
        zan_endpoint="https://ai.zan.top/v1/chat/completions",
        zan_api_key="zan-secret",
        zan_model="gpt-5.6-luna",
        request_timeout=12.0,
    )

    result = asyncio.run(
        _chat_completion_zan(
            settings=settings,
            messages=[{"role": "user", "content": "hello"}],
            model=None,
            response_format={"type": "json_object"},
            temperature=None,
            max_tokens=None,
            top_p=None,
            tools=None,
            tool_choice=None,
            extra_body=None,
        )
    )

    assert result.provider == LLMProvider.ZAN
    assert result.content == "ok"
    assert captured["url"] == "https://ai.zan.top/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer zan-secret"
    assert captured["json"]["model"] == "gpt-5.6-luna"
    assert captured["json"]["response_format"] == {"type": "json_object"}
