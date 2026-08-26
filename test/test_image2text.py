import asyncio

import app.ai_model as ai_model
from app import image2text
from app.image2text import (
    _build_sticker_prompt,
    _build_sticker_understanding_prompt,
    _extract_text_from_chat_payload,
    _extract_text_from_responses_payload,
    _guess_mime_type,
    _parse_sticker_understanding,
)


def test_guess_mime_type_by_extension():
    assert _guess_mime_type("a.png") == "image/png"
    assert _guess_mime_type("a.webp") == "image/webp"
    assert _guess_mime_type("a.gif") == "image/gif"
    assert _guess_mime_type("a.jpg") == "image/jpeg"


def test_extract_text_prefers_output_text_field():
    payload = {"output_text": "  hello world  "}
    assert _extract_text_from_responses_payload(payload) == "hello world"


def test_extract_text_from_nested_output_blocks():
    payload = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ]
            }
        ]
    }
    assert _extract_text_from_responses_payload(payload) == "line1\nline2"


def test_extract_text_returns_empty_for_unrecognized_payload():
    assert _extract_text_from_responses_payload({"output": []}) == ""


def test_extract_text_from_openai_chat_payload():
    payload = {"choices": [{"message": {"content": "  image text  "}}]}
    assert _extract_text_from_chat_payload(payload) == "image text"


def test_build_sticker_prompt_includes_optional_hints():
    prompt = _build_sticker_prompt(emoji="🙂", set_name="mio_pack")
    assert "Known sticker emoji: 🙂." in prompt
    assert "Sticker set name: mio_pack." in prompt


def test_build_sticker_understanding_prompt_requests_quality_json():
    prompt = _build_sticker_understanding_prompt(emoji="🙂", set_name="mio_pack")
    assert "description, tags, mood, safe_for_reply" in prompt
    assert "Known sticker emoji: 🙂." in prompt


def test_parse_sticker_understanding_from_json():
    parsed = _parse_sticker_understanding(
        '{"description":"Happy cat waving", "tags":["happy", "wave", "cat"], "mood":"happy", "safe_for_reply":true}'
    )

    assert parsed is not None
    assert parsed.description == "Happy cat waving"
    assert parsed.tags == ["happy", "wave", "cat"]
    assert parsed.mood == "happy"
    assert parsed.safe_for_reply is True


def test_parse_sticker_understanding_falls_back_to_description_line():
    parsed = _parse_sticker_understanding("smiling cat waving hello")

    assert parsed is not None
    assert parsed.description == "smiling cat waving hello"
    assert parsed.tags == []
    assert parsed.safe_for_reply is True


def test_image_to_text_reads_file_in_worker_thread(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image")

    captured = {"to_thread": False, "url": None}

    async def fake_to_thread(func, *args, **kwargs):
        captured["to_thread"] = True
        return func(*args, **kwargs)

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output_text": " image text "}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *args, **kwargs):
            captured["url"] = url
            return _FakeResponse()

    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("ARK_API_ENDPOINT", "https://example.test/api/v3/chat/completions")
    monkeypatch.setenv("LLM_PROVIDER", "ark")
    monkeypatch.setattr(ai_model, "_settings", None)
    monkeypatch.setattr(image2text.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(image2text.httpx, "AsyncClient", _FakeAsyncClient)

    result = asyncio.run(image2text.image_to_text(str(image_path)))

    assert captured["to_thread"] is True
    assert captured["url"] == "https://example.test/api/v3/responses"
    assert result == "image text"


def test_image_to_text_uses_openai_chat_vision_for_zan(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image")
    captured = {"url": None, "payload": None}

    async def fake_to_thread(func, *args):
        return func(*args)

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "image text"}}]}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *args, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            return _FakeResponse()

    monkeypatch.setenv("LLM_PROVIDER", "zan")
    monkeypatch.setenv("ZAN_OPENAI_BASE_URL", "https://ai.zan.top/v1")
    monkeypatch.setenv("ZAN_API_KEY", "test-key")
    monkeypatch.setenv("ZAN_MODEL", "gpt-5.6-luna")
    monkeypatch.setenv("ZAN_VISION_MODEL", "gpt-5.6-luna")
    monkeypatch.setattr(ai_model, "_settings", None)
    monkeypatch.setattr(image2text.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(image2text.httpx, "AsyncClient", _FakeAsyncClient)

    result = asyncio.run(image2text.image_to_text(str(image_path)))

    assert captured["url"] == "https://ai.zan.top/v1/chat/completions"
    assert captured["payload"]["model"] == "gpt-5.6-luna"
    assert captured["payload"]["messages"][0]["content"][1]["type"] == "image_url"
    assert result == "image text"
