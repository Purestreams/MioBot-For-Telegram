import asyncio

import app.text2md as text2md


def test_plain_text_to_markdown_strips_result(monkeypatch):
    captured = {}

    async def fake_chat_completion_text(*, messages, model=None):
        captured["messages"] = messages
        captured["model"] = model
        return "  # Title\ncontent  "

    monkeypatch.setattr(text2md, "chat_completion_text", fake_chat_completion_text)

    result = asyncio.run(text2md.plain_text_to_markdown("hello", model="m1"))
    assert result == "# Title\ncontent"
    assert captured["model"] == "m1"
    assert "hello" in captured["messages"][1]["content"]
