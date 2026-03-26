import asyncio
import json

import app.reply2message as reply2message


class _Completion:
    def __init__(self, content: str):
        self.content = content


def test_build_user_prompt_contains_all_sections():
    prompt = reply2message._build_user_prompt(
        ["m1", "m2"],
        rag_related_messages=["r1"],
        additional_context=["a1"],
    )
    assert "PART 1: HISTORY MESSAGE" in prompt
    assert "m1\nm2" in prompt
    assert "PART 2: RAG RELATED MESSAGE" in prompt
    assert "r1" in prompt
    assert "PART 3: ADDITIONAL IMPORTANT CONTEXT" in prompt
    assert "a1" in prompt


def test_should_reply_and_generate_returns_reply_when_model_says_yes(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("line1\nline2\n", encoding="utf-8")

    called = {}

    async def fake_chat_completion(*, messages, **kwargs):
        called["messages"] = messages
        return _Completion(json.dumps({"should_reply": True, "reply_content": "nya~"}))

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(
        reply2message.should_reply_and_generate(
            ["u: hi"],
            is_reply_to_bot=True,
        )
    )

    assert result == "nya~"
    system_prompt = called["messages"][0]["content"]
    assert "must_reply = True" in system_prompt


def test_should_reply_and_generate_returns_none_on_invalid_json(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion("not-json")

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_reply_and_generate(["u: hi"]))
    assert result is None


def test_should_reply_and_generate_parses_fenced_json(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion("""```json\n{\"should_reply\": true, \"reply_content\": \"nya~\"}\n```""")

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_reply_and_generate(["u: hi"]))
    assert result == "nya~"


def test_should_reply_and_generate_parses_json_with_prefixed_text(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion(
            'Sure, here is the payload: {"should_reply": true, "reply_content": "meow"}'
        )

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_reply_and_generate(["u: hi"]))
    assert result == "meow"
