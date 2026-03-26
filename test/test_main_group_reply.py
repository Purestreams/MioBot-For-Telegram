import asyncio
from typing import Any, Optional

import main


class _FakeMessage:
    def __init__(self, text="", message_id=1):
        self.replies = []
        self.reply_to_message: Optional[Any] = None
        self.text = text
        self.caption = None
        self.from_user: Optional[Any] = None
        self.message_id = message_id

    async def reply_text(self, text):
        self.replies.append(text)


class _FakeChat:
    def __init__(self, chat_id=1):
        self.id = chat_id
        self.type = "group"


class _FakeUser:
    def __init__(self, name="tester", is_bot=False, username=None, user_id=999):
        self.full_name = name
        self.is_bot = is_bot
        self.username = username
        self.id = user_id


class _FakeUpdate:
    def __init__(self):
        self.message = _FakeMessage()
        self.effective_chat = _FakeChat()
        self.effective_user = _FakeUser()


def test_group_reply_pipeline_calls_rag_and_replies(monkeypatch):
    update = _FakeUpdate()

    calls = {"add": 0, "rag": 0, "reply": 0}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        calls["add"] += 1

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        calls["rag"] += 1
        return ["[t] user: hello"], ["[t] user: cats and fish"]

    async def fake_should_reply_and_generate(**kwargs):
        calls["reply"] += 1
        return "nya~"

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_reply_and_generate", fake_should_reply_and_generate)
    monkeypatch.setattr(main.random, "randint", lambda a, b: 1)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "hello cats"))

    # one add for user message, one add for bot reply
    assert calls["add"] == 2
    assert calls["rag"] == 1
    assert calls["reply"] == 1
    assert update.message.replies == ["nya~"]


def test_group_reply_pipeline_includes_reply_relation_context(monkeypatch):
    update = _FakeUpdate()
    update.message = _FakeMessage(text="A replying to B", message_id=101)
    update.message.from_user = _FakeUser(name="UserA", is_bot=False, username="user_a", user_id=10101)

    parent = _FakeMessage(text="Message B content", message_id=55)
    parent.from_user = _FakeUser(name="UserB", is_bot=False, username="user_b", user_id=20202)
    update.message.reply_to_message = parent

    captured = {"added": [], "additional_context": None}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        captured["added"].append(content)

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        return ["[t] user: hello"], []

    async def fake_should_reply_and_generate(**kwargs):
        captured["additional_context"] = kwargs.get("additional_context")
        return None

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_reply_and_generate", fake_should_reply_and_generate)
    monkeypatch.setattr(main.random, "randint", lambda a, b: 1)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "A replying to B"))

    assert captured["added"]
    assert captured["added"][0].startswith("[reply_to: UserB @user_b] Message B content")
    assert captured["additional_context"] is not None
    assert any("message_relation" in line for line in captured["additional_context"])
    assert any(
        "user_reply_relation: UserA @user_a replies to UserB @user_b" in line
        for line in captured["additional_context"]
    )
    assert any("message_reply_relation: message 101 replies to message 55" in line for line in captured["additional_context"])
    assert any("replied_to_author: UserB @user_b" in line for line in captured["additional_context"])
