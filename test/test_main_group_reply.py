import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import main


class _FakeMessage:
    def __init__(self, text="", message_id=1):
        self.replies = []
        self.reply_to_message: Optional[Any] = None
        self.text = text
        self.caption = None
        self.photo = None
        self.sticker = None
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

    calls = {"add": 0, "context": [], "probe": 0, "generate": 0}
    captured = {"add_kwargs": []}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        calls["add"] += 1
        captured["add_kwargs"].append(kwargs)

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        calls["context"].append(query)
        return ["[t] user: hello"], ["[t] user: cats and fish"]

    async def fake_should_activate_reply(**kwargs):
        calls["probe"] += 1
        return True

    async def fake_generate_group_reply(**kwargs):
        calls["generate"] += 1
        assert any("user_personal_memory:" in line for line in (kwargs.get("additional_context") or []))
        return "nya~"

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        assert telegram_user_key == "tg_user:999"
        return "likes short answers"

    def fake_schedule_personal_memory_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "hello cats"))

    assert calls["add"] == 2
    assert calls["context"] == ["", "hello cats | tester @[999]"]
    assert calls["probe"] == 1
    assert calls["generate"] == 1
    assert update.message.replies == ["nya~"]
    assert captured["add_kwargs"][0]["telegram_user_key"] == "tg_user:999"


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

    async def fake_should_activate_reply(**kwargs):
        return True

    async def fake_generate_group_reply(**kwargs):
        captured["additional_context"] = kwargs.get("additional_context")
        return None

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        return "prefers direct answers"

    def fake_schedule_personal_memory_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "A replying to B"))

    assert captured["added"]
    assert captured["added"][0] == "A replying to B"
    assert captured["additional_context"] is not None
    assert any("message_relation" in line for line in captured["additional_context"])
    assert any(
        "user_reply_relation: UserA @user_a replies to UserB @user_b" in line
        for line in captured["additional_context"]
    )
    assert any("message_reply_relation: message 101 replies to message 55" in line for line in captured["additional_context"])
    assert any("replied_to_author: UserB @user_b" in line for line in captured["additional_context"])
    assert any("replied_to_content: Message B content" in line for line in captured["additional_context"])
    assert any("user_personal_memory:" in line for line in captured["additional_context"])


def test_group_reply_pipeline_stops_after_negative_probe(monkeypatch):
    update = _FakeUpdate()

    calls = {"add": 0, "context": [], "probe": 0, "generate": 0}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        calls["add"] += 1

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        calls["context"].append(query)
        return ["[t] user: hello"], []

    async def fake_should_activate_reply(**kwargs):
        calls["probe"] += 1
        return False

    async def fake_generate_group_reply(**kwargs):
        calls["generate"] += 1
        return "should not happen"

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        return None

    def fake_schedule_personal_memory_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "hello cats"))

    assert calls["add"] == 1
    assert calls["context"] == [""]
    assert calls["probe"] == 1
    assert calls["generate"] == 0
    assert update.message.replies == []


def test_group_reply_pipeline_direct_mention_bypasses_probe(monkeypatch):
    update = _FakeUpdate()
    update.message = _FakeMessage(text="mioo look here", message_id=77)
    update.message.from_user = _FakeUser(name="UserA", is_bot=False, username="user_a", user_id=10101)
    update.effective_user = update.message.from_user

    calls = {"probe": 0, "generate": 0, "context": []}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        return None

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        calls["context"].append(query)
        return ["[t] user: hello"], ["[t] user: mioo look here"]

    async def fake_should_activate_reply(**kwargs):
        calls["probe"] += 1
        return True

    async def fake_generate_group_reply(**kwargs):
        calls["generate"] += 1
        assert kwargs["is_mentioned"] is True
        assert kwargs["runtime_state"] is not None
        assert any("trigger_type: alias_mention" in line for line in kwargs["runtime_state"])
        assert any("user_personal_memory:" in line for line in (kwargs.get("additional_context") or []))
        return "在呢"

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        return "often pings Mioo directly"

    def fake_schedule_personal_memory_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "mioo look here"))

    assert calls["probe"] == 0
    assert calls["generate"] == 1
    assert calls["context"] == ["mioo look here | UserA @user_a"]
    assert update.message.replies == ["在呢"]


def test_group_reply_pipeline_uses_cached_memory_and_schedules_refresh(monkeypatch):
    update = _FakeUpdate()
    update.message = _FakeMessage(text="mioo help", message_id=88)
    update.message.from_user = _FakeUser(name="UserA", is_bot=False, username="user_a", user_id=10101)
    update.effective_user = update.message.from_user

    captured = {"additional_context": [], "probe": 0, "generate": 0, "scheduled": None}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        return None

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        return ["[t] UserA @user_a: mioo help"], []

    async def fake_should_activate_reply(**kwargs):
        captured["probe"] += 1
        return True

    async def fake_generate_group_reply(**kwargs):
        captured["generate"] += 1
        captured["additional_context"] = kwargs.get("additional_context") or []
        return "来了"

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        assert telegram_user_key == "tg_user:10101"
        return "structured_facts:\n- [preference] likes direct fixes"

    def fake_schedule_personal_memory_refresh(context, telegram_user_key, latest_display_name):
        captured["scheduled"] = (context, telegram_user_key, latest_display_name)

    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "mioo help"))

    assert captured["probe"] == 0
    assert captured["generate"] == 1
    assert any("user_personal_memory:" in line for line in captured["additional_context"])
    assert captured["scheduled"] == (None, "tg_user:10101", "UserA @user_a")
    assert update.message.replies == ["来了"]


def test_group_reply_pipeline_skips_bot_senders(monkeypatch):
    update = _FakeUpdate()
    bot_user = _FakeUser(name="RelayBot", is_bot=True, username="relay_bot", user_id=4242)
    update.message.from_user = bot_user
    update.effective_user = bot_user

    async def fail_add_message(*args, **kwargs):
        raise AssertionError("bot sender should be ignored before writing history")

    async def fail_get_personal_memory_context(*args, **kwargs):
        raise AssertionError("bot sender should not load personal memory")

    async def fail_should_activate_reply(*args, **kwargs):
        raise AssertionError("bot sender should not trigger the reply probe")

    async def fail_generate_group_reply(*args, **kwargs):
        raise AssertionError("bot sender should never generate a reply")

    monkeypatch.setattr(main, "add_message", fail_add_message)
    monkeypatch.setattr(main, "get_personal_memory_context", fail_get_personal_memory_context)
    monkeypatch.setattr(main, "should_activate_reply", fail_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fail_generate_group_reply)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "status update"))

    assert update.message.replies == []


def test_group_reply_pipeline_reply_to_bot_is_case_insensitive(monkeypatch):
    update = _FakeUpdate()
    update.message = _FakeMessage(text="what do you think", message_id=91)
    update.message.from_user = _FakeUser(name="UserA", is_bot=False, username="user_a", user_id=10101)
    update.effective_user = update.message.from_user
    update.message.reply_to_message = _FakeMessage(text="prior bot reply", message_id=90)
    update.message.reply_to_message.from_user = _FakeUser(name="Mioo", is_bot=True, username="MioBot", user_id=777)

    calls = {"probe": 0, "generate": 0}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        return None

    async def fake_get_prompt_context_parts(chat_id, query, recent_n=None, retrieved_k=None):
        return ["[t] user: hello"], ["[t] mio: prior bot reply"]

    async def fail_should_activate_reply(**kwargs):
        calls["probe"] += 1
        raise AssertionError("reply-to-bot should bypass the reply probe")

    async def fake_generate_group_reply(**kwargs):
        calls["generate"] += 1
        assert kwargs["is_reply_to_bot"] is True
        assert any("trigger_type: reply_to_bot" in line for line in (kwargs.get("runtime_state") or []))
        return "在"

    async def fake_get_personal_memory_context(telegram_user_key, **kwargs):
        return None

    def fake_schedule_personal_memory_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "TELEGRAM_BOT_USERNAME", "miobot")
    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main, "get_prompt_context_parts", fake_get_prompt_context_parts)
    monkeypatch.setattr(main, "should_activate_reply", fail_should_activate_reply)
    monkeypatch.setattr(main, "generate_group_reply", fake_generate_group_reply)
    monkeypatch.setattr(main, "get_personal_memory_context", fake_get_personal_memory_context)
    monkeypatch.setattr(main, "_schedule_personal_memory_refresh", fake_schedule_personal_memory_refresh)

    update_any: Any = update
    asyncio.run(main._handle_group_ai_reply_pipeline(update_any, "what do you think"))

    assert calls == {"probe": 0, "generate": 1}
    assert update.message.replies == ["在"]


def test_handle_sticker_for_group_ai_reply_uses_cached_description(monkeypatch):
    update = _FakeUpdate()
    update.message.sticker = SimpleNamespace(
        file_unique_id="sticker-1",
        file_id="file-1",
        emoji="🙂",
        set_name="mio_pack",
        is_animated=False,
        is_video=False,
        thumbnail=None,
    )
    captured = {}

    async def fake_get_sticker_text(file_unique_id):
        assert file_unique_id == "sticker-1"
        return "smiling cat waving"

    async def fake_pipeline(update_arg, message_text, *, additional_context=None, context=None):
        captured["message_text"] = message_text
        captured["additional_context"] = additional_context

    monkeypatch.setattr(main, "get_sticker_text", fake_get_sticker_text)
    monkeypatch.setattr(main, "_handle_group_ai_reply_pipeline", fake_pipeline)

    update_any: Any = update
    context_any: Any = SimpleNamespace(bot=None)
    asyncio.run(main.handle_sticker_for_group_ai_reply(update_any, context_any))

    assert captured["message_text"] == "sticker: smiling cat waving"
    assert any("input_type: sticker" in line for line in captured["additional_context"])
    assert any("sticker_cached: true" in line for line in captured["additional_context"])


def test_handle_sticker_for_group_ai_reply_reads_and_caches_new_sticker(monkeypatch, tmp_path):
    update = _FakeUpdate()
    update.message.sticker = SimpleNamespace(
        file_unique_id="sticker-2",
        file_id="file-2",
        emoji="😾",
        set_name="mio_pack",
        is_animated=False,
        is_video=False,
        thumbnail=None,
    )

    cached = {"upsert": None, "message_text": None, "context": None, "downloaded": None}
    sticker_path = tmp_path / "sticker.webp"

    class _FakeTelegramFile:
        async def download_to_drive(self, custom_path):
            Path(custom_path).write_bytes(b"img")
            cached["downloaded"] = custom_path
            return custom_path

    class _FakeBot:
        async def get_file(self, file_id):
            assert file_id == "file-2"
            return _FakeTelegramFile()

    async def fake_get_sticker_text(file_unique_id):
        assert file_unique_id == "sticker-2"
        return None

    async def fake_upsert_sticker_text(file_unique_id, **kwargs):
        cached["upsert"] = {"file_unique_id": file_unique_id, **kwargs}

    async def fake_sticker_to_text(image_path, *, emoji=None, set_name=None, model=None):
        assert Path(image_path).exists()
        assert emoji == "😾"
        assert set_name == "mio_pack"
        return "angry cat glaring"

    async def fake_pipeline(update_arg, message_text, *, additional_context=None, context=None):
        cached["message_text"] = message_text
        cached["context"] = additional_context

    monkeypatch.setattr(main, "get_sticker_text", fake_get_sticker_text)
    monkeypatch.setattr(main, "upsert_sticker_text", fake_upsert_sticker_text)
    monkeypatch.setattr(main, "sticker_to_text", fake_sticker_to_text)
    monkeypatch.setattr(main, "_handle_group_ai_reply_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_build_output_path", lambda prefix, message_id, extension="jpg": str(sticker_path))

    update_any: Any = update
    context_any: Any = SimpleNamespace(bot=_FakeBot())
    asyncio.run(main.handle_sticker_for_group_ai_reply(update_any, context_any))

    assert cached["message_text"] == "sticker: angry cat glaring"
    assert cached["upsert"] is not None
    assert cached["upsert"]["description"] == "angry cat glaring"
    assert cached["upsert"]["description_source"] == "sticker_file"
    assert any("sticker_cached: false" in line for line in cached["context"])
    assert not sticker_path.exists()
