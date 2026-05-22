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
        additional_context=["a1", "user_personal_memory:\nlikes tea"],
        runtime_state=["s1"],
        direct_address_state=["is_mentioned: true"],
    )
    assert "PART 1: EARLIER HISTORY" in prompt
    assert "m1" in prompt
    assert "PART 3: RAG RELATED MESSAGES" in prompt
    assert "r1" in prompt
    assert "PART 2: DURABLE CONTEXT" in prompt
    assert "likes tea" in prompt
    assert "PART 4: MESSAGE-SPECIFIC CONTEXT" in prompt
    assert "a1" in prompt
    assert "PART 5: DIRECT ADDRESS FLAGS" in prompt
    assert "is_mentioned: true" in prompt
    assert "PART 6: RUNTIME STATE" in prompt
    assert "s1" in prompt
    assert "PART 7: LATEST MESSAGE TO RESPOND TO" in prompt
    assert prompt.rstrip().endswith("m2")


def test_build_user_prompt_orders_cache_stable_context_before_volatile_context():
    prompt = reply2message._build_user_prompt(
        ["old message", "newest message"],
        rag_related_messages=["dynamic rag"],
        additional_context=[
            "user_personal_memory:\nstable memory",
            "replied_to_content: volatile reply context",
        ],
        runtime_state=["current_date_utc: 2026-04-30"],
        direct_address_state=["directly_addressed: true"],
    )

    assert prompt.index("PART 1: EARLIER HISTORY") < prompt.index("PART 2: DURABLE CONTEXT")
    assert prompt.index("PART 2: DURABLE CONTEXT") < prompt.index("PART 3: RAG RELATED MESSAGES")
    assert prompt.index("PART 3: RAG RELATED MESSAGES") < prompt.index("PART 4: MESSAGE-SPECIFIC CONTEXT")
    assert prompt.index("PART 4: MESSAGE-SPECIFIC CONTEXT") < prompt.index("PART 5: DIRECT ADDRESS FLAGS")
    assert prompt.index("PART 6: RUNTIME STATE") < prompt.index("PART 7: LATEST MESSAGE TO RESPOND TO")
    assert prompt.rstrip().endswith("newest message")


def test_should_activate_reply_returns_true_when_model_says_yes(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("line1\nline2\n", encoding="utf-8")

    called = {}

    async def fake_chat_completion(*, messages, **kwargs):
        called["messages"] = messages
        return _Completion(json.dumps({"should_reply": True, "reply_content": "nya~"}))

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(
        reply2message.should_activate_reply(
            ["u: hi"],
            is_mentioned=True,
        )
    )

    assert result is True
    system_prompt = called["messages"][0]["content"]
    user_prompt = called["messages"][1]["content"]
    assert "LATEST MESSAGE TO RESPOND TO" in system_prompt
    assert "is_mentioned: true" in user_prompt
    assert "directly_addressed: true" in user_prompt


def test_should_activate_reply_returns_false_on_invalid_json(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion("not-json")

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_activate_reply(["u: hi"]))
    assert result is False


def test_should_activate_reply_parses_fenced_json(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion("""```json\n{\"should_reply\": true, \"reason\": \"direct ask\"}\n```""")

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_activate_reply(["u: hi"]))
    assert result is True


def test_should_activate_reply_parses_json_with_prefixed_text(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion('Sure, here is the payload: {"should_reply": true, "reason": "question"}')

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_activate_reply(["u: hi"]))
    assert result is True


def test_should_activate_reply_can_return_generation_plan(monkeypatch):
    called = {}

    async def fake_chat_completion(*, messages, **kwargs):
        called["user_prompt"] = messages[1]["content"]
        return _Completion(
            json.dumps(
                {
                    "should_reply": True,
                    "reason": "direct question",
                    "reply_target": "sender",
                    "memory_focus": ["sender", "replied_to_author", "made_up"],
                    "conversation_intent": "answer_question",
                    "response_mode": "direct_answer",
                    "language_hint": "zh",
                    "needs_rag": True,
                    "rag_query_hint": "deploy pipeline",
                    "sensitivity": "technical",
                    "sticker_hint": "none",
                    "generation_notes": "answer briefly",
                }
            )
        )

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)

    result = asyncio.run(
        reply2message.should_activate_reply(
            ["u: hi"],
            available_memory_subjects=[
                {"key": "sender", "display": "Alice", "telegram_user_key": "tg_user:1", "role": "latest_message_author"},
                {"key": "replied_to_author", "display": "Bob", "telegram_user_key": "tg_user:2", "role": "replied_message_author"},
            ],
            return_decision=True,
        )
    )

    assert isinstance(result, reply2message.ReplyActivationDecision)
    assert result.should_reply is True
    assert result.memory_focus == ["sender", "replied_to_author"]
    assert result.conversation_intent == "answer_question"
    assert result.rag_query_hint == "deploy pipeline"
    assert "AVAILABLE MEMORY SUBJECTS" in called["user_prompt"]
    assert "key: sender" in called["user_prompt"]


def test_should_activate_reply_rejects_string_boolean_values(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion('{"should_reply": "true", "reason": "question"}')

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.should_activate_reply(["u: hi"]))
    assert result is False


def test_should_reply_and_generate_stops_when_probe_says_no(monkeypatch):
    calls = {"probe": 0, "generate": 0}

    async def fake_should_activate_reply(*args, **kwargs):
        calls["probe"] += 1
        return False

    async def fake_generate_group_reply(*args, **kwargs):
        calls["generate"] += 1
        return "nya~"

    monkeypatch.setattr(reply2message, "should_activate_reply", fake_should_activate_reply)
    monkeypatch.setattr(reply2message, "generate_group_reply", fake_generate_group_reply)

    result = asyncio.run(reply2message.should_reply_and_generate(["u: hi"]))

    assert result is None
    assert calls == {"probe": 1, "generate": 0}


def test_should_reply_and_generate_skips_probe_for_direct_trigger(monkeypatch):
    called = {}

    async def fail_should_activate_reply(*args, **kwargs):
        raise AssertionError("probe should not run for direct triggers")

    async def fake_generate_group_reply(*args, **kwargs):
        called["kwargs"] = kwargs
        return "nya~"

    monkeypatch.setattr(reply2message, "should_activate_reply", fail_should_activate_reply)
    monkeypatch.setattr(reply2message, "generate_group_reply", fake_generate_group_reply)

    result = asyncio.run(
        reply2message.should_reply_and_generate(
            ["u: hi"],
            is_mentioned=True,
            runtime_state=["trigger_type: alias_mention"],
        )
    )

    assert result == "nya~"
    assert called["kwargs"]["is_mentioned"] is True
    assert called["kwargs"]["runtime_state"] == ["trigger_type: alias_mention"]


def test_choose_reply_sticker_returns_allowed_candidate(monkeypatch):
    called = {}

    async def fake_chat_completion(*, messages, **kwargs):
        called["messages"] = messages
        called["kwargs"] = kwargs
        return _Completion(json.dumps({"file_unique_id": "sticker-1", "send_text": True, "reason": "playful laugh"}))

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)

    result = asyncio.run(
        reply2message.choose_reply_sticker(
            latest_message="mioo 哈哈",
            reply_text="笑死",
            sticker_candidates=[
                {
                    "file_unique_id": "sticker-1",
                    "emoji": "😂",
                    "set_name": "mio_pack",
                    "description": "laughing reaction",
                    "is_animated": False,
                    "is_video": False,
                }
            ],
            runtime_state=["trigger_type: alias_mention"],
        )
    )

    assert result is not None
    assert result.file_unique_id == "sticker-1"
    assert result.send_text is True
    assert called["kwargs"]["temperature"] == 0
    assert "Candidate stickers" in called["messages"][1]["content"]


def test_choose_reply_sticker_can_choose_sticker_only(monkeypatch):
    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion(json.dumps({"file_unique_id": "sticker-1", "send_text": False, "reason": "pure reaction"}))

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)

    result = asyncio.run(
        reply2message.choose_reply_sticker(
            latest_message="mioo 发个表情",
            reply_text="给你一个",
            sticker_candidates=[{"file_unique_id": "sticker-1", "description": "playful wave"}],
        )
    )

    assert result is not None
    assert result.file_unique_id == "sticker-1"
    assert result.send_text is False


def test_choose_reply_sticker_rejects_unknown_candidate(monkeypatch):
    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion(json.dumps({"file_unique_id": "missing", "send_text": False, "reason": "bad id"}))

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)

    result = asyncio.run(
        reply2message.choose_reply_sticker(
            latest_message="hello",
            reply_text="hi",
            sticker_candidates=[{"file_unique_id": "sticker-1", "description": "wave"}],
        )
    )

    assert result is None
