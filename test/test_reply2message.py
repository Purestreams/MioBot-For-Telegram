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


def test_cacheable_context_messages_preserve_history_prefix_across_turns():
    first_turn = reply2message._build_cacheable_context_messages(
        ["Alice: first", "Bob: second"],
        additional_context=["user_personal_memory:\nlikes tea"],
    )
    second_turn = reply2message._build_cacheable_context_messages(
        ["Alice: first", "Bob: second", "Alice: third"],
        additional_context=["user_personal_memory:\nlikes tea"],
    )

    assert first_turn[0] == second_turn[0]
    assert second_turn[1]["content"].endswith("Bob: second")
    assert all("DURABLE CONTEXT" not in message["content"] for message in second_turn[:2])
    assert second_turn[-1]["content"].endswith("Alice: third")


def test_cacheable_context_messages_keep_latest_message_last():
    messages = reply2message._build_cacheable_context_messages(
        ["old", "latest"],
        available_memory_subjects=[{"key": "sender", "display": "Alice", "role": "latest_message_author"}],
        include_memory_subjects=True,
    )

    available_index = next(index for index, message in enumerate(messages) if "AVAILABLE MEMORY SUBJECTS" in message["content"])
    assert available_index < len(messages) - 1
    assert "LATEST MESSAGE TO RESPOND TO" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("latest")


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
    system_prompt = called["messages"][1]["content"]
    user_prompt = "\n".join(message["content"] for message in called["messages"][2:])
    assert "LATEST MESSAGE TO RESPOND TO" in system_prompt
    assert "is_mentioned: true" in user_prompt
    assert "directly_addressed: true" in user_prompt


def test_generation_context_omits_probe_only_memory_subjects():
    messages = reply2message._build_cacheable_context_messages(
        ["old", "latest"],
        available_memory_subjects=[{"key": "sender", "display": "Alice", "role": "latest_message_author"}],
    )

    assert all("AVAILABLE MEMORY SUBJECTS" not in message["content"] for message in messages)
    assert "LATEST MESSAGE TO RESPOND TO" in messages[-1]["content"]


def test_probe_prompt_is_identical_across_models():
    luna_prompt = reply2message._build_probe_system_prompt(model="gpt-5.6-luna")
    generic_prompt = reply2message._build_probe_system_prompt(model="another-model")

    assert luna_prompt == generic_prompt
    assert "Never transfer medical, self-harm, or violence assumptions between speakers" in luna_prompt
    assert "Never put police" in luna_prompt


def test_generation_prompt_is_identical_across_models_and_uses_json():
    luna_prompt = reply2message._build_generation_system_prompt(
        information_lines=[],
        model="gpt-5.6-luna",
    )
    generic_prompt = reply2message._build_generation_system_prompt(
        information_lines=[],
        model="another-model",
    )

    assert luna_prompt == generic_prompt
    assert "Offer warmth and emotional acknowledgement before advice" in luna_prompt
    assert "Never mention police" in luna_prompt
    assert '"reply_content" and "support_level"' in luna_prompt


def test_normalize_reply_content_removes_thinking_tags_anywhere():
    assert reply2message._normalize_reply_content("你好</think_never_used_abc>") == "你好"
    assert reply2message._normalize_reply_content("<think>hidden</think>最终回复") == "最终回复"


def test_group_reply_violation_detects_thinking_tags_on_unsanitized_text():
    assert reply2message.group_reply_violation("<think>hidden</think>最终回复") == "thinking_tag"
    assert reply2message.group_reply_violation("你好</think_never_used_abc>") == "thinking_tag"
    assert reply2message.group_reply_violation("最终回复") is None


def test_group_reply_guard_detects_forbidden_escalation_and_false_actions():
    assert reply2message.group_reply_violation("请马上拨打119") == "emergency_escalation"
    assert reply2message.group_reply_violation("call the police") == "emergency_escalation"
    assert reply2message.group_reply_violation("这就生成公钥发你") == "false_external_action"
    assert reply2message.group_reply_violation("我在这里听你说") is None
    assert reply2message.group_reply_violation("这是电影里的警察台词") is None
    assert reply2message.group_reply_violation("打不了911啊这是玩梗吧") is None


def test_guard_fallback_does_not_add_trusted_person_advice_for_non_danger():
    assert "信任" not in reply2message._guard_fallback("thinking_tag")
    assert "信任" not in reply2message._guard_fallback("emergency_escalation", support_level="normal")
    assert "信任" in reply2message._guard_fallback(
        "emergency_escalation",
        support_level="explicit_current_danger",
    )
    assert reply2message._guard_fallback("false_external_action") == reply2message.CAPABILITY_FALLBACK


def test_disabled_thinking_extra_body_is_provider_specific(monkeypatch):
    monkeypatch.setattr(
        reply2message,
        "get_settings",
        lambda: type("Settings", (), {"provider": reply2message.LLMProvider.AZURE})(),
    )
    assert reply2message._disabled_thinking_extra_body() is None

    monkeypatch.setattr(
        reply2message,
        "get_settings",
        lambda: type("Settings", (), {"provider": reply2message.LLMProvider.ZAN})(),
    )
    assert reply2message._disabled_thinking_extra_body() == {"thinking": {"type": "disabled"}}

    monkeypatch.setattr(
        reply2message,
        "get_settings",
        lambda: type("Settings", (), {"provider": reply2message.LLMProvider.OLLAMA})(),
    )
    assert reply2message._disabled_thinking_extra_body() is None


def test_generate_group_reply_thinking_tag_fallback_is_not_crisis(monkeypatch, tmp_path):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")

    async def fake_chat_completion(*, messages, **kwargs):
        return _Completion(
            json.dumps({"reply_content": "哈哈 think_never_used", "support_level": "normal"})
        )

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.generate_group_reply(["u: hi"], return_result=True))

    assert isinstance(result, reply2message.GeneratedGroupReply)
    assert result.reply_content == reply2message.LISTENING_FALLBACK
    assert "信任" not in result.reply_content
    assert result.guard_repaired is True
    assert result.forbidden_pattern == "thinking_tag"


def test_generate_group_reply_strips_well_formed_think_tags_without_discarding_reply(
    monkeypatch, tmp_path
):
    info_file = tmp_path / "info.txt"
    info_file.write_text("x\n", encoding="utf-8")
    calls = {"n": 0}

    async def fake_chat_completion(*, messages, **kwargs):
        calls["n"] += 1
        return _Completion(
            json.dumps(
                {
                    "reply_content": "<think>hidden reasoning</think>最终回复",
                    "support_level": "normal",
                }
            )
        )

    monkeypatch.setattr(reply2message, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(reply2message, "INFO_FILE_PATH", info_file)

    result = asyncio.run(reply2message.generate_group_reply(["u: hi"], return_result=True))

    assert isinstance(result, reply2message.GeneratedGroupReply)
    assert result.reply_content == "最终回复"
    assert result.guard_repaired is True
    assert result.forbidden_pattern == "thinking_tag"
    assert calls["n"] == 1


def test_direct_reply_activation_disables_rag_by_default():
    assert reply2message.direct_reply_activation_decision().needs_rag is False
    assert reply2message.direct_reply_activation_decision(needs_rag=True).needs_rag is True


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
        called["user_prompt"] = "\n".join(message["content"] for message in messages[1:])
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
