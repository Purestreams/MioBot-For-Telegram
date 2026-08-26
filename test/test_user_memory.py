import asyncio
import datetime as dt
import json
import sqlite3

from app import database, user_memory


def test_refresh_user_memory_only_runs_once_per_day(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

    async def fake_chat_completion_text(**kwargs):
        return "likes concise answers\ninterested in iOS beta updates"

    monkeypatch.setattr(user_memory, "chat_completion_text", fake_chat_completion_text)

    async def _run():
        await database.add_message(
            10,
            "Alice @alice",
            "I am testing iOS 26.4 again",
            telegram_user_key="tg_user:111",
        )
        await database.add_message(
            10,
            "Alice @alice",
            "I prefer short answers",
            telegram_user_key="tg_user:111",
        )

        with sqlite3.connect(db_path) as db:
            db.execute(
                "UPDATE messages SET timestamp = '2026-04-27 10:00:00' WHERE telegram_user_key = 'tg_user:111'"
            )
            db.commit()

        first = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:111",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        second = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:111",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        row = await database.get_user_memory("tg_user:111")
        return first, second, row

    first, second, row = asyncio.run(_run())

    assert first == "likes concise answers\ninterested in iOS beta updates"
    assert second == first
    assert row is not None
    assert row.last_refreshed_date == "2026-04-27"

    facts = asyncio.run(database.get_user_memory_facts("tg_user:111"))
    assert sorted(fact.fact_text for fact in facts) == sorted([
        "likes concise answers",
        "interested in iOS beta updates",
    ])


def test_refresh_user_memory_marks_day_even_without_messages(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory_empty.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        result = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:222",
            latest_display_name="Bob @bob",
            today_utc=dt.date(2026, 4, 28),
        )
        row = await database.get_user_memory("tg_user:222")
        return result, row

    result, row = asyncio.run(_run())

    assert result is None
    assert row is not None
    assert row.memory_text == ""
    assert row.last_refreshed_date == "2026-04-27"


def test_refresh_user_memory_bootstraps_empty_memory_from_all_history(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory_bootstrap.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    captured = {}

    async def fake_chat_completion_text(**kwargs):
        source_prompt = kwargs["messages"][1]["content"]
        captured["source_prompt"] = source_prompt
        return json.dumps(
            {
                "memory_text": "prefers direct answers",
                "facts": [
                    {
                        "type": "preference",
                        "text": "Prefers direct answers",
                        "confidence": 0.8,
                        "evidence_message_ids": [1, 2],
                    }
                ],
            }
        )

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    monkeypatch.setattr(user_memory, "chat_completion_text", fake_chat_completion_text)

    async def _run():
        await database.add_message(
            10,
            "Alice @alice",
            "older preference message",
            telegram_user_key="tg_user:444",
        )
        await database.add_message(
            10,
            "Alice @alice",
            "second older preference message",
            telegram_user_key="tg_user:444",
        )
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE messages SET id = 1, timestamp = '2026-04-20 10:00:00' WHERE content = 'older preference message'")
            db.execute("UPDATE messages SET id = 2, timestamp = '2026-04-27 10:00:00' WHERE content = 'second older preference message'")
            db.commit()
        await database.upsert_user_memory(
            "tg_user:444",
            latest_display_name="Alice @alice",
            memory_text="",
            last_refreshed_date="2026-04-28",
        )

        memory_text = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:444",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        row = await database.get_user_memory("tg_user:444")
        facts = await database.get_user_memory_facts("tg_user:444")
        return memory_text, row, facts

    memory_text, row, facts = asyncio.run(_run())

    assert "older preference message" in captured["source_prompt"]
    assert "second older preference message" in captured["source_prompt"]
    assert memory_text == "prefers direct answers"
    assert row is not None
    assert row.last_refreshed_date == "2026-04-27"
    assert facts[0].fact_text == "Prefers direct answers"


def test_refresh_user_memory_parses_structured_facts(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory_facts.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

    async def fake_chat_completion_text(**kwargs):
        return json.dumps(
            {
                "memory_text": "prefers implementation-first plans",
                "facts": [
                    {
                        "type": "preference",
                        "text": "Prefers implementation-first plans",
                        "confidence": 0.84,
                        "evidence_message_ids": [1],
                    }
                ],
            }
        )

    monkeypatch.setattr(user_memory, "chat_completion_text", fake_chat_completion_text)

    async def _run():
        await database.add_message(
            10,
            "Alice @alice",
            "plan it, then implement it",
            telegram_user_key="tg_user:333",
        )
        with sqlite3.connect(db_path) as db:
            db.execute(
                "UPDATE messages SET id = 1, timestamp = '2026-04-27 10:00:00' WHERE telegram_user_key = 'tg_user:333'"
            )
            db.commit()

        memory_text = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:333",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        context = await user_memory.get_personal_memory_context("tg_user:333")
        facts = await database.get_user_memory_facts("tg_user:333")
        return memory_text, context, facts

    memory_text, context, facts = asyncio.run(_run())

    assert memory_text == "prefers implementation-first plans"
    assert context is not None
    assert "structured_facts:" in context
    assert "[preference] Prefers implementation-first plans" in context
    assert len(facts) == 1
    assert facts[0].confidence == 0.84


def test_personal_memory_context_prefers_query_relevant_facts(monkeypatch, tmp_path):
    db_path = tmp_path / "memory_relevance.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_user_memory_facts(
            "tg_user:777",
            [
                {
                    "fact_type": "preference",
                    "fact_text": "Prefers tea over coffee",
                    "confidence": 0.99,
                    "evidence_message_ids": [1],
                },
                {
                    "fact_type": "project",
                    "fact_text": "Working on the deploy pipeline",
                    "confidence": 0.55,
                    "evidence_message_ids": [2],
                },
            ],
        )
        return await user_memory.get_personal_memory_context(
            "tg_user:777",
            max_facts=1,
            query_text="help with deploy pipeline",
            intent="help_task",
        )

    context = asyncio.run(_run())

    assert context is not None
    assert "Working on the deploy pipeline" in context
    assert "Prefers tea over coffee" not in context


def test_global_memory_context_uses_same_selector(monkeypatch, tmp_path):
    db_path = tmp_path / "global_memory.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_global_memory_facts(
            -100,
            [
                {
                    "fact_type": "style",
                    "fact_text": "Keep replies warm and concise",
                    "confidence": 0.9,
                    "evidence_message_ids": [],
                },
                {
                    "fact_type": "project",
                    "fact_text": "The group often discusses the deploy pipeline",
                    "confidence": 0.6,
                    "evidence_message_ids": [3],
                },
            ]
        )
        return await user_memory.get_global_memory_context(
            -100,
            max_facts=1,
            query_text="deploy pipeline",
            intent="answer_question",
        )

    context = asyncio.run(_run())

    assert context is not None
    assert context.startswith("global_memory[chat_id=-100]:")
    assert "deploy pipeline" in context


def test_parse_memory_refresh_payload_coerces_list_memory_text():
    payload = user_memory._parse_memory_refresh_payload(
        json.dumps(
            {
                "memory_text": [
                    "prefers concise answers",
                    "tracks iOS beta releases",
                ],
                "facts": [],
                "archive_fact_ids": [],
            }
        ),
        [],
    )

    assert payload.memory_text == "prefers concise answers\ntracks iOS beta releases"


def test_parse_memory_refresh_payload_salvages_truncated_json_memory_text():
    payload = user_memory._parse_memory_refresh_payload(
        '{"memory_text": ["prefers concise answers", "tracks iOS beta releases"], "facts": [{"type": "preference"',
        [],
    )

    assert payload.memory_text == "prefers concise answers\ntracks iOS beta releases"
    assert [fact["fact_text"] for fact in payload.facts] == [
        "prefers concise answers",
        "tracks iOS beta releases",
    ]


def test_audit_user_memory_texts_flags_malformed_summaries(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory_audit.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_user_memory(
            "tg_user:111",
            latest_display_name="Alice @alice",
            memory_text='{"memory_text": ["prefers concise answers"]',
            last_refreshed_date="2026-04-29",
        )
        return await user_memory.audit_user_memory_texts(limit=10)

    findings = asyncio.run(_run())

    assert len(findings) == 1
    assert findings[0].telegram_user_key == "tg_user:111"
    assert findings[0].issue_types == ["json-blob", "normalizes-differently"]
    assert findings[0].normalized_length < findings[0].stored_length


def test_memory_refresh_prompt_keeps_metadata_after_source_context():
    messages = user_memory._build_memory_messages(
        display_name="Alice @alice",
        existing_memory="stable summary",
        existing_facts="stable facts",
        pending_candidates="candidate facts",
        source_messages="source message",
        target_end_date="2026-04-29",
    )
    prompt = messages[1]["content"]

    assert prompt.index("Existing memory:") < prompt.index("Existing structured facts:")
    assert prompt.index("Existing structured facts:") < prompt.index("Pending memory candidates:")
    assert prompt.index("Pending memory candidates:") < prompt.index("New source messages to fold in:")
    assert prompt.index("New source messages to fold in:") < prompt.index("Update metadata:")
    assert prompt.rstrip().endswith("Update coverage end date (UTC): 2026-04-29")


def test_memory_prompt_rejects_future_bot_instructions():
    messages = user_memory._build_memory_messages(
        display_name="Alice",
        existing_memory="",
        existing_facts="",
        pending_candidates="",
        source_messages="",
        target_end_date="2026-08-11",
    )
    system_prompt = messages[0]["content"]
    assert "Never store instructions about how the bot should respond" in system_prompt
    assert "police or ambulance guidance" in system_prompt


def test_sanitize_memory_lines_filters_directive_like_crisis_rules():
    lines = user_memory._sanitize_memory_lines(
        [
            "Likes short answers.",
            "If severe distress recurs, prioritize immediate safety assessment and crisis support.",
            "以后需要早睡。",
        ]
    )
    assert lines == ["Likes short answers.", "以后需要早睡。"]


def test_memory_candidate_extraction_and_admin_accept(monkeypatch, tmp_path):
    db_path = tmp_path / "candidate_extract.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        candidate_id = await user_memory.extract_user_memory_candidate_from_message(
            telegram_user_key="tg_user:555",
            message_text="以后回答请短一点，我喜欢直接答案",
            message_id=99,
        )
        accepted = await user_memory.accept_user_memory_candidate(candidate_id)
        facts = await database.get_user_memory_facts("tg_user:555")
        candidate = await database.get_user_memory_candidate(candidate_id)
        return candidate_id, accepted, facts, candidate

    candidate_id, accepted, facts, candidate = asyncio.run(_run())

    assert candidate_id is not None
    assert accepted is True
    assert len(facts) == 1
    assert facts[0].fact_type in {"preference", "style"}
    assert facts[0].evidence_message_ids == [99]
    assert candidate is not None
    assert candidate.status == "accepted"


def test_refresh_user_memory_consolidates_pending_candidates_and_archives(monkeypatch, tmp_path):
    db_path = tmp_path / "candidate_refresh.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    captured = {}

    async def fake_chat_completion_text(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        captured["prompt"] = prompt
        return json.dumps(
            {
                "memory_text": "prefers short direct answers",
                "facts": [
                    {
                        "type": "style",
                        "text": "Prefers short direct answers",
                        "confidence": 0.9,
                        "evidence_message_ids": [2],
                    }
                ],
                "archive_fact_ids": [1],
            }
        )

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    monkeypatch.setattr(user_memory, "chat_completion_text", fake_chat_completion_text)

    async def _run():
        await database.add_message(
            10,
            "Alice @alice",
            "以后回答请短一点",
            telegram_user_key="tg_user:666",
        )
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE messages SET id = 2, timestamp = '2026-04-28 10:00:00' WHERE telegram_user_key = 'tg_user:666'")
            db.commit()
        await database.upsert_user_memory_facts(
            "tg_user:666",
            [
                {
                    "fact_type": "style",
                    "fact_text": "Prefers long detailed answers",
                    "confidence": 0.6,
                    "evidence_message_ids": [1],
                }
            ],
        )
        await database.upsert_user_memory_candidate(
            "tg_user:666",
            fact_type="style",
            fact_text="Prefers short direct answers",
            confidence=0.9,
            evidence_message_ids=[2],
            source_message_id=2,
            priority="fast",
        )

        memory_text = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:666",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        facts = await database.get_user_memory_facts("tg_user:666", limit=10)
        candidates = await database.list_user_memory_candidates("tg_user:666", status="accepted")
        return memory_text, facts, candidates

    memory_text, facts, candidates = asyncio.run(_run())

    assert "candidate #" in captured["prompt"]
    assert memory_text == "prefers short direct answers"
    assert [fact.fact_text for fact in facts] == ["Prefers short direct answers"]
    assert len(candidates) == 1
