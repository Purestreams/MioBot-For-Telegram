import asyncio
import sqlite3

import numpy as np

from app import database
from app.database import MessageRow, _cosine_top_k, _format_message
from app.rag_embeddings import EmbeddingMetadata


def test_format_message_truncates_long_content():
    row = MessageRow(
        id=1,
        chat_id=10,
        username="alice",
        content="x" * 20,
        timestamp="2026-03-24 00:00:00",
    )
    text = _format_message(row, max_chars=10)
    assert text.startswith("[2026-03-24 00:00:00] alice: ")
    assert text.endswith("…")


def test_format_message_includes_reply_to_username_without_changing_stored_content():
    row = MessageRow(
        id=1,
        chat_id=10,
        username="alice",
        content="hello there",
        timestamp="2026-03-24 00:00:00",
        reply_to_username="bob @bob",
    )

    text = _format_message(row)

    assert text == "[2026-03-24 00:00:00] alice: [reply to bob @bob] hello there"


def test_cosine_top_k_returns_highest_similarity_indices():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    idx = _cosine_top_k(query, matrix, top_k=2)
    # First vector should rank highest, second vector should be next.
    assert list(idx) == [0, 1]


def test_get_env_int_falls_back_for_invalid_value(monkeypatch):
    from app.database import _get_env_int

    monkeypatch.setenv("RAG_TOP_K", "not-an-int")
    assert _get_env_int("RAG_TOP_K", 12) == 12


def test_sticker_text_round_trip(monkeypatch, tmp_path):
    db_path = tmp_path / "stickers.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run() -> str | None:
        await database.upsert_sticker_text(
            "sticker-1",
            file_id="file-1",
            emoji="🙂",
            set_name="mio_pack",
            description="smiling cat waving",
            description_source="sticker_file",
            tags=["smile", "wave"],
            mood="happy",
        )
        return await database.get_sticker_text("sticker-1")

    assert asyncio.run(_run()) == "smiling cat waving"


def test_webadmin_login_token_is_single_use(monkeypatch, tmp_path):
    db_path = tmp_path / "webadmin_token.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        created = await database.create_webadmin_login_token(
            "hash-1",
            admin_user_id=42,
            admin_username="admin",
            ttl_seconds=600,
        )
        first = await database.consume_webadmin_login_token("hash-1")
        second = await database.consume_webadmin_login_token("hash-1")
        return created, first, second

    created, first, second = asyncio.run(_run())

    assert created.admin_user_id == 42
    assert first is not None
    assert first.admin_username == "admin"
    assert second is None


def test_webadmin_chat_messages_filter_by_search(monkeypatch, tmp_path):
    db_path = tmp_path / "webadmin_messages.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    database.init_db()

    async def _run():
        await database.add_message(100, "Alice @alice", "keep this message", telegram_user_key="tg_user:1")
        await database.add_message(100, "Bob @bob", "something else", telegram_user_key="tg_user:2")
        await database.add_message(200, "Carol @carol", "keep in other chat", telegram_user_key="tg_user:3")
        return await database.list_webadmin_chat_messages(100, search="keep", limit=10)

    rows = asyncio.run(_run())

    assert len(rows) == 1
    assert rows[0].chat_id == 100
    assert rows[0].username == "Alice @alice"
    assert rows[0].telegram_user_key == "tg_user:1"


def test_find_sticker_reply_candidates_prefers_matching_descriptions(monkeypatch, tmp_path):
    db_path = tmp_path / "sticker_candidates.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_sticker_text(
            "sticker-laugh",
            file_id="file-laugh",
            emoji="😂",
            set_name="mio_pack",
            description="laughing reaction with big smile",
            description_source="sticker_file",
            tags=["laugh", "smile"],
            mood="happy",
        )
        await database.upsert_sticker_text(
            "sticker-sad",
            file_id="file-sad",
            emoji="😢",
            set_name="mio_pack",
            description="sad face crying",
            description_source="sticker_file",
            tags=["sad", "cry"],
            mood="sad",
        )
        await database.upsert_sticker_text(
            "sticker-without-file",
            file_id=None,
            emoji="🙂",
            set_name="mio_pack",
            description="happy smile but cannot be sent",
            description_source="sticker_file",
            tags=["happy"],
            mood="happy",
        )
        return await database.find_sticker_reply_candidates("哈哈 that was funny", limit=2)

    candidates = asyncio.run(_run())

    assert [candidate.file_unique_id for candidate in candidates] == ["sticker-laugh"]
    assert candidates[0].file_id == "file-laugh"
    assert candidates[0].tags == ["laugh", "smile"]
    assert candidates[0].mood == "happy"


def test_sticker_reply_candidates_skip_unsafe_and_deprioritize_recently_used(monkeypatch, tmp_path):
    db_path = tmp_path / "sticker_quality.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    monkeypatch.setenv("STICKER_REPLY_COOLDOWN_MINUTES", "60")
    database.init_db()

    async def _run():
        await database.upsert_sticker_text(
            "safe-used",
            file_id="file-used",
            emoji="🙂",
            set_name="mio_pack",
            description="happy smile reaction",
            description_source="sticker_file",
            tags=["happy", "smile"],
            mood="happy",
        )
        await database.upsert_sticker_text(
            "safe-fresh",
            file_id="file-fresh",
            emoji="🙂",
            set_name="mio_pack",
            description="happy smile reaction",
            description_source="sticker_file",
            tags=["happy", "smile"],
            mood="happy",
        )
        await database.upsert_sticker_text(
            "unsafe",
            file_id="file-unsafe",
            emoji="🙂",
            set_name="mio_pack",
            description="happy smile reaction",
            description_source="sticker_file",
            tags=["happy", "smile"],
            mood="happy",
            safe_for_reply=False,
        )
        await database.record_sticker_reply_usage("safe-used")
        return await database.find_sticker_reply_candidates("happy smile", limit=3)

    candidates = asyncio.run(_run())

    assert [candidate.file_unique_id for candidate in candidates] == ["safe-fresh", "safe-used"]
    assert candidates[1].use_count == 1
    assert candidates[1].last_used_at is not None


def test_add_message_releases_write_lock_before_embedding(monkeypatch, tmp_path):
    db_path = tmp_path / "messages.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run() -> None:
        embedding_started = asyncio.Event()
        release_embedding = asyncio.Event()

        async def fake_embed_message_content(username: str, content: str):
            embedding_started.set()
            await release_embedding.wait()
            return (
                np.array([1.0, 0.0], dtype=np.float32),
                EmbeddingMetadata(
                    backend="test",
                    model="test-model",
                    dim=2,
                    signature="test:2",
                ),
            )

        monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

        task = asyncio.create_task(database.add_message(1, "alice", "hello"))
        await asyncio.wait_for(embedding_started.wait(), timeout=1.0)

        with sqlite3.connect(db_path, timeout=0.1) as db:
            db.execute(
                "INSERT INTO messages (chat_id, username, content) VALUES (?, ?, ?)",
                (1, "bob", "write while embedding is paused"),
            )
            db.commit()

        release_embedding.set()
        await task

    asyncio.run(_run())


def test_user_memory_facts_round_trip_and_merge(monkeypatch, tmp_path):
    db_path = tmp_path / "facts.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_user_memory_facts(
            "tg_user:1",
            [
                {
                    "fact_type": "preference",
                    "fact_text": "Prefers concise engineering answers",
                    "confidence": 0.7,
                    "evidence_message_ids": [1, 2],
                }
            ],
        )
        await database.upsert_user_memory_facts(
            "tg_user:1",
            [
                {
                    "type": "preference",
                    "text": "Prefers concise engineering answers",
                    "confidence": 0.9,
                    "evidence_message_ids": [2, 3],
                }
            ],
        )
        return await database.get_user_memory_facts("tg_user:1")

    facts = asyncio.run(_run())

    assert len(facts) == 1
    assert facts[0].fact_type == "preference"
    assert facts[0].fact_text == "Prefers concise engineering answers"
    assert facts[0].confidence == 0.9
    assert facts[0].evidence_message_ids == [1, 2, 3]


def test_global_memory_facts_round_trip_and_merge(monkeypatch, tmp_path):
    db_path = tmp_path / "global_facts.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        await database.upsert_global_memory_facts(
            -100,
            [
                {
                    "fact_type": "style",
                    "fact_text": "Keep group replies concise",
                    "confidence": 0.7,
                    "evidence_message_ids": [1],
                }
            ]
        )
        await database.upsert_global_memory_facts(
            -100,
            [
                {
                    "type": "style",
                    "text": "Keep group replies concise",
                    "confidence": 0.95,
                    "evidence_message_ids": [1, 2],
                }
            ]
        )
        return await database.get_global_memory_facts(-100)

    facts = asyncio.run(_run())

    assert len(facts) == 1
    assert facts[0].chat_id == -100
    assert facts[0].fact_type == "style"
    assert facts[0].fact_text == "Keep group replies concise"
    assert facts[0].confidence == 0.95
    assert facts[0].evidence_message_ids == [1, 2]


def test_list_global_memory_chat_overviews(monkeypatch, tmp_path):
    db_path = tmp_path / "global_memory_chats.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

    async def _run():
        await database.add_message(-100, "Alice @alice", "hello from group one")
        await database.add_message(-200, "Bob @bob", "hello from group two")
        await database.upsert_global_memory_facts(
            -100,
            [{"fact_type": "style", "fact_text": "Keep replies concise", "confidence": 0.9}],
        )
        await database.upsert_global_memory_facts(
            -300,
            [{"fact_type": "note", "fact_text": "Memory-only chat", "confidence": 0.8}],
        )
        return await database.list_global_memory_chat_overviews(limit=10)

    rows = asyncio.run(_run())
    by_chat = {row.chat_id: row for row in rows}

    assert set(by_chat) == {-100, -200, -300}
    assert by_chat[-100].message_count == 1
    assert by_chat[-100].global_fact_count == 1
    assert by_chat[-100].latest_message_username == "Alice @alice"
    assert by_chat[-300].message_count == 0
    assert by_chat[-300].global_fact_count == 1


def test_user_memory_admin_overview_search_and_display_lookup(monkeypatch, tmp_path):
    db_path = tmp_path / "memory_admin.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

    async def _run():
        await database.add_message(
            1,
            "Alice @alice",
            "I prefer concise answers",
            telegram_user_key="tg_user:1",
        )
        await database.add_message(
            1,
            "Bob @bob",
            "I am working on the deploy pipeline",
            telegram_user_key="tg_user:2",
        )
        await database.upsert_user_memory(
            "tg_user:1",
            latest_display_name="Alice @alice",
            memory_text="Prefers concise engineering answers",
            last_refreshed_date="2026-04-29",
        )
        await database.upsert_user_memory_facts(
            "tg_user:2",
            [
                {
                    "fact_type": "project",
                    "fact_text": "Working on the deploy pipeline",
                    "confidence": 0.8,
                    "evidence_message_ids": [2],
                }
            ],
        )
        overviews = await database.list_user_memory_overviews(limit=10)
        search_rows = await database.search_user_memories("deploy", limit=10)
        display_name = await database.get_latest_display_name_for_user("tg_user:2")
        return overviews, search_rows, display_name

    overviews, search_rows, display_name = asyncio.run(_run())

    by_key = {row.telegram_user_key: row for row in overviews}
    assert set(by_key) == {"tg_user:1", "tg_user:2"}
    assert by_key["tg_user:1"].memory_text == "Prefers concise engineering answers"
    assert by_key["tg_user:2"].fact_count == 1
    assert search_rows[0].telegram_user_key == "tg_user:2"
    assert search_rows[0].source == "fact:project"
    assert display_name == "Bob @bob"


def test_user_memory_candidates_and_fact_admin_helpers(monkeypatch, tmp_path):
    db_path = tmp_path / "memory_candidates.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        candidate_id = await database.upsert_user_memory_candidate(
            "tg_user:1",
            fact_type="preference",
            fact_text="Prefers direct answers",
            confidence=0.7,
            evidence_message_ids=[10],
            source_message_id=10,
            priority="slow",
        )
        duplicate_id = await database.upsert_user_memory_candidate(
            "tg_user:1",
            fact_type="preference",
            fact_text="Prefers direct answers",
            confidence=0.9,
            evidence_message_ids=[11],
            source_message_id=11,
            priority="fast",
        )
        candidates = await database.list_user_memory_candidates("tg_user:1")
        count = await database.count_pending_user_memory_candidates("tg_user:1")

        await database.upsert_user_memory_facts(
            "tg_user:1",
            [
                {
                    "fact_type": "style",
                    "fact_text": "Prefers long answers",
                    "confidence": 0.6,
                    "evidence_message_ids": [1],
                }
            ],
        )
        fact = (await database.get_user_memory_facts("tg_user:1"))[0]
        updated = await database.update_user_memory_fact(fact.id, fact_text="Prefers short answers")
        updated_fact = await database.get_user_memory_fact_by_id(fact.id)
        archived = await database.archive_user_memory_fact(fact.id)
        archived_fact = await database.get_user_memory_fact_by_id(fact.id)
        return candidate_id, duplicate_id, candidates, count, updated, updated_fact, archived, archived_fact

    candidate_id, duplicate_id, candidates, count, updated, updated_fact, archived, archived_fact = asyncio.run(_run())

    assert candidate_id == duplicate_id
    assert count == 1
    assert len(candidates) == 1
    assert candidates[0].priority == "fast"
    assert candidates[0].confidence == 0.9
    assert candidates[0].evidence_message_ids == [10, 11]
    assert updated is True
    assert updated_fact is not None
    assert updated_fact.fact_text == "Prefers short answers"
    assert archived is True
    assert archived_fact is None
