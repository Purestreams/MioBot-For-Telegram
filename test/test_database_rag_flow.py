import asyncio
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app import database
from app.rag_embeddings import EmbeddingMetadata


class _EmbedVector:
    def __init__(self, values):
        self.values = values


@pytest.fixture
def isolated_db_copy(monkeypatch, tmp_path):
    source_db = Path(database._db_file_path())
    copied_db = tmp_path / "message_history.test.db"

    if source_db.exists():
        shutil.copy2(source_db, copied_db)

    monkeypatch.setenv("DB_FILE", str(copied_db))
    monkeypatch.setattr(database, "DB_FILE", str(copied_db))
    database.init_db()

    yield copied_db

    if copied_db.exists():
        copied_db.unlink()


def test_get_prompt_context_parts_includes_retrieved_history(monkeypatch, isolated_db_copy):
    monkeypatch.setenv("RAG_ENABLED", "1")

    async def fake_embed_text_with_metadata(text: str, *, model_name=None):
        # Simple deterministic vectors so cat-related query matches cat-related messages.
        if "cat" in text.lower() or "cats" in text.lower() or "fish" in text.lower():
            vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return vector, EmbeddingMetadata(
            backend="test",
            model="test-model",
            dim=3,
            signature="test:3",
        )

    monkeypatch.setattr(database, "embed_text_with_metadata", fake_embed_text_with_metadata)

    async def _run() -> tuple[list[str], list[str]]:
        await database.add_message(100, "u1", "cats and fish are great")
        await database.add_message(100, "u2", "python asyncio tips")
        await database.add_message(100, "u3", "hello group")
        return await database.get_prompt_context_parts(100, "cat fish", recent_n=2, retrieved_k=2)

    recent, rag = asyncio.run(_run())

    assert len(recent) == 2
    assert len(rag) >= 1
    assert "cats and fish" in rag[0].lower()


def test_get_prompt_context_parts_uses_keyword_retrieval_when_embeddings_miss(monkeypatch, tmp_path):
    db_path = tmp_path / "keyword_rag.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    monkeypatch.setenv("RAG_ENABLED", "1")
    database.init_db()

    async def fake_embed_message_content(username: str, content: str):
        return np.array([0.0, 1.0], dtype=np.float32), EmbeddingMetadata(
            backend="test",
            model="test-model",
            dim=2,
            signature="test:2",
        )

    async def fake_embed_text_with_metadata(text: str, *, model_name=None):
        return np.array([1.0, 0.0], dtype=np.float32), EmbeddingMetadata(
            backend="test",
            model="test-model",
            dim=2,
            signature="test:2",
        )

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    monkeypatch.setattr(database, "embed_text_with_metadata", fake_embed_text_with_metadata)

    async def _run() -> tuple[list[str], list[str]]:
        await database.add_message(200, "u1", "the launch checklist mentions sqlite lock handling")
        await database.add_message(200, "u2", "unrelated cats and fish")
        await database.add_message(200, "u3", "current message")
        return await database.get_prompt_context_parts(200, "sqlite lock", recent_n=1, retrieved_k=2)

    recent, rag = asyncio.run(_run())

    assert len(recent) == 1
    assert any("sqlite lock handling" in line for line in rag)


def test_get_prompt_context_parts_keeps_stable_oldest_anchor(monkeypatch, tmp_path):
    db_path = tmp_path / "anchor_context.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    monkeypatch.setenv("RAG_ENABLED", "0")
    database.init_db()

    async def _run() -> tuple[list[str], list[str]]:
        for number in range(1, 7):
            await database.add_message(300, "u", f"message-{number}")
        return await database.get_prompt_context_parts(300, "", recent_n=4, retrieved_k=0, cache_anchor_n=2)

    recent, rag = asyncio.run(_run())

    assert rag == []
    assert "message-1" in recent[0]
    assert "message-2" in recent[1]
    assert "message-5" in recent[-2]
    assert "message-6" in recent[-1]


def test_get_prompt_context_parts_excludes_anchor_and_recent_from_rag(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "1")
    anchor_row = database.MessageRow(1, 9, "Alice", "oldest anchor context", "2026-08-01")
    middle_row = database.MessageRow(2, 9, "Bob", "retrieved only context", "2026-08-02")
    recent_row = database.MessageRow(3, 9, "Cara", "latest recent context", "2026-08-03")

    async def fake_oldest(chat_id, limit):
        return [anchor_row][:limit]

    async def fake_recent(chat_id, limit):
        return [recent_row]

    async def fake_vector(chat_id, query, top_k):
        return [anchor_row, middle_row, recent_row]

    async def fake_keyword(chat_id, query, top_k):
        return []

    monkeypatch.setattr(database, "get_oldest_messages", fake_oldest)
    monkeypatch.setattr(database, "get_recent_messages", fake_recent)
    monkeypatch.setattr(database, "vector_search_messages", fake_vector)
    monkeypatch.setattr(database, "keyword_search_messages", fake_keyword)

    recent, rag = asyncio.run(
        database.get_prompt_context_parts(9, "context", recent_n=2, retrieved_k=3, cache_anchor_n=1)
    )
    assert any("oldest anchor context" in line for line in recent)
    assert any("latest recent context" in line for line in recent)
    assert any("retrieved only context" in line for line in rag)
    assert all("oldest anchor context" not in line for line in rag)
    assert all("latest recent context" not in line for line in rag)


def test_get_prompt_context_parts_excludes_bot_replies_from_rag(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "1")
    bot_row = database.MessageRow(1, 7, database.BOT_HISTORY_USERNAME, "old emergency script", "2026-08-01")
    human_row = database.MessageRow(2, 7, "Alice", "useful human context", "2026-08-02")
    current_row = database.MessageRow(3, 7, "Bob", "current", "2026-08-03")

    async def fake_oldest(chat_id, limit):
        return []

    async def fake_recent(chat_id, limit):
        return [current_row]

    async def fake_vector(chat_id, query, top_k):
        return [bot_row, human_row]

    async def fake_keyword(chat_id, query, top_k):
        return []

    monkeypatch.setattr(database, "get_oldest_messages", fake_oldest)
    monkeypatch.setattr(database, "get_recent_messages", fake_recent)
    monkeypatch.setattr(database, "vector_search_messages", fake_vector)
    monkeypatch.setattr(database, "keyword_search_messages", fake_keyword)

    _, rag = asyncio.run(database.get_prompt_context_parts(7, "context", recent_n=1, retrieved_k=2))
    assert any("useful human context" in line for line in rag)
    assert all("old emergency script" not in line for line in rag)


def test_get_embedding_health_report_flags_signature_drift(monkeypatch, tmp_path):
    db_path = tmp_path / "health.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _seed() -> None:
        async with database.aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                "INSERT INTO messages (id, chat_id, username, content) VALUES (1, 42, 'u1', 'hello world')"
            )
            await db.execute(
                '''
                INSERT INTO message_embeddings (
                    message_id, chat_id, embedding, dim, model, backend, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (1, 42, np.array([1.0, 0.0], dtype=np.float32).tobytes(), 2, "old-model", "fastembed", "fastembed:old-model"),
            )
            await db.commit()

    asyncio.run(_seed())

    async def fake_runtime_metadata(*, model_name=None):
        return EmbeddingMetadata(
            backend="fastembed",
            model="new-model",
            dim=384,
            signature="fastembed:new-model",
        )

    monkeypatch.setattr(database, "get_runtime_embedding_metadata", fake_runtime_metadata)

    report = asyncio.run(database.get_embedding_health_report())

    assert report["needs_reindex"] is True
    assert any("runtime signature fastembed:new-model is absent" in reason for reason in report["reasons"])


def test_reindex_message_embeddings_rewrites_signature(monkeypatch, tmp_path):
    db_path = tmp_path / "reindex.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _seed() -> None:
        async with database.aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                "INSERT INTO messages (id, chat_id, username, content) VALUES (1, 42, 'u1', 'hello world')"
            )
            await db.execute(
                '''
                INSERT INTO message_embeddings (
                    message_id, chat_id, embedding, dim, model, backend, signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (1, 42, np.array([1.0, 0.0], dtype=np.float32).tobytes(), 2, "old-model", "fastembed", "fastembed:old-model"),
            )
            await db.commit()

    asyncio.run(_seed())

    async def fake_embed_message_content(username: str, content: str):
        vector = np.array([0.25, 0.75], dtype=np.float32)
        return vector, EmbeddingMetadata(
            backend="fastembed",
            model="new-model",
            dim=2,
            signature="fastembed:new-model",
        )

    async def fake_runtime_metadata(*, model_name=None):
        return EmbeddingMetadata(
            backend="fastembed",
            model="new-model",
            dim=2,
            signature="fastembed:new-model",
        )

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    monkeypatch.setattr(database, "get_runtime_embedding_metadata", fake_runtime_metadata)

    result = asyncio.run(database.reindex_message_embeddings())

    assert result["reindexed"] == 1

    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT backend, signature, model, dim FROM message_embeddings WHERE message_id = 1"
        ).fetchone()

    assert row == ("fastembed", "fastembed:new-model", "new-model", 2)
