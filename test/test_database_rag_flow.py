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
