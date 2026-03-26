import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest

from app import database


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

    async def fake_embed_text(text: str, *, model_name=None):
        # Simple deterministic vectors so cat-related query matches cat-related messages.
        if "cat" in text.lower() or "cats" in text.lower() or "fish" in text.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(database, "embed_text", fake_embed_text)

    async def _run() -> tuple[list[str], list[str]]:
        await database.add_message(100, "u1", "cats and fish are great")
        await database.add_message(100, "u2", "python asyncio tips")
        await database.add_message(100, "u3", "hello group")
        return await database.get_prompt_context_parts(100, "cat fish", recent_n=2, retrieved_k=2)

    recent, rag = asyncio.run(_run())

    assert len(recent) == 2
    assert len(rag) >= 1
    assert "cats and fish" in rag[0].lower()
