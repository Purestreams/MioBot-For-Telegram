import asyncio

from app import startup


def test_prepare_embedding_index_schedules_background_reindex(monkeypatch):
    calls = {"scheduled": 0}

    async def fake_health():
        return {"needs_reindex": True, "db_file": "data/test.db"}

    async def fake_reindex():
        return {"reindexed": 4, "signature": "fastembed:test"}

    class _Application:
        def create_task(self, coroutine):
            calls["scheduled"] += 1
            coroutine.close()

    monkeypatch.setenv("RAG_REINDEX_ON_STARTUP", "background")
    monkeypatch.setattr(startup, "log_embedding_health_report", fake_health)
    monkeypatch.setattr(startup, "reindex_message_embeddings", fake_reindex)

    asyncio.run(startup.prepare_embedding_index(_Application()))

    assert calls["scheduled"] == 1


def test_prepare_embedding_index_can_be_disabled(monkeypatch):
    calls = {"scheduled": 0}

    async def fake_health():
        return {"needs_reindex": True, "db_file": "data/test.db"}

    class _Application:
        def create_task(self, coroutine):
            calls["scheduled"] += 1
            coroutine.close()

    monkeypatch.setenv("RAG_REINDEX_ON_STARTUP", "disabled")
    monkeypatch.setattr(startup, "log_embedding_health_report", fake_health)

    asyncio.run(startup.prepare_embedding_index(_Application()))

    assert calls["scheduled"] == 0
