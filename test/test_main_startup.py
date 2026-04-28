import pytest

import main


def test_main_raises_when_fastembed_check_fails(monkeypatch):
    async def fake_ensure_fastembed_ready(*, model_name=None):
        raise RuntimeError("fastembed startup failure")

    monkeypatch.setattr(main, "ensure_fastembed_ready", fake_ensure_fastembed_ready)
    monkeypatch.setattr(main, "init_db", lambda: None)

    class _UnexpectedApplication:
        @staticmethod
        def builder():
            raise AssertionError("application builder should not run after embedding failure")

    monkeypatch.setattr(main, "Application", _UnexpectedApplication)

    with pytest.raises(RuntimeError, match="fastembed startup failure"):
        main.main()


def test_main_auto_reindexes_embeddings_when_health_requires_it(monkeypatch):
    calls = {"health": 0, "reindex": 0, "register": 0, "run_polling": 0}

    async def fake_ensure_fastembed_ready(*, model_name=None):
        return None

    def fake_init_db():
        return None

    async def fake_log_embedding_health_report():
        calls["health"] += 1
        if calls["health"] == 1:
            return {
                "needs_reindex": True,
                "db_file": "data/message_history.db",
                "runtime": {"signature": "fastembed:model"},
            }
        return {
            "needs_reindex": False,
            "db_file": "data/message_history.db",
            "runtime": {"signature": "fastembed:model"},
        }

    async def fake_reindex_message_embeddings(*, chat_id=None):
        calls["reindex"] += 1
        assert chat_id is None
        return {"reindexed": 5, "signature": "fastembed:model"}

    class _FakeBuiltApplication:
        def add_error_handler(self, handler):
            return None

        def run_polling(self):
            calls["run_polling"] += 1

    class _FakeApplicationBuilder:
        def token(self, token):
            return self

        def read_timeout(self, timeout):
            return self

        def write_timeout(self, timeout):
            return self

        def build(self):
            return _FakeBuiltApplication()

    class _FakeApplication:
        @staticmethod
        def builder():
            return _FakeApplicationBuilder()

    def fake_register_handlers(application):
        calls["register"] += 1

    monkeypatch.setattr(main, "ensure_fastembed_ready", fake_ensure_fastembed_ready)
    monkeypatch.setattr(main, "init_db", fake_init_db)
    monkeypatch.setattr(main, "log_embedding_health_report", fake_log_embedding_health_report)
    monkeypatch.setattr(main, "reindex_message_embeddings", fake_reindex_message_embeddings)
    monkeypatch.setattr(main, "register_handlers", fake_register_handlers)
    monkeypatch.setattr(main, "Application", _FakeApplication)

    main.main()

    assert calls == {"health": 2, "reindex": 1, "register": 1, "run_polling": 1}