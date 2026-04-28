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