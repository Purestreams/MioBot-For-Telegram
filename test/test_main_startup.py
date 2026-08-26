import pytest

import main


class _FakeProcess:
    def __init__(self):
        self.pid = 4321

    def is_alive(self):
        return False

    def terminate(self):
        return None

    def join(self, timeout=None):
        return None

    def kill(self):
        return None


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


def test_main_registers_nonblocking_embedding_initializer(monkeypatch):
    calls = {"register": 0, "run_polling": 0, "post_init": None}

    async def fake_ensure_fastembed_ready(*, model_name=None):
        return None

    def fake_init_db():
        return None

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

        def post_init(self, callback):
            calls["post_init"] = callback
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
    monkeypatch.setattr(main, "_start_webadmin_process", lambda: _FakeProcess())
    monkeypatch.setattr(main, "_stop_webadmin_process", lambda process: None)
    monkeypatch.setattr(main, "register_handlers", fake_register_handlers)
    monkeypatch.setattr(main, "Application", _FakeApplication)

    main.main()

    assert calls == {
        "register": 1,
        "run_polling": 1,
        "post_init": main.prepare_embedding_index,
    }


def test_main_starts_and_stops_webadmin_process(monkeypatch):
    calls = {"start": 0, "stop": 0, "register": 0, "run_polling": 0}
    process = _FakeProcess()

    async def fake_ensure_fastembed_ready(*, model_name=None):
        return None

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

        def post_init(self, callback):
            return self

        def build(self):
            return _FakeBuiltApplication()

    class _FakeApplication:
        @staticmethod
        def builder():
            return _FakeApplicationBuilder()

    monkeypatch.setattr(main, "ensure_fastembed_ready", fake_ensure_fastembed_ready)
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "register_handlers", lambda application: calls.__setitem__("register", calls["register"] + 1))
    monkeypatch.setattr(main, "Application", _FakeApplication)

    def fake_start_webadmin_process():
        calls["start"] += 1
        return process

    def fake_stop_webadmin_process(value):
        calls["stop"] += 1
        assert value is process

    monkeypatch.setattr(main, "_start_webadmin_process", fake_start_webadmin_process)
    monkeypatch.setattr(main, "_stop_webadmin_process", fake_stop_webadmin_process)

    main.main()

    assert calls == {"start": 1, "stop": 1, "register": 1, "run_polling": 1}
