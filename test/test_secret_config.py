import os
import tempfile
from pathlib import Path

from app import runtime_config


def test_load_env_file_parses_key_values(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / "runtime.env"
        env_file.write_text(
            "# comment\nA=1\nB = hello\nC='quoted value'\nINVALID_LINE\n",
            encoding="utf-8",
        )

        monkeypatch.delenv("A", raising=False)
        monkeypatch.delenv("B", raising=False)
        monkeypatch.delenv("C", raising=False)

        runtime_config.load_env_file(env_file)

        assert os.getenv("A") == "1"
        assert os.getenv("B") == "hello"
        assert os.getenv("C") == "quoted value"


def test_set_environment_reads_from_config_runtime_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Path(tmpdir)
        runtime_env = cfg / "runtime.env"
        runtime_env.write_text(
            "TELEGRAM_BOT_KEY=abc123\n"
            "TELEGRAM_BOT_USERNAME=botuser\n"
            "ARK_API_KEY=ark-key\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(runtime_config, "CONFIG_DIR", cfg)
        monkeypatch.setattr(runtime_config, "ENV_FILES", (cfg / "runtime.env", cfg / "runtime.local.env"))

        monkeypatch.delenv("TELEGRAM_BOT_KEY", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_USERNAME", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)

        runtime_config.bootstrap_runtime_environment(force=True)

        assert os.getenv("TELEGRAM_BOT_KEY") == "abc123"
        assert os.getenv("TELEGRAM_BOT_USERNAME") == "botuser"
        assert os.getenv("ARK_API_KEY") == "ark-key"


def test_ark_endpoint_helpers_derive_chat_and_responses_urls(monkeypatch):
    monkeypatch.setenv("ARK_API_ENDPOINT", "https://example.test/api/v3/chat/completions")

    assert runtime_config.get_ark_chat_completions_endpoint() == "https://example.test/api/v3/chat/completions"
    assert runtime_config.get_ark_responses_endpoint() == "https://example.test/api/v3/responses"

    monkeypatch.setenv("ARK_API_ENDPOINT", "https://example.test/api/v3/responses")

    assert runtime_config.get_ark_chat_completions_endpoint() == "https://example.test/api/v3/chat/completions"
    assert runtime_config.get_ark_responses_endpoint() == "https://example.test/api/v3/responses"
