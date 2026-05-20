import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
ENV_FILES = (
    CONFIG_DIR / "runtime.env",
    CONFIG_DIR / "runtime.local.env",
)
_BOOTSTRAPPED = False

# Centralized runtime defaults. Keep config/runtime.env(.template) aligned.
DEFAULTS = {
    "LLM_PROVIDER": "ark",
    "LLM_ENABLE_THINKING": "0",
    "AZURE_OPENAI_API_VERSION": "2024-04-01-preview",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5-mini",
    "ARK_API_ENDPOINT": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "ARK_MODEL": "doubao-seed-1-8-251228",
    "ARK_VISION_MODEL": "doubao-seed-1-6-251015",
    "OLLAMA_ENDPOINT": "http://100.69.97.8:11434",
    "OLLAMA_MODEL": "gpt-oss:20b",
    "TELEGRAM_BOT_USERNAME": "MioooooooooBot",
    "TELEGRAM_ADMIN_USER_IDS": "",
    "DB_FILE": "data/message_history.db",
    "MESSAGE_REVIEW_BACK": "80",
    "RAG_TOP_K": "12",
    "STICKER_REPLY_ENABLED": "1",
    "STICKER_REPLY_CANDIDATE_LIMIT": "12",
    "STICKER_REPLY_COOLDOWN_MINUTES": "30",
    "MEMORY_CANDIDATE_EXTRACTION_ENABLED": "1",
    "MEMORY_CANDIDATE_AUTO_REFRESH_COUNT": "3",
    "EMBED_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def load_env_file(file_path: Path) -> None:
    """Load KEY=VALUE lines from a dotenv-like file into os.environ.

    Existing environment variables are preserved and not overwritten.
    """
    if not file_path.is_file():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)


def bootstrap_runtime_environment(*, force: bool = False) -> None:
    """Load runtime env files and apply project defaults."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not force:
        return

    for env_file in ENV_FILES:
        load_env_file(env_file)

    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    os.environ.setdefault("ARK_ENDPOINT", os.environ["ARK_API_ENDPOINT"])

    _BOOTSTRAPPED = True


def get_runtime_value(name: str) -> str:
    bootstrap_runtime_environment()
    return os.getenv(name, "")


def _derive_ark_endpoint(endpoint: str, suffix: str) -> str:
    raw_endpoint = (endpoint or "").strip().rstrip("/")
    if not raw_endpoint:
        return ""

    parsed = urlsplit(raw_endpoint)
    path = parsed.path.rstrip("/")
    for known_suffix in ("/chat/completions", "/responses"):
        if path.endswith(known_suffix):
            path = path[: -len(known_suffix)]
            break

    return urlunsplit(parsed._replace(path=f"{path}/{suffix.strip('/')}", query="", fragment=""))


def get_ark_chat_completions_endpoint() -> str:
    return _derive_ark_endpoint(get_runtime_value("ARK_API_ENDPOINT"), "chat/completions")


def get_ark_responses_endpoint() -> str:
    return _derive_ark_endpoint(get_runtime_value("ARK_API_ENDPOINT"), "responses")


def get_runtime_int(name: str, default: int) -> int:
    bootstrap_runtime_environment()
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_runtime_bool(name: str, default: bool = False) -> bool:
    bootstrap_runtime_environment()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
