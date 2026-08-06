"""Authentication helpers for the MioBot web admin."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from ipaddress import ip_address
from typing import Any, Optional
from urllib.parse import urlsplit

from app.runtime_config import get_runtime_bool, get_runtime_int, get_runtime_value

SESSION_COOKIE_NAME = "miobot_webadmin_session"
DEFAULT_LOGIN_TOKEN_TTL_SECONDS = 600
MAX_LOGIN_TOKEN_TTL_SECONDS = 1800
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60
_EPHEMERAL_SESSION_SECRET = secrets.token_bytes(32)


def generate_login_token() -> str:
    return secrets.token_urlsafe(32)


def hash_login_token(token: str) -> str:
    return hashlib.sha256((token or "").strip().encode("utf-8")).hexdigest()


def parse_login_token_ttl_seconds(raw_value: Optional[str]) -> int:
    value = (raw_value or "").strip().lower()
    if not value:
        return DEFAULT_LOGIN_TOKEN_TTL_SECONDS

    match = re.fullmatch(r"(\d{1,4})\s*([smh]?)", value)
    if not match:
        return DEFAULT_LOGIN_TOKEN_TTL_SECONDS

    amount = int(match.group(1))
    suffix = match.group(2)
    if suffix == "s":
        seconds = amount
    elif suffix == "h":
        seconds = amount * 60 * 60
    else:
        seconds = amount * 60
    return max(60, min(seconds, MAX_LOGIN_TOKEN_TTL_SECONDS))


def format_ttl(seconds: int) -> str:
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    return f"{seconds} seconds"


def webadmin_base_url() -> str:
    return (get_runtime_value("WEBADMIN_BASE_URL") or "http://127.0.0.1:8765").rstrip("/")


def webadmin_host() -> str:
    return get_runtime_value("WEBADMIN_HOST") or "127.0.0.1"


def webadmin_port() -> int:
    return get_runtime_int("WEBADMIN_PORT", 8765)


def webadmin_cookie_secure() -> bool:
    # HTTPS deployments must never allow the session cookie over HTTP.  A
    # local HTTP server may opt in explicitly if needed for development.
    return webadmin_base_url().lower().startswith("https://") or get_runtime_bool("WEBADMIN_COOKIE_SECURE", False)


def session_ttl_seconds() -> int:
    return max(300, get_runtime_int("WEBADMIN_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS))


def _session_secret() -> bytes:
    secret = get_runtime_value("WEBADMIN_SECRET_KEY")
    # A random process-local value is safe for local development and avoids a
    # predictable fallback. Sessions intentionally expire after a restart.
    return secret.encode("utf-8") if secret else _EPHEMERAL_SESSION_SECRET


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_webadmin_security_configuration() -> None:
    """Reject public insecure configurations before serving private data."""
    host = webadmin_host()
    base_url = webadmin_base_url()
    scheme = urlsplit(base_url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise RuntimeError("WEBADMIN_BASE_URL must use http or https.")
    if _is_loopback_host(host):
        return
    if scheme != "https":
        raise RuntimeError("A non-loopback web admin requires an HTTPS WEBADMIN_BASE_URL.")
    if not get_runtime_value("WEBADMIN_SECRET_KEY"):
        raise RuntimeError("A non-loopback web admin requires WEBADMIN_SECRET_KEY.")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _signature(payload_b64: str) -> str:
    digest = hmac.new(_session_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64encode(digest)


def create_session_cookie(*, admin_user_id: Optional[int], admin_username: str, ttl_seconds: Optional[int] = None) -> str:
    now = int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else session_ttl_seconds()
    payload = {
        "admin_user_id": admin_user_id,
        "admin_username": admin_username or "",
        "iat": now,
        "exp": now + max(60, ttl),
        "nonce": secrets.token_urlsafe(12),
    }
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_signature(payload_b64)}"


def parse_session_cookie(cookie_value: str) -> Optional[dict[str, Any]]:
    if not cookie_value or "." not in cookie_value:
        return None
    payload_b64, supplied_signature = cookie_value.split(".", 1)
    expected_signature = _signature(payload_b64)
    if not hmac.compare_digest(supplied_signature, expected_signature):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None

    try:
        expires_at = int(payload.get("exp", 0))
    except (TypeError, ValueError):
        return None
    if expires_at < int(time.time()):
        return None
    return payload if isinstance(payload, dict) else None
