import pytest

from webadmin import security


def test_session_secret_does_not_fall_back_to_a_known_value(monkeypatch):
    monkeypatch.delenv("WEBADMIN_SECRET_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_KEY", raising=False)

    assert security._session_secret() != b"miobot-webadmin-dev-secret"


def test_public_webadmin_requires_https_and_an_explicit_secret(monkeypatch):
    monkeypatch.setenv("WEBADMIN_HOST", "0.0.0.0")
    monkeypatch.setenv("WEBADMIN_BASE_URL", "http://admin.example.test")
    monkeypatch.delenv("WEBADMIN_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="HTTPS"):
        security.validate_webadmin_security_configuration()

    monkeypatch.setenv("WEBADMIN_BASE_URL", "https://admin.example.test")
    with pytest.raises(RuntimeError, match="WEBADMIN_SECRET_KEY"):
        security.validate_webadmin_security_configuration()

    monkeypatch.setenv("WEBADMIN_SECRET_KEY", "test-secret")
    security.validate_webadmin_security_configuration()
    monkeypatch.setenv("WEBADMIN_COOKIE_SECURE", "0")
    assert security.webadmin_cookie_secure() is True
