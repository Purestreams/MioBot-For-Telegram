import asyncio

from app import database


def test_webadmin_login_and_management_api(monkeypatch, tmp_path):
    db_path = tmp_path / "webadmin_api.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setenv("WEBADMIN_SECRET_KEY", "test-secret")
    monkeypatch.setenv("WEBADMIN_COOKIE_SECURE", "0")
    monkeypatch.setenv("WEBADMIN_HOST", "127.0.0.1")
    monkeypatch.setenv("WEBADMIN_BASE_URL", "http://127.0.0.1:8765")
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    database.init_db()

    async def seed_data():
        await database.add_message(-100, "Alice @alice", "hello from group", telegram_user_key="tg_user:1")
        await database.upsert_user_memory(
            "tg_user:1",
            latest_display_name="Alice @alice",
            memory_text="likes compact admin tools",
            last_refreshed_date="2026-05-21",
        )
        await database.upsert_user_memory_facts(
            "tg_user:1",
            [{"fact_type": "preference", "fact_text": "Likes compact admin tools", "confidence": 0.8}],
        )
        await database.upsert_global_memory_facts(
            -100,
            [{"fact_type": "style", "fact_text": "Group prefers concise replies", "confidence": 0.7}],
        )

    asyncio.run(seed_data())

    from fastapi.testclient import TestClient
    from webadmin.app import create_app
    from webadmin.security import hash_login_token

    async def create_token():
        await database.create_webadmin_login_token(
            hash_login_token("test-token"),
            admin_user_id=42,
            admin_username="admin",
            ttl_seconds=600,
        )

    asyncio.run(create_token())

    client = TestClient(create_app())

    assert client.get("/api/dashboard").status_code == 401

    login = client.post("/api/auth/login", json={"token": "test-token"})
    assert login.status_code == 200
    assert login.json()["authenticated"] is True

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["message_count"] == 1

    chats = client.get("/api/chats")
    assert chats.status_code == 200
    assert chats.json()["chats"][0]["chat_id"] == -100

    messages = client.get("/api/chats/-100/messages")
    assert messages.status_code == 200
    assert messages.json()["messages"][0]["content"] == "hello from group"

    summary = client.put(
        "/api/memory/users/tg_user:1/summary",
        json={"memory_text": "prefers bilingual admin tools"},
    )
    assert summary.status_code == 200
    assert summary.json()["memory"]["memory_text"] == "prefers bilingual admin tools"

    detail = client.get("/api/memory/users/tg_user:1")
    assert detail.status_code == 200
    assert detail.json()["facts"][0]["fact_type"] == "preference"

    created_global = client.post(
        "/api/global-memory/chats/-100/facts",
        json={"fact_type": "topic", "fact_text": "Talks about ops tooling", "confidence": 0.9},
    )
    assert created_global.status_code == 200
    facts = created_global.json()["facts"]
    assert any(fact["fact_text"] == "Talks about ops tooling" for fact in facts)

    fact_id = next(fact["id"] for fact in facts if fact["fact_text"] == "Talks about ops tooling")
    patched = client.patch(
        f"/api/global-memory/chats/-100/facts/{fact_id}",
        json={"fact_text": "Talks about admin tooling"},
    )
    assert patched.status_code == 200

    updated = client.get("/api/global-memory/chats/-100")
    assert any(fact["fact_text"] == "Talks about admin tooling" for fact in updated.json()["facts"])


def test_webadmin_static_page_contains_language_controls(monkeypatch, tmp_path):
    db_path = tmp_path / "webadmin_static.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    from fastapi.testclient import TestClient
    from webadmin.app import create_app

    client = TestClient(create_app())
    response = client.get("/")

    assert response.status_code == 200
    assert "中文" in response.text
    assert "EN" in response.text
    assert "app.js" in response.text
