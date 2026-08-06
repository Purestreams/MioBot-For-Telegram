import asyncio
from types import SimpleNamespace

import main


class _FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def _update(*, user_id=42, username="Natsume_Mio", chat_type="private", chat_id=123):
    message = _FakeMessage()
    return SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=user_id, username=username),
        effective_chat=SimpleNamespace(type=chat_type, id=chat_id),
    )


def _context(args=None):
    return SimpleNamespace(args=args or [])


def test_admin_ids_accept_only_numeric_and_tg_user_tokens(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42, tg_user:100 @Natsume_Mio")

    assert main._configured_admin_user_ids() == {42, 100}


def test_admin_check_rejects_configured_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "@Natsume_Mio")

    assert main._is_admin_update(_update(user_id=999, username="natsume_mio")) is False


def test_memory_admin_help_requires_private_admin(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    group_update = _update(user_id=42, chat_type="group")
    non_admin_update = _update(user_id=7, chat_type="private")

    asyncio.run(main.handle_memory_admin_help(group_update, _context()))
    asyncio.run(main.handle_memory_admin_help(non_admin_update, _context()))

    assert group_update.message.replies == ["Memory admin commands are only available in a private chat."]
    assert non_admin_update.message.replies == ["You are not allowed to use memory admin commands."]


def test_webadmin_token_creates_private_admin_login_url(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    monkeypatch.setenv("WEBADMIN_BASE_URL", "http://admin.local")
    update = _update(username="AdminUser")
    captured = {}

    async def fake_create_webadmin_login_token(token_hash, *, admin_user_id=None, admin_username="", ttl_seconds=600):
        captured.update(
            {
                "token_hash": token_hash,
                "admin_user_id": admin_user_id,
                "admin_username": admin_username,
                "ttl_seconds": ttl_seconds,
            }
        )
        return SimpleNamespace(expires_at="2026-05-22 12:00:00")

    monkeypatch.setattr(main, "generate_login_token", lambda: "raw-token")
    monkeypatch.setattr(main, "hash_login_token", lambda token: f"hash:{token}")
    monkeypatch.setattr(main, "create_webadmin_login_token", fake_create_webadmin_login_token)

    asyncio.run(main.handle_webadmin_token(update, _context(["30m"])))

    reply = update.message.replies[0]
    assert captured == {
        "token_hash": "hash:raw-token",
        "admin_user_id": 42,
        "admin_username": "AdminUser",
        "ttl_seconds": 1800,
    }
    assert "http://admin.local/?token=raw-token" in reply
    assert "single-use" in reply


def test_memory_admin_list_formats_overviews(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_list_user_memory_overviews(*, limit=40):
        return [
            SimpleNamespace(
                telegram_user_key="tg_user:1",
                latest_display_name="Alice @alice",
                memory_text="Prefers concise answers",
                last_refreshed_date="2026-04-29",
                fact_count=2,
                latest_message_at="2026-04-29 10:00:00",
            )
        ]

    monkeypatch.setattr(main, "list_user_memory_overviews", fake_list_user_memory_overviews)

    asyncio.run(main.handle_memory_admin_list(update, _context()))

    assert "Alice @alice" in update.message.replies[0]
    assert "tg_user:1" in update.message.replies[0]
    assert "facts=2" in update.message.replies[0]


def test_memory_admin_audit_reports_malformed_summaries(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_audit_user_memory_texts(*, limit=200):
        assert limit == 50
        return [
            SimpleNamespace(
                telegram_user_key="tg_user:1",
                latest_display_name="Alice @alice",
                stored_length=220,
                normalized_length=88,
                issue_types=["json-blob", "normalizes-differently"],
                preview='{"memory_text": ["Prefers concise answers"]',
            )
        ]

    monkeypatch.setattr(main, "audit_user_memory_texts", fake_audit_user_memory_texts)

    asyncio.run(main.handle_memory_admin_audit(update, _context(["50"])))

    reply = update.message.replies[0]
    assert "Alice @alice" in reply
    assert "issues=json-blob,normalizes-differently" in reply
    assert "len=220->88" in reply


def test_memory_admin_audit_reports_clean_state(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_audit_user_memory_texts(*, limit=200):
        assert limit == 200
        return []

    monkeypatch.setattr(main, "audit_user_memory_texts", fake_audit_user_memory_texts)

    asyncio.run(main.handle_memory_admin_audit(update, _context()))

    assert update.message.replies == ["Memory audit found no malformed summaries in the latest 200 rows."]


def test_memory_admin_audit_zero_runs_full_scan(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_audit_user_memory_texts(*, limit=200):
        assert limit is None
        return []

    monkeypatch.setattr(main, "audit_user_memory_texts", fake_audit_user_memory_texts)

    asyncio.run(main.handle_memory_admin_audit(update, _context(["0"])))

    assert update.message.replies == ["Memory audit found no malformed summaries in all rows."]


def test_memory_admin_view_shows_summary_and_facts(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_get_user_memory(telegram_user_key):
        assert telegram_user_key == "tg_user:1"
        return SimpleNamespace(
            latest_display_name="Alice @alice",
            memory_text="Prefers concise answers",
            last_refreshed_date="2026-04-29",
        )

    async def fake_get_user_memory_facts(telegram_user_key, *, limit=25, min_confidence=0.0):
        return [
            SimpleNamespace(
                id=5,
                fact_type="preference",
                fact_text="Prefers concise answers",
                confidence=0.9,
                evidence_message_ids=[11, 12],
            )
        ]

    async def fake_get_latest_display_name_for_user(telegram_user_key):
        return "Alice @alice"

    monkeypatch.setattr(main, "get_user_memory", fake_get_user_memory)
    monkeypatch.setattr(main, "get_user_memory_facts", fake_get_user_memory_facts)
    monkeypatch.setattr(main, "get_latest_display_name_for_user", fake_get_latest_display_name_for_user)

    asyncio.run(main.handle_memory_admin_view(update, _context(["1"])))

    reply = update.message.replies[0]
    assert "Memory for Alice @alice" in reply
    assert "key: tg_user:1" in reply
    assert "[preference] Prefers concise answers" in reply
    assert "confidence=0.90" in reply
    assert "evidence=" not in reply


def test_memory_admin_search_formats_matches(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()

    async def fake_search_user_memories(query, *, limit=25):
        assert query == "deploy pipeline"
        return [
            SimpleNamespace(
                telegram_user_key="tg_user:2",
                latest_display_name="Bob @bob",
                source="fact:project",
                text="Working on the deploy pipeline",
            )
        ]

    monkeypatch.setattr(main, "search_user_memories", fake_search_user_memories)

    asyncio.run(main.handle_memory_admin_search(update, _context(["deploy", "pipeline"])))

    assert "Bob @bob" in update.message.replies[0]
    assert "fact:project" in update.message.replies[0]


def test_memory_admin_refresh_forces_refresh(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update()
    captured = {}

    async def fake_get_latest_display_name_for_user(telegram_user_key):
        return "Alice @alice"

    async def fake_refresh_user_memory_if_due(**kwargs):
        captured.update(kwargs)
        return "Prefers concise answers"

    monkeypatch.setattr(main, "get_latest_display_name_for_user", fake_get_latest_display_name_for_user)
    monkeypatch.setattr(main, "refresh_user_memory_if_due", fake_refresh_user_memory_if_due)

    asyncio.run(main.handle_memory_admin_refresh(update, _context(["1"])))

    assert captured["telegram_user_key"] == "tg_user:1"
    assert captured["latest_display_name"] == "Alice @alice"
    assert captured["force"] is True
    assert len(update.message.replies) == 2
    assert "Memory refresh finished" in update.message.replies[1]


def test_memory_admin_set_replaces_summary(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "999")
    update = _update(user_id=999, username="Natsume_Mio")
    captured = {}

    async def fake_get_user_memory(telegram_user_key):
        return SimpleNamespace(latest_display_name="Alice @alice", last_refreshed_date="2026-04-29")

    async def fake_get_latest_display_name_for_user(telegram_user_key):
        return "Alice @alice"

    async def fake_upsert_user_memory(telegram_user_key, **kwargs):
        captured["telegram_user_key"] = telegram_user_key
        captured.update(kwargs)

    monkeypatch.setattr(main, "get_user_memory", fake_get_user_memory)
    monkeypatch.setattr(main, "get_latest_display_name_for_user", fake_get_latest_display_name_for_user)
    monkeypatch.setattr(main, "upsert_user_memory", fake_upsert_user_memory)

    asyncio.run(main.handle_memory_admin_set(update, _context(["1", "Prefers", "direct", "answers"])))

    assert captured == {
        "telegram_user_key": "tg_user:1",
        "latest_display_name": "Alice @alice",
        "memory_text": "Prefers direct answers",
        "last_refreshed_date": "2026-04-29",
    }
    assert "Memory summary updated" in update.message.replies[0]


def test_memory_admin_candidate_review_commands(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "999")
    update = _update(user_id=999, username="Natsume_Mio")
    calls = {"accept": None, "reject": None}

    async def fake_list_user_memory_candidates(telegram_user_key=None, *, status="pending", limit=30):
        return [
            SimpleNamespace(
                id=7,
                telegram_user_key="tg_user:1",
                priority="fast",
                fact_type="style",
                fact_text="Prefers short answers",
                confidence=0.88,
                evidence_message_ids=[12],
            )
        ]

    async def fake_accept_user_memory_candidate(candidate_id):
        calls["accept"] = candidate_id
        return True

    async def fake_reject_user_memory_candidate(candidate_id):
        calls["reject"] = candidate_id
        return True

    monkeypatch.setattr(main, "list_user_memory_candidates", fake_list_user_memory_candidates)
    monkeypatch.setattr(main, "accept_user_memory_candidate", fake_accept_user_memory_candidate)
    monkeypatch.setattr(main, "reject_user_memory_candidate", fake_reject_user_memory_candidate)

    asyncio.run(main.handle_memory_admin_candidates(update, _context(["1"])))
    asyncio.run(main.handle_memory_admin_accept(update, _context(["7"])))
    asyncio.run(main.handle_memory_admin_reject(update, _context(["8"])))

    assert "#7" in update.message.replies[0]
    assert "Prefers short answers" in update.message.replies[0]
    assert calls == {"accept": 7, "reject": 8}
    assert "accepted" in update.message.replies[1]
    assert "rejected" in update.message.replies[2]


def test_memory_admin_fact_edit_commands(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "999")
    update = _update(user_id=999, username="Natsume_Mio")
    calls = {"update": None, "archive": None}

    async def fake_update_user_memory_fact(fact_id, *, fact_text=None, fact_type=None, confidence=None):
        calls["update"] = (fact_id, fact_text)
        return True

    async def fake_archive_user_memory_fact(fact_id):
        calls["archive"] = fact_id
        return True

    monkeypatch.setattr(main, "update_user_memory_fact", fake_update_user_memory_fact)
    monkeypatch.setattr(main, "archive_user_memory_fact", fake_archive_user_memory_fact)

    asyncio.run(main.handle_memory_admin_fact_set(update, _context(["5", "Prefers", "short", "answers"])))
    asyncio.run(main.handle_memory_admin_fact_delete(update, _context(["5"])))

    assert calls == {"update": (5, "Prefers short answers"), "archive": 5}
    assert "updated" in update.message.replies[0]
    assert "archived" in update.message.replies[1]


def test_global_memory_admin_commands_require_private_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update(chat_type="group", chat_id=-100)

    asyncio.run(main.handle_global_memory_admin_view(update, _context(["-100"])))

    assert update.message.replies == ["Memory admin commands are only available in a private chat."]


def test_global_memory_admin_commands_use_private_explicit_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update(chat_type="private")
    calls = {"upsert": None, "archive": None}

    async def fake_get_global_memory_facts(chat_id, *, limit=25, min_confidence=0.0):
        assert chat_id == -100
        return [
            SimpleNamespace(
                id=9,
                chat_id=chat_id,
                fact_type="style",
                fact_text="Keep group replies concise",
                confidence=0.9,
            )
        ]

    async def fake_upsert_global_memory_facts(chat_id, facts):
        calls["upsert"] = (chat_id, facts)

    async def fake_archive_global_memory_fact(chat_id, fact_id):
        calls["archive"] = (chat_id, fact_id)
        return True

    monkeypatch.setattr(main, "get_global_memory_facts", fake_get_global_memory_facts)
    monkeypatch.setattr(main, "upsert_global_memory_facts", fake_upsert_global_memory_facts)
    monkeypatch.setattr(main, "archive_global_memory_fact", fake_archive_global_memory_fact)

    asyncio.run(main.handle_global_memory_admin_view(update, _context(["-100"])))
    asyncio.run(main.handle_global_memory_admin_set(update, _context(["-100", "style", "Keep", "it", "short"])))
    asyncio.run(main.handle_global_memory_admin_delete(update, _context(["-100", "9"])))

    assert "chat_id=-100" in update.message.replies[0]
    assert "Keep group replies concise" in update.message.replies[0]
    assert calls["upsert"][0] == -100
    assert calls["upsert"][1][0]["fact_type"] == "style"
    assert calls["upsert"][1][0]["fact_text"] == "Keep it short"
    assert calls["archive"] == (-100, 9)


def test_global_memory_admin_private_chat_requires_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update(chat_type="private")

    async def fake_list_global_memory_chat_overviews(*, limit=40):
        return [
            SimpleNamespace(
                chat_id=-100,
                message_count=12,
                global_fact_count=2,
                latest_message_at="2026-05-22 10:00:00",
                latest_message_username="Alice @alice",
                latest_message_preview="latest group message",
            )
        ]

    monkeypatch.setattr(main, "list_global_memory_chat_overviews", fake_list_global_memory_chat_overviews)

    asyncio.run(main.handle_global_memory_admin_view(update, _context()))

    reply = update.message.replies[0]
    assert "Usage: /global_memory <chat_id>" in reply
    assert "Available chat_ids" in reply
    assert "chat_id=-100" in reply
    assert "messages=12" in reply
    assert "global_facts=2" in reply


def test_global_memory_admin_set_and_delete_missing_args_show_chat_list(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    update = _update(chat_type="private")

    async def fake_list_global_memory_chat_overviews(*, limit=40):
        return [
            SimpleNamespace(
                chat_id=-200,
                message_count=5,
                global_fact_count=1,
                latest_message_at="2026-05-22 11:00:00",
                latest_message_username="Bob @bob",
                latest_message_preview="another group message",
            )
        ]

    monkeypatch.setattr(main, "list_global_memory_chat_overviews", fake_list_global_memory_chat_overviews)

    asyncio.run(main.handle_global_memory_admin_set(update, _context()))
    asyncio.run(main.handle_global_memory_admin_delete(update, _context()))

    assert "Usage: /global_memory_set <chat_id>" in update.message.replies[0]
    assert "chat_id=-200" in update.message.replies[0]
    assert "Usage: /global_memory_delete <chat_id>" in update.message.replies[1]
    assert "chat_id=-200" in update.message.replies[1]
