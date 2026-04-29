import asyncio
from types import SimpleNamespace

import main


class _FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def _update(*, user_id=42, username="Natsume_Mio", chat_type="private"):
    message = _FakeMessage()
    return SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=user_id, username=username),
        effective_chat=SimpleNamespace(type=chat_type),
    )


def _context(args=None):
    return SimpleNamespace(args=args or [])


def test_admin_ids_accept_numeric_and_tg_user_tokens(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42, tg_user:100 @Natsume_Mio")

    assert main._configured_admin_user_ids() == {42, 100}
    assert main._configured_admin_usernames() == {"natsume_mio"}


def test_admin_check_accepts_configured_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "@Natsume_Mio")

    assert main._is_admin_update(_update(user_id=999, username="natsume_mio")) is True


def test_memory_admin_help_requires_private_admin(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "42")
    group_update = _update(user_id=42, chat_type="group")
    non_admin_update = _update(user_id=7, chat_type="private")

    asyncio.run(main.handle_memory_admin_help(group_update, _context()))
    asyncio.run(main.handle_memory_admin_help(non_admin_update, _context()))

    assert group_update.message.replies == ["Memory admin commands are only available in a private chat."]
    assert non_admin_update.message.replies == ["You are not allowed to use memory admin commands."]


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
    assert "evidence=11,12" in reply


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
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "@Natsume_Mio")
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
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "@Natsume_Mio")
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
    monkeypatch.setenv("TELEGRAM_ADMIN_USER_IDS", "@Natsume_Mio")
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