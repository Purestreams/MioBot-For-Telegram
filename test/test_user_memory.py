import asyncio
import datetime as dt
import sqlite3

from app import database, user_memory


def test_refresh_user_memory_only_runs_once_per_day(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)

    async def fake_chat_completion_text(**kwargs):
        return "likes concise answers\ninterested in iOS beta updates"

    monkeypatch.setattr(user_memory, "chat_completion_text", fake_chat_completion_text)

    async def _run():
        await database.add_message(
            10,
            "Alice @alice",
            "I am testing iOS 26.4 again",
            telegram_user_key="tg_user:111",
        )
        await database.add_message(
            10,
            "Alice @alice",
            "I prefer short answers",
            telegram_user_key="tg_user:111",
        )

        with sqlite3.connect(db_path) as db:
            db.execute(
                "UPDATE messages SET timestamp = '2026-04-27 10:00:00' WHERE telegram_user_key = 'tg_user:111'"
            )
            db.commit()

        first = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:111",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        second = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:111",
            latest_display_name="Alice @alice",
            today_utc=dt.date(2026, 4, 28),
        )
        row = await database.get_user_memory("tg_user:111")
        return first, second, row

    first, second, row = asyncio.run(_run())

    assert first == "likes concise answers\ninterested in iOS beta updates"
    assert second == first
    assert row is not None
    assert row.last_refreshed_date == "2026-04-27"


def test_refresh_user_memory_marks_day_even_without_messages(monkeypatch, tmp_path):
    db_path = tmp_path / "user_memory_empty.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))
    database.init_db()

    async def _run():
        result = await user_memory.refresh_user_memory_if_due(
            telegram_user_key="tg_user:222",
            latest_display_name="Bob @bob",
            today_utc=dt.date(2026, 4, 28),
        )
        row = await database.get_user_memory("tg_user:222")
        return result, row

    result, row = asyncio.run(_run())

    assert result is None
    assert row is not None
    assert row.memory_text == ""
    assert row.last_refreshed_date == "2026-04-27"