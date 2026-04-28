import asyncio
import sqlite3

from app import database


def test_init_db_auto_migrates_old_messages_schema(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    # Simulate old schema from earlier versions.
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.commit()

    database.init_db()

    with sqlite3.connect(db_path) as db:
        cols = {row[1] for row in db.execute("PRAGMA table_info(messages)").fetchall()}

    assert "telegram_message_id" in cols
    assert "reply_to_telegram_message_id" in cols
    assert "reply_to_db_message_id" in cols
    assert "reply_to_username" in cols
    assert "telegram_user_key" in cols


def test_add_message_resolves_reply_to_db_message_id(monkeypatch, tmp_path):
    db_path = tmp_path / "reply_chain.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    async def fake_embed_message_content(*args, **kwargs):
        raise RuntimeError("skip embeddings in this test")

    monkeypatch.setattr(database, "_embed_message_content", fake_embed_message_content)
    database.init_db()

    async def _run():
        await database.add_message(
            42,
            "UserA @user_a",
            "Parent",
            telegram_message_id=1001,
        )
        await database.add_message(
            42,
            "UserB @user_b",
            "Child",
            telegram_message_id=1002,
            reply_to_telegram_message_id=1001,
            reply_to_username="UserA @user_a",
        )

    asyncio.run(_run())

    with sqlite3.connect(db_path) as db:
        row = db.execute(
            """
            SELECT child.reply_to_db_message_id, parent.id
            FROM messages child
            JOIN messages parent ON parent.chat_id = child.chat_id
            WHERE child.telegram_message_id = 1002
              AND parent.telegram_message_id = 1001
            """
        ).fetchone()

    assert row is not None
    assert row[0] == row[1]


def test_init_db_auto_migrates_old_message_embeddings_schema(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy_embed.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    with sqlite3.connect(db_path) as db:
        db.execute(
            '''
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        db.execute(
            '''
            CREATE TABLE message_embeddings (
                message_id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        db.commit()

    database.init_db()

    with sqlite3.connect(db_path) as db:
        embed_cols = {row[1] for row in db.execute("PRAGMA table_info(message_embeddings)").fetchall()}
        sticker_tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sticker_descriptions'"
        ).fetchone()
        memory_tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'user_memories'"
        ).fetchone()

    assert "backend" in embed_cols
    assert "signature" in embed_cols
    assert sticker_tables is not None
    assert memory_tables is not None
