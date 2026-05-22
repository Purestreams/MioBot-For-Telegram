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


def test_init_db_auto_migrates_old_sticker_schema(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy_stickers.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    with sqlite3.connect(db_path) as db:
        db.execute(
            '''
            CREATE TABLE sticker_descriptions (
                file_unique_id TEXT PRIMARY KEY,
                file_id TEXT,
                emoji TEXT,
                set_name TEXT,
                description TEXT NOT NULL,
                description_source TEXT NOT NULL DEFAULT 'fallback',
                is_animated INTEGER NOT NULL DEFAULT 0,
                is_video INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        db.commit()

    database.init_db()

    with sqlite3.connect(db_path) as db:
        cols = {row[1] for row in db.execute("PRAGMA table_info(sticker_descriptions)").fetchall()}

    assert "sticker_tags" in cols
    assert "mood" in cols
    assert "safe_for_reply" in cols
    assert "use_count" in cols
    assert "last_used_at" in cols


def test_init_db_versions_and_migrates_unscoped_global_memory(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy_global_memory.db"
    monkeypatch.setenv("DB_FILE", str(db_path))
    monkeypatch.setattr(database, "DB_FILE", str(db_path))

    with sqlite3.connect(db_path) as db:
        db.execute(
            '''
            CREATE TABLE global_memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                evidence_message_ids TEXT NOT NULL DEFAULT '[]',
                first_observed_at TEXT,
                last_confirmed_at TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(fact_type, fact_text)
            )
            '''
        )
        db.execute(
            "INSERT INTO global_memory_facts (fact_type, fact_text, confidence) VALUES (?, ?, ?)",
            ("style", "Legacy unscoped memory", 0.8),
        )
        db.commit()

    database.init_db()

    with sqlite3.connect(db_path) as db:
        cols = {row[1] for row in db.execute("PRAGMA table_info(global_memory_facts)").fetchall()}
        version = db.execute("SELECT value FROM db_metadata WHERE key = 'schema_version'").fetchone()
        legacy_backup = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'global_memory_facts_unscoped_backup'"
        ).fetchone()
        legacy_row = db.execute(
            "SELECT chat_id, fact_type, fact_text FROM global_memory_facts WHERE fact_text = 'Legacy unscoped memory'"
        ).fetchone()

    assert "chat_id" in cols
    assert version == (str(database.DB_SCHEMA_VERSION),)
    assert legacy_backup is not None
    assert legacy_row == (0, "style", "Legacy unscoped memory")

    with sqlite3.connect(db_path) as db:
        webadmin_table = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'webadmin_login_tokens'"
        ).fetchone()

    assert webadmin_table is not None

    async def _run():
        await database.upsert_global_memory_facts(
            -100,
            [{"fact_type": "style", "fact_text": "Legacy unscoped memory", "confidence": 0.9}],
        )
        await database.upsert_global_memory_facts(
            -200,
            [{"fact_type": "style", "fact_text": "Legacy unscoped memory", "confidence": 0.7}],
        )
        return await database.get_global_memory_facts(-100), await database.get_global_memory_facts(-200)

    chat_100_facts, chat_200_facts = asyncio.run(_run())

    assert chat_100_facts[0].chat_id == -100
    assert chat_100_facts[0].confidence == 0.9
    assert chat_200_facts[0].chat_id == -200
