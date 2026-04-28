import argparse
import asyncio
import aiosqlite
import sqlite3
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.rag_embeddings import (
    EmbeddingMetadata,
    embed_text_with_metadata,
    get_runtime_embedding_metadata,
    pack_embedding,
    unpack_embedding,
)
from app.runtime_config import get_runtime_bool, get_runtime_int, get_runtime_value

DB_FILE = get_runtime_value("DB_FILE")
logger = logging.getLogger(__name__)


def _db_file_path() -> str:
    # Prefer runtime env override while preserving compatibility with test monkeypatching.
    return get_runtime_value("DB_FILE") or DB_FILE


def _ensure_db_parent_dir(db_file: str) -> None:
    parent_dir = os.path.dirname(db_file)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _get_env_int(name: str, default: int) -> int:
    value = get_runtime_int(name, default)
    if value == default:
        raw = get_runtime_value(name)
        if raw and raw != str(default):
            logger.warning("Invalid %s=%r, using default=%d", name, raw, default)
    return value


def _rag_top_k() -> int:
    return _get_env_int("RAG_TOP_K", 12)


def _rag_enabled() -> bool:
    return get_runtime_bool("RAG_ENABLED", True)


def _message_review_back() -> int:
    return _get_env_int("MESSAGE_REVIEW_BACK", 80)


@dataclass(frozen=True)
class MessageRow:
    id: int
    chat_id: int
    username: str
    content: str
    timestamp: str
    reply_to_username: Optional[str] = None


@dataclass(frozen=True)
class UserMemoryRow:
    telegram_user_key: str
    latest_display_name: str
    memory_text: str
    last_refreshed_date: Optional[str] = None


def _format_message(row: MessageRow, *, max_chars: int = 800) -> str:
    content = (row.content or "").replace("\r\n", "\n").strip()
    if len(content) > max_chars:
        content = content[: max_chars - 1] + "…"
    if row.reply_to_username:
        content = f"[reply to {row.reply_to_username}] {content}"
    return f"[{row.timestamp}] {row.username}: {content}"


async def _enable_foreign_keys(db: aiosqlite.Connection) -> None:
    try:
        await db.execute("PRAGMA foreign_keys = ON")
    except Exception:
        # Best effort; if it fails, DB still works but cascade deletes won't.
        pass


def _get_message_columns(db: sqlite3.Connection) -> set[str]:
    cursor = db.execute("PRAGMA table_info(messages)")
    return {str(row[1]) for row in cursor.fetchall()}


def _migrate_messages_table(db: sqlite3.Connection) -> None:
    columns = _get_message_columns(db)

    if "telegram_message_id" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN telegram_message_id INTEGER")
    if "reply_to_telegram_message_id" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN reply_to_telegram_message_id INTEGER")
    if "reply_to_db_message_id" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN reply_to_db_message_id INTEGER")
    if "reply_to_username" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN reply_to_username TEXT")
    if "telegram_user_key" not in columns:
        db.execute("ALTER TABLE messages ADD COLUMN telegram_user_key TEXT")

    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_tg ON messages (chat_id, telegram_message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_reply_db ON messages (reply_to_db_message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_date ON messages (telegram_user_key, timestamp)")

    # Best-effort backfill for old rows that embedded reply target in content.
    db.execute(
        '''
        UPDATE messages
        SET reply_to_username = TRIM(SUBSTR(content, 11, INSTR(SUBSTR(content, 11), ']') - 1))
        WHERE reply_to_username IS NULL
          AND content LIKE '[reply_to:%'
          AND INSTR(SUBSTR(content, 11), ']') > 0
        '''
    )


def _get_message_embedding_columns(db: sqlite3.Connection) -> set[str]:
    cursor = db.execute("PRAGMA table_info(message_embeddings)")
    return {str(row[1]) for row in cursor.fetchall()}


def _migrate_message_embeddings_table(db: sqlite3.Connection) -> None:
    columns = _get_message_embedding_columns(db)
    if "backend" not in columns:
        db.execute("ALTER TABLE message_embeddings ADD COLUMN backend TEXT")
    if "signature" not in columns:
        db.execute("ALTER TABLE message_embeddings ADD COLUMN signature TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_embed_chat_signature ON message_embeddings (chat_id, signature)")


def _init_stickers_table(db: sqlite3.Connection) -> None:
    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS sticker_descriptions (
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
    db.execute("CREATE INDEX IF NOT EXISTS idx_sticker_set_name ON sticker_descriptions (set_name)")


def _init_user_memories_table(db: sqlite3.Connection) -> None:
    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS user_memories (
            telegram_user_key TEXT PRIMARY KEY,
            latest_display_name TEXT NOT NULL,
            memory_text TEXT NOT NULL DEFAULT '',
            last_refreshed_date TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )


async def _embed_message_content(username: str, content: str) -> tuple[np.ndarray, EmbeddingMetadata]:
    return await embed_text_with_metadata(f"{username}: {content}")

def init_db():
    """Initializes the database and creates the messages table if it doesn't exist."""
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    with sqlite3.connect(db_file) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                telegram_message_id INTEGER,
                reply_to_telegram_message_id INTEGER,
                reply_to_db_message_id INTEGER,
                reply_to_username TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        _migrate_messages_table(db)
        db.execute('CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON messages (chat_id, timestamp)')

        db.execute('''
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT,
                backend TEXT,
                signature TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        ''')
        _migrate_message_embeddings_table(db)
        db.execute('CREATE INDEX IF NOT EXISTS idx_embed_chat ON message_embeddings (chat_id)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_embed_chat_msg ON message_embeddings (chat_id, message_id)')
        _init_stickers_table(db)
        _init_user_memories_table(db)

        db.commit()
        logger.info("Database initialized: %s", db_file)

async def add_message(
    chat_id: int,
    username: str,
    content: str,
    *,
    telegram_user_key: Optional[str] = None,
    telegram_message_id: Optional[int] = None,
    reply_to_telegram_message_id: Optional[int] = None,
    reply_to_username: Optional[str] = None,
):
    """Adds a message to the history with optional reply-chain metadata."""
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)
        # Add the new message
        cursor = await db.execute(
            """
            INSERT INTO messages (
                chat_id,
                username,
                content,
                telegram_user_key,
                telegram_message_id,
                reply_to_telegram_message_id,
                reply_to_username
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chat_id, username, content, telegram_user_key, telegram_message_id, reply_to_telegram_message_id, reply_to_username)
        )

        lastrowid = cursor.lastrowid
        if lastrowid is None:
            await db.commit()
            return
        message_id = int(lastrowid)

        # Resolve parent DB row for reply chain when Telegram parent id is known.
        if reply_to_telegram_message_id is not None:
            parent_cursor = await db.execute(
                '''
                SELECT id
                FROM messages
                WHERE chat_id = ? AND telegram_message_id = ?
                ORDER BY id DESC
                LIMIT 1
                ''',
                (chat_id, reply_to_telegram_message_id),
            )
            parent_row = await parent_cursor.fetchone()
            if parent_row:
                await db.execute(
                    "UPDATE messages SET reply_to_db_message_id = ? WHERE id = ?",
                    (int(parent_row[0]), message_id),
                )

        # Store local embedding (best-effort)
        try:
            vec, metadata = await _embed_message_content(username, content)
            blob, dim = pack_embedding(vec)
            await db.execute(
                "INSERT OR REPLACE INTO message_embeddings (message_id, chat_id, embedding, dim, model, backend, signature) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, blob, dim, metadata.model, metadata.backend, metadata.signature),
            )
        except Exception as e:
            logging.warning(f"Embedding failed for message {message_id}: {e}")
        
        await db.commit()


async def get_recent_messages(chat_id: int, *, limit: Optional[int] = None) -> list[MessageRow]:
    effective_limit = _message_review_back() if limit is None else limit
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)
        cursor = await db.execute(
            '''
            SELECT id, chat_id, username, content, timestamp, reply_to_username FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            ''',
            (chat_id, effective_limit),
        )
        rows = list(await cursor.fetchall())
        # reverse to chronological
        rows.reverse()
        return [MessageRow(*row) for row in rows]


def _cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, *, top_k: int) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-8

    mn = np.linalg.norm(matrix, axis=1) + 1e-8
    sims = (matrix @ q) / (mn * qn)

    if top_k <= 0:
        top_k = 1
    top_k = min(top_k, sims.shape[0])

    # argpartition for speed, then sort selected
    idx = np.argpartition(-sims, top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return idx


async def vector_search_messages(
    chat_id: int,
    query: str,
    *,
    top_k: Optional[int] = None,
) -> list[MessageRow]:
    if not query.strip():
        return []

    query_vec, query_metadata = await embed_text_with_metadata(query)

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)

        cursor = await db.execute(
            '''
            SELECT m.id, m.chat_id, m.username, m.content, m.timestamp, m.reply_to_username, e.embedding, e.dim, e.signature
            FROM message_embeddings e
            JOIN messages m ON m.id = e.message_id
            WHERE e.chat_id = ?
            ''',
            (chat_id,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return []

    query_dim = int(query_metadata.dim)

    message_rows: list[MessageRow] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        msg = MessageRow(id=row[0], chat_id=row[1], username=row[2], content=row[3], timestamp=row[4], reply_to_username=row[5])
        blob = row[6]
        dim = int(row[7])
        signature = row[8]
        if signature:
            if signature != query_metadata.signature:
                continue
        elif dim != query_dim:
            continue
        vec = unpack_embedding(blob, dim)
        message_rows.append(msg)
        vectors.append(vec)

    if not vectors:
        return []

    try:
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
    except Exception:
        # Fallback if shapes inconsistent
        return []

    effective_top_k = _rag_top_k() if top_k is None else top_k
    idx = _cosine_top_k(query_vec, matrix, top_k=effective_top_k)
    selected = [message_rows[int(i)] for i in idx]
    selected.sort(key=lambda r: r.id)
    return selected


async def get_rag_context(
    chat_id: int,
    query: str,
    *,
    recent_n: Optional[int] = None,
    retrieved_k: Optional[int] = None,
) -> list[str]:
    """Return context lines where the last line is the newest message.

    We place retrieved history first, then recent chat, so the model's
    "last message is most recent" rule stays true.
    """
    recent_lines, retrieved_lines = await get_prompt_context_parts(
        chat_id,
        query,
        recent_n=recent_n,
        retrieved_k=retrieved_k,
    )

    lines: list[str] = []
    if retrieved_lines:
        lines.append("### RETRIEVED RELEVANT HISTORY")
        lines.extend(retrieved_lines)

    lines.append("### RECENT CHAT")
    lines.extend(recent_lines)
    return lines


async def get_prompt_context_parts(
    chat_id: int,
    query: str,
    *,
    recent_n: Optional[int] = None,
    retrieved_k: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Return context split into recent history and retrieved RAG lines.

    Returns:
        (recent_lines, retrieved_lines)
    """
    effective_recent_n = _message_review_back() if recent_n is None else recent_n
    effective_retrieved_k = _rag_top_k() if retrieved_k is None else retrieved_k

    recent = await get_recent_messages(chat_id, limit=effective_recent_n)

    retrieved: list[MessageRow] = []
    if _rag_enabled():
        try:
            retrieved = await vector_search_messages(chat_id, query, top_k=effective_retrieved_k)
        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            retrieved = []

    recent_ids = {m.id for m in recent}
    retrieved = [m for m in retrieved if m.id not in recent_ids]

    recent_lines = [_format_message(m) for m in recent]
    retrieved_lines = [_format_message(m) for m in retrieved]
    return recent_lines, retrieved_lines

async def get_messages(chat_id: int) -> list[str]:
    """Retrieves the last messages for a given chat, formatted as strings."""
    # Back-compat: return recent chat only.
    recent = await get_recent_messages(chat_id, limit=_message_review_back())
    logger.info(recent[-1] if recent else "No messages found for this chat.")
    return [_format_message(m) for m in recent]


async def get_user_memory(telegram_user_key: str) -> Optional[UserMemoryRow]:
    if not telegram_user_key:
        return None

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT telegram_user_key, latest_display_name, memory_text, last_refreshed_date
            FROM user_memories
            WHERE telegram_user_key = ?
            ''',
            (telegram_user_key,),
        )
        row = await cursor.fetchone()

    if not row:
        return None
    return UserMemoryRow(*row)


async def upsert_user_memory(
    telegram_user_key: str,
    *,
    latest_display_name: str,
    memory_text: str,
    last_refreshed_date: Optional[str],
) -> None:
    if not telegram_user_key:
        return

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            '''
            INSERT INTO user_memories (
                telegram_user_key,
                latest_display_name,
                memory_text,
                last_refreshed_date,
                updated_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(telegram_user_key) DO UPDATE SET
                latest_display_name = excluded.latest_display_name,
                memory_text = excluded.memory_text,
                last_refreshed_date = excluded.last_refreshed_date,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (telegram_user_key, latest_display_name, memory_text, last_refreshed_date),
        )
        await db.commit()


async def get_user_messages_for_memory(
    telegram_user_key: str,
    *,
    start_date_exclusive: Optional[str],
    end_date_inclusive: str,
    limit: int = 200,
) -> list[MessageRow]:
    if not telegram_user_key:
        return []

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    params: list[Any] = [telegram_user_key]
    query = (
        '''
        SELECT id, chat_id, username, content, timestamp, reply_to_username
        FROM messages
        WHERE telegram_user_key = ?
        '''
    )
    if start_date_exclusive:
        query += " AND date(timestamp) > date(?)"
        params.append(start_date_exclusive)
    query += " AND date(timestamp) <= date(?) ORDER BY id ASC LIMIT ?"
    params.extend([end_date_inclusive, limit])

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(query, tuple(params))
        rows = await cursor.fetchall()

    return [MessageRow(*row) for row in rows]


async def get_sticker_text(file_unique_id: str) -> Optional[str]:
    if not file_unique_id:
        return None

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            "SELECT description FROM sticker_descriptions WHERE file_unique_id = ?",
            (file_unique_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    description = row[0]
    return description if isinstance(description, str) and description.strip() else None


async def upsert_sticker_text(
    file_unique_id: str,
    *,
    file_id: Optional[str],
    emoji: Optional[str],
    set_name: Optional[str],
    description: str,
    description_source: str,
    is_animated: bool = False,
    is_video: bool = False,
) -> None:
    if not file_unique_id or not description.strip():
        return

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await db.execute(
            '''
            INSERT INTO sticker_descriptions (
                file_unique_id,
                file_id,
                emoji,
                set_name,
                description,
                description_source,
                is_animated,
                is_video,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(file_unique_id) DO UPDATE SET
                file_id = excluded.file_id,
                emoji = excluded.emoji,
                set_name = excluded.set_name,
                description = excluded.description,
                description_source = excluded.description_source,
                is_animated = excluded.is_animated,
                is_video = excluded.is_video,
                updated_at = CURRENT_TIMESTAMP
            ''',
            (
                file_unique_id,
                file_id,
                emoji,
                set_name,
                description,
                description_source,
                int(is_animated),
                int(is_video),
            ),
        )
        await db.commit()


async def get_embedding_health_report() -> dict[str, Any]:
    runtime = await get_runtime_embedding_metadata()
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    report: dict[str, Any] = {
        "db_file": db_file,
        "runtime": {
            "backend": runtime.backend,
            "model": runtime.model,
            "dim": runtime.dim,
            "signature": runtime.signature,
        },
        "messages": 0,
        "embeddings": 0,
        "messages_without_embedding": 0,
        "stored_profiles": [],
        "needs_reindex": False,
        "reasons": [],
    }

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)

        cursor = await db.execute("SELECT COUNT(*) FROM messages")
        report["messages"] = int((await cursor.fetchone() or [0])[0])

        cursor = await db.execute("SELECT COUNT(*) FROM message_embeddings")
        report["embeddings"] = int((await cursor.fetchone() or [0])[0])

        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE id NOT IN (SELECT message_id FROM message_embeddings)"
        )
        report["messages_without_embedding"] = int((await cursor.fetchone() or [0])[0])

        cursor = await db.execute(
            '''
            SELECT COALESCE(signature, ''), COALESCE(backend, ''), COALESCE(model, ''), dim, COUNT(*)
            FROM message_embeddings
            GROUP BY signature, backend, model, dim
            ORDER BY COUNT(*) DESC
            '''
        )
        rows = await cursor.fetchall()

    stored_profiles = [
        {
            "signature": row[0],
            "backend": row[1],
            "model": row[2],
            "dim": int(row[3]),
            "count": int(row[4]),
        }
        for row in rows
    ]
    report["stored_profiles"] = stored_profiles

    reasons: list[str] = []
    if report["messages_without_embedding"]:
        reasons.append(f"{report['messages_without_embedding']} messages do not have embeddings")

    if stored_profiles:
        stored_signatures = {profile["signature"] for profile in stored_profiles if profile["signature"]}
        if runtime.signature not in stored_signatures:
            reasons.append(
                f"runtime signature {runtime.signature} is absent from stored embeddings"
            )
        if any(not profile["signature"] for profile in stored_profiles):
            reasons.append("legacy embeddings without signature metadata were found")
        if len(stored_signatures) > 1:
            reasons.append("multiple embedding signatures are mixed in the same database")

    report["needs_reindex"] = bool(reasons)
    report["reasons"] = reasons
    return report


async def log_embedding_health_report() -> dict[str, Any]:
    report = await get_embedding_health_report()
    if report["needs_reindex"]:
        logger.warning(
            "Embedding health check flagged reindex need. runtime=%s, reasons=%s",
            report["runtime"]["signature"],
            "; ".join(report["reasons"]),
        )
    else:
        logger.info(
            "Embedding health check OK. runtime=%s, embeddings=%s",
            report["runtime"]["signature"],
            report["embeddings"],
        )
    return report


async def reindex_message_embeddings(*, chat_id: Optional[int] = None) -> dict[str, Any]:
    runtime = await get_runtime_embedding_metadata()
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)
        if chat_id is None:
            cursor = await db.execute(
                "SELECT id, chat_id, username, content FROM messages ORDER BY id ASC"
            )
        else:
            cursor = await db.execute(
                "SELECT id, chat_id, username, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,),
            )
        rows = await cursor.fetchall()

        processed = 0
        for message_id, row_chat_id, username, content in rows:
            vec, metadata = await _embed_message_content(str(username), str(content))
            blob, dim = pack_embedding(vec)
            await db.execute(
                '''
                INSERT OR REPLACE INTO message_embeddings (
                    message_id,
                    chat_id,
                    embedding,
                    dim,
                    model,
                    backend,
                    signature,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''',
                (
                    int(message_id),
                    int(row_chat_id),
                    blob,
                    dim,
                    metadata.model,
                    metadata.backend,
                    metadata.signature,
                ),
            )
            processed += 1
            if processed % 50 == 0:
                await db.commit()

        await db.commit()

    return {
        "reindexed": processed,
        "chat_id": chat_id,
        "signature": runtime.signature,
        "backend": runtime.backend,
        "model": runtime.model,
        "dim": runtime.dim,
    }


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="RAG maintenance helpers for MioBot.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Show embedding health and drift status.")

    reindex_parser = subparsers.add_parser("reindex", help="Rebuild message embeddings with the current runtime backend.")
    reindex_parser.add_argument("--chat-id", type=int, default=None, help="Optional chat id to reindex.")

    args = parser.parse_args()
    init_db()

    if args.command == "health":
        report = asyncio.run(get_embedding_health_report())
        print(f"db_file={report['db_file']}")
        print(f"runtime_signature={report['runtime']['signature']}")
        print(f"messages={report['messages']} embeddings={report['embeddings']} missing={report['messages_without_embedding']}")
        for profile in report["stored_profiles"]:
            print(
                "stored_profile="
                f"signature:{profile['signature'] or '(legacy)'} "
                f"backend:{profile['backend'] or '(unknown)'} "
                f"model:{profile['model'] or '(unknown)'} "
                f"dim:{profile['dim']} count:{profile['count']}"
            )
        if report["needs_reindex"]:
            print("needs_reindex=yes")
            for reason in report["reasons"]:
                print(f"reason={reason}")
        else:
            print("needs_reindex=no")
        return

    result = asyncio.run(reindex_message_embeddings(chat_id=args.chat_id))
    print(
        f"reindexed={result['reindexed']} chat_id={result['chat_id']} "
        f"signature={result['signature']} backend={result['backend']} model={result['model']} dim={result['dim']}"
    )


if __name__ == "__main__":
    cli_main()