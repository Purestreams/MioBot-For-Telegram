import aiosqlite
import sqlite3
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.rag_embeddings import embed_text, pack_embedding, unpack_embedding
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


def _format_message(row: MessageRow, *, max_chars: int = 800) -> str:
    content = (row.content or "").replace("\r\n", "\n").strip()
    if len(content) > max_chars:
        content = content[: max_chars - 1] + "…"
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

    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_tg ON messages (chat_id, telegram_message_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_reply_db ON messages (reply_to_db_message_id)")

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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
            )
        ''')
        db.execute('CREATE INDEX IF NOT EXISTS idx_embed_chat ON message_embeddings (chat_id)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_embed_chat_msg ON message_embeddings (chat_id, message_id)')

        db.commit()
        logger.info("Database initialized: %s", db_file)

async def add_message(
    chat_id: int,
    username: str,
    content: str,
    *,
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
                telegram_message_id,
                reply_to_telegram_message_id,
                reply_to_username
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, username, content, telegram_message_id, reply_to_telegram_message_id, reply_to_username)
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
            vec = await embed_text(f"{username}: {content}")
            blob, dim = pack_embedding(vec)
            await db.execute(
                "INSERT OR REPLACE INTO message_embeddings (message_id, chat_id, embedding, dim, model) VALUES (?, ?, ?, ?, ?)",
                (message_id, chat_id, blob, dim, get_runtime_value("EMBED_MODEL")),
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
            SELECT id, chat_id, username, content, timestamp FROM messages
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

    query_vec = await embed_text(query)

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)

        cursor = await db.execute(
            '''
            SELECT m.id, m.chat_id, m.username, m.content, m.timestamp, e.embedding, e.dim
            FROM message_embeddings e
            JOIN messages m ON m.id = e.message_id
            WHERE e.chat_id = ?
            ''',
            (chat_id,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return []

    query_dim = int(query_vec.shape[0])

    message_rows: list[MessageRow] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        msg = MessageRow(id=row[0], chat_id=row[1], username=row[2], content=row[3], timestamp=row[4])
        blob = row[5]
        dim = int(row[6])
        if dim != query_dim:
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