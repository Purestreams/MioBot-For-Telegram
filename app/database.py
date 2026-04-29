import argparse
import asyncio
import aiosqlite
import json
import sqlite3
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

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


def _rag_keyword_scan_back() -> int:
    return _get_env_int("RAG_KEYWORD_SCAN_BACK", 500)


def _recent_context_max_chars() -> int:
    return _get_env_int("RECENT_CONTEXT_MAX_CHARS", 12000)


def _rag_context_max_chars() -> int:
    return _get_env_int("RAG_CONTEXT_MAX_CHARS", 6000)


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


@dataclass(frozen=True)
class UserMemoryFactRow:
    id: int
    telegram_user_key: str
    fact_type: str
    fact_text: str
    confidence: float
    evidence_message_ids: list[int]
    first_observed_at: Optional[str] = None
    last_confirmed_at: Optional[str] = None


@dataclass(frozen=True)
class UserMemoryOverviewRow:
    telegram_user_key: str
    latest_display_name: str
    memory_text: str
    last_refreshed_date: Optional[str]
    fact_count: int
    latest_message_at: Optional[str]


@dataclass(frozen=True)
class UserMemorySearchRow:
    telegram_user_key: str
    latest_display_name: str
    source: str
    text: str


@dataclass(frozen=True)
class UserMemoryCandidateRow:
    id: int
    telegram_user_key: str
    fact_type: str
    fact_text: str
    confidence: float
    evidence_message_ids: list[int]
    source_message_id: Optional[int]
    priority: str
    status: str
    created_at: Optional[str]
    updated_at: Optional[str]


def _format_message(row: MessageRow, *, max_chars: int = 800) -> str:
    content = (row.content or "").replace("\r\n", "\n").strip()
    if len(content) > max_chars:
        content = content[: max_chars - 1] + "…"
    if row.reply_to_username:
        content = f"[reply to {row.reply_to_username}] {content}"
    return f"[{row.timestamp}] {row.username}: {content}"


def _trim_context_lines(lines: list[str], *, max_chars: int, keep: str = "last") -> list[str]:
    if max_chars <= 0 or not lines:
        return lines

    selected: list[str] = []
    total = 0
    source = reversed(lines) if keep == "last" else iter(lines)

    for line in source:
        text = str(line)
        cost = len(text) + 1
        if selected and total + cost > max_chars:
            break
        if cost > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
            cost = len(text) + 1
        selected.append(text)
        total += cost

    if keep == "last":
        selected.reverse()
    return selected


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


def _init_user_memory_facts_table(db: sqlite3.Connection) -> None:
    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS user_memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_user_key TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_message_ids TEXT NOT NULL DEFAULT '[]',
            first_observed_at TEXT,
            last_confirmed_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(telegram_user_key, fact_type, fact_text)
        )
        '''
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_memory_facts_user ON user_memory_facts "
        "(telegram_user_key, is_active, fact_type, confidence)"
    )


def _init_user_memory_candidates_table(db: sqlite3.Connection) -> None:
    db.execute(
        '''
        CREATE TABLE IF NOT EXISTS user_memory_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_user_key TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            evidence_message_ids TEXT NOT NULL DEFAULT '[]',
            source_message_id INTEGER,
            priority TEXT NOT NULL DEFAULT 'slow',
            status TEXT NOT NULL DEFAULT 'pending',
            review_note TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            reviewed_at DATETIME
        )
        '''
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_memory_candidates_user_status ON user_memory_candidates "
        "(telegram_user_key, status, priority, id)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_memory_candidates_status ON user_memory_candidates "
        "(status, priority, id)"
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
        _init_user_memory_facts_table(db)
        _init_user_memory_candidates_table(db)

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
) -> Optional[int]:
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
            return None
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

        await db.commit()

    # Store local embedding best-effort after committing the message row. This
    # keeps slow model work from holding a SQLite write transaction open.
    try:
        vec, metadata = await _embed_message_content(username, content)
        blob, dim = pack_embedding(vec)
        async with aiosqlite.connect(db_file) as db:
            await _enable_foreign_keys(db)
            await db.execute(
                "INSERT OR REPLACE INTO message_embeddings (message_id, chat_id, embedding, dim, model, backend, signature) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, chat_id, blob, dim, metadata.model, metadata.backend, metadata.signature),
            )
            await db.commit()
    except Exception as e:
        logger.warning("Embedding failed for message %s: %s", message_id, e)

    return message_id


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


RAG_TERM_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "you", "your",
    "are", "was", "were", "been", "being", "about", "just", "what", "when", "where", "why",
}


def _query_terms(query: str, *, max_terms: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", (query or "").lower()):
        if token in RAG_TERM_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _keyword_score(row: MessageRow, terms: list[str], raw_query: str) -> int:
    content = (row.content or "").lower()
    username = (row.username or "").lower()
    reply_to = (row.reply_to_username or "").lower()
    haystack = f"{username} {reply_to} {content}"
    score = 0
    normalized_query = (raw_query or "").strip().lower()
    if normalized_query and len(normalized_query) >= 4 and normalized_query in content:
        score += 6
    for term in terms:
        if term in content:
            score += 3
        elif term in username or term in reply_to:
            score += 1
        elif term in haystack:
            score += 1
    return score


async def keyword_search_messages(
    chat_id: int,
    query: str,
    *,
    top_k: Optional[int] = None,
    scan_limit: Optional[int] = None,
) -> list[MessageRow]:
    terms = _query_terms(query)
    if not terms and not (query or "").strip():
        return []

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    effective_scan_limit = _rag_keyword_scan_back() if scan_limit is None else scan_limit

    async with aiosqlite.connect(db_file) as db:
        await _enable_foreign_keys(db)
        cursor = await db.execute(
            '''
            SELECT id, chat_id, username, content, timestamp, reply_to_username
            FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
            ''',
            (chat_id, effective_scan_limit),
        )
        rows = [MessageRow(*row) for row in await cursor.fetchall()]

    scored = [(_keyword_score(row, terms, query), row) for row in rows]
    scored = [(score, row) for score, row in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], -item[1].id))

    effective_top_k = _rag_top_k() if top_k is None else top_k
    selected = [row for _, row in scored[:effective_top_k]]
    selected.sort(key=lambda row: row.id)
    return selected


def _merge_retrieved_messages(*groups: list[MessageRow], top_k: int) -> list[MessageRow]:
    seen: set[int] = set()
    merged: list[MessageRow] = []
    for group in groups:
        for row in group:
            if row.id in seen:
                continue
            seen.add(row.id)
            merged.append(row)
            if len(merged) >= top_k:
                return merged
    return merged


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
            search_k = max(effective_retrieved_k * 2, effective_retrieved_k)
            vector_retrieved = await vector_search_messages(chat_id, query, top_k=search_k)
            keyword_retrieved = await keyword_search_messages(chat_id, query, top_k=search_k)
            retrieved = _merge_retrieved_messages(vector_retrieved, keyword_retrieved, top_k=search_k)
        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            retrieved = []

    recent_ids = {m.id for m in recent}
    retrieved = [m for m in retrieved if m.id not in recent_ids]
    retrieved = retrieved[:effective_retrieved_k]
    retrieved.sort(key=lambda m: m.id)

    recent_lines = [_format_message(m) for m in recent]
    retrieved_lines = [_format_message(m) for m in retrieved]
    recent_lines = _trim_context_lines(recent_lines, max_chars=_recent_context_max_chars(), keep="last")
    retrieved_lines = _trim_context_lines(retrieved_lines, max_chars=_rag_context_max_chars(), keep="first")
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


def _decode_evidence_ids(raw_value: Any) -> list[int]:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        source = raw_value
    else:
        try:
            source = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return []
    evidence_ids: list[int] = []
    for item in source if isinstance(source, list) else []:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value not in evidence_ids:
            evidence_ids.append(value)
    return evidence_ids


def _encode_evidence_ids(values: list[int]) -> str:
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return json.dumps(unique, ensure_ascii=False)


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return min(1.0, max(0.0, confidence))


async def get_user_memory_facts(
    telegram_user_key: str,
    *,
    limit: int = 8,
    min_confidence: float = 0.0,
) -> list[UserMemoryFactRow]:
    if not telegram_user_key:
        return []

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT id, telegram_user_key, fact_type, fact_text, confidence,
                   evidence_message_ids, first_observed_at, last_confirmed_at
            FROM user_memory_facts
            WHERE telegram_user_key = ? AND is_active = 1 AND confidence >= ?
            ORDER BY confidence DESC, updated_at DESC, id DESC
            LIMIT ?
            ''',
            (telegram_user_key, min_confidence, limit),
        )
        rows = await cursor.fetchall()

    return [
        UserMemoryFactRow(
            id=int(row[0]),
            telegram_user_key=str(row[1]),
            fact_type=str(row[2]),
            fact_text=str(row[3]),
            confidence=float(row[4]),
            evidence_message_ids=_decode_evidence_ids(row[5]),
            first_observed_at=row[6],
            last_confirmed_at=row[7],
        )
        for row in rows
    ]


async def get_user_memory_fact_by_id(fact_id: int) -> Optional[UserMemoryFactRow]:
    if fact_id <= 0:
        return None

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT id, telegram_user_key, fact_type, fact_text, confidence,
                   evidence_message_ids, first_observed_at, last_confirmed_at
            FROM user_memory_facts
            WHERE id = ? AND is_active = 1
            ''',
            (fact_id,),
        )
        row = await cursor.fetchone()

    if not row:
        return None
    return UserMemoryFactRow(
        id=int(row[0]),
        telegram_user_key=str(row[1]),
        fact_type=str(row[2]),
        fact_text=str(row[3]),
        confidence=float(row[4]),
        evidence_message_ids=_decode_evidence_ids(row[5]),
        first_observed_at=row[6],
        last_confirmed_at=row[7],
    )


async def update_user_memory_fact(
    fact_id: int,
    *,
    fact_text: Optional[str] = None,
    fact_type: Optional[str] = None,
    confidence: Optional[float] = None,
) -> bool:
    if fact_id <= 0:
        return False

    assignments: list[str] = []
    params: list[Any] = []
    if fact_text is not None:
        cleaned_text = fact_text.strip()
        if not cleaned_text:
            return False
        assignments.append("fact_text = ?")
        params.append(cleaned_text)
    if fact_type is not None:
        cleaned_type = fact_type.strip().lower() or "note"
        assignments.append("fact_type = ?")
        params.append(cleaned_type)
    if confidence is not None:
        assignments.append("confidence = ?")
        params.append(_clamp_confidence(confidence))
    if not assignments:
        return False

    assignments.append("updated_at = CURRENT_TIMESTAMP")
    params.append(fact_id)

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            f"UPDATE user_memory_facts SET {', '.join(assignments)} WHERE id = ? AND is_active = 1",
            tuple(params),
        )
        await db.commit()
        return cursor.rowcount > 0


async def archive_user_memory_fact(fact_id: int) -> bool:
    if fact_id <= 0:
        return False

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            UPDATE user_memory_facts
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND is_active = 1
            ''',
            (fact_id,),
        )
        await db.commit()
        return cursor.rowcount > 0


async def archive_user_memory_facts(fact_ids: list[int]) -> int:
    valid_ids = [int(fact_id) for fact_id in fact_ids if int(fact_id) > 0]
    if not valid_ids:
        return 0

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    placeholders = ",".join("?" for _ in valid_ids)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            f'''
            UPDATE user_memory_facts
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders}) AND is_active = 1
            ''',
            tuple(valid_ids),
        )
        await db.commit()
        return cursor.rowcount


async def upsert_user_memory_candidate(
    telegram_user_key: str,
    *,
    fact_type: str,
    fact_text: str,
    confidence: float = 0.5,
    evidence_message_ids: Optional[list[int]] = None,
    source_message_id: Optional[int] = None,
    priority: str = "slow",
    status: str = "pending",
) -> Optional[int]:
    if not telegram_user_key or not fact_text.strip():
        return None

    cleaned_type = fact_type.strip().lower() or "note"
    cleaned_text = fact_text.strip()
    cleaned_priority = priority.strip().lower() if priority else "slow"
    if cleaned_priority not in {"fast", "slow"}:
        cleaned_priority = "slow"
    cleaned_status = status.strip().lower() if status else "pending"

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT id, confidence, evidence_message_ids, priority
            FROM user_memory_candidates
            WHERE telegram_user_key = ?
              AND fact_type = ?
              AND fact_text = ?
              AND status = 'pending'
            ORDER BY id DESC
            LIMIT 1
            ''',
            (telegram_user_key, cleaned_type, cleaned_text),
        )
        existing = await cursor.fetchone()
        evidence_ids = _decode_evidence_ids(evidence_message_ids or [])
        if source_message_id is not None and source_message_id not in evidence_ids:
            evidence_ids.append(source_message_id)

        if existing:
            candidate_id = int(existing[0])
            merged_evidence_ids = _decode_evidence_ids(existing[2]) + evidence_ids
            merged_priority = "fast" if cleaned_priority == "fast" or str(existing[3]) == "fast" else "slow"
            await db.execute(
                '''
                UPDATE user_memory_candidates
                SET confidence = ?,
                    evidence_message_ids = ?,
                    source_message_id = COALESCE(?, source_message_id),
                    priority = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                ''',
                (
                    max(float(existing[1] or 0.0), _clamp_confidence(confidence)),
                    _encode_evidence_ids(merged_evidence_ids),
                    source_message_id,
                    merged_priority,
                    candidate_id,
                ),
            )
            await db.commit()
            return candidate_id

        cursor = await db.execute(
            '''
            INSERT INTO user_memory_candidates (
                telegram_user_key,
                fact_type,
                fact_text,
                confidence,
                evidence_message_ids,
                source_message_id,
                priority,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                telegram_user_key,
                cleaned_type,
                cleaned_text,
                _clamp_confidence(confidence),
                _encode_evidence_ids(evidence_ids),
                source_message_id,
                cleaned_priority,
                cleaned_status,
            ),
        )
        await db.commit()
        return int(cursor.lastrowid) if cursor.lastrowid is not None else None


def _candidate_from_row(row) -> UserMemoryCandidateRow:
    return UserMemoryCandidateRow(
        id=int(row[0]),
        telegram_user_key=str(row[1]),
        fact_type=str(row[2]),
        fact_text=str(row[3]),
        confidence=float(row[4]),
        evidence_message_ids=_decode_evidence_ids(row[5]),
        source_message_id=int(row[6]) if row[6] is not None else None,
        priority=str(row[7]),
        status=str(row[8]),
        created_at=row[9],
        updated_at=row[10],
    )


async def get_user_memory_candidate(candidate_id: int) -> Optional[UserMemoryCandidateRow]:
    if candidate_id <= 0:
        return None

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT id, telegram_user_key, fact_type, fact_text, confidence,
                   evidence_message_ids, source_message_id, priority, status, created_at, updated_at
            FROM user_memory_candidates
            WHERE id = ?
            ''',
            (candidate_id,),
        )
        row = await cursor.fetchone()

    return _candidate_from_row(row) if row else None


async def list_user_memory_candidates(
    telegram_user_key: Optional[str] = None,
    *,
    status: str = "pending",
    limit: int = 20,
) -> list[UserMemoryCandidateRow]:
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    params: list[Any] = []
    query = '''
        SELECT id, telegram_user_key, fact_type, fact_text, confidence,
               evidence_message_ids, source_message_id, priority, status, created_at, updated_at
        FROM user_memory_candidates
        WHERE 1 = 1
    '''
    if telegram_user_key:
        query += " AND telegram_user_key = ?"
        params.append(telegram_user_key)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY CASE priority WHEN 'fast' THEN 0 ELSE 1 END, id DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(query, tuple(params))
        rows = await cursor.fetchall()

    return [_candidate_from_row(row) for row in rows]


async def count_pending_user_memory_candidates(telegram_user_key: str) -> int:
    if not telegram_user_key:
        return 0

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT COUNT(*)
            FROM user_memory_candidates
            WHERE telegram_user_key = ? AND status = 'pending'
            ''',
            (telegram_user_key,),
        )
        row = await cursor.fetchone()
    return int(row[0] or 0) if row else 0


async def update_user_memory_candidate_status(
    candidate_id: int,
    status: str,
    *,
    review_note: Optional[str] = None,
) -> bool:
    if candidate_id <= 0:
        return False
    cleaned_status = status.strip().lower() or "pending"

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            UPDATE user_memory_candidates
            SET status = ?,
                review_note = ?,
                reviewed_at = CASE WHEN ? != 'pending' THEN CURRENT_TIMESTAMP ELSE reviewed_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            ''',
            (cleaned_status, review_note, cleaned_status, candidate_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def mark_user_memory_candidates_status(
    candidate_ids: list[int],
    status: str,
    *,
    review_note: Optional[str] = None,
) -> int:
    valid_ids = [int(candidate_id) for candidate_id in candidate_ids if int(candidate_id) > 0]
    if not valid_ids:
        return 0
    cleaned_status = status.strip().lower() or "pending"
    placeholders = ",".join("?" for _ in valid_ids)

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            f'''
            UPDATE user_memory_candidates
            SET status = ?,
                review_note = ?,
                reviewed_at = CASE WHEN ? != 'pending' THEN CURRENT_TIMESTAMP ELSE reviewed_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders})
            ''',
            (cleaned_status, review_note, cleaned_status, *valid_ids),
        )
        await db.commit()
        return cursor.rowcount


async def list_user_memory_overviews(*, limit: int = 50) -> list[UserMemoryOverviewRow]:
    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            WITH memory_keys AS (
                SELECT telegram_user_key FROM user_memories
                UNION
                SELECT telegram_user_key FROM user_memory_facts WHERE is_active = 1
            ),
            fact_counts AS (
                SELECT telegram_user_key, COUNT(*) AS fact_count
                FROM user_memory_facts
                WHERE is_active = 1
                GROUP BY telegram_user_key
            ),
            latest_message_ids AS (
                SELECT telegram_user_key, MAX(id) AS latest_id
                FROM messages
                WHERE telegram_user_key IS NOT NULL
                GROUP BY telegram_user_key
            ),
            latest_messages AS (
                SELECT m.telegram_user_key, m.username, m.timestamp AS latest_message_at
                FROM messages m
                JOIN latest_message_ids l ON l.latest_id = m.id
            ),
            message_keys AS (
                SELECT telegram_user_key FROM messages WHERE telegram_user_key IS NOT NULL
            )
            SELECT
                k.telegram_user_key,
                COALESCE(NULLIF(um.latest_display_name, ''), lm.username, ''),
                COALESCE(um.memory_text, ''),
                um.last_refreshed_date,
                COALESCE(fc.fact_count, 0),
                lm.latest_message_at
            FROM (
                SELECT telegram_user_key FROM memory_keys
                UNION
                SELECT telegram_user_key FROM message_keys
            ) k
            LEFT JOIN user_memories um ON um.telegram_user_key = k.telegram_user_key
            LEFT JOIN fact_counts fc ON fc.telegram_user_key = k.telegram_user_key
            LEFT JOIN latest_messages lm ON lm.telegram_user_key = k.telegram_user_key
            ORDER BY COALESCE(um.updated_at, lm.latest_message_at, '') DESC, k.telegram_user_key ASC
            LIMIT ?
            ''',
            (limit,),
        )
        rows = await cursor.fetchall()

    return [
        UserMemoryOverviewRow(
            telegram_user_key=str(row[0]),
            latest_display_name=str(row[1] or ''),
            memory_text=str(row[2] or ''),
            last_refreshed_date=row[3],
            fact_count=int(row[4] or 0),
            latest_message_at=row[5],
        )
        for row in rows
    ]


async def search_user_memories(query: str, *, limit: int = 20) -> list[UserMemorySearchRow]:
    needle = (query or '').strip().lower()
    if not needle:
        return []

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)
    like_query = f"%{needle}%"

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            WITH display_names AS (
                SELECT telegram_user_key, latest_display_name FROM user_memories
                UNION
                SELECT telegram_user_key, username FROM messages WHERE telegram_user_key IS NOT NULL
            )
            SELECT um.telegram_user_key,
                   COALESCE(NULLIF(um.latest_display_name, ''), ''),
                   'summary' AS source,
                   um.memory_text
            FROM user_memories um
            WHERE LOWER(um.telegram_user_key) LIKE ?
               OR LOWER(um.latest_display_name) LIKE ?
               OR LOWER(um.memory_text) LIKE ?
            UNION ALL
            SELECT f.telegram_user_key,
                   COALESCE((
                       SELECT latest_display_name
                       FROM display_names d
                       WHERE d.telegram_user_key = f.telegram_user_key
                         AND COALESCE(d.latest_display_name, '') != ''
                       LIMIT 1
                   ), ''),
                   'fact:' || f.fact_type AS source,
                   f.fact_text
            FROM user_memory_facts f
            WHERE f.is_active = 1
              AND (
                  LOWER(f.telegram_user_key) LIKE ?
                  OR LOWER(f.fact_type) LIKE ?
                  OR LOWER(f.fact_text) LIKE ?
              )
            LIMIT ?
            ''',
            (like_query, like_query, like_query, like_query, like_query, like_query, limit),
        )
        rows = await cursor.fetchall()

    return [
        UserMemorySearchRow(
            telegram_user_key=str(row[0]),
            latest_display_name=str(row[1] or ''),
            source=str(row[2]),
            text=str(row[3] or ''),
        )
        for row in rows
    ]


async def get_latest_display_name_for_user(telegram_user_key: str) -> Optional[str]:
    if not telegram_user_key:
        return None

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute(
            '''
            SELECT latest_display_name
            FROM user_memories
            WHERE telegram_user_key = ? AND COALESCE(latest_display_name, '') != ''
            ''',
            (telegram_user_key,),
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return str(row[0])

        cursor = await db.execute(
            '''
            SELECT username
            FROM messages
            WHERE telegram_user_key = ? AND COALESCE(username, '') != ''
            ORDER BY id DESC
            LIMIT 1
            ''',
            (telegram_user_key,),
        )
        row = await cursor.fetchone()

    return str(row[0]) if row and row[0] else None


async def upsert_user_memory_facts(
    telegram_user_key: str,
    facts: list[Mapping[str, Any]],
) -> None:
    if not telegram_user_key or not facts:
        return

    db_file = _db_file_path()
    _ensure_db_parent_dir(db_file)

    async with aiosqlite.connect(db_file) as db:
        for fact in facts:
            fact_type = str(fact.get("fact_type") or fact.get("type") or "note").strip().lower() or "note"
            fact_text = str(fact.get("fact_text") or fact.get("text") or "").strip()
            if not fact_text:
                continue

            confidence = _clamp_confidence(fact.get("confidence", 0.5))
            new_evidence_ids = _decode_evidence_ids(fact.get("evidence_message_ids") or fact.get("evidence_ids"))
            observed_at = fact.get("first_observed_at") or fact.get("observed_at")
            confirmed_at = fact.get("last_confirmed_at") or fact.get("observed_at")

            cursor = await db.execute(
                '''
                SELECT evidence_message_ids, confidence, first_observed_at
                FROM user_memory_facts
                WHERE telegram_user_key = ? AND fact_type = ? AND fact_text = ?
                ''',
                (telegram_user_key, fact_type, fact_text),
            )
            existing = await cursor.fetchone()

            if existing:
                merged_evidence_ids = _decode_evidence_ids(existing[0]) + new_evidence_ids
                merged_confidence = max(float(existing[1] or 0.0), confidence)
                first_observed_at = existing[2] or observed_at
                await db.execute(
                    '''
                    UPDATE user_memory_facts
                    SET confidence = ?,
                        evidence_message_ids = ?,
                        first_observed_at = ?,
                        last_confirmed_at = COALESCE(?, CURRENT_TIMESTAMP),
                        is_active = 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE telegram_user_key = ? AND fact_type = ? AND fact_text = ?
                    ''',
                    (
                        merged_confidence,
                        _encode_evidence_ids(merged_evidence_ids),
                        first_observed_at,
                        confirmed_at,
                        telegram_user_key,
                        fact_type,
                        fact_text,
                    ),
                )
                continue

            await db.execute(
                '''
                INSERT INTO user_memory_facts (
                    telegram_user_key,
                    fact_type,
                    fact_text,
                    confidence,
                    evidence_message_ids,
                    first_observed_at,
                    last_confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                ''',
                (
                    telegram_user_key,
                    fact_type,
                    fact_text,
                    confidence,
                    _encode_evidence_ids(new_evidence_ids),
                    observed_at,
                    confirmed_at,
                ),
            )
        await db.commit()


async def get_user_messages_for_memory(
    telegram_user_key: str,
    *,
    start_date_exclusive: Optional[str],
    end_date_inclusive: str,
    limit: Optional[int] = 200,
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
    query += " AND date(timestamp) <= date(?) ORDER BY id ASC"
    params.append(end_date_inclusive)
    if limit and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

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