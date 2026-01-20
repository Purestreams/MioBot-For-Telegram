import aiosqlite
import sqlite3
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.rag_embeddings import embed_text, pack_embedding, unpack_embedding

DB_FILE = "message_history.db"
MESSAGE_REVIEW_BACK = 80

# RAG defaults
RAG_RECENT_N = int(os.getenv("RAG_RECENT_N", "20"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "1") not in {"0", "false", "False"}


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

def init_db():
    """Initializes the database and creates the messages table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
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
        # Ensure the table is created
        print("Database initialized successfully.")
        #print the latest 5 messages
        print(db.execute('SELECT * FROM messages ORDER BY timestamp DESC LIMIT 5').fetchall())

async def add_message(chat_id: int, username: str, content: str):
    """Adds a message to the history and culls old messages."""
    async with aiosqlite.connect(DB_FILE) as db:
        await _enable_foreign_keys(db)
        # Add the new message
        cursor = await db.execute(
            "INSERT INTO messages (chat_id, username, content) VALUES (?, ?, ?)",
            (chat_id, username, content)
        )

        message_id = int(cursor.lastrowid)

        # Store local embedding (best-effort)
        try:
            vec = await embed_text(f"{username}: {content}")
            blob, dim = pack_embedding(vec)
            await db.execute(
                "INSERT OR REPLACE INTO message_embeddings (message_id, chat_id, embedding, dim, model) VALUES (?, ?, ?, ?, ?)",
                (message_id, chat_id, blob, dim, os.getenv("EMBED_MODEL")),
            )
        except Exception as e:
            logging.warning(f"Embedding failed for message {message_id}: {e}")
        
        # Keep only the last MESSAGE_REVIEW_BACK messages for the chat
        await db.execute('''
            DELETE FROM messages
            WHERE id IN (
                SELECT id FROM messages
                WHERE chat_id = ?
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
            )
        ''', (chat_id, MESSAGE_REVIEW_BACK))
        await db.commit()


async def get_recent_messages(chat_id: int, *, limit: int = RAG_RECENT_N) -> list[MessageRow]:
    async with aiosqlite.connect(DB_FILE) as db:
        await _enable_foreign_keys(db)
        cursor = await db.execute(
            '''
            SELECT id, chat_id, username, content, timestamp FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''',
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
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
    top_k: int = RAG_TOP_K,
) -> list[MessageRow]:
    if not query.strip():
        return []

    query_vec = await embed_text(query)

    async with aiosqlite.connect(DB_FILE) as db:
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

    idx = _cosine_top_k(query_vec, matrix, top_k=top_k)
    selected = [message_rows[int(i)] for i in idx]
    selected.sort(key=lambda r: r.timestamp)
    return selected


async def get_rag_context(
    chat_id: int,
    query: str,
    *,
    recent_n: int = RAG_RECENT_N,
    retrieved_k: int = RAG_TOP_K,
) -> list[str]:
    """Return context lines where the last line is the newest message.

    We place retrieved history first, then recent chat, so the model's
    "last message is most recent" rule stays true.
    """
    recent = await get_recent_messages(chat_id, limit=recent_n)

    retrieved: list[MessageRow] = []
    if RAG_ENABLED:
        try:
            retrieved = await vector_search_messages(chat_id, query, top_k=retrieved_k)
        except Exception as e:
            logging.warning(f"Vector search failed: {e}")
            retrieved = []

    recent_ids = {m.id for m in recent}
    retrieved = [m for m in retrieved if m.id not in recent_ids]

    lines: list[str] = []
    if retrieved:
        lines.append("### RETRIEVED RELEVANT HISTORY")
        lines.extend(_format_message(m) for m in retrieved)

    lines.append("### RECENT CHAT")
    lines.extend(_format_message(m) for m in recent)
    return lines

async def get_messages(chat_id: int) -> list[str]:
    """Retrieves the last messages for a given chat, formatted as strings."""
    # Back-compat: return recent chat only.
    recent = await get_recent_messages(chat_id, limit=MESSAGE_REVIEW_BACK)
    logging.info(recent[-1] if recent else "No messages found for this chat.")
    return [_format_message(m) for m in recent]