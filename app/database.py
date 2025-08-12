import aiosqlite
import sqlite3
import logging

DB_FILE = "message_history.db"
MESSAGE_REVIEW_BACK = 50

def init_db():
    """Initializes the database and creates the messages table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as db:
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
        db.commit()
        # Ensure the table is created
        print("Database initialized successfully.")
        #print the latest 5 messages
        print(db.execute('SELECT * FROM messages ORDER BY timestamp DESC LIMIT 5').fetchall())

async def add_message(chat_id: int, username: str, content: str):
    """Adds a message to the history and culls old messages."""
    async with aiosqlite.connect(DB_FILE) as db:
        # Add the new message
        await db.execute(
            "INSERT INTO messages (chat_id, username, content) VALUES (?, ?, ?)",
            (chat_id, username, content)
        )
        
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

async def get_messages(chat_id: int) -> list[str]:
    """Retrieves the last messages for a given chat, formatted as strings."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            '''
            SELECT username, content, timestamp FROM messages
            WHERE chat_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
            ''',
            (chat_id, MESSAGE_REVIEW_BACK)
        )
        rows = await cursor.fetchall()

        logging.info(rows[-1] if rows else "No messages found for this chat.")
        return [f"username: {row[0]} \n content: {row[1]} \n time: {row[2]}" for row in rows]