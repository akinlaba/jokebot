from pathlib import Path
import sqlite3
from datetime import datetime

# Save DB inside data/ directory
DB_PATH = Path(__file__).parent.parent / "data" / "chat_logs.db"

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)  # Ensure the 'data/' folder exists
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                retrieved_jokes TEXT
            );
        """)
        conn.commit()

def log_chat(user_input: str, bot_response: str, session_id: str = None, retrieved_jokes: list[str] = None):
    timestamp = datetime.utcnow().isoformat()
    jokes_str = "\n---\n".join(retrieved_jokes) if retrieved_jokes else None
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO chat_logs (timestamp, session_id, user_input, bot_response, retrieved_jokes)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, session_id, user_input, bot_response, jokes_str))
        conn.commit()

init_db()