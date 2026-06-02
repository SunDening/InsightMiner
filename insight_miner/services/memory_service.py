"""SQLite-persistent conversation history."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from insight_miner.config import DATA_DIR


class MemoryService:
    """Stores conversations in SQLite, one table per thread_id."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or DATA_DIR / "conversations.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                kb_id TEXT DEFAULT 'default',
                message_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT,
                FOREIGN KEY (thread_id) REFERENCES conversations(thread_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_kb ON conversations(kb_id);
        """)
        conn.commit()

    def get_thread_kb_id(self, thread_id: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT kb_id FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return row["kb_id"] if row else None

    def create_thread(self, thread_id: str, kb_id: str = "default") -> bool:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute(
                "INSERT INTO conversations (thread_id, kb_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, kb_id, now, now),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # already exists

    def save_message(self, thread_id: str, role: str, content: str):
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO messages (thread_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (thread_id, role, content[:10000], now),
        )
        conn.execute(
            "UPDATE conversations SET message_count = message_count + 1, updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )
        conn.commit()

    def get_history(self, thread_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE thread_id = ? ORDER BY id ASC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_threads(self, kb_id: str | None = None) -> list[dict]:
        conn = self._get_conn()
        if kb_id:
            rows = conn.execute(
                "SELECT thread_id, title, message_count, updated_at FROM conversations WHERE kb_id = ? ORDER BY updated_at DESC",
                (kb_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT thread_id, title, message_count, updated_at FROM conversations ORDER BY updated_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_thread(self, thread_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
        conn.commit()

    def update_title(self, thread_id: str, title: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE conversations SET title = ? WHERE thread_id = ?",
            (title[:200], thread_id),
        )
        conn.commit()
