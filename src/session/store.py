"""SQLite 会话存储 — 持久化 + TTL 清理"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings
from logger import logger
from models.session import Message, Session


class SessionStore:
    """SQLite 会话持久化存储（线程安全）"""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = settings.session.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._create_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        """惰性连接（调用方已持有 _lock）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        """关闭数据库连接（测试清理用）"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        with self._lock:
            conn = self.conn
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    context_summary TEXT,
                    current_topic TEXT,
                    topic_embedding TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
            """)

    # ---- Session CRUD ----

    def create(self, session_id: str) -> Session:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        with self._lock:
            conn = self.conn
            conn.execute(
                "INSERT INTO sessions(session_id, created_at, last_active) VALUES (?, ?, ?)",
                (session_id, now_iso, now_iso),
            )
        return Session(session_id=session_id, created_at=now, last_active=now)

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            conn = self.conn
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            messages = self._load_messages(conn, session_id)
            return Session(
                session_id=row["session_id"],
                messages=messages,
                context_summary=row["context_summary"],
                current_topic=row["current_topic"],
                topic_embedding=json.loads(row["topic_embedding"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                last_active=datetime.fromisoformat(row["last_active"]),
            )

    def get_or_create(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            try:
                session = self.create(session_id)
            except sqlite3.IntegrityError:
                session = self.get(session_id)
                if session is None:
                    raise RuntimeError(f"无法创建或获取会话: {session_id}")
        return session

    def update(self, session: Session) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self.conn
            conn.execute(
                """UPDATE sessions
                   SET context_summary=?, current_topic=?,
                       topic_embedding=?, last_active=?
                   WHERE session_id=?""",
                (
                    session.context_summary,
                    session.current_topic,
                    json.dumps(session.topic_embedding, ensure_ascii=False),
                    now,
                    session.session_id,
                ),
            )

    def delete(self, session_id: str) -> bool:
        with self._lock:
            conn = self.conn
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    # ---- Message CRUD ----

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self.conn
            conn.execute(
                "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (now, session_id),
            )
        return Message(role=role, content=content, timestamp=datetime.fromisoformat(now))

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[Message]:
        with self._lock:
            conn = self.conn
            return self._load_messages(conn, session_id, limit)

    def _load_messages(
        self, conn: sqlite3.Connection, session_id: str, limit: Optional[int] = None
    ) -> list[Message]:
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (session_id,)).fetchall()
        return [
            Message(role=r["role"], content=r["content"],
                    timestamp=datetime.fromisoformat(r["timestamp"]))
            for r in rows
        ]

    # ---- TTL 清理 ----

    def cleanup_expired(self) -> int:
        """删除超过 TTL 的会话，返回删除数量"""
        ttl_hours = settings.session.ttl_hours
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_hours * 3600
        cutoff_str = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        with self._lock:
            conn = self.conn
            expired = conn.execute(
                "SELECT session_id FROM sessions WHERE last_active < ?", (cutoff_str,)
            ).fetchall()
            for row in expired:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (row["session_id"],))
            conn.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff_str,))
            count = len(expired)
            if count:
                logger.info("TTL 清理: %d 个过期会话", count)
            return count
