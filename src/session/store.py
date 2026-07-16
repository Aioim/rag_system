"""SQLite 会话存储 — 持久化 + TTL 清理"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Generator

from config import settings
from logger import logger
from models.session import Message, Session


class TransactionScope:
    """事务代理 — 在已持锁+事务内调用 _locked_* 方法，不重复加锁。

    TransactionScope 只在 SessionStore.transaction() 上下文中使用，
    确保所有操作共享同一个锁和同一个 SQLite 事务。
    """

    __slots__ = ("_store",)

    def __init__(self, store: "SessionStore") -> None:
        self._store = store

    def get(self, session_id: str) -> Optional[Session]:
        return self._store._locked_get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        return self._store._locked_get_or_create(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        return self._store._locked_add_message(session_id, role, content)

    def update(self, session: Session) -> None:
        self._store._locked_update(session)

    def delete_messages_keep_last(self, session_id: str, keep_count: int) -> None:
        self._store._locked_delete_messages_keep_last(session_id, keep_count)

    def delete(self, session_id: str) -> bool:
        return self._store._locked_delete(session_id)


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

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def _get_conn(self) -> sqlite3.Connection:
        """惰性连接（调用方已持有 _lock）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """关闭数据库连接（测试清理用）"""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # 事务上下文管理器
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[TransactionScope, None, None]:
        """持锁 + SQLite 事务的原子操作上下文。

        用法::

            with store.transaction() as tx:
                session = tx.get(sid)
                msg = tx.add_message(sid, "user", text)
                tx.update(session)
        """
        with self._lock:
            conn = self._get_conn
            with conn:  # BEGIN … COMMIT / ROLLBACK
                yield TransactionScope(self)

    # ------------------------------------------------------------------
    # 表创建
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock:
            conn = self._get_conn
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

    # ------------------------------------------------------------------
    # Session CRUD（公有方法 = 加锁 → _locked_*）
    # ------------------------------------------------------------------

    def create(self, session_id: str) -> Session:
        with self._lock:
            return self._locked_create(session_id)

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._locked_get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            return self._locked_get_or_create(session_id)

    def update(self, session: Session) -> None:
        with self._lock:
            self._locked_update(session)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            conn = self._get_conn
            with conn:
                return self._locked_delete(session_id)

    # ---- 锁内实现 ----

    def _locked_create(self, session_id: str) -> Session:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        self._get_conn.execute(
            "INSERT INTO sessions(session_id, created_at, last_active) VALUES (?, ?, ?)",
            (session_id, now_iso, now_iso),
        )
        return Session(session_id=session_id, created_at=now, last_active=now)

    def _locked_get(self, session_id: str) -> Optional[Session]:
        conn = self._get_conn
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        messages = self._locked_load_messages(session_id)
        return Session(
            session_id=row["session_id"],
            messages=messages,
            context_summary=row["context_summary"],
            current_topic=row["current_topic"],
            topic_embedding=json.loads(row["topic_embedding"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
        )

    def _locked_get_or_create(self, session_id: str) -> Session:
        session = self._locked_get(session_id)
        if session is None:
            try:
                session = self._locked_create(session_id)
            except sqlite3.IntegrityError:
                session = self._locked_get(session_id)
                if session is None:
                    raise RuntimeError(f"无法创建或获取会话: {session_id}")
        return session

    def _locked_update(self, session: Session) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn
        cur = conn.execute(
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
        if cur.rowcount == 0:
            logger.warning(
                "update() 未影响任何行: session=%s 可能已被删除",
                session.session_id,
            )

    def _locked_delete(self, session_id: str) -> bool:
        conn = self._get_conn
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Message CRUD（公有方法 = 加锁 → _locked_*，多语句 DML 用 with conn:）
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        with self._lock:
            conn = self._get_conn
            with conn:
                return self._locked_add_message(session_id, role, content)

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> list[Message]:
        with self._lock:
            return self._locked_load_messages(session_id, limit)

    # ---- 锁内实现 ----

    def _locked_add_message(
        self, session_id: str, role: str, content: str
    ) -> Message:
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        conn = self._get_conn
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso),
        )
        conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (now_iso, session_id),
        )
        return Message(role=role, content=content, timestamp=now)

    def _locked_delete_messages_keep_last(
        self, session_id: str, keep_count: int
    ) -> None:
        """只保留最近 keep_count 条消息，删除更早的消息。

        调用方必须持有 _lock 并负责事务管理。
        """
        if keep_count <= 0:
            return
        conn = self._get_conn
        row = conn.execute(
            "SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1 OFFSET ?",
            (session_id, keep_count - 1),
        ).fetchone()
        if row is not None:
            conn.execute(
                "DELETE FROM messages WHERE session_id=? AND id < ?",
                (session_id, row["id"]),
            )

    def _locked_load_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> list[Message]:
        conn = self._get_conn
        sql = "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC"
        if limit is not None:
            # SQLite 不支持 LIMIT ? 参数化；int() 确保无注入风险
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql, (session_id,)).fetchall()
        return [
            Message(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # TTL 清理
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """删除超过 TTL 的会话，返回删除数量"""
        ttl_hours = settings.session.ttl_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        cutoff_str = cutoff.isoformat()
        with self._lock:
            conn = self._get_conn
            # 先统计过期会话数
            count = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE last_active < ?",
                (cutoff_str,),
            ).fetchone()[0]
            if count:
                with conn:
                    # 用子查询避免 IN (?, ...) 参数超 SQLite 999 限制
                    conn.execute(
                        "DELETE FROM messages WHERE session_id IN "
                        "(SELECT session_id FROM sessions WHERE last_active < ?)",
                        (cutoff_str,),
                    )
                    conn.execute(
                        "DELETE FROM sessions WHERE last_active < ?",
                        (cutoff_str,),
                    )
                logger.info("TTL 清理: %d 个过期会话", count)
            return count
