"""SQLite 会话存储 — 持久化 + TTL 清理"""

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from config import settings
from logger import logger
from models.session import Message, Session

# TTL 批量清理每批删除的会话数（< SQLite 999 参数上限）
_CLEANUP_BATCH_SIZE = 500


class TransactionScope:
    """事务代理 — 在已持锁+事务内调用 _locked_* 方法，不重复加锁。

    TransactionScope 只在 SessionStore.transaction() 上下文中使用，
    确保所有操作共享同一个锁和同一个 SQLite 事务。
    """

    __slots__ = ("_store",)

    def __init__(self, store: "SessionStore") -> None:
        self._store = store

    def get(self, session_id: str) -> Session | None:
        return self._store._locked_get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        return self._store._locked_get_or_create(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        return self._store._locked_add_message(session_id, role, content)

    def update(self, session: Session) -> None:
        self._store._locked_update(session)

    def archive_messages_keep_last(self, session_id: str, keep_count: int) -> None:
        self._store._locked_archive_messages_keep_last(session_id, keep_count)

    def delete(self, session_id: str) -> bool:
        return self._store._locked_delete(session_id)


class SessionStore:
    """SQLite 会话持久化存储（线程安全）

    注意：`threading.Lock` 只保证进程内互斥；多进程场景依赖 SQLite
    文件锁与 `_locked_get_or_create` 的 INSERT OR IGNORE 冲突处理。
    """

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = settings.session.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
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
        """关闭数据库连接（幂等；关闭前执行 WAL checkpoint 收缩日志）"""
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as e:
                    logger.debug("WAL checkpoint 失败（忽略）: %s", e)
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            # __init__ 中途失败时 _lock 可能尚未赋值
            if hasattr(self, "_lock"):
                self.close()
        except Exception:
            # 解释器关闭阶段部分资源可能已不可用，忽略清理失败
            pass

    # ------------------------------------------------------------------
    # 事务上下文管理器
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[TransactionScope, None, None]:
        """持锁 + SQLite 事务的原子操作上下文。

        警告：事务块内必须通过 ``tx.*`` 操作数据；直接调用 SessionStore
        的公共方法（get/add_message 等）会重复获取 _lock 导致死锁。

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
                    archived INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
            """)
            # 旧库迁移：补充 archived 软删除标记列
            cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            if "archived" not in cols:
                conn.execute(
                    "ALTER TABLE messages"
                    " ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                )

    # ------------------------------------------------------------------
    # Session CRUD（公有方法 = 加锁 → _locked_*）
    # ------------------------------------------------------------------

    def create(self, session_id: str) -> Session:
        with self._lock:
            return self._locked_create(session_id)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._locked_get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            return self._locked_get_or_create(session_id)

    def update(self, session: Session) -> None:
        with self._lock:
            self._locked_update(session)

    def delete(self, session_id: str) -> bool:
        # 统一走 transaction()，全库只有一处开启锁+事务的入口
        with self.transaction() as tx:
            return tx.delete(session_id)

    # ---- 锁内实现 ----

    def _locked_create(self, session_id: str) -> Session:
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        self._get_conn.execute(
            "INSERT INTO sessions(session_id, created_at, last_active) VALUES (?, ?, ?)",
            (session_id, now_iso, now_iso),
        )
        return Session(session_id=session_id, created_at=now, last_active=now)

    def _locked_get(self, session_id: str) -> Session | None:
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
            topic_embedding=json.loads(row["topic_embedding"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
        )

    def _locked_get_or_create(self, session_id: str) -> Session:
        session = self._locked_get(session_id)
        if session is not None:
            return session
        # INSERT OR IGNORE 避免多进程竞态下的 IntegrityError 控制流
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        cur = self._get_conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at, last_active)"
            " VALUES (?, ?, ?)",
            (session_id, now_iso, now_iso),
        )
        if cur.rowcount > 0:
            return Session(session_id=session_id, created_at=now, last_active=now)
        # 其他进程抢先插入，重新读取
        session = self._locked_get(session_id)
        if session is None:
            raise RuntimeError(f"无法创建或获取会话: {session_id}")
        return session

    def _locked_update(self, session: Session) -> None:
        now = datetime.now(UTC).isoformat()
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
    # Message CRUD（公有方法统一经 transaction() 加锁+开事务 → _locked_*）
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        with self.transaction() as tx:
            return tx.add_message(session_id, role, content)

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
        include_archived: bool = False,
    ) -> list[Message]:
        """读取会话消息列表（默认只含活跃消息；include_archived 含归档历史）。

        Raises:
            ValueError: 会话不存在（区别于"会话存在但无消息"的空列表）
        """
        with self._lock:
            row = self._get_conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"会话不存在: {session_id}")
            return self._locked_load_messages(
                session_id, limit, include_archived=include_archived
            )

    # ---- 锁内实现 ----

    def _locked_add_message(
        self, session_id: str, role: str, content: str
    ) -> Message:
        """INSERT 消息 + 刷新 last_active。

        调用方必须持有 _lock 并负责事务管理——本方法不自行开启事务，
        由 TransactionScope 或公共 add_message 的 ``with conn:`` 提供。
        """
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        conn = self._get_conn
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now_iso),
        )
        conn.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (now_iso, session_id),
        )
        return Message(
            role=role, content=content, timestamp=now, row_id=cur.lastrowid or 0
        )

    def _locked_archive_messages_keep_last(
        self, session_id: str, keep_count: int
    ) -> None:
        """只保留最近 keep_count 条活跃消息，更早的标记 archived（软删除）。

        物理数据保留在库中，可通过 get_messages(include_archived=True)
        溯源。调用方必须持有 _lock 并负责事务管理。
        """
        if keep_count <= 0:
            return
        conn = self._get_conn
        row = conn.execute(
            "SELECT id FROM messages"
            " WHERE session_id=? AND archived=0"
            " ORDER BY id DESC LIMIT 1 OFFSET ?",
            (session_id, keep_count - 1),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE messages SET archived=1"
                " WHERE session_id=? AND archived=0 AND id < ?",
                (session_id, row["id"]),
            )

    def _locked_load_messages(
        self,
        session_id: str,
        limit: int | None = None,
        include_archived: bool = False,
    ) -> list[Message]:
        conn = self._get_conn
        sql = "SELECT id, role, content, timestamp FROM messages WHERE session_id = ?"
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY id ASC"
        params: list[str | int] = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [
            Message(
                row_id=r["id"],
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
        """删除超过 TTL 的会话，返回删除数量。

        分批处理（每批 _CLEANUP_BATCH_SIZE 个会话），限制单个事务的
        删除行数，并在批与批之间短暂释放锁，缓解大量过期会话时对其他
        操作的阻塞。
        """
        ttl_hours = settings.session.ttl_hours
        cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
        cutoff_str = cutoff.isoformat()
        total = 0
        while True:
            with self._lock:
                conn = self._get_conn
                batch = [
                    r[0]
                    for r in conn.execute(
                        "SELECT session_id FROM sessions WHERE last_active < ? LIMIT ?",
                        (cutoff_str, _CLEANUP_BATCH_SIZE),
                    ).fetchall()
                ]
                if not batch:
                    break
                # 占位符仅由 "?" 组成，参数全部走绑定，无注入风险
                placeholders = ",".join("?" for _ in batch)
                with conn:
                    conn.execute(
                        f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                        batch,
                    )
                    conn.execute(
                        f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                        batch,
                    )
                total += len(batch)
        if total:
            logger.info("TTL 清理: %d 个过期会话", total)
        return total
