"""SessionStore 测试"""
import tempfile
from pathlib import Path

import pytest

from session.store import SessionStore


@pytest.fixture
def store():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    s = SessionStore(db_path=db_path)
    yield s
    s.close()


class TestSessionCRUD:
    def test_create_and_get(self, store):
        session = store.create("s1")
        assert session.session_id == "s1"
        assert session.messages == []

        fetched = store.get("s1")
        assert fetched is not None
        assert fetched.session_id == "s1"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_get_or_create(self, store):
        s1 = store.get_or_create("new-session")
        assert s1.session_id == "new-session"
        s2 = store.get_or_create("new-session")
        assert s2.session_id == "new-session"

    def test_delete(self, store):
        store.create("s-del")
        assert store.delete("s-del") is True
        assert store.get("s-del") is None
        assert store.delete("nonexistent") is False

    def test_update(self, store):
        session = store.create("s-update")
        session.current_topic = "RAG"
        session.topic_embedding = [0.1, 0.2]
        session.context_summary = "摘要内容"
        store.update(session)

        fetched = store.get("s-update")
        assert fetched.current_topic == "RAG"
        assert fetched.topic_embedding == [0.1, 0.2]
        assert fetched.context_summary == "摘要内容"


class TestMessageCRUD:
    def test_add_and_get_messages(self, store):
        store.create("s-msg")
        store.add_message("s-msg", "user", "hello")
        store.add_message("s-msg", "assistant", "hi there")
        store.add_message("s-msg", "user", "question?")

        msgs = store.get_messages("s-msg")
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert msgs[0].content == "hello"
        assert msgs[2].role == "user"

    def test_add_message_returns_row_id(self, store):
        store.create("s-rowid")
        msg = store.add_message("s-rowid", "user", "hello")
        assert msg.row_id > 0  # 已持久化，row_id 必须为 DB 生成的 ID

        fetched = store.get_messages("s-rowid")
        assert fetched[0].row_id == msg.row_id

    def test_get_messages_nonexistent_session(self, store):
        with pytest.raises(ValueError, match="不存在"):
            store.get_messages("no-such-session")

    def test_message_limit(self, store):
        store.create("s-limit")
        for i in range(10):
            store.add_message("s-limit", "user", f"msg{i}")

        msgs = store.get_messages("s-limit", limit=3)
        assert len(msgs) == 3
        assert msgs[0].content == "msg0"


class TestTTLCleanup:
    def test_cleanup_expired(self, store):
        from config import settings
        original = settings.session.ttl_hours
        settings.session.ttl_hours = -1  # 立即过期

        try:
            store.create("s-old")
            count = store.cleanup_expired()
            assert count == 1
            assert store.get("s-old") is None
        finally:
            settings.session.ttl_hours = original


class TestLifecycle:
    def test_context_manager_closes_connection(self):
        db_path = Path(tempfile.mkdtemp()) / "ctx.db"
        with SessionStore(db_path=db_path) as store:
            store.create("s-ctx")
        # close 幂等：重复调用不报错
        store.close()

    def test_close_idempotent(self, store):
        store.close()
        store.close()


class TestArchiveMessagesKeepLast:
    """审查 H9：压缩应软删除（archived 标记），历史保留可溯源"""

    def test_archive_hides_old_but_preserves_in_db(self, store):
        # Arrange
        store.create("s-arch")
        for i in range(5):
            store.add_message("s-arch", "user", f"m{i}")

        # Act
        with store.transaction() as tx:
            tx.archive_messages_keep_last("s-arch", 2)

        # Assert
        active = store.get_messages("s-arch")
        assert [m.content for m in active] == ["m3", "m4"], "默认只见最近 2 条"
        full = store.get_messages("s-arch", include_archived=True)
        assert len(full) == 5, "归档消息应保留在库中可溯源"

    def test_repeated_archive_counts_only_active(self, store):
        """重复归档时 keep_count 只针对活跃消息计算"""
        store.create("s-re")
        for i in range(4):
            store.add_message("s-re", "user", f"m{i}")
        with store.transaction() as tx:
            tx.archive_messages_keep_last("s-re", 2)
        store.add_message("s-re", "user", "m4")

        with store.transaction() as tx:
            tx.archive_messages_keep_last("s-re", 2)

        active = store.get_messages("s-re")
        assert [m.content for m in active] == ["m3", "m4"]
        assert len(store.get_messages("s-re", include_archived=True)) == 5
