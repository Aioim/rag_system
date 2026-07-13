"""SessionStore 测试"""
import tempfile
from pathlib import Path

import pytest

from session.store import SessionStore


@pytest.fixture
def store():
    import tempfile
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
