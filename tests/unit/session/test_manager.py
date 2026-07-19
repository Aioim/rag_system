"""SessionManager 测试"""
import tempfile
from pathlib import Path

import pytest

from session.store import SessionStore
from session.manager import SessionManager


@pytest.fixture
def manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestSessionLifecycle:
    def test_get_or_create_new(self, manager):
        s = manager.get_or_create()
        assert s.session_id
        assert s.messages == []

    def test_get_or_create_existing(self, manager):
        s1 = manager.get_or_create("fixed-id")
        s2 = manager.get_or_create("fixed-id")
        assert s1.session_id == s2.session_id

    def test_delete(self, manager):
        s = manager.get_or_create("to-delete")
        manager.add_message("to-delete", "user", "hi")
        assert manager.delete("to-delete") is True
        assert manager.get("to-delete") is None


class TestMessages:
    def test_add_message(self, manager):
        manager.get_or_create("s1")
        msg = manager.add_message("s1", "user", "什么是RAG？")
        assert msg.role == "user"
        assert msg.content == "什么是RAG？"

        session = manager.get("s1")
        assert len(session.messages) == 1

    def test_add_message_nonexistent_session(self, manager):
        with pytest.raises(ValueError, match="不存在"):
            manager.add_message("no-such-session", "user", "hi")

    def test_add_message_invalid_role(self, manager):
        manager.get_or_create("s-role")
        with pytest.raises(ValueError, match="role"):
            manager.add_message("s-role", "bot", "hi")


class TestTopicDetection:
    def test_topic_switch(self, manager):
        manager.get_or_create("s-topic")

        # 第一条消息带 embedding
        manager.add_message("s-topic", "user", "Python 怎么学？",
                            embedding=[1.0, 0.0, 0.0])

        # 完全不同的话题
        manager.add_message("s-topic", "user", "晚饭吃什么？",
                            embedding=[0.0, 0.0, 1.0])

        session = manager.get("s-topic")
        # 话题切换后只保留最新 2 条
        assert len(session.messages) <= 4

    def test_no_switch_same_topic(self, manager):
        manager.get_or_create("s-same")
        manager.add_message("s-same", "user", "Python 基础",
                            embedding=[1.0, 0.1, 0.0])
        manager.add_message("s-same", "user", "Python 进阶",
                            embedding=[1.0, 0.0, 0.1])
        session = manager.get("s-same")
        assert session.context_summary is None  # 未触发压缩

    def test_dimension_mismatch_does_not_raise(self, manager):
        """维度不匹配应跳过检测而非回滚事务"""
        manager.get_or_create("s-dim")
        manager.add_message("s-dim", "user", "A", embedding=[1.0, 0.0, 0.0])
        msg = manager.add_message("s-dim", "user", "B", embedding=[1.0, 0.0])
        assert msg.content == "B"
        session = manager.get("s-dim")
        assert len(session.messages) == 2  # 消息正常写入


class TestContextCompression:
    def test_compress_long_messages(self, manager):
        from config import settings
        original = settings.session.max_context_tokens
        settings.session.max_context_tokens = 50  # 很小的限制

        try:
            manager.get_or_create("s-compress")
            long_text = "这是一个很长的消息，" * 20  # ~140 chars
            manager.add_message("s-compress", "user", long_text)
            manager.add_message("s-compress", "assistant", long_text)
            manager.add_message("s-compress", "user", "短消息")

            session = manager.get("s-compress")
            # 应触发压缩，长消息被移除
            assert session.context_summary is not None
        finally:
            settings.session.max_context_tokens = original

    def test_single_overlong_message_not_compressed(self, manager):
        """单条超长消息无早期消息可删，压缩静默跳过但不崩溃"""
        from config import settings
        original = settings.session.max_context_tokens
        settings.session.max_context_tokens = 10

        try:
            manager.get_or_create("s-single")
            manager.add_message("s-single", "user", "超长消息" * 50)
            session = manager.get("s-single")
            assert len(session.messages) == 1  # 消息保留
            assert session.context_summary is None  # 未生成摘要
        finally:
            settings.session.max_context_tokens = original


class TestGetContext:
    def test_get_context(self, manager):
        manager.get_or_create("s-ctx")
        manager.add_message("s-ctx", "user", "Q1")
        manager.add_message("s-ctx", "assistant", "A1")
        manager.add_message("s-ctx", "user", "Q2")

        ctx = manager.get_context("s-ctx", max_tokens=10)
        assert len(ctx["messages"]) <= 3
        assert ctx["summary"] is None
        assert ctx["topic"] is None

    def test_get_context_nonexistent(self, manager):
        ctx = manager.get_context("no-session")
        assert ctx["messages"] == []

    def test_get_context_max_tokens_zero(self, manager):
        """max_tokens=0 不应回退到默认值，仅保留最新一条"""
        manager.get_or_create("s-zero")
        manager.add_message("s-zero", "user", "Q1")
        manager.add_message("s-zero", "assistant", "A1")

        ctx = manager.get_context("s-zero", max_tokens=0)
        assert len(ctx["messages"]) == 1
        assert ctx["messages"][0].content == "A1"


class TestLifecycle:
    def test_close(self, manager):
        manager.get_or_create("s-close")
        manager.close()
        manager.close()  # 幂等


class TestCompressionPreservesHistory:
    """审查 H9：压缩/话题切换应归档旧消息而非物理删除"""

    def test_compress_archives_instead_of_deleting(self, manager):
        from config import settings
        original = settings.session.max_context_tokens
        settings.session.max_context_tokens = 50
        try:
            manager.get_or_create("s-keep")
            long_text = "这是一个很长的消息，" * 20
            manager.add_message("s-keep", "user", long_text)
            manager.add_message("s-keep", "assistant", long_text)
            manager.add_message("s-keep", "user", "短消息")

            session = manager.get("s-keep")
            assert session.context_summary is not None  # 前置：压缩已触发

            active = manager.store.get_messages("s-keep")
            full = manager.store.get_messages("s-keep", include_archived=True)
            assert len(full) == 3, "被压缩的消息应仍保留在库中"
            assert len(active) < len(full)
        finally:
            settings.session.max_context_tokens = original

    def test_topic_switch_archives_old_messages(self, manager):
        manager.get_or_create("s-swarch")
        for i in range(5):
            manager.add_message(
                "s-swarch", "user", f"python 话题 {i}", embedding=[1.0, 0.0, 0.0]
            )

        manager.add_message(
            "s-swarch", "user", "晚饭吃什么", embedding=[0.0, 0.0, 1.0]
        )

        active = manager.store.get_messages("s-swarch")
        full = manager.store.get_messages("s-swarch", include_archived=True)
        assert len(full) == 6, "话题切换应归档而非物理删除"
        assert len(active) == 5  # 保留 4 条 + 新消息 1 条
