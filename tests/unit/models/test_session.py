"""Session 测试"""
from models.session import Message, Session


class TestMessage:
    def test_construction(self):
        msg = Message(role="user", content="什么是RAG？")
        assert msg.role == "user"
        assert msg.timestamp is not None

    def test_assistant(self):
        msg = Message(role="assistant", content="RAG是检索增强生成")
        assert msg.role == "assistant"


class TestSession:
    def test_minimal(self):
        s = Session(session_id="s1")
        assert s.messages == []
        assert s.context_summary is None
        assert s.created_at is not None
        assert s.last_active is not None

    def test_add_message(self):
        s = Session(session_id="s1")
        s.messages.append(Message(role="user", content="hello"))
        s.messages.append(Message(role="assistant", content="hi"))
        assert len(s.messages) == 2

    def test_topic_tracking(self):
        s = Session(session_id="s1", current_topic="RAG基础",
                     topic_embedding=[0.1, 0.2, 0.3])
        assert s.current_topic == "RAG基础"
        assert s.topic_embedding == [0.1, 0.2, 0.3]
