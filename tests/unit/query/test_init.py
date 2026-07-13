"""Query 模块 __init__.py 测试"""
import tempfile
from pathlib import Path

import pytest
from session.store import SessionStore
from session.manager import SessionManager
import query
from query.layer import QueryUnderstandingLayer


class FakeLLM:
    async def generate(self, prompt, **kwargs):
        return "response"


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestModuleExports:
    def test_query_layer_exported(self):
        assert hasattr(query, "QueryUnderstandingLayer")

    def test_get_query_layer_exported(self):
        assert hasattr(query, "get_query_layer")
        assert callable(query.get_query_layer)

    def test_reset_query_layer_exported(self):
        assert hasattr(query, "reset_query_layer")
        assert callable(query.reset_query_layer)

    def test_intent_result_exported(self):
        from query.intent_classifier import IntentResult
        assert hasattr(query, "IntentResult")


class TestGetQueryLayer:
    def test_returns_singleton(self, session_manager):
        query.reset_query_layer()
        llm = FakeLLM()
        layer1 = query.get_query_layer(llm, session_manager)
        layer2 = query.get_query_layer(llm, session_manager)
        assert layer1 is layer2

    def test_reset_creates_new_instance(self, session_manager):
        query.reset_query_layer()
        llm = FakeLLM()
        layer1 = query.get_query_layer(llm, session_manager)
        query.reset_query_layer()
        layer2 = query.get_query_layer(llm, session_manager)
        assert layer1 is not layer2
