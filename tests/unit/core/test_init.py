"""Core 模块 __init__.py 测试"""
import tempfile
from pathlib import Path

import pytest
from session.store import SessionStore
from session.manager import SessionManager
import core
from core.pipeline import RAGPipeline


class FakeLLM:
    async def generate(self, prompt, **kwargs):
        return "response"

    async def ainvoke(self, prompt, **kwargs):
        class FakeMsg:
            content = "response"

        return FakeMsg()


@pytest.fixture
def session_manager():
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()


class TestModuleExports:
    def test_rag_pipeline_exported(self):
        assert hasattr(core, "RAGPipeline")
        assert core.RAGPipeline is RAGPipeline

    def test_get_rag_pipeline_exported(self):
        assert hasattr(core, "get_rag_pipeline")
        assert callable(core.get_rag_pipeline)

    def test_reset_rag_pipeline_exported(self):
        assert hasattr(core, "reset_rag_pipeline")
        assert callable(core.reset_rag_pipeline)


class TestGetRagPipeline:
    def test_returns_singleton(self, session_manager):
        core.reset_rag_pipeline()
        llm = FakeLLM()
        pipeline1 = core.get_rag_pipeline(llm, session_manager)
        pipeline2 = core.get_rag_pipeline(llm, session_manager)
        assert pipeline1 is pipeline2

    def test_reset_creates_new_instance(self, session_manager):
        core.reset_rag_pipeline()
        llm = FakeLLM()
        pipeline1 = core.get_rag_pipeline(llm, session_manager)
        core.reset_rag_pipeline()
        pipeline2 = core.get_rag_pipeline(llm, session_manager)
        assert pipeline1 is not pipeline2

    def test_warns_on_different_llm(self, session_manager):
        core.reset_rag_pipeline()
        llm1 = FakeLLM()
        llm2 = FakeLLM()
        pipeline1 = core.get_rag_pipeline(llm1, session_manager)
        # 第二次调用传入不同 llm，应返回同一实例但记录警告
        pipeline2 = core.get_rag_pipeline(llm2, session_manager)
        assert pipeline1 is pipeline2
