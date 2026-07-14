"""Query 模块测试共享 fixtures"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from session.store import SessionStore
from session.manager import SessionManager


class MockLLM:
    """Mock LLM 客户端 — 所有 query 模块测试共享。

    支持通过 response 参数编程控制返回值，通过 should_fail 控制失败场景。
    """

    def __init__(self, response="", should_fail=False):
        self.response = response
        self.should_fail = should_fail
        self.calls = []

    async def ainvoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.should_fail:
            raise RuntimeError("LLM timeout")
        return SimpleNamespace(content=self.response)


@pytest.fixture
def session_manager():
    """创建临时数据库的 SessionManager"""
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = SessionStore(db_path=db_path)
    mgr = SessionManager(store=store)
    yield mgr
    store.close()
