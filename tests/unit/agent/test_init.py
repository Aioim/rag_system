"""agent/__init__.py 单例工厂测试"""
from unittest.mock import MagicMock

from agent import get_react_agent, reset_react_agent


class TestSingleton:
    def test_get_returns_same_instance(self):
        llm = MagicMock()
        search = MagicMock()
        web = MagicMock()

        reset_react_agent()
        a1 = get_react_agent(llm, search, web)
        a2 = get_react_agent(llm, search, web)
        assert a1 is a2

    def test_reset_creates_new_instance(self):
        llm = MagicMock()
        search = MagicMock()
        web = MagicMock()

        reset_react_agent()
        a1 = get_react_agent(llm, search, web)
        reset_react_agent()
        a2 = get_react_agent(llm, search, web)
        assert a1 is not a2
