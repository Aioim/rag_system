"""ReAct Agent 共享测试 fixtures"""
from unittest.mock import MagicMock

import pytest


class ProgrammableLLM:
    """可编程 Mock LLM — 每次 ainvoke 按顺序返回预设响应"""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[tuple[str, float]] = []  # (prompt, temperature)

    async def ainvoke(self, prompt: str, temperature: float = 0.0):
        self.calls.append((prompt, temperature))
        if self.call_count >= len(self.responses):
            return MagicMock(content="THOUGHT: 默认响应\nACTION: FINISH")
        response = self.responses[self.call_count]
        self.call_count += 1
        return MagicMock(content=response)


@pytest.fixture
def agent_config():
    """使用 AgentConfig 默认值；测试可按需覆盖特定字段"""
    from config.settings import AgentConfig
    return AgentConfig()


@pytest.fixture
def programmable_llm():
    """返回工厂函数，避免跨测试状态污染"""
    def _make(responses: list[str]):
        return ProgrammableLLM(responses)
    return _make
