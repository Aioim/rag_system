"""LocalLLMAdapter 测试"""
import pytest

from model.llm_adapter import LocalLLMAdapter, _FakeMessage


class _FakeLocalLLM:
    """模拟 LocalLLM，可直接记录调用参数"""

    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return f"回复: {prompt[:20]}"


class TestLocalLLMAdapter:
    def test_ainvoke_returns_fake_message(self):
        """ainvoke() 返回 _FakeMessage，有 .content 属性"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        result = adapter.ainvoke("你好")
        # 需要 asyncio 运行
        import asyncio
        msg = asyncio.run(result)
        assert isinstance(msg, _FakeMessage)
        assert msg.content == "回复: 你好"

    def test_ainvoke_passes_temperature(self):
        """ainvoke() 透传 temperature"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        import asyncio
        asyncio.run(adapter.ainvoke("测试", temperature=0.5))
        assert fake_llm.calls[0]["temperature"] == 0.5

    def test_default_temperature(self):
        """未指定 temperature 时使用默认值"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm, default_temperature=0.3)
        import asyncio
        asyncio.run(adapter.ainvoke("测试"))
        assert fake_llm.calls[0]["temperature"] == 0.3

    def test_llm_property(self):
        """llm 属性返回注入的 LocalLLM"""
        fake_llm = _FakeLocalLLM()
        adapter = LocalLLMAdapter(fake_llm)
        assert adapter.llm is fake_llm

    def test_fake_message_repr(self):
        """_FakeMessage.__repr__ 包含内容预览"""
        msg = _FakeMessage("这是一条很长的消息内容" * 10)
        assert "..." in repr(msg)
