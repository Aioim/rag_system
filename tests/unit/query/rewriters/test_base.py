"""BaseRewriter 测试"""
import pytest
from query.rewriters.base import BaseRewriter


def test_default_build_prompt_raises_not_implemented():
    """未覆写 _build_prompt 或 rewrite() 时，rewrite() 抛出 NotImplementedError"""
    r = BaseRewriter.__new__(BaseRewriter)

    with pytest.raises(NotImplementedError):
        r._build_prompt("test")


def test_subclass_overriding_rewrite_works():
    """直接覆写 rewrite() 的子类可以正常使用"""

    class CustomRewriter(BaseRewriter):
        async def rewrite(self, query: str) -> list[str]:
            return [query + "_modified"]

    r = CustomRewriter.__new__(CustomRewriter)
    assert isinstance(r, BaseRewriter)


def test_subclass_overriding_build_prompt_works():
    """覆写 _build_prompt 的子类使用模板方法"""

    class PromptRewriter(BaseRewriter):
        def _build_prompt(self, query: str) -> str:
            return f"PROMPT: {query}"

    r = PromptRewriter.__new__(PromptRewriter)
    assert isinstance(r, BaseRewriter)
    assert r._build_prompt("hello") == "PROMPT: hello"


@pytest.mark.asyncio
async def test_unimplemented_subclass_raises_in_orchestrator():
    """未覆写 _build_prompt 的子类在编排器中应被感知，原始 query 保留"""
    from query.rewriters import QueryRewriter

    class IncompleteRewriter(BaseRewriter):
        pass  # 忘记覆写 _build_prompt

    orchestrator = QueryRewriter.__new__(QueryRewriter)
    orchestrator._rewriters = [IncompleteRewriter()]

    result = await orchestrator.rewrite("test query")
    assert result == ["test query"]


@pytest.mark.asyncio
async def test_base_rewriter_none_llm_raises():
    """BaseRewriter 在 llm=None 时调用 rewrite() 应抛出 ValueError"""
    r = BaseRewriter.__new__(BaseRewriter)
    r._llm = None

    with pytest.raises(ValueError, match="未收到 LLM 实例"):
        await r.rewrite("test")
