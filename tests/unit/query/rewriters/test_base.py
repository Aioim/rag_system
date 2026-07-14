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
