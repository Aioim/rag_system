"""BaseRewriter 测试"""
import pytest
from query.rewriters.base import BaseRewriter


def test_cannot_instantiate_abstract():
    """不能直接实例化抽象基类"""
    with pytest.raises(TypeError):
        BaseRewriter()


def test_concrete_subclass_must_implement_rewrite():
    """子类必须实现 rewrite 方法"""

    class BadRewriter(BaseRewriter):
        pass

    with pytest.raises(TypeError):
        BadRewriter()


def test_valid_subclass_instantiates():
    """正确实现 rewrite 的子类可以实例化"""

    class GoodRewriter(BaseRewriter):
        async def rewrite(self, query: str) -> list[str]:
            return [query]

    r = GoodRewriter()
    assert isinstance(r, BaseRewriter)
