"""查询改写器抽象基类"""
from abc import ABC, abstractmethod


class BaseRewriter(ABC):
    """查询改写器基类

    所有改写器需实现 rewrite 方法，返回改写后的查询列表（可为空）。
    LLM 通过构造函数注入，需有 async generate(prompt, **kwargs) -> str 方法。
    """

    @abstractmethod
    async def rewrite(self, query: str) -> list[str]:
        """返回改写后的查询列表（可为空）"""
        ...
