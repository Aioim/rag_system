"""LLM 客户端协议 — 统一各模块的 LLM 参数类型"""

from typing import Any, Protocol


class LLMProtocol(Protocol):
    """LLM 客户端接口协议

    各模块通过此 Protocol 获得一致的 LLM 类型契约，
    避免各处使用 Any 或遗漏类型标注。

    需要满足：
    - async ainvoke(prompt, **kwargs) 方法
    - 返回对象有 .content 属性（如 LangChain 的 BaseMessage）
    """

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any: ...
