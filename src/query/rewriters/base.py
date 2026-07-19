"""查询改写器基类 — 提供模板方法消除子类重复代码"""
from logger import logger
from models.llm import LLMProtocol


class BaseRewriter:
    """查询改写器基类

    提供模板方法 rewrite()：调用 _build_prompt → LLM.ainvoke → _parse_response。
    子类只需覆写 _build_prompt()，可选覆写 _parse_response()。
    也可直接覆写 rewrite() 完全自定义行为。
    LLM 通过构造函数注入，需满足 LLMProtocol。
    """

    def __init__(self, llm: LLMProtocol | None = None, temperature: float | None = None) -> None:
        self._llm = llm
        self._temperature = temperature

    async def rewrite(self, query: str) -> list[str]:
        """模板方法：构建 prompt → 调用 LLM.ainvoke → 解析响应"""
        if self._llm is None:
            raise ValueError(
                f"{type(self).__name__} 未收到 LLM 实例，请通过构造函数传入"
            )
        prompt = self._build_prompt(query)
        try:
            kwargs: dict[str, float] = {}
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = await self._llm.ainvoke(prompt, **kwargs)
            return self._parse_response(response.content)
        except Exception:
            logger.warning(
                "%s LLM 调用失败，返回空列表", type(self).__name__, exc_info=True
            )
            return []

    def _build_prompt(self, query: str) -> str:
        """构建发送给 LLM 的 prompt（子类必须覆写，否则运行时抛出 NotImplementedError）"""
        raise NotImplementedError(
            f"{type(self).__name__} 必须覆写 _build_prompt() 或 rewrite()"
        )

    def _parse_response(self, response: str) -> list[str]:
        """解析 LLM 响应为查询列表（子类可选覆写）"""
        result = response.strip()
        return [result] if result else []
