"""LocalLLMAdapter — 将 LocalLLM 适配为 LLMProtocol，无缝接入现有生成管线

使用示例：
    from model.inference import LocalLLM
    from model.llm_adapter import LocalLLMAdapter

    llm = LocalLLM("models/Qwen3-0.6B-Q4_K_M.gguf")
    adapter = LocalLLMAdapter(llm)

    # 直接替换云端 LLM，注入 Pipeline
    from core import get_rag_pipeline
    pipeline = get_rag_pipeline(adapter, session_manager)
    ctx = await pipeline.run("什么是RAG？")
"""

from typing import Any


class _FakeMessage:
    """模拟 LangChain BaseMessage，提供 .content 属性"""

    def __init__(self, content: str):
        self.content = content

    def __repr__(self) -> str:
        return f"_FakeMessage(content={self.content[:50]!r}...)"


class LocalLLMAdapter:
    """将 LocalLLM 适配为 LLMProtocol

    符合 models.llm.LLMProtocol 协议：
    - async ainvoke(prompt, **kwargs) -> object with .content

    可直接注入 GenerationLayer / RAGPipeline / FactChecker 等需要 LLMProtocol 的组件。
    """

    def __init__(self, llm: "LocalLLM", default_temperature: float = 0.0):
        """
        Args:
            llm: LocalLLM 实例（可以是已加载或未加载的）
            default_temperature: 默认温度（优先级低于 kwargs 显式传入的 temperature）
        """
        self._llm = llm
        self._default_temperature = default_temperature

    @property
    def llm(self) -> "LocalLLM":
        return self._llm

    async def ainvoke(self, prompt: str, **kwargs: Any) -> _FakeMessage:
        """异步 LLM 调用，返回 _FakeMessage（有 .content 属性）

        Args:
            prompt: 输入提示
            **kwargs: 透传给 LocalLLM.ainvoke()
                - temperature: 温度（默认 self._default_temperature）
                - max_tokens: 最大 token 数

        Returns:
            _FakeMessage(content=generated_text)
        """
        temperature = kwargs.pop("temperature", self._default_temperature)
        content = await self._llm.ainvoke(prompt, temperature=temperature, **kwargs)
        return _FakeMessage(content)
