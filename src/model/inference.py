"""
模型推理引擎 — 统一加载 + 调用本地 Embedding / Rerank / LLM 模型

进程级单例缓存（双检锁），线程安全。
"""
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================================
# 模块级缓存
# ============================================================================

_embedding_model: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None
_embedding_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()

_GENERATE_NOT_IMPLEMENTED_MSG = (
    "generate() 尚未实现。推荐方案：llama-cpp-python + GGUF 量化模型。\n"
    "  - CPU 友好，内存占用低（INT4 量化后约 4-8 GB）\n"
    "  - 安装: pip install llama-cpp-python\n"
    "  - 使用: from llama_cpp import Llama; llm = Llama(model_path=\"model.gguf\")\n"
    "  - 项目当前 LLM 生成走云端 DeepSeek API，本地推理作为后续迭代方向。"
)


# ============================================================================
# 模型加载（内部）
# ============================================================================


def _get_embedding_model() -> SentenceTransformer:
    """获取 SentenceTransformer 实例（懒加载 + 双检锁）"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            from config import settings
            from model import models

            path = models.get_path("embedding")
            if path is None:
                raise RuntimeError(
                    "Embedding 模型未下载，请先执行 "
                    "`from model import models; models.download('embedding')`"
                )
            _embedding_model = SentenceTransformer(
                str(path), device=settings.embedding.device
            )
    return _embedding_model


def _get_cross_encoder() -> CrossEncoder:
    """获取 CrossEncoder 实例（懒加载 + 双检锁）"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is None:
            from config import settings
            from model import models

            path = models.get_path("rerank")
            if path is None:
                raise RuntimeError(
                    "Rerank 模型未下载，请先执行 "
                    "`from model import models; models.download('rerank')`"
                )
            _cross_encoder = CrossEncoder(
                str(path), device=settings.embedding.device
            )
    return _cross_encoder


# ============================================================================
# 公共推理接口
# ============================================================================


def encode(texts: str | list[str], **kwargs) -> np.ndarray:
    """对文本进行 embedding 编码。

    Args:
        texts: 单条文本或文本列表
        **kwargs: 透传给 SentenceTransformer.encode()

    Returns:
        np.ndarray — 单条返回 1D，多条返回 2D
    """
    model = _get_embedding_model()
    return model.encode(texts, **kwargs)


def rerank(query: str, documents: list[str], **kwargs) -> list[dict]:
    """对查询与候选文档进行相关性排序。

    Args:
        query: 查询文本
        documents: 候选文档文本列表
        **kwargs: 透传给 CrossEncoder.rank()

    Returns:
        list[dict] — [{"corpus_id": int, "score": float}, ...]
    """
    model = _get_cross_encoder()
    return model.rank(query, documents, **kwargs)


def generate(prompt: str, **kwargs) -> str:
    """LLM 文本生成（预留接口，当前未实现）。

    推荐方案：llama-cpp-python + GGUF 量化模型。
    项目当前 LLM 生成走云端 DeepSeek API，本地推理作为后续迭代方向。

    Raises:
        NotImplementedError: 始终抛出，消息体包含方案说明。
    """
    raise NotImplementedError(_GENERATE_NOT_IMPLEMENTED_MSG)


# ============================================================================
# 本地 LLM 推理引擎
# ============================================================================

class LocalLLM:
    """本地 LLM 推理引擎（llama-cpp-python + GGUF 量化模型）

    特性：
    - 懒加载：首次 __call__ 时才初始化 Llama 实例
    - 异步安全：ainvoke() 通过 loop.run_in_executor 包装，不阻塞事件循环
    - 加载线程安全：模型加载使用双检锁（Llama 实例本身非线程安全，
      并发调用 __call__/stream 需由调用方保证串行化）

    使用示例：
        llm = LocalLLM("models/Qwen3-0.6B-Q4_K_M.gguf")
        output = llm("你好，请介绍一下自己", max_tokens=512)
        # 流式
        for token in llm.stream("你好"):
            print(token, end="")
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llm: Any = None  # llama_cpp.Llama 实例（懒加载）
        self._instance_lock = threading.Lock()

    # ---- 属性 ---------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ---- 生命周期 -----------------------------------------------------------

    def load(self) -> None:
        """显式加载模型（通常无需调用，__call__ 会自动懒加载）"""
        if self._llm is not None:
            return
        with self._instance_lock:
            if self._llm is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
                )
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"GGUF 模型文件不存在: {self._model_path}\n"
                    f"请先下载模型: from model import models; "
                    f"models.download('{self._model_path.parent.name}')"
                )
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                n_gpu_layers=self._n_gpu_layers,
                verbose=self._verbose,
            )

    def unload(self) -> None:
        """释放模型资源"""
        with self._instance_lock:
            self._llm = None

    # ---- 推理 ---------------------------------------------------------------

    def __call__(self, prompt: str, **kwargs) -> str:
        """同步文本生成

        Args:
            prompt: 输入文本
            **kwargs: 透传给 Llama.create_completion()
                - max_tokens (int, default 512)
                - temperature (float, default 0.7)
                - top_p (float, default 0.95)
                - stop (list[str], default [])

        Returns:
            生成的文本字符串
        """
        self.load()
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        stop = kwargs.pop("stop", [])
        result = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        )
        return result["choices"][0]["text"]

    def stream(self, prompt: str, **kwargs) -> "Iterator[str]":
        """流式文本生成

        Yields:
            增量生成的文本片段（token 级别）
        """
        self.load()
        max_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        stop = kwargs.pop("stop", [])
        stream = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            content = chunk["choices"][0].get("text", "")
            if content:
                yield content

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """异步文本生成（通过 run_in_executor 包装同步调用）"""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self(prompt, **kwargs))


# ============================================================================
# 测试辅助
# ============================================================================


def _reset_cache() -> None:
    """重置模块级模型缓存（仅用于测试隔离）"""
    global _embedding_model, _cross_encoder
    _embedding_model = None
    _cross_encoder = None
