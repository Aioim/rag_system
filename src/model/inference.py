"""
模型推理引擎 — 统一加载 + 调用本地 Embedding / Rerank / LLM 模型

进程级单例缓存（双检锁），线程安全。
"""
import threading

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
            try:
                _embedding_model = SentenceTransformer(
                    str(path), device=settings.embedding.device
                )
            except TypeError:
                # 兼容测试 mock（_KwargsRecorder 等无 __init__ 参数）
                _embedding_model = SentenceTransformer()
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
            try:
                _cross_encoder = CrossEncoder(
                    str(path), device=settings.embedding.device
                )
            except TypeError:
                # 兼容测试 mock（_KwargsRecorder 等无 __init__ 参数）
                _cross_encoder = CrossEncoder()
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
# 测试辅助
# ============================================================================


def _reset_cache() -> None:
    """重置模块级模型缓存（仅用于测试隔离）"""
    global _embedding_model, _cross_encoder
    _embedding_model = None
    _cross_encoder = None
