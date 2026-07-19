"""RAG 核心 Pipeline — 查询理解 → 检索 → 兜底 → 生成 → 会话记录"""
import threading
from typing import Any

from core.fallback import FallbackHandler
from core.pipeline import RAGPipeline
from logger import logger
from session.manager import SessionManager

# 全局单例
_rag_pipeline: RAGPipeline | None = None
_lock = threading.Lock()
_init_llm_id: int | None = None
_init_sm_id: int | None = None


def get_rag_pipeline(llm: Any, session_manager: SessionManager) -> RAGPipeline:
    """获取 RAG Pipeline 全局单例

    首次调用时用传入的 llm/session_manager 初始化单例。
    后续调用若传入不同对象，会记录警告但仍返回已缓存的实例。

    Args:
        llm: LLM 实例（需有 async generate / ainvoke 方法）
        session_manager: 会话管理器实例

    Returns:
        RAGPipeline 全局单例
    """
    global _rag_pipeline, _init_llm_id, _init_sm_id

    # 快速路径：已初始化，无锁检查
    if _rag_pipeline is not None:
        if id(llm) != _init_llm_id or id(session_manager) != _init_sm_id:
            logger.warning(
                "get_rag_pipeline 已初始化，忽略不同的 llm/session_manager 参数"
            )
        return _rag_pipeline

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _rag_pipeline is None:
            _rag_pipeline = RAGPipeline(llm, session_manager)
            _init_llm_id = id(llm)
            _init_sm_id = id(session_manager)
        return _rag_pipeline


def reset_rag_pipeline() -> None:
    """重置全局单例（测试用）"""
    global _rag_pipeline, _init_llm_id, _init_sm_id
    with _lock:
        _rag_pipeline = None
        _init_llm_id = None
        _init_sm_id = None


__all__ = [
    "FallbackHandler",
    "RAGPipeline",
    "get_rag_pipeline",
    "reset_rag_pipeline",
]
