"""检索层 — 混合召回 / RRF 融合 / 上下文扩展 / Rerank+MMR / Self-RAG 自评"""
import threading

from retrieval.layer import RetrievalLayer
from retrieval.store import reset_stores

# 全局单例
_retrieval_layer: RetrievalLayer | None = None
_lock = threading.Lock()


def get_retrieval_layer() -> RetrievalLayer:
    """获取检索层全局单例（模型懒加载，首次 retrieve 时初始化）"""
    global _retrieval_layer

    # 快速路径：已初始化，无锁检查
    if _retrieval_layer is not None:
        return _retrieval_layer

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _retrieval_layer is None:
            _retrieval_layer = RetrievalLayer()
        return _retrieval_layer


def reset_retrieval_layer() -> None:
    """重置全局单例并清空 store 缓存（测试用）"""
    global _retrieval_layer
    with _lock:
        _retrieval_layer = None
        reset_stores()


__all__ = [
    "RetrievalLayer",
    "get_retrieval_layer",
    "reset_retrieval_layer",
]
