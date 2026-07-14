"""查询理解层 — 意图分类 / 清晰度判断 / 上下文融合 / 查询改写"""
import threading

from logger import logger
from query.layer import QueryUnderstandingLayer
from query.intent_classifier import IntentResult

# 全局单例
_query_layer: QueryUnderstandingLayer | None = None
_lock = threading.Lock()
_init_llm_id: int | None = None
_init_sm_id: int | None = None


def get_query_layer(llm, session_manager) -> QueryUnderstandingLayer:
    """获取查询理解层全局单例

    首次调用时用传入的 llm/session_manager 初始化单例。
    后续调用若传入不同对象，会记录警告但仍返回已缓存的实例。
    """
    global _query_layer, _init_llm_id, _init_sm_id

    # 快速路径：已初始化，无锁检查
    if _query_layer is not None:
        if id(llm) != _init_llm_id or id(session_manager) != _init_sm_id:
            logger.warning(
                "get_query_layer 已初始化，忽略不同的 llm/session_manager 参数"
            )
        return _query_layer

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _query_layer is None:
            _query_layer = QueryUnderstandingLayer(llm, session_manager)
            _init_llm_id = id(llm)
            _init_sm_id = id(session_manager)
        return _query_layer


def reset_query_layer() -> None:
    """重置全局单例（测试用）"""
    global _query_layer, _init_llm_id, _init_sm_id
    with _lock:
        _query_layer = None
        _init_llm_id = None
        _init_sm_id = None


__all__ = [
    "QueryUnderstandingLayer",
    "IntentResult",
    "get_query_layer",
    "reset_query_layer",
]
