"""查询理解层 — 意图分类 / 清晰度判断 / 上下文融合 / 查询改写"""
import threading
from typing import TYPE_CHECKING

from logger import logger
from models.llm import LLMProtocol
from query.intent_classifier import IntentResult
from query.layer import QueryUnderstandingLayer

if TYPE_CHECKING:
    from session.manager import SessionManager

# 全局单例
_query_layer: QueryUnderstandingLayer | None = None
_lock = threading.Lock()
_init_llm: LLMProtocol | None = None
_init_sm: "SessionManager | None" = None


def get_query_layer(llm: LLMProtocol, session_manager: "SessionManager | None" = None) -> QueryUnderstandingLayer:
    """获取查询理解层全局单例

    首次调用时用传入的 llm/session_manager 初始化单例。
    后续调用若传入不同对象，会记录警告但仍返回已缓存的实例。
    """
    global _query_layer, _init_llm, _init_sm

    # 快速路径：已初始化，无锁检查
    if _query_layer is not None:
        if llm is not _init_llm or session_manager is not _init_sm:
            logger.warning(
                "get_query_layer 已初始化，忽略不同的 llm/session_manager 参数"
            )
        return _query_layer

    with _lock:
        # 双重检查：可能另一个线程刚完成初始化
        if _query_layer is None:
            _query_layer = QueryUnderstandingLayer(llm, session_manager)
            _init_llm = llm
            _init_sm = session_manager
        return _query_layer


def reset_query_layer() -> None:
    """重置全局单例（测试用）"""
    global _query_layer, _init_llm, _init_sm
    with _lock:
        _query_layer = None
        _init_llm = None
        _init_sm = None


__all__ = [
    "IntentResult",
    "QueryUnderstandingLayer",
    "get_query_layer",
    "reset_query_layer",
]
