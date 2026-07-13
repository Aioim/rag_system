"""查询理解层 — 意图分类 / 清晰度判断 / 上下文融合 / 查询改写"""
from query.layer import QueryUnderstandingLayer
from query.intent_classifier import IntentResult

# 全局单例
_query_layer: QueryUnderstandingLayer | None = None


def get_query_layer(llm, session_manager) -> QueryUnderstandingLayer:
    """获取查询理解层全局单例"""
    global _query_layer
    if _query_layer is None:
        _query_layer = QueryUnderstandingLayer(llm, session_manager)
    return _query_layer


def reset_query_layer() -> None:
    """重置全局单例（测试用）"""
    global _query_layer
    _query_layer = None


__all__ = [
    "QueryUnderstandingLayer",
    "IntentResult",
    "get_query_layer",
    "reset_query_layer",
]
