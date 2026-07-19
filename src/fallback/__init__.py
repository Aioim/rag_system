"""三级兜底模块 — 补充检索 → 联网搜索 → 诚实告知"""
import threading

from fallback.handler import FallbackHandler
from fallback.supplementary import SupplementaryRetriever
from fallback.web_search import WebSearcher

# 全局单例
_fallback_handler: FallbackHandler | None = None
_lock = threading.Lock()


def get_fallback_handler() -> FallbackHandler:
    """获取 FallbackHandler 全局单例

    首次调用时创建单例（含 WebSearcher + SupplementaryRetriever）。
    后续调用返回已缓存的实例。

    Returns:
        FallbackHandler 全局单例
    """
    global _fallback_handler

    if _fallback_handler is not None:
        return _fallback_handler

    with _lock:
        if _fallback_handler is None:
            _fallback_handler = FallbackHandler(
                web_searcher=WebSearcher(),
                supplementary=SupplementaryRetriever(),
            )
        return _fallback_handler


def reset_fallback_handler() -> None:
    """重置全局单例（测试用）"""
    global _fallback_handler
    with _lock:
        _fallback_handler = None


__all__ = [
    "FallbackHandler",
    "SupplementaryRetriever",
    "WebSearcher",
    "get_fallback_handler",
    "reset_fallback_handler",
]
