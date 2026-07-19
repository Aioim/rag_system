"""三级兜底处理 — 委托给 src/fallback/ 模块

保留此文件以维持向后兼容：from core.fallback import FallbackHandler 仍然可用。
"""
from fallback.handler import FallbackHandler
from fallback.supplementary import SupplementaryRetriever
from fallback.web_search import WebSearcher

__all__ = ["FallbackHandler", "SupplementaryRetriever", "WebSearcher"]
