"""ReAct Agent 工具定义：SearchTool / WebSearchTool"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import settings
from models.chunk import Chunk

if TYPE_CHECKING:
    from fallback.web_search import WebSearcher
    from retrieval.layer import RetrievalLayer

_logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具调用结果"""
    tool: str           # "search" | "web_search"
    query: str          # 实际执行的 query
    content: str        # 搜索结果文本（格式化后）
    chunk_count: int    # 返回的 chunk 数量
    elapsed_ms: float   # 耗时


class SearchTool:
    """search(query) — 封装 RetrievalLayer，走完整混合检索"""

    def __init__(self, retrieval_layer: RetrievalLayer) -> None:
        self._retrieval = retrieval_layer

    async def run(self, query: str, collection: str) -> ToolResult:
        # 延迟导入避免循环引用（agent.tools → models.context → models.react_trace）
        from models.context import PipelineContext

        t0 = time.perf_counter()
        ctx = PipelineContext(query=query, collection=collection)
        ctx.rewritten_queries = [query]  # Agent 自行改写，不依赖 QueryRewriter
        try:
            ctx = await self._retrieval.retrieve(
                ctx, top_k=settings.agent.search_top_k
            )
        except Exception as e:
            _logger.warning("SearchTool 检索失败: %s", e)
            return ToolResult(
                tool="search", query=query, content="",
                chunk_count=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        chunks = ctx.reranked or []
        content = self._format_chunks(chunks)
        return ToolResult(
            tool="search", query=query, content=content,
            chunk_count=len(chunks),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _format_chunks(chunks: list[Chunk]) -> str:
        lines: list[str] = []
        for c in chunks:
            text = c.text.replace("\n", " ")[:settings.agent.max_observation_chars]
            lines.append(f"[来源: {c.doc_id}] {text}")
        return "\n".join(lines)


class WebSearchTool:
    """web_search(query) — 封装 WebSearcher"""

    def __init__(self, web_searcher: WebSearcher) -> None:
        self._searcher = web_searcher

    async def run(self, query: str) -> ToolResult:
        t0 = time.perf_counter()
        try:
            content = await self._searcher.search(query)
        except Exception as e:
            _logger.warning("WebSearchTool 失败: %s", e)
            content = ""

        return ToolResult(
            tool="web_search", query=query,
            content=content or "",
            chunk_count=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
