"""Tool 定义单元测试：ToolResult / SearchTool / WebSearchTool"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools import SearchTool, ToolResult, WebSearchTool


class TestToolResult:
    def test_create_tool_result(self):
        r = ToolResult(tool="search", query="test", content="results",
                       chunk_count=3, elapsed_ms=100.0)
        assert r.tool == "search"
        assert r.chunk_count == 3


class TestSearchTool:
    @pytest.fixture
    def mock_retrieval_layer(self):
        layer = MagicMock()
        return layer

    @pytest.mark.asyncio
    async def test_run_returns_tool_result(self, mock_retrieval_layer):
        """search 正常返回时获得 ToolResult，内容包含来源标识"""
        from models.chunk import Chunk
        from models.enums import RetrievalEval

        chunk = Chunk(
            chunk_id="c1", doc_id="d1", text="RAG是检索增强生成",
            chunk_index=0, embedding=None
        )
        chunk.rerank_score = 0.9

        # 构造 PipelineContext 返回值（async side_effect 用 AsyncMock）
        mock_retrieval_layer.retrieve = AsyncMock()
        mock_retrieval_layer.retrieve.side_effect = [
            _make_ctx([chunk], RetrievalEval.SUFFICIENT)
        ]

        tool = SearchTool(mock_retrieval_layer)
        result = await tool.run("RAG架构", "default")

        assert result.tool == "search"
        assert result.chunk_count == 1
        assert "RAG是检索增强生成" in result.content
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_run_handles_exception(self, mock_retrieval_layer):
        """retrieval 异常时返回空内容，不抛异常"""
        mock_retrieval_layer.retrieve = AsyncMock(
            side_effect=RuntimeError("store error")
        )

        tool = SearchTool(mock_retrieval_layer)
        result = await tool.run("test", "default")

        assert result.content == ""
        assert result.chunk_count == 0


class TestWebSearchTool:
    @pytest.fixture
    def mock_web_searcher(self):
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value="联网搜索结果文本")
        return searcher

    @pytest.mark.asyncio
    async def test_run_returns_tool_result(self, mock_web_searcher):
        tool = WebSearchTool(mock_web_searcher)
        result = await tool.run("Python RAG")

        assert result.tool == "web_search"
        assert "联网搜索结果" in result.content

    @pytest.mark.asyncio
    async def test_run_handles_exception(self, mock_web_searcher):
        mock_web_searcher.search = AsyncMock(
            side_effect=RuntimeError("network error")
        )
        tool = WebSearchTool(mock_web_searcher)
        result = await tool.run("test")
        assert result.content == ""


def _make_ctx(reranked: list[Chunk], retrieval_eval: RetrievalEval) -> PipelineContext:
    """Helper：构造含指定 reranked 和 eval 值的 PipelineContext"""
    from models.chunk import Chunk
    from models.context import PipelineContext
    ctx = PipelineContext(query="test", collection="default")
    ctx.reranked = reranked
    ctx.retrieval_eval = retrieval_eval
    return ctx
