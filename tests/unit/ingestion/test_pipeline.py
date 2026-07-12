"""IngestionPipeline 编排器测试"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.context import Document, PipelineContext, StageError
from ingestion.pipeline import IngestionPipeline


class FakeFatalStage:
    """模拟 fatal 阶段：首次调用抛出异常"""
    name = "parser"
    fatal = True

    def __init__(self):
        self.call_count = 0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("文件损坏，无法解析")
        ctx.document.raw_text = "parsed"
        return ctx


class FakeNonFatalStage:
    """模拟非 fatal 阶段：出错后继续"""
    name = "embedder"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        raise RuntimeError("部分 batch 失败")


class FakeSuccessStage:
    """模拟正常阶段"""
    name = "chunker"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        from ingestion.context import Chunk
        ctx.chunks = [
            Chunk(chunk_id="c-1", doc_id=ctx.document.doc_id,
                  text="chunk text", chunk_index=0),
        ]
        return ctx


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        """正常流程：所有 stage 成功 + index_writer 被调用"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/test.md"))

        assert ctx.status == "done"
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].text == "chunk text"
        assert ctx.document.file_type == "md"
        assert ctx.document.title == "test"
        mock_writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_fatal_error_stops_pipeline(self):
        """fatal 错误应立即停止 pipeline，不调用 index_writer"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeFatalStage(), FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/bad.pdf"))

        assert ctx.status == "failed"
        assert len(ctx.errors) == 1
        assert ctx.errors[0].stage == "parser"
        assert "文件损坏" in ctx.errors[0].error
        # index_writer 不应被调用
        mock_writer.write.assert_not_called()
        # FakeSuccessStage 不应被执行
        assert ctx.chunks == []

    @pytest.mark.asyncio
    async def test_non_fatal_error_continues(self):
        """非 fatal 错误记录后继续执行"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage(), FakeNonFatalStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/doc.md"))

        assert ctx.status == "done"
        assert len(ctx.errors) == 1
        assert ctx.errors[0].stage == "embedder"
        assert ctx.errors[0].fatal is False
        mock_writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_records_stage_timing(self):
        """每个 stage 的耗时应记录到 ctx.metadata"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/tmp/doc.md"))

        assert "chunker_ms" in ctx.metadata
        assert isinstance(ctx.metadata["chunker_ms"], float)

    @pytest.mark.asyncio
    async def test_document_construction(self):
        """验证 Document 由 run() 从 file_path 自动构造"""
        mock_writer = MagicMock()
        pipeline = IngestionPipeline(
            stages=[FakeSuccessStage()],
            index_writer=mock_writer,
        )

        ctx = await pipeline.run(Path("/data/年度报告.docx"), collection="tech")

        assert ctx.document.file_type == "docx"
        assert ctx.document.title == "年度报告"
        assert ctx.document.collection == "tech"
        assert ctx.document.source_path == Path("/data/年度报告.docx")
