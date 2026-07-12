"""Stage 协议测试"""

from pathlib import Path

import pytest

from ingestion.context import Document, PipelineContext
from ingestion.stage import Stage


class FakeParserStage:
    """符合 Stage 协议的示例实现"""
    name = "parser"
    fatal = True

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.document.raw_text = "parsed content"
        ctx.current_stage = self.name
        return ctx


class FakeChunkerStage:
    """非 fatal 的 Stage"""
    name = "chunker"
    fatal = False

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.document.raw_text:
            from ingestion.context import StageError
            ctx.errors.append(StageError(stage=self.name, error="empty text"))
            return ctx
        ctx.current_stage = self.name
        return ctx


class TestStageProtocol:
    def test_structural_subtyping(self):
        """FakeParserStage 应该通过 Stage 协议的结构类型检查"""
        stage: Stage = FakeParserStage()
        assert stage.name == "parser"
        assert stage.fatal is True

    @pytest.mark.asyncio
    async def test_run_modifies_context(self):
        stage = FakeParserStage()
        doc = Document(doc_id="d-1", source_path=Path("test.pdf"), file_type="pdf")
        ctx = PipelineContext(document=doc)

        result = await stage.run(ctx)
        assert result.document.raw_text == "parsed content"
        assert result.current_stage == "parser"

    @pytest.mark.asyncio
    async def test_non_fatal_error_continues(self):
        stage = FakeChunkerStage()
        doc = Document(doc_id="d-1", source_path=Path("test.pdf"), file_type="pdf")
        ctx = PipelineContext(document=doc)

        result = await stage.run(ctx)
        assert len(result.errors) == 1
        assert result.errors[0].stage == "chunker"
