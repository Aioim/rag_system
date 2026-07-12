"""ParserStage 测试"""

import tempfile
from pathlib import Path

import pytest

from ingestion.context import Document, PipelineContext
from ingestion.parser import ParserStage


class TestParserStage:
    @pytest.mark.asyncio
    async def test_parse_markdown_file(self):
        """解析 Markdown 文件"""
        stage = ParserStage()
        doc = Document(
            doc_id="d-md",
            source_path=Path("/dev/null"),  # 会被 ctx 中的覆盖
            file_type="md",
        )
        ctx = PipelineContext(document=doc)

        # 创建临时 markdown 文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# 测试标题\n\n这是一段**测试**内容。\n\n- 列表项 1\n- 列表项 2\n")
            tmp_path = Path(f.name)

        try:
            ctx.document.source_path = tmp_path
            result = await stage.run(ctx)
            assert "# 测试标题" in result.document.raw_text
            assert "测试" in result.document.raw_text
            assert len(result.document.raw_text) > 0
        finally:
            tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self):
        """解析不存在的文件应抛出异常（fatal）"""
        stage = ParserStage()
        doc = Document(
            doc_id="d-bad",
            source_path=Path("/nonexistent/file.pdf"),
            file_type="pdf",
        )
        ctx = PipelineContext(document=doc)

        with pytest.raises(Exception):
            await stage.run(ctx)

    @pytest.mark.asyncio
    async def test_name_and_fatal(self):
        """验证 Stage 元数据"""
        stage = ParserStage()
        assert stage.name == "parser"
        assert stage.fatal is True
