"""ParserStage 测试"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.context import Document, PipelineContext
from ingestion.parser import ParserStage


class TestParserStage:
    @pytest.mark.asyncio
    async def test_parse_markdown_file(self):
        """解析 Markdown 文件 — 默认使用 direct parser"""
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


class TestParserStageDispatch:
    """ParserStage 解析器调度测试"""

    @pytest.mark.asyncio
    async def test_metadata_includes_parser_name(self):
        """验证 ctx.document.metadata["parser"] 记录了解析器名称"""
        stage = ParserStage()
        doc = Document(
            doc_id="d-meta",
            source_path=Path("/dev/null"),
            file_type="md",
        )
        ctx = PipelineContext(document=doc)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("content")
            tmp_path = Path(f.name)

        try:
            ctx.document.source_path = tmp_path
            result = await stage.run(ctx)
            assert result.document.metadata["parser"] == "direct"
        finally:
            tmp_path.unlink()

    @pytest.mark.asyncio
    async def test_fallback_to_docling_for_unknown_format(self):
        """未知文件格式 fallback 到 docling parser（需 mock 避免真实调用）"""
        from ingestion.parsers import DoclingParser, reset_parser_cache

        reset_parser_cache()
        saved_converter = DoclingParser._converter
        mock_result = type("R", (), {"document": type("D", (), {"export_to_markdown": lambda self: "mock"})()})()  # noqa: E501
        mock_converter = type("C", (), {"convert": lambda self, p: mock_result})()
        DoclingParser._converter = mock_converter

        try:
            stage = ParserStage()
            doc = Document(
                doc_id="d-unknown",
                source_path=Path("/dev/null"),
                file_type="xyz",  # 不在配置中的格式
            )
            ctx = PipelineContext(document=doc)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xyz", delete=False, encoding="utf-8"
            ) as f:
                f.write("unknown")
                tmp_path = Path(f.name)

            try:
                ctx.document.source_path = tmp_path
                result = await stage.run(ctx)
                assert result.document.raw_text == "mock"
                assert result.document.metadata["parser"] == "docling"
            finally:
                tmp_path.unlink()
        finally:
            DoclingParser._converter = saved_converter

    @pytest.mark.asyncio
    async def test_configured_pdf_uses_direct_for_mock(self):
        """验证配置 pdf: direct 时，PDF 文件使用 DirectParser"""
        from ingestion.parsers import reset_parser_cache

        reset_parser_cache()
        stage = ParserStage()
        doc = Document(
            doc_id="d-pdf-direct",
            source_path=Path("/dev/null"),
            file_type="pdf",
        )
        ctx = PipelineContext(document=doc)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdf", delete=False, encoding="utf-8"
        ) as f:
            f.write("# mock pdf as direct")
            tmp_path = Path(f.name)

        try:
            # 通过全局 settings 单例临时覆盖 parsers 配置
            from config import settings as cfg

            cfg.initialize()  # 确保已初始化
            saved = cfg._config.ingestion
            mock_cfg = MagicMock()
            mock_cfg.parsers = {"pdf": "direct", "md": "direct"}
            mock_cfg.parsed_doc_dir = Path(tempfile.gettempdir()) / "test_parsed_docs"
            cfg._config.ingestion = mock_cfg

            try:
                ctx.document.source_path = tmp_path
                result = await stage.run(ctx)
                assert result.document.metadata["parser"] == "direct"
                assert "# mock pdf as direct" in result.document.raw_text
            finally:
                cfg._config.ingestion = saved
        finally:
            tmp_path.unlink()
            reset_parser_cache()
