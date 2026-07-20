"""MinerUParser 单元测试"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMinerUParserRegistry:
    """注册表测试"""

    def test_get_parser_returns_mineru_instance(self):
        from ingestion.parsers import get_parser, reset_parser_cache

        reset_parser_cache()
        parser = get_parser("mineru")
        from ingestion.parsers.mineru_parser import MinerUParser

        assert isinstance(parser, MinerUParser)
        assert parser.name == "mineru"
        assert parser.supported_formats == ("pdf",)

    def test_mineru_is_registered(self):
        from ingestion.parsers import _PARSER_CLASSES

        assert "mineru" in _PARSER_CLASSES


class TestMinerUParserLazyImport:
    """延迟导入测试"""

    def test_instantiation_without_magic_pdf(self):
        from ingestion.parsers.mineru_parser import MinerUParser

        MinerUParser._initialized = False
        parser = MinerUParser()
        assert parser.name == "mineru"
        assert MinerUParser._initialized is False

    def test_parse_without_magic_pdf_raises_clear_error(self):
        from ingestion.parsers.mineru_parser import MinerUParser

        MinerUParser._initialized = False
        parser = MinerUParser()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake")
            tmp_path = Path(f.name)

        try:
            with patch.object(
                MinerUParser,
                "_ensure_initialized",
                side_effect=ImportError(
                    "MinerU (magic-pdf) 未安装。请运行: pip install magic-pdf[full-cpu]"
                ),
            ):
                with pytest.raises(ImportError, match="magic-pdf"):
                    parser.parse(tmp_path)
        finally:
            tmp_path.unlink()


class TestMinerUParserMockedParse:
    """Mock MinerU 外部依赖的 parse() 测试"""

    @pytest.fixture(autouse=True)
    def reset_mineru_state(self):
        from ingestion.parsers import reset_parser_cache
        from ingestion.parsers.mineru_parser import MinerUParser

        reset_parser_cache()
        MinerUParser._initialized = False
        yield
        reset_parser_cache()
        MinerUParser._initialized = False

    @staticmethod
    def _make_mock_modules(use_ocr=True):
        """构建 mock magic_pdf 模块并注入 sys.modules"""
        mock_pdf = MagicMock()
        mock_data = MagicMock()
        mock_rw = MagicMock()
        mock_dataset = MagicMock()
        mock_doc = MagicMock()
        mock_model = MagicMock()
        mock_enums = MagicMock()
        mock_pdf.data = mock_data
        mock_pdf.model = mock_model
        mock_pdf.config = MagicMock()
        mock_pdf.config.enums = mock_enums
        mock_data.data_reader_writer = mock_rw
        mock_data.dataset = mock_dataset
        mock_model.doc_analyze_by_custom_model = mock_doc

        if use_ocr:
            mock_enums.SupportedPdfParseMethod = MagicMock()
            mock_enums.SupportedPdfParseMethod.OCR = "ocr"

        return {
            "magic_pdf": mock_pdf,
            "magic_pdf.data": mock_data,
            "magic_pdf.data.data_reader_writer": mock_rw,
            "magic_pdf.data.dataset": mock_dataset,
            "magic_pdf.model": mock_model,
            "magic_pdf.model.doc_analyze_by_custom_model": mock_doc,
            "magic_pdf.config": mock_pdf.config,
            "magic_pdf.config.enums": mock_enums,
        }

    def _setup_pipeline_mocks(self, mock_modules, md_content, out_dir):
        """装配 mock pipeline：Dataset → InferenceResult → PipeResult"""
        mock_modules["magic_pdf.data.data_reader_writer"].FileBasedDataWriter = \
            lambda p: MagicMock()
        mock_modules["magic_pdf.model.doc_analyze_by_custom_model"].doc_analyze = \
            MagicMock()

        mock_pipe_result = MagicMock()
        mock_pipe_result.get_markdown = lambda image_dir: md_content
        mock_pipe_result.dump_md = lambda w, f, d: (out_dir / f).write_text(
            md_content, encoding="utf-8"
        )

        mock_infer_result = MagicMock()
        mock_infer_result.pipe_ocr_mode = lambda iw: mock_pipe_result
        mock_infer_result.pipe_txt_mode = lambda iw: mock_pipe_result

        mock_ds = MagicMock()
        mock_ds.classify.return_value = "ocr"
        mock_ds.apply.return_value = mock_infer_result

        mock_modules["magic_pdf.data.dataset"].PymuDocDataset = \
            lambda pdf_bytes: mock_ds

        return mock_ds

    def test_parse_returns_markdown_via_get_markdown(self):
        """验证 parse() 通过 get_markdown() 返回 Markdown"""
        from ingestion.parsers.mineru_parser import MinerUParser

        md_content = "# 测试文档\n\n![图片](test_images/img_001.png)\n\n正文内容。"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "parsed_docs"
            out_dir.mkdir()
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            mock_modules = self._make_mock_modules()
            self._setup_pipeline_mocks(mock_modules, md_content, out_dir)

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                result = parser.parse(pdf_path, output_dir=out_dir)

            assert result == md_content
            assert "test_images" in result

    def test_images_dir_created(self):
        """验证图片目录被创建"""
        from ingestion.parsers.mineru_parser import MinerUParser

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "parsed_docs"
            out_dir.mkdir()
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            mock_modules = self._make_mock_modules()
            self._setup_pipeline_mocks(mock_modules, "# test", out_dir)

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                parser.parse(pdf_path, output_dir=out_dir)

            images_dir = out_dir / "test_images"
            assert images_dir.exists()
            assert images_dir.is_dir()

    def test_output_dir_none_falls_back_to_source_parent(self):
        """验证 output_dir=None 时图片写入源文件同级目录"""
        from ingestion.parsers.mineru_parser import MinerUParser

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            mock_modules = self._make_mock_modules()
            self._setup_pipeline_mocks(mock_modules, "# test", Path(tmpdir))

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                parser.parse(pdf_path, output_dir=None)

            images_dir = Path(tmpdir) / "test_images"
            assert images_dir.exists()

    def test_fallback_to_dump_md_when_get_markdown_missing(self):
        """验证 get_markdown 不存在时 fallback 到 dump_md + 文件读取"""
        from ingestion.parsers.mineru_parser import MinerUParser

        md_content = "# fallback test"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "parsed_docs"
            out_dir.mkdir()
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            mock_modules = self._make_mock_modules()
            mock_modules["magic_pdf.data.data_reader_writer"].FileBasedDataWriter = \
                lambda p: MagicMock()
            mock_modules["magic_pdf.model.doc_analyze_by_custom_model"].doc_analyze = \
                MagicMock()

            # pipe_result 只有 dump_md，没有 get_markdown
            class FakePipeResult:
                def dump_md(self, writer, filename, image_dir):
                    (out_dir / filename).write_text(md_content, encoding="utf-8")

            mock_pr = FakePipeResult()

            mock_ir = MagicMock()
            mock_ir.pipe_ocr_mode = lambda iw: mock_pr
            mock_ir.pipe_txt_mode = lambda iw: mock_pr

            mock_ds = MagicMock()
            mock_ds.classify.return_value = "ocr"
            mock_ds.apply.return_value = mock_ir
            mock_modules["magic_pdf.data.dataset"].PymuDocDataset = \
                lambda pdf_bytes: mock_ds

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                result = parser.parse(pdf_path, output_dir=out_dir)

            assert result == md_content

    def test_missing_md_file_raises_runtime_error(self):
        """验证 MinerU dump_md 未产出 .md 文件时抛出 RuntimeError"""
        from ingestion.parsers.mineru_parser import MinerUParser

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "parsed_docs"
            out_dir.mkdir()
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 fake")

            mock_modules = self._make_mock_modules()
            mock_modules["magic_pdf.data.data_reader_writer"].FileBasedDataWriter = \
                lambda p: MagicMock()
            mock_modules["magic_pdf.model.doc_analyze_by_custom_model"].doc_analyze = \
                MagicMock()

            mock_pipe_result = MagicMock()
            del mock_pipe_result.get_markdown  # 无此属性
            mock_pipe_result.dump_md = lambda w, f, d: None  # 不写入文件

            mock_infer_result = MagicMock()
            mock_infer_result.pipe_ocr_mode = lambda iw: mock_pipe_result
            mock_infer_result.pipe_txt_mode = lambda iw: mock_pipe_result

            mock_ds = MagicMock()
            mock_ds.classify.return_value = "ocr"
            mock_ds.apply.return_value = mock_infer_result
            mock_modules["magic_pdf.data.dataset"].PymuDocDataset = \
                lambda pdf_bytes: mock_ds

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                with pytest.raises(RuntimeError, match="未找到输出的 Markdown"):
                    parser.parse(pdf_path, output_dir=out_dir)

    def test_txt_mode_used_for_non_ocr_pdf(self):
        """验证非 OCR PDF 使用 pipe_txt_mode"""
        from ingestion.parsers.mineru_parser import MinerUParser

        md_content = "# txt mode doc"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "parsed_docs"
            out_dir.mkdir()
            pdf_path = Path(tmpdir) / "test.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 text native")

            mock_modules = self._make_mock_modules(use_ocr=False)
            mock_modules["magic_pdf.data.data_reader_writer"].FileBasedDataWriter = \
                lambda p: MagicMock()
            mock_modules["magic_pdf.model.doc_analyze_by_custom_model"].doc_analyze = \
                MagicMock()

            mock_pipe_result = MagicMock()
            mock_pipe_result.get_markdown = lambda image_dir: md_content

            mock_infer_result = MagicMock()
            mock_infer_result.pipe_txt_mode = MagicMock(return_value=mock_pipe_result)
            mock_infer_result.pipe_ocr_mode = MagicMock()

            mock_ds = MagicMock()
            mock_ds.classify.return_value = "txt"
            mock_ds.apply.return_value = mock_infer_result
            mock_modules["magic_pdf.data.dataset"].PymuDocDataset = \
                lambda pdf_bytes: mock_ds

            with patch.dict(sys.modules, mock_modules):
                MinerUParser._initialized = True
                parser = MinerUParser()
                result = parser.parse(pdf_path, output_dir=out_dir)

            assert result == md_content
            mock_infer_result.pipe_txt_mode.assert_called_once()
            mock_infer_result.pipe_ocr_mode.assert_not_called()


class TestMinerUConfig:
    """配置模型测试"""

    def test_default_config_values(self):
        from config import settings

        cfg = settings.ingestion.mineru
        assert cfg.device == "cpu"
        assert cfg.models_dir == "local_models/mineru"
