"""解析器注册表 + 各后端单元测试"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.parsers import (
    BaseParser,
    DirectParser,
    DoclingParser,
    PyMuPDF4LLMParser,
    get_parser,
    reset_parser_cache,
)


class TestRegistry:
    """解析器注册表 & 工厂函数测试"""

    def test_all_builtin_parsers_registered(self):
        """三个内置解析器均可通过 get_parser 获取"""
        for name in ("direct", "docling", "pymupdf4llm"):
            parser = get_parser(name)
            assert parser.name == name

    def test_get_parser_returns_cached_instance(self):
        """同一名称多次调用返回同一实例"""
        reset_parser_cache()
        a = get_parser("direct")
        b = get_parser("direct")
        assert a is b

    def test_get_parser_unknown_raises(self):
        """未知解析器名抛出 ValueError"""
        with pytest.raises(ValueError, match="未知解析器"):
            get_parser("nonexistent")

    def test_reset_parser_cache(self):
        """reset_parser_cache 后获取新实例"""
        reset_parser_cache()
        a = get_parser("direct")
        reset_parser_cache()
        b = get_parser("direct")
        assert a is not b

    def test_base_parser_is_abstract(self):
        """BaseParser 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseParser()  # type: ignore[abstract]


class TestDirectParser:
    """DirectParser 测试"""

    def test_name_and_formats(self):
        p = DirectParser()
        assert p.name == "direct"
        assert "md" in p.supported_formats
        assert "txt" in p.supported_formats

    def test_parse_markdown(self):
        p = DirectParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# 标题\n\n内容文本")
            tmp = Path(f.name)
        try:
            result = p.parse(tmp)
            assert "# 标题" in result
            assert "内容文本" in result
        finally:
            tmp.unlink()

    def test_parse_utf8(self):
        p = DirectParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("中文字符 ✅")
            tmp = Path(f.name)
        try:
            result = p.parse(tmp)
            assert "中文字符 ✅" in result
        finally:
            tmp.unlink()

    def test_parse_empty_file(self):
        p = DirectParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            tmp = Path(f.name)
        try:
            result = p.parse(tmp)
            assert result == ""
        finally:
            tmp.unlink()


class TestDoclingParser:
    """DoclingParser 测试（mock，不依赖 docling）"""

    def test_name_and_formats(self):
        p = DoclingParser()
        assert p.name == "docling"
        assert "pdf" in p.supported_formats
        assert "docx" in p.supported_formats

    def test_parse_calls_converter(self):
        """验证 parse() 调用 docling DocumentConverter"""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Mocked MD"
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(DoclingParser, "_converter", mock_converter):
            p = DoclingParser()
            result = p.parse(Path("/fake/test.pdf"))
            mock_converter.convert.assert_called_once_with(str(Path("/fake/test.pdf")))
            assert result == "# Mocked MD"

    def test_converter_singleton(self):
        """多次 parse() 复用同一个 converter 实例"""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "x"
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.object(DoclingParser, "_converter", mock_converter):
            p1 = DoclingParser()
            p2 = DoclingParser()
            p1.parse(Path("/a.pdf"))
            p2.parse(Path("/b.pdf"))
            assert mock_converter.convert.call_count == 2

    def test_converter_lazy_loaded(self):
        """converter 在首次 parse() 时才初始化"""
        saved = DoclingParser._converter
        DoclingParser._converter = None
        try:
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = "lazy"
            mock_converter = MagicMock()
            mock_converter.convert.return_value = mock_result

            # mock docling 模块级别的 DocumentConverter 类
            with patch.dict(
                "sys.modules",
                {"docling": MagicMock(), "docling.document_converter": MagicMock()},
            ):
                docling_mod = __import__("sys").modules["docling.document_converter"]
                docling_mod.DocumentConverter = MagicMock(return_value=mock_converter)

                p = DoclingParser()
                result = p.parse(Path("/test.pdf"))
                assert result == "lazy"
        finally:
            DoclingParser._converter = saved


class TestPyMuPDF4LLMParser:
    """PyMuPDF4LLMParser 测试（mock，不依赖 pymupdf4llm）"""

    def test_name_and_formats(self):
        p = PyMuPDF4LLMParser()
        assert p.name == "pymupdf4llm"
        assert "pdf" in p.supported_formats

    def test_parse_calls_to_markdown(self):
        """验证 parse() 调用 pymupdf4llm.to_markdown"""
        mock_pymupdf = MagicMock()
        mock_pymupdf.to_markdown.return_value = "# PDF Content"

        with patch.dict("sys.modules", {"pymupdf4llm": mock_pymupdf}):
            p = PyMuPDF4LLMParser()
            result = p.parse(Path("/fake/doc.pdf"))
            mock_pymupdf.to_markdown.assert_called_once_with(str(Path("/fake/doc.pdf")))
            assert result == "# PDF Content"

    def test_parse_import_error(self):
        """pymupdf4llm 未安装时抛出清晰的 ImportError"""
        with patch.dict("sys.modules", {"pymupdf4llm": None}):
            # 确保 import 会失败
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                p = PyMuPDF4LLMParser()
                with pytest.raises(ImportError, match="PyMuPDF4LLM"):
                    p.parse(Path("/fake/doc.pdf"))
