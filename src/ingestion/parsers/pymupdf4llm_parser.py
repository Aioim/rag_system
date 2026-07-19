"""PyMuPDF4LLMParser — 基于 PyMuPDF4LLM 的 PDF 解析"""

from pathlib import Path

from ingestion.parsers.base import BaseParser


class PyMuPDF4LLMParser(BaseParser):
    """使用 PyMuPDF4LLM 将 PDF 转换为 Markdown

    pymupdf4llm 在 parse() 中延迟导入，仅在实际使用此解析器时才需要安装。
    """

    name = "pymupdf4llm"
    supported_formats = ("pdf",)

    def parse(self, source_path: Path) -> str:
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyMuPDF4LLM 未安装。请运行: pip install rag-service[ingestion]"
            ) from None
        return pymupdf4llm.to_markdown(str(source_path))
