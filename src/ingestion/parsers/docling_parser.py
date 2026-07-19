"""DoclingParser — 基于 docling 的多格式文档解析（PDF/Word/PPT/HTML）"""

import threading
from pathlib import Path

from ingestion.parsers.base import BaseParser


class DoclingParser(BaseParser):
    """使用 docling DocumentConverter 解析文档 → Markdown

    DocumentConverter 实例在类级别懒加载并以线程安全的双检锁模式缓存，
    跨所有 DoclingParser 实例共享。
    """

    name = "docling"
    supported_formats = ("pdf", "docx", "doc", "pptx", "ppt", "html")

    _converter: object | None = None
    _lock = threading.Lock()

    def parse(self, source_path: Path, output_dir: Path | None = None) -> str:
        if DoclingParser._converter is None:
            with DoclingParser._lock:
                if DoclingParser._converter is None:
                    from docling.document_converter import DocumentConverter

                    DoclingParser._converter = DocumentConverter()

        result = DoclingParser._converter.convert(str(source_path))
        return result.document.export_to_markdown()
