"""DirectParser — 直接读取 .md/.txt 等纯文本文件"""

from pathlib import Path

from ingestion.parsers.base import BaseParser


class DirectParser(BaseParser):
    """直接读取纯文本/Markdown 文件，不经过任何转换"""

    name = "direct"
    supported_formats = ("md", "markdown", "txt")

    def parse(self, source_path: Path) -> str:
        return source_path.read_text(encoding="utf-8")
