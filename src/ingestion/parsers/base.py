"""BaseParser — 文档解析器抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """解析器基类 — parse() 是同步方法，由 ParserStage 通过 run_in_executor 调用"""

    name: str = ""  # "docling", "pymupdf4llm", "direct"
    supported_formats: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source_path: Path) -> str:
        """解析文档，返回 Markdown 文本（同步方法，在 executor 线程中执行）"""
        ...
