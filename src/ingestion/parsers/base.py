"""BaseParser — 文档解析器抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """解析器基类 — parse() 是同步方法，由 ParserStage 通过 to_thread 调用"""

    name: str = ""  # "docling", "pymupdf4llm", "direct", "mineru"
    supported_formats: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source_path: Path, output_dir: Path | None = None) -> str:
        """解析文档，返回 Markdown 文本（同步方法，通过 to_thread 在线程中执行）

        Args:
            source_path: 源文档路径
            output_dir: 产物输出目录（图片等）。默认为 None，仅需写入额外产物
                        的解析器（如 MinerU）使用此参数。
        """
        ...
