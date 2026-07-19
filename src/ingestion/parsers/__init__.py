"""可插拔文档解析器 — 注册表 + 工厂函数"""

from ingestion.parsers.base import BaseParser
from ingestion.parsers.direct_parser import DirectParser
from ingestion.parsers.docling_parser import DoclingParser
from ingestion.parsers.mineru_parser import MinerUParser
from ingestion.parsers.pymupdf4llm_parser import PyMuPDF4LLMParser

_PARSER_CLASSES: dict[str, type[BaseParser]] = {
    "docling": DoclingParser,
    "pymupdf4llm": PyMuPDF4LLMParser,
    "mineru": MinerUParser,
    "direct": DirectParser,
}

_instances: dict[str, BaseParser] = {}


def get_parser(name: str) -> BaseParser:
    """获取解析器实例（按名称缓存，懒初始化）"""
    if name not in _PARSER_CLASSES:
        raise ValueError(
            f"未知解析器: {name}，可用: {list(_PARSER_CLASSES)}"
        )
    if name not in _instances:
        _instances[name] = _PARSER_CLASSES[name]()
    return _instances[name]


def reset_parser_cache() -> None:
    """清理解析器实例缓存（测试用）"""
    _instances.clear()


__all__ = [
    "BaseParser",
    "DirectParser",
    "DoclingParser",
    "MinerUParser",
    "PyMuPDF4LLMParser",
    "get_parser",
    "reset_parser_cache",
]
