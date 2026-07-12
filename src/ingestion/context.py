"""Ingestion Pipeline 数据模型 — Document、Chunk、PipelineContext、StageError"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """待处理的文档"""
    doc_id: str
    source_path: Path
    file_type: str                      # 源文件扩展名，第一期启用: pdf / docx / md
    title: str = ""
    raw_text: str = ""
    collection: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """分块后的文本片段"""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    context_summary: str | None = None  # Contextual Retrieval 预留
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class StageError:
    """Stage 执行错误"""
    stage: str
    error: str
    fatal: bool = False


@dataclass
class PipelineContext:
    """贯穿全链路的数据容器"""
    document: Document
    chunks: list[Chunk] = field(default_factory=list)
    current_stage: str = ""
    status: str = "pending"             # pending → running → done / failed
    errors: list[StageError] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
